import io
import logging
import os
from datetime import date

import pypdfium2 as pdfium
import pytesseract
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps

import mrz

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_PAGES = 10
PDF_RENDER_DPI = 300
# The machine-readable zone is printed in OCR-B: a single block of uppercase
# text, so a whitelist and a block layout give Tesseract far less room to guess.
MRZ_OCR_CONFIG = "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"

logger = logging.getLogger("passport-ocr")

app = FastAPI(title="Passport OCR API")

# "*" with credentials is rejected by browsers, so credentials are only enabled
# when ALLOWED_ORIGINS names the sites explicitly.
origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

EMPTY_FIELDS = {
    "first_name": "",
    "last_name": "",
    "passport": "",
    "nationality": "",
    "date_of_birth": "",
    "sex": "",
    "date_of_expiry": "",
    "personal_id_number": "",
}


def _prepare(image: Image.Image) -> Image.Image:
    """Straighten and upscale an image so the MRZ is large enough to read."""
    image = ImageOps.exif_transpose(image).convert("L")
    if image.width < 1600:
        ratio = 1600 / image.width
        image = image.resize((1600, max(1, round(image.height * ratio))), Image.LANCZOS)
    return image


def _score(result: dict | None) -> int:
    if result is None:
        return -1
    return sum(1 for ok in result["checks"].values() if ok)


def _read_orientation(image: Image.Image) -> dict | None:
    """OCR one orientation with default then MRZ-tuned settings; return the better read."""
    result = mrz.parse(pytesseract.image_to_string(image))
    if result is None or not all(result["checks"].values()):
        second = mrz.parse(pytesseract.image_to_string(image, config=MRZ_OCR_CONFIG))
        if _score(second) > _score(result):
            result = second
    return result


def _ocr(image: Image.Image) -> dict | None:
    """OCR the image; try 90/180/270-degree rotations when the first read is unverified."""
    prepared = _prepare(image)
    best = _read_orientation(prepared)
    # Scanned or rotated PDF pages often place the MRZ vertically. Only rotate if
    # what we have is missing checks - a fully-verified read is trusted as-is.
    if best is not None and all(best["checks"].values()):
        return best
    for angle in (270, 90, 180):
        candidate = _read_orientation(prepared.rotate(angle, expand=True))
        if _score(candidate) > _score(best):
            best = candidate
            if all(best["checks"].values()):
                break
    return best


def _pdf_pages(document: "pdfium.PdfDocument"):
    """Yield (page number, embedded text, renderer) for each page of a PDF."""
    try:
        for index in range(min(len(document), MAX_PAGES)):
            page = document[index]
            text_page = page.get_textpage()
            try:
                embedded = text_page.get_text_range()
            finally:
                text_page.close()
            # Rendering is deferred: a PDF generated digitally already carries
            # the MRZ as text, and never needs to be rasterised and OCR'd.
            yield index + 1, embedded, lambda p=page: p.render(
                scale=PDF_RENDER_DPI / 72
            ).to_pil()
    finally:
        document.close()


def _pages(data: bytes, filename: str):
    """Return the pages of the upload as (page number, embedded text, renderer)."""
    if data[:5] == b"%PDF-":
        try:
            return _pdf_pages(pdfium.PdfDocument(data))
        except Exception:
            raise HTTPException(status_code=400, detail=f"Could not read the PDF '{filename}'.")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file '{filename}': expected an image or a PDF.",
        )
    return [(1, None, lambda: image)]


@app.get("/health")
def health():
    try:
        version = str(pytesseract.get_tesseract_version())
    except Exception as error:
        return {"status": "degraded", "tesseract": None, "detail": str(error)}
    return {"status": "ok", "tesseract": version}


@app.post("/extract-text/")
async def extract_text(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    try:
        result, page_number = None, None
        for number, embedded, render in _pages(data, file.filename):
            result = mrz.parse(embedded) if embedded else None
            if result is None:
                result = _ocr(render())
            if result is not None:
                page_number = number
                break
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("OCR failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=f"An error occurred: {error}")

    return _response(file.filename, result, page_number)


def _response(filename: str | None, result: dict | None, page_number: int | None) -> dict:
    """Build the reply, keeping the original `data` keys the front end reads."""
    payload = {
        "status": "success",
        "filename": filename,
        "data": dict(EMPTY_FIELDS),
        "mrz_found": result is not None,
        "document_format": None,
        "page": page_number,
        "checks": {},
        "valid": False,
        "expired": None,
        "mrz": [],
    }
    if result is None:
        return payload

    fields = result["fields"]
    birth_iso = mrz.to_iso(fields["date_of_birth"], past=True)
    expiry_iso = mrz.to_iso(fields["date_of_expiry"], past=False)

    payload["data"] = {
        **fields,
        # Added alongside the raw YYMMDD values, which keep their original format.
        "date_of_birth_iso": birth_iso,
        "date_of_expiry_iso": expiry_iso,
    }
    payload["document_format"] = result["document_format"]
    payload["checks"] = result["checks"]
    payload["valid"] = all(result["checks"].values())
    payload["expired"] = expiry_iso < date.today().isoformat() if expiry_iso else None
    payload["mrz"] = result["mrz"]
    return payload
