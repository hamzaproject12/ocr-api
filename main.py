from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
from PIL import Image
import io
import re

app = FastAPI(title="Passport OCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def parse_mrz(mrz_lines):
    """
    Slices the two 44-character MRZ lines into standard passport fields.
    """
    line1, line2 = mrz_lines
    
    # Extract Names from Line 1
    # Format: P<MAR[LASTNAME]<<[FIRSTNAME]<<<<<<...
    names_part = line1[5:].split('<<')
    last_name = names_part[0].replace('<', ' ').strip()
    first_name = names_part[1].replace('<', ' ').strip() if len(names_part) > 1 else ""

    # Extract Data from Line 2
    passport_number = line2[0:9].replace('<', '')
    nationality = line2[10:13]
    dob_yymmdd = line2[13:19]
    sex = line2[20]
    expiry_yymmdd = line2[21:27]
    personal_number = line2[28:42].replace('<', '') # Usually contains the CNIE in Morocco

    return {
        "last_name": last_name,
        "first_name": first_name,
        "passport_number": passport_number,
        "nationality": nationality,
        "date_of_birth": dob_yymmdd,
        "sex": sex,
        "date_of_expiry": expiry_yymmdd,
        "personal_id_number": personal_number
    }

@app.post("/extract-text/")
async def extract_text(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Extract raw text from the image
        raw_text = pytesseract.image_to_string(image)
        
        # Clean up the text: remove spaces and find lines with exactly 44 characters 
        # containing only uppercase letters, numbers, and the '<' symbol.
        cleaned_text = raw_text.replace(' ', '')
        mrz_matches = re.findall(r'([A-Z0-9<]{44})', cleaned_text)

        if len(mrz_matches) >= 2:
            # Pass the last two valid MRZ lines to our parser
            extracted_data = parse_mrz(mrz_matches[-2:])
            return {
                "status": "success",
                "filename": file.filename,
                "data": extracted_data
            }
        else:
             return {
                "status": "partial_success",
                "message": "Could not cleanly read the MRZ lines at the bottom of the passport. Here is the raw text instead.",
                "raw_text": raw_text
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")