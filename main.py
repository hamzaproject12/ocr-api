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

@app.post("/extract-text/")
async def extract_text(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # 1. Extract raw text from the image
        raw_text = pytesseract.image_to_string(image)
        
        # 2. Define the exact JSON structure you want to return
        extracted_data = {
            "first_name": "",
            "last_name": "",
            "passport": "",
            "nationality": "",
            "date_of_birth": "",
            "sex": "",
            "date_of_expiry": "",
            "personal_id_number": ""
        }
        
        # 3. Aggressive Cleaning: Remove all spaces and replace common OCR error 'K' with '<'
        cleaned_lines = [line.replace(' ', '').replace('K', '<') for line in raw_text.split('\n')]
        
        line1 = ""
        line2 = ""
        
        # 4. Hunt for the MRZ lines anywhere in the document
        for line in cleaned_lines:
            # Line 1 always starts with P<
            if line.startswith('P<'):
                line1 = line
            # Line 2 usually starts with 9 alphanumeric characters followed by a digit and a 3-letter country code
            elif re.search(r'^[A-Z0-9]{9}[0-9][A-Z<]{3}', line):
                line2 = line

        # 5. Map the data to your JSON if the lines were found
        if line1:
            # P<MARLASTNAME<<FIRSTNAME<<<<<
            names_part = line1[5:].split('<<')
            if len(names_part) > 0:
                extracted_data["last_name"] = names_part[0].replace('<', '')
            if len(names_part) > 1:
                extracted_data["first_name"] = names_part[1].replace('<', '')

        if line2 and len(line2) >= 28:
            extracted_data["passport"] = line2[0:9].replace('<', '')
            extracted_data["nationality"] = line2[10:13].replace('<', '')
            extracted_data["date_of_birth"] = line2[13:19]
            extracted_data["sex"] = line2[20]
            extracted_data["date_of_expiry"] = line2[21:27]
            # Usually the Moroccan CNIE is in this section
            extracted_data["personal_id_number"] = line2[28:42].replace('<', '')

        # 6. Always return the formatted JSON
        return {
            "status": "success",
            "filename": file.filename,
            "data": extracted_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")