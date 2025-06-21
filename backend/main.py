from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv
from backend.file_processor import upload_to_s3, extract_text_from_s3
from backend.language_utils import transcribe_audio, translate_text, speak_text
from backend.model_config import summarize_text
import openai

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="HealthTech V1")

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.get("/ping")
async def ping():
    """Health check endpoint"""
    return {"message": "pong"}

@app.post("/process-file")
async def process_file(file: UploadFile = File(...)):
    """
    Process uploaded medical report file.
    Uploads to S3, extracts text via Textract, and generates summary.
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Read file bytes
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file provided")
        
        # Upload to S3
        try:
            s3_key = upload_to_s3(file_bytes, file.filename)
        except RuntimeError as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to upload file: {str(e)}"}
            )
        
        # Extract text from S3 using Textract
        try:
            raw_text = extract_text_from_s3(s3_key)
        except RuntimeError as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to extract text: {str(e)}"}
            )
        
        # Generate summary using model_config
        try:
            summary = summarize_text(raw_text)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to generate summary: {str(e)}"}
            )
        
        return {
            "raw_text": raw_text,
            "summary": summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Unexpected error: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
