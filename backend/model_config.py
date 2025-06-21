import os
import requests
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
USE_SARVAM = os.getenv("USE_SARVAM", "false").lower() == "true"
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_MODEL = os.getenv("SARVAM_MODEL", "sarvamai/sarvam-2b-v0.5")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# Configure OpenAI
openai.api_key = OPENAI_API_KEY

def summarize_text(text: str) -> str:
    """
    Summarize text using either Sarvam API or OpenAI based on environment configuration.
    
    Args:
        text: The text to summarize
        
    Returns:
        Summarized text
        
    Raises:
        RuntimeError: If summarization fails
    """
    if USE_SARVAM:
        return _summarize_with_sarvam(text)
    else:
        return _summarize_with_openai(text)

def _summarize_with_sarvam(text: str) -> str:
    """
    Summarize text using Sarvam API.
    
    Args:
        text: The text to summarize
        
    Returns:
        Summarized text
        
    Raises:
        RuntimeError: If Sarvam API call fails
    """
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY not found in environment variables")
    
    try:
        # Prepare the prompt for medical summarization
        prompt = f"""You are a medical assistant. Please provide a clear, concise summary of the following medical report or text. Focus on key findings, diagnoses, and recommendations. Keep it professional and easy to understand.

Text to summarize:
{text}

Summary:"""

        # Prepare request payload
        payload = {
            "model": SARVAM_MODEL,
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.7,
            "stream": False
        }
        
        headers = {
            "Authorization": f"Bearer {SARVAM_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Make API request
        response = requests.post(
            "https://api.sarvam.ai/api/infer",
            json=payload,
            headers=headers,
            timeout=30
        )
        
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        # Extract the generated text
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0].get("text", "").strip()
        elif "text" in result:
            return result["text"].strip()
        else:
            raise RuntimeError("Unexpected response format from Sarvam API")
            
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Sarvam API request failed: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Sarvam summarization failed: {str(e)}")

def _summarize_with_openai(text: str) -> str:
    """
    Summarize text using OpenAI ChatCompletion API.
    
    Args:
        text: The text to summarize
        
    Returns:
        Summarized text
        
    Raises:
        RuntimeError: If OpenAI API call fails
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not found in environment variables")
    
    try:
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical assistant. Provide a clear, concise summary of medical reports or health-related text. Focus on key findings, diagnoses, and recommendations. Keep it professional and easy to understand."
                },
                {
                    "role": "user",
                    "content": f"Please summarize this medical text:\n\n{text}"
                }
            ],
            max_tokens=512,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        raise RuntimeError(f"OpenAI summarization failed: {str(e)}")
