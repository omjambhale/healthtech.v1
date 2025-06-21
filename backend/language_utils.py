from typing import List
import re
import os
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

def transcribe_audio(audio_file_bytes: bytes, filename: str) -> str:
    """
    Transcribe audio file to text using OpenAI Whisper.
    
    Args:
        audio_file_bytes: Raw bytes of audio file
        filename: Original filename for format detection
        
    Returns:
        Transcribed text
        
    Raises:
        RuntimeError: If transcription fails
    """
    try:
        # Create a temporary file-like object for OpenAI API
        import io
        audio_file = io.BytesIO(audio_file_bytes)
        audio_file.name = filename  # OpenAI needs filename for format detection
        
        # Use OpenAI Whisper for transcription
        response = openai.Audio.transcribe(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
        
        return response.strip()
        
    except Exception as e:
        raise RuntimeError(f"Audio transcription failed: {str(e)}")

def translate_text(text: str, target_language: str = "English") -> str:
    """
    Translate text to target language using OpenAI.
    
    Args:
        text: Text to translate
        target_language: Target language (default: English)
        
    Returns:
        Translated text
        
    Raises:
        RuntimeError: If translation fails
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a professional translator. Translate the following text to {target_language}. Preserve the meaning and tone. Return only the translation without any additional commentary."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            max_tokens=1000,
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        raise RuntimeError(f"Text translation failed: {str(e)}")

def speak_text(text: str) -> bytes:
    """
    Convert text to speech using OpenAI TTS.
    
    Args:
        text: Text to convert to speech
        
    Returns:
        Audio bytes (MP3 format)
        
    Raises:
        RuntimeError: If text-to-speech fails
    """
    try:
        response = openai.Audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text,
            response_format="mp3"
        )
        
        return response.content
        
    except Exception as e:
        raise RuntimeError(f"Text-to-speech failed: {str(e)}")

def detect_language(text: str) -> str:
    """
    Detect the language of the given text using OpenAI.
    
    Args:
        text: Text to analyze
        
    Returns:
        Detected language name
        
    Raises:
        RuntimeError: If language detection fails
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a language detection expert. Identify the language of the given text. Return only the language name (e.g., 'English', 'Spanish', 'French', etc.). Be concise."
                },
                {
                    "role": "user",
                    "content": f"What language is this text written in?\n\n{text[:500]}"  # Limit to first 500 chars
                }
            ],
            max_tokens=50,
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        raise RuntimeError(f"Language detection failed: {str(e)}")

def chunk_text(text: str, max_chars: int = 2000) -> List[str]:
    """
    Split long text into ~max_chars slices on sentence boundaries.
    
    Args:
        text: The text to chunk
        max_chars: Maximum characters per chunk (default 2000)
        
    Returns:
        List of text chunks, each approximately max_chars in length
    """
    if not text or not text.strip():
        return []
    
    # If text is shorter than max_chars, return as single chunk
    if len(text) <= max_chars:
        return [text.strip()]
    
    chunks = []
    
    # Split text into sentences using regex
    # This regex looks for sentence endings (.!?) followed by whitespace and capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If adding this sentence would exceed max_chars
        if len(current_chunk) + len(sentence) + 1 > max_chars:
            # If current_chunk has content, save it
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # If single sentence is longer than max_chars, split it by words
            if len(sentence) > max_chars:
                word_chunks = _split_by_words(sentence, max_chars)
                chunks.extend(word_chunks[:-1])  # Add all but last
                current_chunk = word_chunks[-1]  # Start new chunk with last part
            else:
                current_chunk = sentence
        else:
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    # Add final chunk if it has content
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def _split_by_words(text: str, max_chars: int) -> List[str]:
    """
    Helper function to split text by words when sentences are too long.
    
    Args:
        text: Text to split
        max_chars: Maximum characters per chunk
        
    Returns:
        List of word-based chunks
    """
    words = text.split()
    chunks = []
    current_chunk = ""
    
    for word in words:
        # If adding this word would exceed max_chars
        if len(current_chunk) + len(word) + 1 > max_chars:
            # Save current chunk if it has content
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = word
            else:
                # Single word is longer than max_chars
                chunks.append(word)
        else:
            # Add word to current chunk
            if current_chunk:
                current_chunk += " " + word
            else:
                current_chunk = word
    
    # Add final chunk if it has content
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks
