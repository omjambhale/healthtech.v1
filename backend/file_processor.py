import boto3
import os
import uuid
import time
from botocore.exceptions import ClientError, NoCredentialsError

# Read AWS configuration from environment variables
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET = os.getenv("S3_BUCKET")

# Initialize AWS clients only if credentials are available
s3 = None
textract = None

# Only initialize clients if AWS configuration is available
if AWS_REGION and S3_BUCKET:
    try:
        s3 = boto3.client("s3", region_name=AWS_REGION)
        textract = boto3.client("textract", region_name=AWS_REGION)
        print(f"✅ AWS clients initialized for region: {AWS_REGION}")
    except Exception as e:
        print(f"⚠️  Failed to initialize AWS clients: {e}")
        s3 = None
        textract = None
else:
    print("⚠️  AWS credentials not configured. S3/Textract functionality will be disabled.")


def upload_to_s3(file_bytes: bytes, filename: str) -> str:
    """Upload file to S3 under reports/..., return key."""
    
    # Check if S3 client is initialized
    if s3 is None:
        raise RuntimeError("S3 client not initialized. This code must run in the AWS sandbox with proper credentials.")
    
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET environment variable not set. This code must run in the AWS sandbox.")
    
    try:
        # Generate unique key with reports/ prefix
        unique_id = str(uuid.uuid4())
        file_extension = filename.split('.')[-1] if '.' in filename else 'bin'
        key = f"reports/{unique_id}_{filename}"
        
        # Upload file to S3
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=file_bytes,
            ContentType=_get_content_type(file_extension)
        )
        
        return key
        
    except NoCredentialsError:
        raise RuntimeError("AWS credentials not found. This code must run in the AWS sandbox.")
    except ClientError as e:
        raise RuntimeError(f"Failed to upload file to S3: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during S3 upload: {e}")


def extract_text_from_s3(key: str) -> str:
    """Start a Textract job on the given S3 key, poll until done, collect all LINE blocks' text, and return as one string."""
    
    # Check if Textract client is initialized
    if textract is None:
        raise RuntimeError("Textract client not initialized. This code must run in the AWS sandbox with proper credentials.")
    
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET environment variable not set. This code must run in the AWS sandbox.")
    
    try:
        # Start document text detection job
        response = textract.start_document_text_detection(
            DocumentLocation={
                'S3Object': {
                    'Bucket': S3_BUCKET,
                    'Name': key
                }
            }
        )
        
        job_id = response['JobId']
        
        # Poll for job completion
        max_attempts = 60  # 5 minutes with 5-second intervals
        attempt = 0
        
        while attempt < max_attempts:
            try:
                result = textract.get_document_text_detection(JobId=job_id)
                status = result['JobStatus']
                
                if status == 'SUCCEEDED':
                    # Collect all text from LINE blocks
                    text_lines = []
                    
                    # Process first page of results
                    for block in result.get('Blocks', []):
                        if block['BlockType'] == 'LINE':
                            text_lines.append(block['Text'])
                    
                    # Handle pagination if there are more results
                    while 'NextToken' in result:
                        result = textract.get_document_text_detection(
                            JobId=job_id,
                            NextToken=result['NextToken']
                        )
                        
                        for block in result.get('Blocks', []):
                            if block['BlockType'] == 'LINE':
                                text_lines.append(block['Text'])
                    
                    return '\n'.join(text_lines)
                
                elif status == 'FAILED':
                    error_msg = result.get('StatusMessage', 'Unknown error')
                    raise RuntimeError(f"Textract job failed: {error_msg}")
                
                elif status == 'IN_PROGRESS':
                    # Wait before polling again
                    time.sleep(5)
                    attempt += 1
                    continue
                    
            except ClientError as e:
                if attempt < max_attempts - 1:
                    time.sleep(5)
                    attempt += 1
                    continue
                else:
                    raise RuntimeError(f"Failed to get Textract job status: {e}")
        
        raise RuntimeError("Textract job timed out - took longer than expected to complete")
        
    except ClientError as e:
        raise RuntimeError(f"Failed to start Textract job: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during text extraction: {e}")


def _get_content_type(file_extension: str) -> str:
    """Helper function to determine content type based on file extension."""
    content_types = {
        'pdf': 'application/pdf',
        'png': 'image/png',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'tiff': 'image/tiff',
        'tif': 'image/tiff'
    }
    return content_types.get(file_extension.lower(), 'application/octet-stream')
