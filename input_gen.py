import modal
import os
import sqlite3
import uuid
from typing import Optional
import torch
from fasthtml.common import (
    fast_app, H1, P, Div, Form, Input, Button, Group,
    Title, Main
)

# Core PDF support - required
import PyPDF2

# Optional format support
import whisper

from langchain_community.document_loaders import WebBaseLoader

class BaseIngestor:
    """Base class for all ingestors"""
    def validate(self, source: str) -> bool:
        pass
    def extract_text(self, source: str, max_chars: int = 100000) -> Optional[str]:
        pass

class PDFIngestor(BaseIngestor):
    """PDF ingestion - core functionality"""
    def validate(self, file_path: str) -> bool:
        if not os.path.exists(file_path):
            print(f"Error: File not found at path: {file_path}")
            return False
        if not file_path.lower().endswith('.pdf'):
            print("Error: File is not a PDF")
            return False
        return True
    def extract_text(self, file_path: str, max_chars: int = 100000) -> Optional[str]:
        if not self.validate(file_path):
            return None
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                print(f"Processing PDF with {num_pages} pages...")
                extracted_text = []
                total_chars = 0
                for page_num in range(num_pages):
                    page_text = pdf_reader.pages[page_num].extract_text()
                    if page_text:
                        total_chars += len(page_text)
                        if total_chars > max_chars:
                            remaining_chars = max_chars - total_chars
                            extracted_text.append(page_text[:remaining_chars])
                            print(f"Reached {max_chars} character limit at page {page_num + 1}")
                            break
                        extracted_text.append(page_text)
                        print(f"Processed page {page_num + 1}/{num_pages}")
                return "\n".join(extracted_text)
        except PyPDF2.PdfReadError:
            print("Error: Invalid or corrupted PDF file")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            return None

class WebsiteIngestor(BaseIngestor):
    """Website ingestion using LangChain's WebBaseLoader"""
    def validate(self, url: str) -> bool:
        if not url.startswith(('http://', 'https://')):
            print("Error: Invalid URL format")
            return False
        return True
    def extract_text(self, url: str, max_chars: int = 100000) -> Optional[str]:
        if not self.validate(url):
            return None
        try:
            loader = WebBaseLoader(url)
            documents = loader.load()
            extracted_text = "\n".join([doc.page_content for doc in documents])
            if len(extracted_text) > max_chars:
                extracted_text = extracted_text[:max_chars]
                print(f"Truncated extracted text to {max_chars} characters")
            print(f"Extracted text from website: {url}")
            return extracted_text
        except Exception as e:
            print(f"An error occurred while extracting from website: {str(e)}")
            return None

class AudioIngestor(BaseIngestor):
    """Audio ingestion using OpenAI's Whisper model"""
    def __init__(self, model_type: str = "base"):
        self.model_type = model_type
        self.model = whisper.load_model(self.model_type)
    def validate(self, audio_file: str) -> bool:
        if not os.path.exists(audio_file):
            print(f"Error: Audio file not found at path: {audio_file}")
            return False
        if not audio_file.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
            print("Error: Unsupported audio format. Supported formats are .mp3, .wav, .flac, .m4a")
            return False
        return True
    def extract_text(self, audio_file: str, max_chars: int = 100000) -> Optional[str]:
        if not self.validate(audio_file):
            return None
        try:
            result = self.model.transcribe(audio_file)
            transcription = result["text"]
            if len(transcription) > max_chars:
                transcription = transcription[:max_chars]
                print(f"Truncated transcription to {max_chars} characters")
            print(f"Transcribed audio file: {audio_file}")
            return transcription
        except Exception as e:
            print(f"An error occurred during audio transcription: {str(e)}")
            return None

class IngestorFactory:
    """Factory to create appropriate ingestor based on input type"""
    @staticmethod
    def get_ingestor(input_type: str, **kwargs) -> Optional[BaseIngestor]:
        input_type = input_type.lower()
        if input_type == "pdf":
            return PDFIngestor()
        elif input_type == "website":
            return WebsiteIngestor()
        elif input_type == "audio":
            return AudioIngestor(**kwargs)
        else:
            print(f"Unsupported input type: {input_type}")
            return None

# Create Modal app and configure image
app = modal.App("content_injection")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "PyPDF2",
        "python-fasthtml==0.12.0",
        "langchain",
        "langchain-community",
        "openai-whisper",
        "beautifulsoup4",
        "requests",
        "pydub"
    )
)

# Create/lookup volume for persistent storage
try:
    content_volume = modal.Volume.lookup("content_volume", create_if_missing=True)
except modal.exception.NotFoundError:
    content_volume = modal.Volume.persisted("content_volume")

UPLOAD_DIR = "/data/uploads"
OUTPUT_DIR = "/data/processed"
DB_PATH = "/data/injections.db"

def setup_database(db_path: str):
    """Initialize SQLite database for tracking injections"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS injections (
            id TEXT PRIMARY KEY,
            original_filename TEXT NOT NULL,
            input_type TEXT NOT NULL,
            processed_path TEXT,
            status TEXT DEFAULT 'pending',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    return conn

def get_input_type(filename: str) -> str:
    """Determine input type based on file extension or URL pattern"""
    lower_filename = filename.lower()
    
    if lower_filename.endswith('.pdf'):
        return 'pdf'
    elif lower_filename.endswith(('.mp3', '.wav', '.m4a', '.flac')):
        return 'audio'
    elif lower_filename.startswith(('http://', 'https://')):
        return 'website'
    elif lower_filename.endswith(('.txt', '.md')):
        return 'text'
    else:
        raise ValueError(f"Unsupported file type: {filename}")

def process_content(
    source_path: str,
    input_type: str,
    max_chars: int = 100000
) -> Optional[str]:
    """Process content using appropriate ingestor"""
    try:
        ingestor = IngestorFactory.get_ingestor(input_type)
        if not ingestor:
            raise ValueError(f"No ingestor found for type: {input_type}")
        
        return ingestor.extract_text(source_path, max_chars)
    except Exception as e:
        print(f"Error processing content: {str(e)}")
        return None

@app.function(
    image=image,
    gpu=modal.gpu.T4(count=1),
    volumes={"/data": content_volume},
    timeout=3600
)
@modal.asgi_app()
def serve():
    # Initialize directories and database
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    conn = setup_database(DB_PATH)
    
    fasthtml_app, rt = fast_app()
    
    @rt("/")
    def homepage():
        """Render the upload form"""
        upload_input = Input(
            type="file",
            name="content",
            accept=".pdf,.txt,.md,.mp3,.wav,.m4a,.flac",
            required=False
        )
        url_input = Input(
            type="text",
            name="url",
            placeholder="Or enter website URL",
            cls="w-full px-3 py-2 border rounded"
        )
        
        form = Form(
            Group(
                upload_input,
                url_input,
                Button("Process Content"),
                cls="space-y-4"
            ),
            hx_post="/inject",
            hx_swap="afterbegin",
            enctype="multipart/form-data",
            method="post",
        )
        
        return Title("Content Injection"), Main(
            H1("Upload Content for Podcast Generation"),
            P("Upload a file or provide a URL to process content for podcast generation."),
            form,
            Div(id="injection-status")
        )
    
    @rt("/inject", methods=["POST"])
    async def inject_content(request):
        """Handle content injection request"""
        form = await request.form()
        
        # Generate unique ID for this injection
        injection_id = uuid.uuid4().hex
        
        try:
            # Handle file upload or URL
            if "content" in form and form["content"].filename:
                file_field = form["content"]
                original_filename = file_field.filename
                input_type = get_input_type(original_filename)
                save_path = os.path.join(UPLOAD_DIR, f"upload_{injection_id}{os.path.splitext(original_filename)[1]}")
                content = await file_field.read()
                with open(save_path, "wb") as f:
                    f.write(content)
            elif "url" in form and form["url"].strip():
                url = form["url"].strip()
                input_type = get_input_type(url)
                save_path = url
                original_filename = url
            else:
                return Div(
                    P("Please select a file or provide a URL"),
                    id="injection-status",
                    cls="text-red-500"
                )
            
            # Preview processed content
            processed_text = process_content(save_path, input_type)
            if not processed_text:
                return Div(
                    P("Failed to process content"),
                    id="injection-status",
                    cls="text-red-500"
                )
            
            # Record injection in database
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO injections (id, original_filename, input_type, status)
                VALUES (?, ?, ?, 'processing')
                """,
                (injection_id, original_filename, input_type)
            )
            conn.commit()
            
            # Process the content
            processed_text = process_content(save_path, input_type)
            if not processed_text:
                raise Exception("Failed to process content")
            
            # Save processed text
            output_path = os.path.join(OUTPUT_DIR, f"processed_{injection_id}.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(processed_text)
            
            # Update database with success
            cursor.execute(
                """
                UPDATE injections
                SET status = 'completed', processed_path = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (output_path, injection_id)
            )
            conn.commit()
            
            return Div(
                P(f"✅ Content processed successfully! ID: {injection_id}"),
                P(f"Output saved to: {output_path}"),
                P("Preview:", cls="font-bold mt-4"),
                P(processed_text, cls="font-mono text-sm whitespace-pre-wrap"),
                id="injection-status",
                cls="text-green-500"
            )
            
        except Exception as e:
            # Update database with error status
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE injections
                SET status = 'failed', updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (injection_id,)
            )
            conn.commit()
            
            return Div(
                P(f"Error processing content: {str(e)}"),
                id="injection-status",
                cls="text-red-500"
            )
    
    return fasthtml_app

if __name__ == "__main__":
    serve()