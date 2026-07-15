import os
import io
import json
import uuid
import logging
import requests
import fitz  # pymupdf
import pytesseract
import ollama
import chromadb
import pandas as pd
from urllib.parse import urlparse, parse_qs
from PIL import Image
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb import Documents, EmbeddingFunction, Embeddings

import sys
from dotenv import load_dotenv

load_dotenv()

# force=True ensures this takes effect even after uvicorn has already
# configured the root logger (basicConfig is normally a no-op if called twice).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

# ---------- CONFIG ----------
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

from google import genai
import json
import re
import time

GEMINI_API = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API)

def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = []

    for text in texts:
        response = client.models.embed_content(
            model="gemini-embedding-2",
            contents=text,
        )
        embeddings.append(response.embeddings[0].values)

    return embeddings

# DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
# chroma = chromadb.PersistentClient(path=DB_PATH)

DB_PATH = os.getenv(
    "CHROMA_DB_PATH",
    "/opt/render/project/src/chroma_db"
)

os.makedirs(DB_PATH, exist_ok=True)

chroma = chromadb.PersistentClient(path=DB_PATH)

collection = chroma.get_or_create_collection(
    name="medical_documents"
)

app = FastAPI()

class DownloadRequest(BaseModel):
    url: str

# ---------- PDF EXTRACTION (native text + OCR fallback) ----------
def extract_content(file_path: str) -> str:
    doc = fitz.open(file_path)
    extracted_text = []

    for page in doc:
        text = page.get_text().strip()
        if len(text) > 50:
            extracted_text.append(text)
            continue

        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        extracted_text.append(pytesseract.image_to_string(img))

    doc.close()
    return "\n\n".join(extracted_text)


# ---------- METADATA EXTRACTION (LLM) ----------
def extract_metadata(document_text: str) -> dict:
    prompt = f"""
Extract metadata from the medical document. If a piece of information is not present in the document, use null.

Return ONLY valid JSON.

Schema:
{{
    "title": "",
    "document_type": "",
    "specialty": "",
    "patient_name": "",
    "age": "",
    "gender": "",
    "diagnosis": "",
    "symptoms": [],
    "medications": [],
    "doctor": "",
    "hospital": "",
    "date": "",
    "disease": [],
    "keywords": [],
    "summary": ""
}}

Document:
{document_text[:12000]}
"""

    last_error = None

    for attempt in range(3):
        try:
            logger.info("Calling Gemini attempt=%d", attempt + 1)

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )

            content = response.text.strip()
            logger.info("Gemini response: %s", content)

            try:
                metadata = json.loads(content)
            except json.JSONDecodeError:
                clean = re.sub(r"```(?:json)?|```", "", content).strip()
                metadata = json.loads(clean)

            # Chroma accepts only primitive metadata values
            for k, v in metadata.items():
                if isinstance(v, list):
                    metadata[k] = ", ".join(map(str, v))

            return metadata

        except Exception as e:
            last_error = e
            logger.warning("Gemini attempt=%d failed: %s", attempt + 1, e)
            time.sleep(5)

    raise RuntimeError("Failed to extract metadata.") from last_error


# ---------- CHUNK + STORE IN CHROMA ----------
def store_document(document_text: str, file_name: str, metadata: dict) -> tuple[str, int]:
    document_id = str(uuid.uuid4())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_text(document_text)
    embeddings = embed_texts(chunks)

    ids = [f"{document_id}_{i}" for i in range(len(chunks))]

    metadatas = [
        {
            "document_id": document_id,
            "chunk_number": i,
            "file_name": file_name,
            **metadata,
        }
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return document_id, len(chunks)

# ---------- MAIN ENDPOINT ----------
@app.post("/download-document")
async def download_document(request: DownloadRequest):
    try:
        url = request.url
        logger.info("Received request for url: %s", url)

        # Determine download URL
        if "docs.google.com/document" in url:
            doc_id = url.split("/d/")[1].split("/")[0]
            download_url = f"https://docs.google.com/document/d/{doc_id}/export?format=pdf"

        elif "drive.google.com/file/d/" in url:
            file_id = url.split("/d/")[1].split("/")[0]
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

        elif "drive.google.com/open" in url and "id=" in url:
            file_id = parse_qs(urlparse(url).query)["id"][0]
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

        elif "drive.google.com/uc" in url:
            download_url = url

        else:
            download_url = url

        logger.info("Downloading from: %s", download_url)

        # Download
        response = requests.get(download_url, timeout=60)
        response.raise_for_status()

        logger.info(
            "Download complete | Status=%s | Content-Type=%s | Bytes=%d",
            response.status_code,
            response.headers.get("Content-Type"),
            len(response.content),
        )

        # Validate response
        content_type = response.headers.get("Content-Type", "").lower()

        if "text/html" in content_type:
            raise HTTPException(
                status_code=400,
                detail="Unable to download the Google Drive file. Ensure it is shared publicly ('Anyone with the link') and that the URL points to a downloadable file."
            )

        # Generate filename
        filename = os.path.basename(urlparse(download_url).path)

        if not filename or "." not in filename:
            filename = f"{uuid.uuid4()}.pdf"

        file_path = os.path.join(DOWNLOAD_DIR, filename)

        with open(file_path, "wb") as f:
            f.write(response.content)

        logger.info("Saved file to %s", file_path)

        logger.info("Extracting text...")
        extracted_text = extract_content(file_path)
        logger.info("Extracted %d characters", len(extracted_text))

        logger.info("Extracting metadata via LLM...")
        metadata = extract_metadata(extracted_text)
        logger.info("Metadata: %s", metadata)

        logger.info("Chunking + storing in Chroma...")
        document_id, chunk_count = store_document(extracted_text, filename, metadata)
        logger.info("Stored document_id=%s chunks=%d", document_id, chunk_count)

        return {
            "status": "Success",
            "document_id": document_id,
            "document_type": metadata.get("document_type", "PDF"),
            "embedding_generated": True,
            "stored_in_chromadb": True,
            "chromadb_collection": "medical_documents",
            "metadata": {
                "file_name": filename,
                "file_type": "pdf",
                "processing_status": "Success",
                "title": metadata.get("title", ""),
                "specialty": metadata.get("specialty", ""),
                "disease": metadata.get("disease", ""),
                "keywords": metadata.get("keywords", ""),
                "summary": metadata.get("summary", ""),
            },
            "document": {
                "patient_name": metadata.get("patient_name"),
                "age": metadata.get("age"),
                "gender": metadata.get("gender"),
                "document_type": metadata.get("document_type", "Report"),
                "diagnosis": metadata.get("diagnosis"),
                "symptoms": metadata.get("symptoms"),
                "medications": metadata.get("medications"),
                "doctor": metadata.get("doctor"),
                "hospital": metadata.get("hospital"),
                "date": metadata.get("date"),
                "raw_text": extracted_text
            }
        }

    except Exception as e:
        logger.exception("Request failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/search-medical-documents")
def search_medical_documents(request: SearchRequest):
    try:
        query_embedding = embed_texts([request.query])[0]

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=request.top_k,
            include=["documents", "metadatas", "distances"]
        )
        documents = []

        for doc, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            documents.append({
                "score": round(1 - distance, 4),   # optional similarity score
                "metadata": meta,
                "content": doc,
            })

        return {
            "query": request.query,
            "count": len(documents),
            "results": documents,
        }

    except Exception as e:
        logger.exception("Medical document search failed")
        raise HTTPException(status_code=500, detail=str(e))

# ---------- CSV INGESTION INTEGRATION ----------
class CsvIngestRequest(BaseModel):
    limit: int = 10  # Default to limit so we don't blow up the Gemini API quota accidentally

def process_csv_in_background(limit: int):
    logger.info(f"Starting background CSV ingestion. Limit: {limit}")
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "unified_patient_data.csv")
    
    if not os.path.exists(csv_path):
        logger.error(f"Dataset not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path, low_memory=False).fillna("")
    
    if limit > 0:
        df = df.head(limit)
        
    documents = []
    metadatas = []
    ids = []
    
    for i, row in df.iterrows():
        # Represent patient row as a document string
        row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if str(val).strip() != ""])
        if not row_text:
            continue
            
        documents.append(row_text)
        
        meta = {"source": "unified_patient_data.csv", "row_index": i}
        if "subject_id" in row: meta["subject_id"] = str(row["subject_id"])
        if "hadm_id" in row: meta["hadm_id"] = str(row["hadm_id"])
        if "drg_code" in row: meta["drg_code"] = str(row["drg_code"])
        
        metadatas.append(meta)
        ids.append(f"unified_csv_row_{i}")
        
        # Batch insert every 50 records to avoid huge Gemini embedding payload
        if len(documents) >= 50:
            logger.info(f"Embedding and inserting batch of {len(documents)} records...")
            try:
                embeddings = embed_texts(documents)
                collection.upsert(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings)
            except Exception as e:
                logger.error(f"Failed to embed/upsert batch: {e}")
                
            documents, metadatas, ids = [], [], []
            time.sleep(1) # Rate limit protection
            
    # Insert remaining records
    if documents:
        try:
            embeddings = embed_texts(documents)
            collection.upsert(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings)
        except Exception as e:
            logger.error(f"Failed to embed/upsert final batch: {e}")
            
    logger.info("Finished background CSV ingestion process.")

@app.post("/ingest-csv-dataset")
async def ingest_csv_dataset(request: CsvIngestRequest, background_tasks: BackgroundTasks):
    """
    Ingests the local unified_patient_data.csv file into ChromaDB Cloud.
    Uses Gemini to embed the rows. Runs as a background task to prevent timeouts.
    """
    background_tasks.add_task(process_csv_in_background, request.limit)
    return {
        "success": True,
        "message": f"CSV ingestion started in the background. Limit set to {request.limit} records. Check server logs for progress."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, log_config=None)
