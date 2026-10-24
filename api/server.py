#!/usr/bin/env python3
"""
RAG Knowledge Base API Server - Minimal version
"""

import os
import json
import uuid
import hashlib
from pathlib import Path
from fastapi import FastAPI, HTTPException, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

app = FastAPI(title="RAG Knowledge Base API", description="A local RAG system for your documents")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

DATA_DIR = Path("/home/medo/Desktop/projects/RAG-Knowledge-Base/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR = DATA_DIR / "storage"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_CACHE = "/tmp/sentence-transformers-all-MiniLM-L6-v2"
EMBEDDING_LOADED = False

def get_embedding_model():
    """Get or create the embedding model."""
    global embedding_model, EMBEDDING_LOADED
    if EMBEDDING_LOADED:
        return embedding_model
    
    try:
        from sentence_transformers import SentenceTransformer
        if os.path.exists(EMBEDDING_CACHE):
            embedding_model = SentenceTransformer(EMBEDDING_CACHE)
            EMBEDDING_LOADED = True
            print("  ✓ Model loaded from cache")
            return embedding_model
        else:
            print(f"  ⚠ Model not cached, downloading {EMBEDDING_MODEL}...")
            embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            EMBEDDING_LOADED = True
            print("  ✓ Model loaded successfully")
            return embedding_model
    except ImportError as e:
        print(f"  ⚠ sentence-transformers not installed: {e}")
        print("  Installing now...")
        import subprocess
        subprocess.run(["pip", "install", "sentence-transformers", "--break-system-packages"], 
                      capture_output=True, check=True)
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        EMBEDDING_LOADED = True
        print("  ✓ Model loaded successfully")
        return embedding_model
    except Exception as e:
        print(f"  ⚠ Model load error: {e}")
        print("  ⚠ Using fallback (no embeddings)")
        return None

embedding_model = None

def get_embedding(text):
    """Get embedding for text."""
    if embedding_model is None:
        return None
    try:
        return embedding_model.encode(text.tolist() if isinstance(text, list) else [text]).tolist()[0]
    except Exception as e:
        print(f"  Embedding error: {e}")
        return None

def get_query_embedding(query):
    """Get embedding for a query."""
    if embedding_model is None:
        return None
    return embedding_model.encode([query]).tolist()[0]

def get_doc_metadata():
    """List indexed documents from storage."""
    docs_path = STORAGE_DIR / "documents.json"
    docs = []
    if docs_path.exists():
        with open(docs_path) as f:
            docs = json.load(f)
    return docs

@app.get("/")
def root():
    return {"status": "ok", "service": "RAG Knowledge Base API", "llm": "ollama/llama3.2", "embeddings": EMBEDDING_MODEL, "model_loaded": EMBEDDING_LOADED}

@app.get("/docs")
def docs():
    return {"docs": "/docs", "reDoc": "/redoc"}

@app.get("/health")
def health():
    return {"status": "healthy", "embedding_model": "loaded" if embedding_model else "loading"}

@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    try:
        filepath = STORAGE_DIR / file.filename
        content = await file.read()
        with open(filepath, "wb") as f:
            f.write(content)
        
        suffix = Path(file.filename).suffix.lower()
        text = ""
        if suffix == ".pdf":
            try:
                import fitz
                doc = fitz.open(filepath)
                text = "\n\n".join([page.get_text() for page in doc])
                doc.close()
            except:
                try:
                    import PyPDF2
                    with open(filepath, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        text = "\n\n".join([page.extract_text() or "" for page in reader.pages])
                except:
                    text = f"PDF: {file.filename}"
        elif suffix in [".txt", ".md"]:
            with open(filepath, "r") as f:
                text = f.read()
        elif suffix == ".json":
            with open(filepath, "r") as f:
                text = json.dumps(json.load(f))
        elif suffix == ".html":
            with open(filepath, "r") as f:
                text = f.read()
        else:
            text = f"File: {file.filename}"
        
        # Chunk
        chunk_size = 512
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            if chunk.strip():
                chunks.append({"text": chunk, "filename": file.filename, "chunk_num": i // chunk_size})
        
        # Embed and store
        global vector_index
        vector_index = {}
        for chunk in chunks:
            emb = get_embedding(chunk["text"])
            if emb:
                key = hashlib.md5(chunk["text"].encode()).hexdigest()
                vector_index[key] = {"text": chunk["text"], "filename": chunk["filename"]}
        
        # Save metadata
        docs_path = STORAGE_DIR / "documents.json"
        docs = get_doc_metadata()
        docs.append({"id": str(uuid.uuid4()), "filename": file.filename, "size": str(len(content) / 1024 / 1024) + " MB", "chunks": len(chunks), "date": "just now"})
        with open(docs_path, "w") as f:
            json.dump(docs, f)
        
        return {"success": True, "chunks": len(chunks), "filename": file.filename}
    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    return get_doc_metadata()

@app.get("/search")
async def search(query: str, top_k: int = 5):
    try:
        query_emb = get_query_embedding(query)
        if query_emb is None:
            return {"response": "Embedding model not loaded yet.", "citations": [], "sources": []}
        
        # Cosine similarity
        scores = []
        for key, data in vector_index.items():
            emb = data["embedding"]
            dot = sum(a*b for a,b in zip(query_emb, emb))
            norm_q = (sum(a*a for a in query_emb))**0.5
            norm_e = (sum(b*b for b in emb))**0.5
            score = dot / (norm_q * norm_e) if norm_q and norm_e else 0
            scores.append((score, key))
        scores.sort(reverse=True)
        top_keys = [k for _, k in scores[:top_k]]
        
        results = []
        citations = []
        for key in top_keys:
            data = vector_index[key]
            results.append({"id": key, "title": data["filename"], "score": round(scores[scores.index((score, key) for score, k in scores if k == key)[0]][0], 4)})
            citations.append(f"Source: {data['filename']}\n{data['text'][:300]}...")
        
        return {"query": query, "response": f"Found {len(top_keys)} relevant chunks.", "citations": citations, "sources": results}
    except Exception as e:
        print(f"Search error: {e}")
        return {"response": f"Error: {str(e)}", "citations": [], "sources": []}

@app.post("/chat")
async def chat(query: str):
    try:
        query_emb = get_query_embedding(query)
        if query_emb is None:
            return {"response": "Embedding model not loaded yet.", "sources": []}
        
        scores = []
        for key, data in vector_index.items():
            emb = data["embedding"]
            dot = sum(a*b for a,b in zip(query_emb, emb))
            norm_q = (sum(a*a for a in query_emb))**0.5
            norm_e = (sum(b*b for b in emb))**0.5
            score = dot / (norm_q * norm_e) if norm_q and norm_e else 0
            scores.append((score, key))
        scores.sort(reverse=True)
        top_keys = [k for _, k in scores[:5]]
        
        context = "\n\n".join([vector_index[k]["text"] for k in top_keys])
        
        try:
            from llama_index.llms.ollama import Ollama
            llm = Ollama(model="llama3.2", base_url="http://localhost:11434", temperature=0.7)
            response = llm.complete(prompt=f"Based on the following context, answer the question concisely. Cite your sources.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:")
            return {"response": response.response, "sources": [{"id": k, "text": vector_index[k]["text"][:200]} for k in top_keys]}
        except ImportError:
            from llama_index.llms.ollama import Ollama
            llm = Ollama(model="llama3.2", base_url="http://localhost:11434", temperature=0.7)
            response = llm.complete(prompt=f"Based on the following context, answer the question concisely. Cite your sources.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:")
            return {"response": response.response, "sources": [{"id": k, "text": vector_index[k]["text"][:200]} for k in top_keys]}
    except Exception as e:
        print(f"Chat error: {e}")
        return {"response": f"Error: {str(e)}", "sources": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
