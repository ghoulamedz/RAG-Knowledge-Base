# RAG Knowledge Base

A modern, beautiful RAG (Retrieval-Augmented Generation) application built with React + Tailwind CSS (frontend) and Python + LlamaIndex (backend).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React + Tailwind)               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Chat Tab    │  │  Search Tab  │  │  Documents Tab   │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    API Server (FastAPI + Python)              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Routes: /documents, /search, /chat, /docs          │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    LlamaIndex RAG Engine                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Embeddings  │  │   Vector DB  │  │    LLM (Ollama)  │  │
│  │ (sentence-    │  │  (FAISS)     │  │                   │  │
│  │  transformers)│  │              │  │                   │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### UI Design (Stripe/Linear/Vercel Inspired)
- **Dark mode default** with subtle glassmorphism effects
- **Generous whitespace** and rounded corners
- **Subtle borders** and micro-interactions
- **Gradient accents** (indigo → purple → pink)
- **Dense but clean** information density

### RAG Architecture
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (120M params, fast, ~85BM)
- **LLM**: Ollama (local LLM like Llama 3.2)
- **Vector Store**: FAISS (local, no external dependencies)
- **Chunking**: Sentence splitter (512 token chunks with 50 overlap)

### Why This Stack?
- **No cloud dependencies** — everything runs locally
- **Lightweight** — ~500MB total size
- **Privacy-first** — no data leaves your machine
- **Extensible** — easy to swap LLM/embedding models

## Running

### 1. Start Ollama
```bash
ollama pull llama3.2
ollama serve
```

### 2. Start the API
```bash
cd /home/medo/Desktop/projects/RAG-Knowledge-Base/api
python server.py
```

### 3. Open the UI
```bash
cd /home/medo/Desktop/projects/RAG-Knowledge-Base/ui
python -m http.server 3000
```

Then open `http://localhost:3000` in your browser.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/docs` | OpenAPI docs |
| POST | `/documents` | Upload a document |
| GET | `/documents` | List indexed documents |
| GET | `/search?q=...` | Search the knowledge base |
| POST | `/chat` | Chat with the RAG system |

## Adding Documents

1. Go to the **Documents** tab in the UI
2. Click **Upload** and select a file (PDF, Markdown, TXT, JSON, HTML)
3. The document is parsed, chunked, and indexed automatically

## How RAG Works Here

1. **Ingestion** → Document is read, split into chunks, embedded
2. **Retrieval** → Query is embedded and searched in the vector store
3. **Generation** → Retrieved chunks are used as context for the LLM
4. **Response** → LLM generates a grounded answer with citations

## Project Structure

```
RAG-Knowledge-Base/
├── api/
│   └── server.py        # FastAPI backend
├── ui/
│   └── index.html       # React + Tailwind frontend
├── docs/
│   └── README.md        # This file
└── data/
    └── storage/         # Persisted vector index
```
