# api.py
import os
import shutil
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from rag_core import (
    load_documents,
    chunk_documents,
    create_vectorstore,
    get_retriever,
    answer_question,
)

app = FastAPI(title="RAG API")

DOCS_DIR = "docs"
DB_DIR = "db/Chroma"

retriever = None

class AskBody(BaseModel):
    question: str
    top_k: int = 3

class ReindexBody(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 100
    top_k: int = 3

def build_retriever(chunk_size=1000, chunk_overlap=100, top_k=3):
    documents = load_documents(DOCS_DIR)
    if not documents:
        raise HTTPException(status_code=400, detail="No .txt files found in docs/")
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vectordb = create_vectorstore(chunks, persist_dir=DB_DIR)
    return get_retriever(vectordb, k=top_k)

@app.get("/health")
def health():
    return {"ok": True, "docs_dir": DOCS_DIR, "indexed": retriever is not None}

@app.post("/reindex")
def reindex(body: ReindexBody):
    global retriever
    os.makedirs(DOCS_DIR, exist_ok=True)
    retriever = build_retriever(body.chunk_size, body.chunk_overlap, body.top_k)
    return {"status": "reindexed", "chunk_size": body.chunk_size, "chunk_overlap": body.chunk_overlap, "top_k": body.top_k}

@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    os.makedirs(DOCS_DIR, exist_ok=True)
    for f in files:
        if not f.filename.lower().endswith(".txt"):
            raise HTTPException(status_code=400, detail=f"Only .txt allowed: {f.filename}")
        dest = os.path.join(DOCS_DIR, f.filename)
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)

    return {"status": "uploaded", "files": [f.filename for f in files], "note": "Call /reindex to index them."}

@app.post("/ask")
def ask(body: AskBody):
    global retriever
    if retriever is None:
        # Auto-index existing docs if not indexed yet
        retriever = build_retriever(chunk_size=1000, chunk_overlap=100, top_k=body.top_k)

    answer, docs = answer_question(body.question, retriever)

    # basic "outside docs" UI help
    outside_docs = answer.strip().lower() == "i don't know"

    return {
        "answer": answer,
        "outside_docs": outside_docs,
        "retrieved": [
            {
                "source": d.metadata.get("source"),
                "chunk_id": d.metadata.get("chunk_id"),
                "preview": d.page_content[:220]
            }
            for d in docs
        ],
    }
