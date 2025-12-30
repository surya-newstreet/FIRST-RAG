# RAG — Retrieval-Augmented Generation Example

This folder contains a small RAG (retrieval-augmented generation) example application built with Streamlit.

## Contents
- `app.py` — Streamlit UI to upload/index documents and ask questions.
- `rag_core.py` — helper functions for loading, chunking, and creating the vectorstore.
- `api.py` — (optional) small API wrapper used by the app.
- `docs/` — sample documents used for indexing.
- `db/Chroma` — persisted Chroma vectorstore (large binary files).

## Requirements
- Python 3.10+
- Install dependencies (example):

```bash
python -m pip install -r requirements.txt
# if you don't have a requirements file, at minimum:
python -m pip install streamlit chromadb sentence-transformers
```

## Run the app
1. From this folder run:

```bash
streamlit run app.py
```

2. In the sidebar click `Index existing docs/` to load the sample `docs/` into the vectorstore (or upload your own .txt/.pdf files).
3. Ask questions in the chat area — retrieved chunks will be shown in each result.

## Notes
- The `db/Chroma` folder contains persisted vectorstore files (binary blobs). Consider adding `db/Chroma/` to `.gitignore` to avoid committing large binary data.
- If you push this repository to a remote, remove or ignore large database files to keep the repository small.

## Suggested `.gitignore` entries
```
# Chroma DB
db/Chroma/
db/chroma/

# Python
__pycache__/
.venv/
env/
*.pyc
```

## License
This project is provided as-is for demonstration purposes.
