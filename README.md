# FIRST-RAG

This repository contains several experiments and demos. The primary folder for the
Retrieval-Augmented Generation (RAG) demonstration is the `RAG/` directory.

Project layout
- RAG/: Streamlit RAG app, core logic, sample documents, and persisted Chroma DB.
- RAG_Practice/: alternative/practice versions of the RAG app.
- building_rag/: notebooks and helper scripts for experimenting with RAG pipelines.
- pytorc/: PyTorch examples and helper modules.
- trail/: miscellaneous notebooks and small experiments.
- tensorflow/: TensorFlow experiments.
- Root-level scripts: `ingestion.py`, `ingestion_test.py`, `rag.practice_1.py`, etc.

Running the RAG demo
1. Change into the `RAG` folder and run the Streamlit app:

```bash
cd RAG
streamlit run app.py
```

2. In the sidebar click "Index existing docs/" to load the sample `docs/` into the vectorstore
	 (or upload your own `.txt`/`.pdf` files). Then ask questions in the chat area.

More information
- See [RAG/README.md](RAG/README.md#L1) for detailed instructions, requirements, and notes about the
	Chroma DB location.

Notes
- The `RAG/db/Chroma` folder contains large binary database files — add it to `.gitignore`
	before committing large data.
- Minimal Python dependencies (example):

```bash
python -m pip install streamlit chromadb sentence-transformers
```

