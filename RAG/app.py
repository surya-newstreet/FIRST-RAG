import streamlit as st
import os

from rag_core import (
    load_documents,
    chunk_documents,
    create_vectorstore,
    get_retriever,
    answer_question
)

st.set_page_config(page_title="RAG App", layout="wide")
st.title("📄 RAG Question Answering App")

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Settings")

chunk_size = st.sidebar.slider("Chunk Size", 200, 2000, 1000)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 500, 100)
top_k = st.sidebar.slider("Top K Results", 1, 10, 3)


DOCS_DIR = "docs"
DB_DIR = "db/Chroma"

st.sidebar.divider()

if st.sidebar.button("📌 Index existing docs/"):
    documents = load_documents(DOCS_DIR)
    if not documents:
        st.sidebar.error("No .txt files found in docs/")
    else:
        chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        vectordb = create_vectorstore(chunks, persist_dir=DB_DIR)
        retriever = get_retriever(vectordb, k=top_k)
        st.session_state["retriever"] = retriever
        st.sidebar.success(f"Indexed ✅ ({len(documents)} docs)")

# ---------- Upload ----------
st.subheader("📤 Upload TXT and PDF documents (optional)")
MAX_DOCS = 10
MAX_FILE_SIZE_MB = 5

uploaded_files = st.file_uploader(
    "Upload .txt and .pdf documents to add into docs/",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

os.makedirs(DOCS_DIR, exist_ok=True)

if uploaded_files:
    if len(uploaded_files) > MAX_DOCS:
        st.error(f"❌ Too many documents (max {MAX_DOCS})")
        st.stop()

    for file in uploaded_files:
        if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.error(f"❌ {file.name} is too large (max {MAX_FILE_SIZE_MB} MB)")
            st.stop()

        with open(os.path.join(DOCS_DIR, file.name), "wb") as f:
            f.write(file.read())

    st.success("✅ Documents saved to docs/. Click 'Index existing docs/' in the sidebar.")

st.divider()

# ---------- Chat ----------
st.header("💬 Ask Questions")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask something from the documents")

if st.button("Ask"):
    if "retriever" not in st.session_state:
        st.warning("⚠️ Click 'Index existing docs/' first (sidebar).")
    elif not query.strip():
        st.warning("⚠️ Type a question first.")
    else:
        answer, retrieved_docs = answer_question(query, st.session_state["retriever"])

        if answer.strip().lower() == "i don't know":
            st.warning("⚠️ Looks like this question is outside your docs (or not found).")

        st.session_state.chat_history.append((query, answer, retrieved_docs))

# ---------- Display chat ----------
for q, a, docs in st.session_state.chat_history:
    st.markdown(f"**🧑 Question:** {q}")
    st.markdown(f"**🤖 Answer:** {a}")

    with st.expander("🔎 Show retrieved chunks"):
        for i, d in enumerate(docs, start=1):
            st.write(f"**Chunk {i}** | source: `{d.metadata.get('source')}` | chunk_id: `{d.metadata.get('chunk_id')}`")
            st.code(d.page_content[:350])

    st.markdown("---")
