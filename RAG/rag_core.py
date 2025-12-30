import os
from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import (
    PyPDFLoader,
    PyMuPDFLoader,
    UnstructuredFileLoader
)

llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
    temperature=0
)

__all__ = [
    "load_documents",
    "chunk_documents",
    "create_vectorstore",
    "get_retriever",
    "answer_question",
]


def load_documents(docs_path: str = "docs"):
    txt_loader = DirectoryLoader(path=docs_path,glob="*.txt",loader_cls=TextLoader)
    pdf_loader = DirectoryLoader(
        path = docs_path,
        loader_cls=PyMuPDFLoader,
        glob = "*.pdf"
    )
    text_load = txt_loader.load()
    pdf_load = pdf_loader.load()   
    documents  =text_load + pdf_load
    return documents

def chunk_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_vectorstore(chunks, persist_dir: str = "db/Chroma"):
    os.makedirs(persist_dir, exist_ok=True)
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    return vectorstore

def get_retriever(vectorstore, k: int = 3):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

def retrieve_docs(query: str, retriever):
    return retriever.invoke(query)

def build_context(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_prompt(context: str, question: str):
    return f"""
You are a helpful assistant.
Answer ONLY using the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
""".strip()

def answer_question(question: str, retriever):
    docs = retrieve_docs(question, retriever)
    context = build_context(docs)
    prompt = build_prompt(context, question)
    response = llm.invoke(prompt)
    return response.content, docs
