from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

def load_documents(file_path: str):
    """Loads documents depending on their type (PDF, Word, TXT)."""

    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Use PDF, DOCX, or TXT.")

    docs = loader.load()
    return docs

from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(documents, chunk_size=500, chunk_overlap=50):
    """Splits documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def store_embeddings(chunks, persist_directory="db"):
    """Converts document chunks into embeddings and stores them in ChromaDB."""

    # Step 1: Load embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Step 2: Store embeddings in Chroma
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    # Step 3: Save (persist) locally
    vectorstore.persist()
    print(f"✅ Embeddings stored in {persist_directory}/")

    return vectorstore
