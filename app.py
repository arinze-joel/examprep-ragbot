import streamlit as st
import os
import hashlib
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.title("📄 Document Q&A Chatbot")
st.sidebar.header("Upload your document")

# --- File uploader ---
uploaded_file = st.sidebar.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

# --- Clear button ---
if st.sidebar.button("🗑️ Clear Upload"):
    uploaded_file = None
    st.experimental_rerun()

def get_file_hash(file_path):
    """Generate a unique hash for the file (to cache embeddings)."""
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def load_and_process(uploaded_file):
    """Load, split, and store embeddings for uploaded document with caching and progress messages."""
    if uploaded_file is None:
        return None

    # Show uploaded file info
    st.sidebar.info(f"📄 File: {uploaded_file.name}")
    st.sidebar.info(f"🗂️ Type: {uploaded_file.type}")
    st.sidebar.info(f"💾 Size: {uploaded_file.size / 1024:.2f} KB")

    # Save uploaded file temporarily
    file_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Generate hash for caching
    file_hash = get_file_hash(file_path)
    persist_dir = os.path.join("vector_cache", file_hash)
    os.makedirs("vector_cache", exist_ok=True)

    if os.path.exists(persist_dir):
        st.info("⚡ Using cached embeddings for this file...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    # Detect file type and set loader
    if file_path.endswith(".pdf"):
        loader_class = PyPDFLoader
        loader_label = "PDF"
    elif file_path.endswith(".txt"):
        loader_class = TextLoader
        loader_label = "TXT"
    else:
        st.error("Unsupported file type.")
        return None

    # Load document
    with st.spinner(f"📥 Loading {loader_label} document..."):
        loader = loader_class(file_path)
        docs = loader.load()

    # Split document into chunks
    with st.spinner("✂️ Splitting document into chunks..."):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)

    # Create embeddings
    with st.spinner("🔎 Creating embeddings (first run may take a while)..."):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(
            chunks,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        vectorstore.persist()

    st.success("✅ Document processed successfully!")
    return vectorstore

vectorstore = None
if uploaded_file:
    st.sidebar.success("✅ File uploaded successfully!")
    vectorstore = load_and_process(uploaded_file)

# --- Chatbot interface ---
query = st.text_input("Ask a question about your document:")

if query and vectorstore:
    with st.spinner("🤔 Thinking..."):
        retriever = vectorstore.as_retriever()
        docs = retriever.get_relevant_documents(query)

        context = "\n\n".join([doc.page_content for doc in docs])

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer based only on the provided document."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            model="llama3-8b-8192",
        )

        answer = response.choices[0].message.content

    st.subheader("💡 Answer:")
    st.write(answer)
