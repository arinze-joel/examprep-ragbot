# Naive RAG Chatbot

A simple Retrieval-Augmented Generation (RAG) chatbot that answers questions based on uploaded documents (PDF, TXT). 
Built using Streamlit, LangChain, HuggingFace embeddings, and Groq API.

## Features
- Upload your own documents (PDF, TXT)  
- Automatically split documents into chunks and create embeddings  
- Ask questions about your documents and get grounded answers  
- Caching of embeddings for faster subsequent queries  

## Tech Stack
- Python  
- Streamlit for the web app interface  
- LangChain for document processing and retrieval  
- HuggingFace Embeddings (`all-MiniLM-L6-v2`)  
- Chroma as vector store  
- Groq API for LLM completions  

## Setup Instructions
1. Clone this repository:
```bash
git clone https://github.com/<YourUsername>/examprep-ragbot.git
cd examprep-ragbot

2. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt
Create a .env file in the root folder and add your GROQ key
GROQ_API_KEY = your_api_key_here

4. Run the Streamlit app
streamlit run app.py

Example Queries

- Upload a PDF of a lecture transcript, then ask:
"What are the key topics covered in this lecture?"
"Explain the main concept of section 2."

- Upload a TXT file of notes, then ask:
"Summarize the notes in 3 points."
"What are the important dates mentioned?"

- Known Issues
DOCX support is currently disabled due to segmentation faults on some systems
Embeddings creation may take time on large documents
