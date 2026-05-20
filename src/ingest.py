import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

def reset_local_database():
    """Cleans up any old vector databases so you start fresh with the new PDF."""
    if os.path.exists("chroma_db"):
        print("🧹 Cleaning up previous document database...")
        shutil.rmtree("chroma_db")

def ingest_pdf():
    print("\n" + "="*50)
    print("🚀 Welcome to the Bulletproof Local PDF Ingestion Core! 🚀")
    print("="*50)
    
    pdf_input = input("💡 Drag & drop your PDF file here, or paste the exact path:\n-> ").strip()
    pdf_path = pdf_input.replace("\\ ", " ").replace("'", "").replace('"', '')
    
    if not os.path.exists(pdf_path):
        print(f"\n❌ Error: Could not find a file at: '{pdf_path}'\n")
        return
        
    if not pdf_path.lower().endswith('.pdf'):
        print("\n❌ Error: The file must be a standard PDF format (.pdf).\n")
        return

    reset_local_database()

    print(f"\n📄 [1/4] Loading document from: {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print("✂️ [2/4] Slicing document into readable fragments...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Generated {len(chunks)} text fragments.")
    
    print("🧠 [3/4] Initializing Local HuggingFace Embedding Engine...")
    print("📥 (Note: This will download a small 120MB model on your first run...)")
    # This downloads and runs a stable, fast open-source embedding model right on your Mac
    embeddings_engine = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("💾 [4/4] Building local vector database folder...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_engine,
        persist_directory="chroma_db"
    )
    print("\n🎉 Success! Your local vector library is built inside 'chroma_db/'")
    print("👉 Now execute: 'python src/query.py' to start chatting with it!\n")

if __name__ == "__main__":
    ingest_pdf()