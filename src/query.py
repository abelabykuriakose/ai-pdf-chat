import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

def ask_pdf():
    # Must match the exact local model we used for ingestion
    embeddings_engine = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if not os.path.exists("chroma_db"):
        print("\n❌ Error: No database index found! Run 'python src/ingest.py' first.\n")
        return
        
    vector_store = Chroma(persist_directory="chroma_db", embedding_function=embeddings_engine)
    
    # Gemini Flash will still answer your questions using the local text context
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.3)
    
    print("\n" + "="*50)
    print("📚 Smart PDF Document Query Engine Active 📚")
    print("Type 'exit' or 'quit' to close the session.")
    print("="*50 + "\n")
    
    while True:
        question = input("Ask a question about your PDF: ")
        if question.strip().lower() in ['exit', 'quit']:
            print("👋 Session closed.")
            break
            
        if not question.strip():
            continue
            
        print("\n🔍 Scanning local document vectors for context...")
        relevant_docs = vector_store.similarity_search(question, k=3)
        context_text = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
        
        print("📝 Generating verified response via Gemini...")
        prompt = (
            "You are a precise document analysis expert. Answer the user's question using ONLY the provided text excerpt "
            "extracted from their document. If the answer cannot be found in the context, say: 'I cannot find that information in the document.'\n\n"
            f"DOCUMENT EXCERPTS:\n{context_text}\n\n"
            f"USER QUESTION: {question}\n\n"
            "ANSWER:"
        )
        
        response = llm.invoke(prompt)
        
        content = response.content
        if isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict):
            content = content[0].get("text", str(content))
            
        print("\n🤖 " + "-"*20 + " DOCUMENT RESPONSE " + "-"*20)
        print(content)
        print("="*50 + "\n")

if __name__ == "__main__":
    ask_pdf()