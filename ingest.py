import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables from your .env file
load_dotenv()

def process_pdf_to_vector_store(pdf_path: str, persist_directory: str = "./chroma_db"):
    """
    Extracts text from a PDF, splits it into chunks, 
    converts it to embeddings via OpenRouter, and saves it to a local ChromaDB store.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Could not find the file: {pdf_path}")
        
    print(f"🔄 Step 1: Loading PDF file from '{pdf_path}'...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"✅ Successfully loaded {len(documents)} pages.")

    print("\n🔄 Step 2: Splitting text into semantic chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Split document into {len(chunks)} individual text chunks.")

    print("\n🔄 Step 3: Generating mathematical embeddings & saving to ChromaDB...")
    
    # FIX: Point explicitly to OpenRouter and use their full prefix identifier
    embedding_model = OpenAIEmbeddings(
        model="openai/text-embedding-3-small", 
        openai_api_base="https://openrouter.ai/api/v1"
    )
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    
    print(f"🎉 Success! Local vector database created at '{persist_directory}'")
    return vector_store

if __name__ == "__main__":
    # Ensure you have a file named 'sample.pdf' in this folder or rename this variable!
    SAMPLE_PDF = "sample.pdf" 
    
    try:
        db = process_pdf_to_vector_store(SAMPLE_PDF)
        
        print("\n🔍 Running a quick verification search query...")
        test_query = "What is the summary of this document?"
        results = db.similarity_search(test_query, k=2)
        
        print("\nTop matching text found in your PDF:")
        for idx, doc in enumerate(results):
            print(f"\n--- Match {idx+1} (Page {doc.metadata.get('page', 0) + 1}) ---")
            print(doc.page_content[:200] + "...")
            
    except Exception as e:
        print(f"\n❌ Error encountered: {e}")