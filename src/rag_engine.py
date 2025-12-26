import os
import uuid
import logging
from typing import List, Dict

# Core RAG dependencies
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from PyPDF2 import PdfReader

# Suppress annoying logs
logging.basicConfig(level=logging.ERROR)

class LocalVectorStore:
    def __init__(self, collection_name="rag_demo", persistence_path="./chroma_db"):
        print(f"Initializing Vector Store...")
        self.client = chromadb.PersistentClient(path=persistence_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        # Load embedding model (Small & Fast for CPU)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
    def ingest_text(self, text: str, chunk_size=500, overlap=50):
        """Splits text and stores embeddings."""
        # Simple sliding window chunking
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk) > 50:  # Filter tiny artifacts
                chunks.append(chunk)
        
        if not chunks:
            return
            
        # Create embeddings
        print(f"Embedding {len(chunks)} chunks...")
        embeddings = self.embedder.encode(chunks).tolist()
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        
        # Store in Chroma
        self.collection.add(documents=chunks, embeddings=embeddings, ids=ids)
        print(f"Indexed {len(chunks)} chunks.")

    def search(self, query: str, top_k=3):
        """Retrieves most relevant chunks."""
        query_vec = self.embedder.encode([query]).tolist()
        results = self.collection.query(query_embeddings=query_vec, n_results=top_k)
        return results['documents'][0] if results['documents'] else []

class RAGSystem:
    def __init__(self, model_path, vector_store: LocalVectorStore):
        self.vector_store = vector_store
        
        print("Loading Llama 3.2 1B (Quantized)...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,        # Context window
            n_threads=4,       # CPU threads
            n_gpu_layers=0,    # STRICT CPU MODE
            verbose=False
        )
        print("Model Loaded.")

    def query(self, user_query: str):
        # 1. Retrieve
        context_chunks = self.vector_store.search(user_query)
        context_str = "\n---\n".join(context_chunks)
        
        # 2. Prompt Engineering (Strict Llama 3 format)
        system_prompt = (
            "You are a helpful assistant. Use the provided context to answer the question. "
            "If the answer is not in the context, say 'I do not know'. "
            "Keep answers concise."
        )
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}

Context:
{context_str}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

        # 3. Generate
        output = self.llm(
            prompt, 
            max_tokens=256, 
            stop=["<|eot_id|>"], 
            echo=False
        )
        return output['choices'][0]['text'].strip()

# Helper to extract text from PDF
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a PDF file."""
    print(f"Extracting text from PDF: {pdf_path}")
    reader = PdfReader(pdf_path)
    text = ""
    for page_num, page in enumerate(reader.pages, 1):
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    print(f"Extracted {len(text)} characters from {len(reader.pages)} pages.")
    return text

# Helper to load file (supports both text and PDF)
def ingest_file(file_path, vector_store):
    """Ingests a text or PDF file into the vector store."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.pdf':
        text = extract_text_from_pdf(file_path)
    elif file_ext in ['.txt', '.md', '.text']:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        raise ValueError(f"Unsupported file type: {file_ext}. Supported types: .pdf, .txt, .md")
    
    vector_store.ingest_text(text)