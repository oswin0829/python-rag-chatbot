import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

app = FastAPI(title="TechNova Brain")

# Enable CORS (Critical for your HTML frontend to talk to this backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State (Simple in-memory storage for this demo)
VECTOR_STORE = None
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

class ChatRequest(BaseModel):
    question: str

@app.get("/")
def health_check():
    return {"status": "neural_link_active", "system": "ready"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global VECTOR_STORE
    
    try:
        # 1. Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name

        # 2. Process PDF
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        
        # 3. Chunking
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # 4. Create Vector Store
        VECTOR_STORE = FAISS.from_documents(documents=splits, embedding=EMBEDDING_MODEL)
        
        # Cleanup
        os.remove(tmp_path)
        
        return {"message": "Knowledge ingestion complete", "chunks": len(splits)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    global VECTOR_STORE
    
    if not VECTOR_STORE:
        raise HTTPException(status_code=400, detail="No knowledge base loaded. Please upload a PDF first.")

    try:
        # Setup LLM
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Server missing Groq API Key")
            
        llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0)
        retriever = VECTOR_STORE.as_retriever()

        # Prompt Engineering
        system_prompt = (
            "You are TechNova Prime, an advanced corporate AI. "
            "Answer purely based on the context provided. "
            "If the answer is missing, state: 'DATA NOT FOUND IN SECTOR 7'. "
            "Keep answers technical and concise."
            "\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "{input}")]
        )

        # Build Chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Run
        response = rag_chain.invoke({"input": request.question})

        # Format Sources
        sources = []
        for doc in response["context"]:
            sources.append({
                "page": doc.metadata.get("page", "N/A"),
                "text": doc.page_content[:150] + "..."
            })

        return {
            "answer": response["answer"],
            "sources": sources
        }

    except Exception as e:
        print(f"Error: {e}") # Log to terminal
        raise HTTPException(status_code=500, detail=str(e))