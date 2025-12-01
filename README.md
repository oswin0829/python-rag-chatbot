# 🧠 Enterprise RAG Chatbot (Python + FastAPI)

## 🚀 Project Overview
This project is a **Retrieval-Augmented Generation (RAG)** system designed to answer questions based on private corporate documents. Unlike standard LLMs, this system cites its sources and ensures zero hallucinations by restricting answers to the provided context.

## 🛠️ Tech Stack
* **Backend:** Python, FastAPI, Uvicorn
* **AI Engine:** Llama 3-70b (via Groq Cloud)
* **Vector DB:** FAISS (Facebook AI Similarity Search)
* **Orchestration:** LangChain v0.3
* **Frontend:** HTML5, CSS3 (Modern SaaS UI), Vanilla JS

## ⚙️ Architecture
1.  **Ingestion:** PDF documents are parsed and split into 1000-character chunks.
2.  **Embedding:** Chunks are converted to vectors using `HuggingFace-MiniLM`.
3.  **Retrieval:** User queries are matched against the FAISS index to find relevant context.
4.  **Synthesis:** Llama 3 generates an answer using *only* the retrieved context.

## 📦 How to Run
1.  Clone the repo:
    ```bash
    git clone [https://github.com/yourname/python-rag-chatbot.git](https://github.com/yourname/python-rag-chatbot.git)
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Set up environment:
    * Create a `.env` file with `GROQ_API_KEY=your_key`.
4.  Start the server:
    ```bash
    uvicorn api:app --reload
    ```
5.  Open `index.html` in your browser.
