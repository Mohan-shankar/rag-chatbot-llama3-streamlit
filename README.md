# rag-chatbot-llama3-streamlit
A full RAG (Retrieval-Augmented Generation) chatbot using FAISS Vector DB, HuggingFace Embeddings, LangChain, Streamlit UI, and Llama 3.1 via Ollama. Supports PDF upload, chat history, and chat bubbles UI.
# 📘 RAG Chatbot with Llama 3.1, FAISS, LangChain & Streamlit

This is a fully functional **Retrieval-Augmented Generation (RAG)** chatbot that can read PDFs and answer questions based on their content — built using:

- **Llama 3.1 (via Ollama)**
- **LangChain 0.2.x**
- **FAISS Vector Database**
- **Sentence Transformers Embeddings**
- **Streamlit**
- **Chat Bubble UI**
- **Advanced Text Extraction (Pro-Max Mode)**

---

## 🚀 Features

### ✔ PDF Upload  
Upload any PDF such as resumes, textbooks, notes, guides, etc.

### ✔ RAG Pipeline  
- Text Extraction  
- Chunk Splitting  
- Embedding using HuggingFace  
- FAISS Vector Storage  
- Querying & Retrieval  
- Answer generation using **Llama 3.1 locally**

### ✔ Chat Interface  
- Chat history  
- Chat bubbles (WhatsApp/ChatGPT style)  
- Clean readability  

### ✔ Pro-Max Text Cleaning  
Fixes common resume PDF issues:  
- Missing spaces  
- Stuck words (e.g., TECHNICALSKILLS)  
- Bullet points  
- Headings merging  
- Bad formatting  

---

## 🛠️ Tech Stack

| Component | Tool |
|----------|------|
| UI | Streamlit |
| LLM | Llama 3.1 (Ollama) |
| Retrieval | FAISS |
| Embeddings | HuggingFace Sentence Transformers |
| Framework | LangChain 0.2+ |
| Local Runtime | Mac M1/M2/M3 optimized |

---

## 📦 Installation

### 1️⃣ Clone the repo
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
2️⃣ Create virtual environment
python3 -m venv ragenv
source ragenv/bin/activate
3️⃣nstall & pull Llama model
Install Ollama:
https://ollama.ai
Then pull Llama 3.1:
ollama pull llama3.1
▶️ Run the Chatbot
streamlit run ragbot2.py
