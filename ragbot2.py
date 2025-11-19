import os
import tempfile
import streamlit as st
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

def clean_extracted_text(text):

    text = re.sub(r"(?<=[a-zA-Z])(?=[A-Z][a-z])", " ", text)

    text = re.sub(r"([A-Z]{2,})([A-Z][a-z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z])([0-9])", r"\1 \2", text)
    text = re.sub(r"([0-9])([A-Za-z])", r"\1 \2", text)

    text = text.replace("•", "\n• ")
    text = re.sub(r"\n\s*[-–]\s*", "\n- ", text)  # normalize dashes
    text = re.sub(r"(?<!\n)(•|-)\s", r"\n\1 ", text)

    text = re.sub(
        r"([A-Z][A-Z ]{3,})([A-Z][a-z])",
        r"\1\n\n\2",
        text
    )

    text = re.sub(r"\s{2,}", " ", text)

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

chat_css = """
<style>
.chat-container {
    width: 100%;
    margin-top: 10px;
}

.user-bubble {
    background-color: #0b93f6;
    color: white;
    padding: 10px 15px;
    border-radius: 15px;
    margin: 5px;
    max-width: 70%;
    float: right;
    clear: both;
}

.bot-bubble {
    background-color: #e5e5ea;
    color: black;
    padding: 10px 15px;
    border-radius: 15px;
    margin: 5px;
    max-width: 70%;
    float: left;
    clear: both;
}
</style>
"""
st.markdown(chat_css, unsafe_allow_html=True)

st.title("RAG Chatbot with Llama 3.1 & Streamlit")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        for d in documents:
            d.page_content = clean_extracted_text(d.page_content)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        db = FAISS.from_documents(chunks, embeddings)
        retriever = db.as_retriever()

        llm = Ollama(model="llama3.1")
        prompt = PromptTemplate(
    input_variables=["context", "input"],
    template="""
Use ONLY the following context to answer the question.
Context:
{context}

Question:
{input}

Answer:
""",
)

        # RetrievalQA chain (old & stable)
        combine_chain = create_stuff_documents_chain(llm,prompt)

        # Final RAG chain
        rag_chain = create_retrieval_chain(retriever, combine_chain)


        question = st.text_input("Ask me anything from the PDF:")

        if question:
            question = clean_extracted_text(question)
            result = rag_chain.invoke({"input": question})
            answer = clean_extracted_text(result["answer"])
            # Save to session chat history
            st.session_state.chat_history.append(("user", question))
            st.session_state.chat_history.append(("bot", answer))

# Display the chat
            for role, msg in st.session_state.chat_history:
                if role == "user":
                    st.markdown(f"<div class='chat-container'><div class='user-bubble'>{msg}</div></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='chat-container'><div class='bot-bubble'>{msg}</div></div>", unsafe_allow_html=True)

            st.markdown("**Answer:**")
            st.write(answer)
    except Exception as e:
        st.error(f"Error while processing file: {e}")

    finally:
        try:
            os.remove(tmp_path)
        except:
            pass

