# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
## RAG Q&A Conversation With PDF Including Chat History
import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import chromadb

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Ensure API key is loaded
if not groq_api_key:
    st.error("Groq API Key not found! Please set it in the .env file.")
    st.stop()

# Clear system cache
chromadb.api.client.SharedSystemClient.clear_system_cache()

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Streamlit UI
st.title("Conversational RAG With PDF Uploads and Chat History")
st.write("Upload PDFs and chat with their content")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

# Session ID input
session_id = st.text_input("Session ID", value="default_session")

# Manage chat history state
if 'store' not in st.session_state:
    st.session_state.store = {}

# PDF Uploads
uploaded_files = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)

if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        temp_pdf = "./temp.pdf"
        with open(temp_pdf, "wb") as file:
            file.write(uploaded_file.getvalue())

        loader = PyPDFLoader(temp_pdf)
        docs = loader.load()
        documents.extend(docs)

    # Split text and create embeddings
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # System prompts
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question, "
        "formulate a standalone question that can be understood "
        "without context. Do NOT answer the question—just rewrite it."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Answering system
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep it concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create RAG chains
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Function to manage session chat history
    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # User input for Q&A
    user_input = st.text_input("Your question:")
    if user_input:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        st.write("Assistant:", response['answer'])
        st.write("Chat History:", session_history.messages)
