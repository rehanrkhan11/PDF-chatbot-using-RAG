import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

st.set_page_config(page_title="PDF Chatbot (RAG)", page_icon="🤖", layout="centered")
st.title("🤖 Your Assignment AI Assistant")
st.write("Ask any question based on the processed `sample.pdf` document.")

DB_DIR = "./faiss_db"
PDF_PATH = "sample.pdf"

@st.cache_resource
def get_vector_store():
    api_key = os.getenv("OPENAI_API_KEY")
    embedding_model = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=api_key
    )
    if os.path.exists(DB_DIR):
        return FAISS.load_local(DB_DIR, embedding_model, allow_dangerous_deserialization=True)
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Missing essential document: '{PDF_PATH}' for processing.")
    st.info("🔄 First-time setup: Processing your document vectors for the cloud server...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(chunks, embedding_model)
    vector_store.save_local(DB_DIR)
    return vector_store

try:
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(
        model="openrouter/free",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.2
    )

    system_prompt = (
        "You are an expert academic assistant. Answer the user's question using ONLY the provided context "
        "extracted from the document. If you do not know the answer or if it's not explicitly in the text, "
        "say 'I cannot find that information in the document.' Do not make up facts.\n\n"
        "Context:\n{context}"
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_question := st.chat_input("Ask something about the assignment (e.g., 'What is question 1?')"):
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})

        with st.chat_message("assistant"):
            with st.spinner("Thinking... searching your PDF..."):
                answer = rag_chain.invoke(user_question)
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.info("💡 Tip: Make sure your OPENAI_API_KEY is set in Streamlit Cloud → App Settings → Secrets.")
