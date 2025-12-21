import streamlit as st
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(
    page_title="HiveAI",
    layout="wide"
)

session_id = "123"
loader = PyPDFLoader("./faq.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 3000,
    chunk_overlap = 300
)
splits = text_splitter.split_documents(docs)
embeddings = OllamaEmbeddings(model="embeddinggemma:300m")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

SYSTEM_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

CONTEXTUALIZE_Q_PROMPT = (
    "Given a chat history and the latest user question"
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

if 'store' not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id]=ChatMessageHistory()
        return st.session_state.store[session_id]
        
def generate_response(user_input, model, temperature, max_tokens):

    llm = ChatOllama(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_Q_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}")
        ]
    )

    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    conversational_rag_chain=RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    
    response = conversational_rag_chain.invoke(
        {"input": user_input},
        config = {
            "configurable": {"session_id": session_id}
        }
    )
    return response["answer"]

if "messages" not in st.session_state:
    st.session_state.messages = []

# PAGE TITLE
st.title("HiveAI - Chatbot")

col1, col2 = st.columns([7.5, 2.5])

with col2:
    st.header("Settings")
    model = st.selectbox("Models", ["llama3.1:8b", "deepseek-r1:14b", "mistral:7b"])
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
    max_tokens = st.slider("Max Token", min_value=50, max_value=300, value=150)

with col1:
    user_input = st.chat_input("What is up?", width=1250)
    with st.container(height=500):
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if user_input:
            with st.chat_message("user"):
                st.write(user_input)

            response = generate_response(user_input, model, temperature, max_tokens)
            with st.chat_message("assistant"):
                st.write(response)

            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": response})



