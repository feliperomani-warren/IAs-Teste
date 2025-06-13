import streamlit as st
from langchain.chains import ConversationChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import tempfile
import os


# Carrega a API Key
api_key = st.secrets["GEMINI"]["api_key"]
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25", google_api_key=api_key)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)


# Configuração da Página
st.set_page_config(page_title="Chatbot Gemini com RAG", page_icon="🤖", layout="wide")
st.logo("assets/LogoWarrenRena.png", icon_image="assets/LogoWRena.png")
st.title("🤖 Chatbot Gemini com RAG")


# Inicialização do vetorstore no session_state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


# Upload do PDF
uploaded_file = st.file_uploader("Carregue um PDF para análise", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Carrega e processa o PDF
    loader = PyPDFLoader(tmp_file_path)
    pages = loader.load()
    
    # Divide o texto em chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(pages)
    
    # Cria o vetorstore
    st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Limpa o arquivo temporário
    os.unlink(tmp_file_path)
    st.success("PDF processado com sucesso!")


if "memory" not in st.session_state:   # Inicializa a memória e a conversação no session_state
    st.session_state.memory = ConversationBufferMemory()

if "conversation" not in st.session_state:
    st.session_state.conversation = ConversationChain(llm=llm, memory=st.session_state.memory)


if "messages" not in st.session_state:# Inicializa o histórico no session_state se ainda não existir
    st.session_state["messages"] = []

for message in st.session_state["messages"]:# Exibe o histórico de mensagens no chat
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_input = st.chat_input("Digite sua pergunta...")# Campo de input do usuário

if user_input:
    
    st.session_state["messages"].append({"role": "user", "content": user_input})# Adiciona a mensagem do usuário ao histórico

    
    # Se houver um vetorstore, usa RAG
    if st.session_state.vectorstore:
        # Busca documentos relevantes
        docs = st.session_state.vectorstore.similarity_search(user_input)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Prepara o prompt com contexto
        prompt = f"""Com base no seguinte contexto, responda à pergunta do usuário.
        
        Contexto:
        {context}
        
        Pergunta: {user_input}
        
        Resposta:"""
        
        response = st.session_state.conversation.run(prompt)
    else:
        # Resposta sem RAG
        response = st.session_state.conversation.run(user_input)

    
    st.session_state["messages"].append({"role": "assistant", "content": response})# Adiciona a resposta ao histórico

    
    with st.chat_message("assistant"):# Exibe a resposta no chat
        st.markdown(response)
        
