import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.chat import ChatPromptTemplate
import streamlit as st
import re
from PyPDF2 import PdfReader
import fitz
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

load_dotenv()

model = os.getenv("MODEL")

llm = ChatGoogleGenerativeAI(
    temperature=0.5,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    model=model,
    max_tokens=100,
)

autiAlly ="./assets/autiAlly.png"
user = "./assets/user.png"

st.set_page_config(page_title="AutiAlly", initial_sidebar_state="auto",  page_icon=f"""{autiAlly}""")
st.markdown(
    f"""
    <div style=" display: flex; justify-content: center; align-items: flex-end; margin-top: 20px;">
        <img src="https://i.imgur.com/DAFvhR3.png" width="150"> 
        <h1 style="margin-left: 20px; margin-bottom: 10px;"> AutiAlly</h1>
    </div>
    <h5 style="margin-top: 20px; justify-content: center; align-items: center;">🧠 Tudo sobre Transtorno do espectro autista! 🧩</h5>
   
    """, unsafe_allow_html=True)
 
pdf_file_paths = [ 
    "./doc/tea.pdf",
    "./doc/TRANSTORNODOESPECTROAUTISTA_red.pdf",
    "./doc/TratamentosTerapêuticosCrianças.pdf",
    "./doc/autiamoAlimentação.pdf",
    "./doc/sintomas.pdf",
    "./doc/inclusodeautistasnomercadodetrabalho.pdf"
]

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Olá! o que você gostaria de saber sobre autismo? Estou aqui para te ajudar!"}]

for message in st.session_state["messages"]:
    st.chat_message(message["role"], avatar=autiAlly if message["role"] == "assistant" else user).write(message["content"])

def is_valid_input(prompt):
    return re.search(r'[a-zA-Z0-9]', prompt) and len(prompt) > 2

def extract_text_from_pdf(pdf_path):
  
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

@st.cache_data
def load_and_process_pdfs(pdf_file_paths):
    all_text = ""
    for file_path in pdf_file_paths:
        all_text += extract_text_from_pdf(file_path) + "\n\n"
        print(all_text)
    return all_text

@st.cache_data
def preprocess_pdfs(pdf_file_paths):
    documents = []
    text = load_and_process_pdfs(pdf_file_paths)
    sentences = text.split('. ')
    current_segment = ""
    current_length = 0
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > 500:
            documents.append(current_segment)
            current_segment = sentence
            current_length = sentence_length
        else:
            current_segment += " " + sentence
            current_length += sentence_length
    if current_segment:
        documents.append(current_segment)
    
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(documents).toarray()
    return embeddings, documents, vectorizer

if not os.path.exists("embeddings.npy") or not os.path.exists("documents.pkl") or not os.path.exists("vectorizer.pkl"):
    embeddings, documents, vectorizer = preprocess_pdfs(pdf_file_paths)
    np.save("embeddings.npy", embeddings)
    joblib.dump(documents, "documents.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
else:
    embeddings = np.load("embeddings.npy")
    documents = joblib.load("documents.pkl")
    vectorizer = joblib.load("vectorizer.pkl")

def similarity_search(query):
    query_embedding = vectorizer.transform([query]).toarray()
    similarities = cosine_similarity(query_embedding, embeddings).flatten()
    indices = similarities.argsort()[-6:][::-1]
    return [documents[idx] for idx in indices]

def build_context():
    context = ""
    for message in st.session_state["messages"]:
        role = message["role"]
        content = message["content"]
        context += f"{role}: {content}\n"
    return context

def chatbot_interaction(question):
    docs = similarity_search(question)
    context = build_context()
    context += " ".join(docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """O seu nome é AutiAlly e você é um assistente virtual especializado em Transtorno do Espectro Autista (TEA). 
            Seu objetivo é fornecer informações precisas e úteis sobre TEA para ajudar autistas e pessoas interessadas no assunto.
            Responda às perguntas abaixo de forma clara e detalhada, em português brasileiro ou inglês, conforme a pergunta do usuário.
            Utilize apenas o texto fornecido para elaborar suas respostas,mas não precisa informar isso ao usuário. 
            Você é um especialista em autismo que só respode sobre autismo, não fale de outro tópico Não traduza o seu contexto para usuario.
            Não fale com o usuario sobre outros tópicos, não faça resumos sobre outros tópicos além do autismo.
            Não invente novas informações e não mencione ao usuário que as respostas são baseadas no texto fornecido.
            Não cite o texto. De a resposta completa mas Não de uma resposta muito grande.

        Contexto da conversa:""" + context ), 
        ("user", f"{question}")
    ])
    prompt_text = prompt.format_prompt(question=question)
    try:
        response = llm.invoke(prompt_text)
    except Exception as e:
        response = f"Erro ao chamar API: {e}"
    
    return response.content

if question := st.chat_input("Qual é a sua dúvida hoje?"):
    question = question.strip().lower()

    if question and is_valid_input(question):
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user", avatar=user).write(question)
        with st.spinner("Processando..."):
            try:
                response = chatbot_interaction(question)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.chat_message("assistant", avatar=autiAlly).write(response)
            except Exception as e:
                print(f"Erro ao chamar API: {e}")
                st.session_state.messages.append({"role": "assistant", "content": "Desculpe, não consegui entender sua pergunta."})
                st.chat_message("assistant", avatar=autiAlly).write("Desculpe, não consegui entender sua pergunta.")
    else:
        st.write("Por favor, insira uma pergunta válida.")