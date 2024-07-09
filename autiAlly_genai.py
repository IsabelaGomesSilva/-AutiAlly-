import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts.chat import ChatPromptTemplate
import streamlit as st
import re
from PyPDF2 import PdfReader
import fitz
import numpy as np
import faiss

load_dotenv()

model = os.getenv("MODEL")

llm = ChatGoogleGenerativeAI(
    temperature=0.3,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    model=model,
    max_tokens=100,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

st.set_page_config(page_title="AutiAlly", initial_sidebar_state="auto",  page_icon="🧩")
st.title("🧠 AutiAlly 🧩")
st.markdown("*Tudo* sobre **Transtorno do espectro autista** a sua ***disposição!***")

pdf_file_paths = [ 
    "./TRANSTORNODOESPECTROAUTISTA.pdf",
    "./TratamentosTerapêuticosCrianças.pdf",
]

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Olá! Como posso ajudar você hoje?"}]

for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])

def is_valid_input(prompt):
    if re.search(r'[a-zA-Z0-9]', prompt) and len(prompt) > 2:
        return True
    return False

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text


def load_and_process_pdfs(pdf_file_paths):
    all_text = ""
    for file_path in pdf_file_paths:
        text = extract_text_from_pdf(file_path)
        all_text += text + "\n\n"
    return all_text

def get_relevant_text(question, pdf_text):
   
    sections = pdf_text.split('\n\n')
    
    question_embedding = embeddings.embed_query(question)
    
    section_embeddings = [embeddings.embed_query(section) for section in sections]
    
    index = faiss.IndexFlatL2(len(question_embedding))
    index.add(np.array(section_embeddings))
    D, I = index.search(np.array([question_embedding]), 1)

    return sections[I[0][0]]

@st.cache_data
def chatbot_interaction(question, relevant_text):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an Autism Spectrum Disorder specialist who helps with education for autistic people or those who want to know about autism"+
         " and answer in Brazilian Portuguese."+ 
         "if not based on the information provided, please answer:" + 
         "Sorry, I can only tell you about autism spectrum disorder. Just use the following text to write a good response and answer the user's question:" + relevant_text),
        ("user", f"{question} ")
    ])
    print(relevant_text)
    prompt_text = prompt.format_prompt(question=question)
    response = llm.invoke(prompt_text)
    return response.content

pdf_text = load_and_process_pdfs(pdf_file_paths)

if question := st.chat_input("Qual é a sua dúvida hoje?"):
    question = question.strip().lower()

    if question and is_valid_input(question):
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)
        
        with st.spinner("Processando..."):
            try:
                relevant_text = get_relevant_text(question, pdf_text)
                response = chatbot_interaction(question, relevant_text)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.chat_message("assistant").write(response)
            except Exception as e:
                print(f"Erro ao chamar API: {e}")
                st.session_state.messages.append({"role": "assistant", "content": "Desculpe, não consegui entender sua pergunta."})
                st.chat_message("assistant").write("Desculpe, não consegui entender sua pergunta.")
    else:
        st.write("Por favor, insira uma pergunta válida.")