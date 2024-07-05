import os
import openai
import dotenv
import fitz
import streamlit as st
from io import BytesIO
import numpy as np
import re
import streamlit.components.v1 as components

dotenv.load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

openai.api_type = "azure"
openai.api_base = endpoint
openai.api_version = "2024-02-01"
openai.api_key = api_key

st.set_page_config(page_title="AutiAlly", initial_sidebar_state="auto")
st.title("🧠 AutiAlly 🧩")
st.markdown("*Tudo* sobre **Transtorno do espectro autista** a sua ***disposição!***")

pdf_file_paths = [ 
    "./TRANSTORNODOESPECTROAUTISTA.pdf",
    "./TratamentosTerapêuticosCrianças.pdf",
]

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

@st.cache_data
def create_embeddings(all_text_segments):
    embeddings = []
    for segment in all_text_segments:
        response = openai.Embedding.create(input=segment, engine=embedding_deployment)
        embeddings.append(response['data'][0]['embedding'])
    return embeddings

def search_with_embeddings(query, embeddings, all_text_segments):
    response = openai.Embedding.create(input=query, engine=embedding_deployment)
    query_embedding = response['data'][0]['embedding']

    scores = [np.dot(query_embedding, emb) for emb in embeddings]
    best_match_index = np.argmax(scores)
    return best_match_index, all_text_segments[best_match_index]

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Olá! Como posso ajudar você hoje?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

def is_valid_input(prompt):
    if re.search(r'[a-zA-Z0-9]', prompt) and len(prompt) > 2:
        return True
    return False

def split_text_into_segments(text, max_tokens):
    words = text.split()
    num_words = len(words)
    segments = []

    for i in range(0, num_words, max_tokens):
        segment = ' '.join(words[i:i + max_tokens])
        segments.append(segment)

    return segments

def load_and_process_pdfs(pdf_file_paths):
    all_text_segments = []
    for file_path in pdf_file_paths:
        text = extract_text_from_pdf(file_path)
        segments = split_text_into_segments(text, 30)
        all_text_segments.extend(segments)
    return all_text_segments

all_text_segments = load_and_process_pdfs(pdf_file_paths)
embeddings = create_embeddings(all_text_segments)

if question:= st.chat_input("Qual é a sua dúvida hoje?"):
    question = question.strip().lower()
    if question and is_valid_input(question):
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)
        
        with st.spinner("Processando..."): 
                try:
                    best_match_index, pdf_text = search_with_embeddings(question, embeddings, all_text_segments)

                    response = openai.ChatCompletion.create(
                        engine=deployment,
                        messages=[
                            {"role": "system", "content": "Você é um especialista em Transtorno do Espectro Autista que auxilía com educação pessoas autistas ou que queiram saber sobre autismo."+ 
                             "caso nao seja com base nas informações fornecidas responda:" + 
                             "Desculpe, só sei informar sobre o transtorno do espectro autista.Use apenas o seguinte texto para elaborar boas respostar e responder à pergunta do usuário: " + pdf_text},
                            {"role": "user", "content": question}
                        ],
                        max_tokens=150,
                        n=1,
                        temperature=0.7,
                    )
                    msg = response['choices'][0]['message']['content']
                    st.session_state.messages.append({"role": "assistant", "content": msg})
                    st.chat_message("assistant").write(msg)
                except Exception as e:
                    print(f"Erro ao chamar API: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": "Desculpe, não consegui entender sua pergunta."})
                    st.chat_message("assistant").write("Desculpe, não consegui entender sua pergunta.")
    else:
        st.write("Por favor, insira uma pergunta válida.")
