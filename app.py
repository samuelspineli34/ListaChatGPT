import os
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import streamlit as st
from annoy import AnnoyIndex
import re
from sklearn.decomposition import PCA
from dotenv import load_dotenv
import json

load_dotenv('chave.env')
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("""Erro: Chave API não encontrada. Verifique se:
                1. Você criou um arquivo .env
                2. Adicionou OPENAI_API_KEY=sua_chave
                3. O arquivo está na pasta correta""")
    st.stop()

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ''
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def split_sentences(text):
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text)
    return sentences

def process_pdfs(pdf_files):
    all_sentences = []
    for pdf_file in pdf_files:
        pdf_text = extract_text_from_pdf(pdf_file)
        if not pdf_text:
            st.warning(f"O arquivo {pdf_file.name} está vazio ou não contém texto extraível.")
            continue
        sentences = split_sentences(pdf_text)
        all_sentences.extend(sentences)

    if not all_sentences:
        st.error("Não foi possível extrair texto dos PDFs fornecidos.")
        return None, None, None, None

    embeddings = model.encode(all_sentences)
    embeddings = np.array(embeddings)

    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        st.error("Os embeddings contêm valores inválidos (NaN ou infinito).")
        return None, None, None, None

    n_samples = embeddings.shape[0]
    n_components = min(100, n_samples)
    if n_components == 0:
        st.error("Não há sentenças suficientes para aplicar o PCA.")
        return None, None, None, None

    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)

    vector_length = reduced_embeddings.shape[1]
    annoy_index = AnnoyIndex(vector_length, 'angular')
    for i, vector in enumerate(reduced_embeddings):
        annoy_index.add_item(i, vector)
    annoy_index.build(10)

    return all_sentences, reduced_embeddings, annoy_index, pca

def search_similar_sentences(query, pca, annoy_index, sentences, top_k=5):
    query_embedding = model.encode([query])[0]
    query_embedding = pca.transform([query_embedding])[0]
    indices = annoy_index.get_nns_by_vector(query_embedding, n=top_k, include_distances=False)
    results = [sentences[idx] for idx in indices]
    return results

async def generate_response(question, context):
    prompt = f"Contexto:\n{context}\n\nPergunta:\n{question}"
    chatHistory = []
    chatHistory.append({"role": "user", "parts": [{"text": prompt}]})
    payload = {"contents": chatHistory}
    apiKey = ""
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={apiKey}"

    try:
        simulated_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Esta é uma resposta simulada da API do Gemini baseada no seu contexto e pergunta."}
                        ]
                    }
                }
            ]
        }
        
        result = simulated_response

        if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
            return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            st.error("Erro: A estrutura da resposta da API do Gemini é inesperada ou o conteúdo está faltando.")
            return "Não foi possível gerar uma resposta para sua pergunta."

    except Exception as e:
        st.error(f"Ocorreu um erro ao chamar a API do Gemini: {e}")
        return "Desculpe, não consegui processar sua solicitação no momento."

def main():
    st.title("Lista Chat GPT")
    st.write("Faça perguntas sobre os documentos PDF.")

    uploaded_files = st.file_uploader(
        "Faça upload dos seus arquivos PDF",
        type="pdf",
        accept_multiple_files=True
    )

    if 'sentences' not in st.session_state:
        st.session_state['sentences'] = None
        st.session_state['annoy_index'] = None
        st.session_state['pca'] = None

    if uploaded_files:
        if st.button("Enviar PDFs"):
            with st.spinner('Processando os arquivos...'):
                sentences, embeddings, annoy_index, pca = process_pdfs(uploaded_files)
                st.session_state['sentences'] = sentences
                st.session_state['annoy_index'] = annoy_index
                st.session_state['pca'] = pca
            st.success("PDFs processados com sucesso!")

    if st.session_state['sentences'] is not None:
        question = st.text_input("Digite sua pergunta:")
        if st.button("Enviar Pergunta"):
            if question:
                with st.spinner('Gerando resposta...'):
                    similar_sentences = search_similar_sentences(
                        question,
                        st.session_state['pca'],
                        st.session_state['annoy_index'],
                        st.session_state['sentences']
                    )
                    context = "\n".join(similar_sentences)
                    
                    st.session_state['answer'] = "Gerando resposta..."
                    try:
                        answer = "Esta é uma resposta simulada para teste local." # Substitua pela chamada real
                        st.session_state['answer'] = answer

                    except Exception as e:
                        st.error(f"Erro ao obter resposta da API: {e}")
                        st.session_state['answer'] = "Não foi possível obter a resposta."

                st.write("**Resposta:**")
                st.write(st.session_state['answer'])
            else:
                st.warning("Por favor, digite uma pergunta.")
    else:
        st.info("Por favor, faça o upload de arquivos PDF e clique em 'Enviar PDFs' para começar.")

if __name__ == "__main__":
    main()
