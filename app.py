import os
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import openai
import streamlit as st
from annoy import AnnoyIndex
import re
from sklearn.decomposition import PCA
from dotenv import load_dotenv # Importa load_dotenv

# Carrega variáveis de ambiente do arquivo 'chave.env'
# Esta é a correção crucial para o problema da chave não encontrada.
load_dotenv('chave.env')

# Obtém a chave da API OpenAI
api_key = os.getenv("OPENAI_API_KEY")

# Verifica se a chave existe
if not api_key:
    st.error("""Erro: Chave API OpenAI não encontrada. Verifique se:
                1. Você criou um arquivo 'chave.env'
                2. Adicionou OPENAI_API_KEY=sua_chave_aqui no arquivo 'chave.env'
                3. O arquivo 'chave.env' está na mesma pasta do 'app.py'""")
    st.stop() # Interrompe a execução do Streamlit se a chave não for encontrada

# Configurar a chave da API para a biblioteca OpenAI
openai.api_key = api_key

# Cachear o modelo SentenceTransformer
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

    # Criar o índice Annoy
    vector_length = reduced_embeddings.shape[1]
    annoy_index = AnnoyIndex(vector_length, 'angular')
    for i, vector in enumerate(reduced_embeddings):
        annoy_index.add_item(i, vector)
    # A variável `num_trees` do Annoy (aqui como 100) é para a precisão da busca,
    # não está diretamente ligada ao PCA. O padrão 10 é geralmente bom,
    # mas 100 pode aumentar a precisão com custo de tempo de construção.
    annoy_index.build(100) # Use 100 como o default para o número de iterações/árvores, conforme sua instrução.

    return all_sentences, reduced_embeddings, annoy_index, pca

def search_similar_sentences(query, pca, annoy_index, sentences, top_k=5):
    query_embedding = model.encode([query])[0]
    query_embedding = pca.transform([query_embedding])[0]
    indices = annoy_index.get_nns_by_vector(query_embedding, n=top_k, include_distances=False)
    results = [sentences[idx] for idx in indices]
    return results

def generate_response(question, context):
    messages = [
        {"role": "system",
         "content": "Você é um assistente inteligente que usa o contexto fornecido para responder às perguntas do usuário de forma clara e concisa."},
        {"role": "user", "content": f"Contexto:\n{context}\n\nPergunta:\n{question}"}
    ]
    try:
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo', # Mantido o modelo original do GPT-3.5-turbo
            messages=messages,
            max_tokens=200,
            temperature=0.7,
            n=1,
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        st.error(f"Ocorreu um erro ao chamar a API OpenAI: {e}")
        return "Desculpe, não consegui processar sua solicitação no momento. Por favor, verifique sua chave API ou o status do serviço."

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
                    try:
                        similar_sentences = search_similar_sentences(
                            question,
                            st.session_state['pca'],
                            st.session_state['annoy_index'],
                            st.session_state['sentences']
                        )
                        context = "\n".join(similar_sentences)
                        answer = generate_response(question, context)
                        
                        st.write("**Resposta:**")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"Ocorreu um erro ao buscar o contexto ou gerar a resposta: {e}")
                        st.write("**Resposta:** Desculpe, não foi possível gerar uma resposta no momento.")
            else:
                st.warning("Por favor, digite uma pergunta.")
    else:
        st.info("Por favor, faça o upload de arquivos PDF e clique em 'Enviar PDFs' para começar.")

if __name__ == "__main__":
    main()
