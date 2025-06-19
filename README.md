# 📄 ListaChatGPT – Assistente Inteligente para PDFs

O **ListaChatGPT** é um assistente conversacional que responde a perguntas com base em documentos PDF enviados pelo usuário. Ele utiliza técnicas de Processamento de Linguagem Natural (PLN), embeddings semânticos e integração com o modelo GPT-3.5 para oferecer respostas contextuais e precisas.

## 🚀 Funcionalidades

- Upload de múltiplos arquivos PDF.
- Extração de texto com `PyPDF2`.
- Segmentação em sentenças com expressões regulares.
- Geração de embeddings com `sentence-transformers` (modelo `all-MiniLM-L6-v2`).
- Redução de dimensionalidade com PCA.
- Indexação e busca semântica com `Annoy`.
- Geração de respostas com GPT-3.5 (via API da OpenAI).
- Interface interativa com `Streamlit`.

## 🧠 Tecnologias Utilizadas

### Backend e IA
- **Python 3.10+**
- **FastAPI** (versões futuras)
- **GPT-3.5-turbo** via OpenAI API
- **Embeddings:** SentenceTransformer (`all-MiniLM-L6-v2`)
- **Indexação:** Annoy (local), compatível com ChromaDB ou FAISS
- **PCA:** Scikit-learn (`sklearn.decomposition.PCA`)

### Frontend
- **Streamlit** (protótipo)
- Futura migração para **React.js** + Vercel

### Infraestrutura e Dados
- Hospedagem futura: Render, Railway ou Google Cloud Run
- Banco de dados local: JSON/CSV
- Alternativas em produção: MongoDB Atlas, PostgreSQL via Supabase
- CI/CD: GitHub Actions
- Logs: Loguru

## 📁 Estrutura do Projeto

📦listachatgpt
┣ 📜app.py # Código principal com interface Streamlit
┣ 📜requirements.txt # Dependências do projeto
┗ 📂data # (Opcional) PDFs enviados

bash
Copy
Edit

## ▶️ Como Rodar Localmente

### 1. Clone o repositório

git clone https://github.com/seu-usuario/listachatgpt.git
cd listachatgpt 

### 2. Instale as dependências
bash
pip install -r requirements.txt

### 3. Configure a chave da API da OpenAI
Crie um arquivo .env ou defina no terminal:
bash
export OPENAI_API_KEY="sua-chave-aqui"

### 4. Execute o projeto
bash
streamlit run app.py
