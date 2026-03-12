# Enterprise Bilingual Document AI 🧠🏢

An intelligent, bilingual (French/English) Retrieval-Augmented Generation (RAG) application. This system allows users to upload complex PDF documents—such as dropshipping supplier catalogs, Etsy store policies, or legal agreements—and chat with them in real-time.

## Features
* **Dynamic PDF Ingestion:** Instantly reads and chunks messy, real-world PDFs on the fly.
* **Conversational Memory:** Maintains chat history for seamless follow-up questions.
* **Bilingual Processing:** Built to comprehend French documents and answer accurately in English (or vice versa).

## Project Structure
This repository showcases a dual-architecture approach:
1. **The Modular Backend:** Individual scripts (`document_loader.py`, `vector_db.py`, `retriever.py`, `generator.py`) demonstrate the step-by-step data pipeline and vector embeddings.
2. **The Unified Frontend:** The `app.py` file combines all logic into a seamless, interactive Streamlit UI with drag-and-drop document processing.

## Tech Stack
* **Frontend:** Streamlit
* **LLM Engine:** Mistral AI (mistral-small-latest)
* **Orchestration:** LangChain
* **Vector Database:** ChromaDB
* **Embeddings:** HuggingFace (all-MiniLM-L6-v2)

## How to Run Locally
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file and add your Mistral API key: `MISTRAL_API_KEY=your_key_here`
4. Launch the web app: `streamlit run app.py`