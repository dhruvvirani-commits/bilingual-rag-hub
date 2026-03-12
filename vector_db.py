from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from document_loader import chunks

print("Downloading the AI memory model... (This turns text into numbers)")
# We use a fast, free local model to do the translation
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Saving the 5 chunks into the Vector Database...")
# This creates a new folder called "chroma_db" and saves the memory there
db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

print("Success! The AI has permanently memorized the French handbook.")