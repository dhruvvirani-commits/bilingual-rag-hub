from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. Load the same translator so we speak the same "number" language
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Connect to the memory folder you just created
print("Connecting to the Vector Database...")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 3. Create the Search Engine! (We tell it to bring back the top 2 best chunks)
search_engine = db.as_retriever(search_kwargs={"k": 2})

# 4. Let's test it with a question in French!
question = "Combien de jours de télétravail sont autorisés ?"
print(f"\nUser Question: {question}")
print("Searching memory...\n")

# Go fetch the answer!
results = search_engine.invoke(question)

print("Found the most relevant chunk:")
print(f"--> {results[0].page_content}")