from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Use the new PDF Loader for your contract
print("Opening the PDF Contract...")
loader = PyPDFLoader("Data/House Contract.pdf")
documents = loader.load()

# 2. Upgrade to an Enterprise Splitter 
# (This one is smarter and avoids cutting sentences in half)
print("Slicing the PDF into smart chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

# 3. Prove it worked!
print(f"Success! The AI ripped the PDF into {len(chunks)} chunks.")