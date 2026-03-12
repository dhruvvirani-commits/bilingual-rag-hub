import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. Unlock the secret AI password
load_dotenv()

st.title("Enterprise AI Document Hub 🏢")
st.write("Chat with your documents in real-time!")

# 2. Connect to the Memory & Brain
@st.cache_resource
def load_ai_tools():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    search_engine = db.as_retriever(search_kwargs={"k": 3})
    llm = ChatMistralAI(model="mistral-small-latest")
    return search_engine, llm

search_engine, llm = load_ai_tools()

# 3. Create the "Short-Term Memory" for the Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all previous messages on the screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. The New ChatGPT-Style Input Box
if user_question := st.chat_input("Ask a question about your PDF..."):
    
    # Save the user's question to memory and display it
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # 5. Generate the AI Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            # Fetch the document context
            results = search_engine.invoke(user_question)
            context = "\n\n".join([doc.page_content for doc in results])
            
            # Grab the last few chat messages so the AI remembers the context
            chat_history = ""
            for msg in st.session_state.messages[-3:]: # Remembers last 3 messages
                chat_history += f"{msg['role']}: {msg['content']}\n"
            
            # The Enterprise Prompt
            prompt = f"""
            You are a highly intelligent bilingual AI assistant. 
            Use the following document Context and the Chat History to answer the user's latest question.
            
            Chat History:
            {chat_history}
            
            Document Context: 
            {context}
            
            Latest Question: {user_question}
            """
            
            # Get the answer and display it
            response = llm.invoke(prompt)
            st.markdown(response.content)
            
            # Save the AI's answer to memory
            st.session_state.messages.append({"role": "assistant", "content": response.content})
            
            # Show the proof
            with st.expander("View Source Paragraphs"):
                st.info(context)