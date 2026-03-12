from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from retriever import search_engine, question

# 1. Unlock the .env file to get your secret password
load_dotenv()

# 2. Wake up the Mistral AI Brain
print("Waking up Mistral AI...")
llm = ChatMistralAI(model="mistral-small-latest")

# 3. Use the search engine from Phase 5 to get the French rule
print("Fetching the French rule from the database...")
results = search_engine.invoke(question)
context = results[0].page_content

# 4. Give Mistral its instructions! 
prompt = f"""
You are a bilingual HR assistant. Read the following French company rule, and answer the user's question in English.

Rule: {context}
Question: {question}
"""

# 5. Send it to Mistral and print the final answer
print("Thinking...\n")
response = llm.invoke(prompt)

print("🤖 FINAL BILINGUAL AI ANSWER:")
print(response.content)