from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()
# 1. Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Load existing vector database
db = Chroma(
    persist_directory="vectorstore",
    embedding_function=embedding_model
)

# 3. Create retriever
retriever = db.as_retriever(search_kwargs={"k": 10})

# 4. Create Groq LLM
llm = ChatGroq(model=os.getenv("MODEL"),temperature=0)

# 5. Ask user
query = input("Ask: ")

# 6. Retrieve relevant chunks
docs = retriever.invoke(query)

# 7. Combine retrieved chunks
context = "\n\n".join(doc.page_content for doc in docs)

# 8. Create prompt
prompt = f"""
You are a helpful assistant answering questions based on the provided document.

Answer the user's question using ONLY the information in the context below.
If the answer is not present in the context, say:
"I don't know based on the provided document."

Context:
{context}

Question:
{query}

Answer:
"""

# 9. Ask Groq
response = llm.invoke(prompt)

# 10. Print answer
print("\nAnswer:")
print(response.content)