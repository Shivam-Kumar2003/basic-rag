from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

loader = PyPDFLoader("data/filename.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)

chunks = splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(documents=chunks,embedding=embedding_model,persist_directory="vectorstore")

print("PDF stored successfully")