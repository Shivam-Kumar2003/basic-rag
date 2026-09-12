# Basic RAG — PDF Question Answering

A beginner-friendly **Retrieval-Augmented Generation (RAG)** application that allows users to ask questions about a PDF document.

This project uses **LangChain**, **Chroma**, **Hugging Face embeddings**, and **Groq** to retrieve relevant information from a PDF and generate answers using an LLM.

## How RAG Works

```text
PDF
 ↓
Document Loader
 ↓
Text Chunking
 ↓
Embeddings
 ↓
Chroma Vector Database
 ↓
User Question
 ↓
Similarity Search
 ↓
Relevant Chunks
 ↓
Groq LLM
 ↓
Final Answer