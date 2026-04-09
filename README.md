# RAG PDF Chatbot

This project implements a Retrieval-Augmented Generation (RAG) system to query PDF documents using Large Language Models.

## Overview

The system allows users to ask questions about a PDF document and receive context-aware answers.

## Features

- PDF document ingestion
- Text chunking and preprocessing
- Embedding generation using OpenAI
- Semantic search via vector store
- LLM-based answer generation

## Tech Stack

- Python
- LangChain
- OpenAI API

## How it works

1. Load a PDF document
2. Split it into chunks
3. Convert chunks into embeddings
4. Store embeddings in a vector database
5. Retrieve relevant chunks based on a query
6. Generate an answer using an LLM

## Example

Query:
"What is this document about?"

Output:
A summary generated from the most relevant parts of the document.
