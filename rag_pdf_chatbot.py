import os

# Disable LangSmith
os.environ["LANGSMITH_TRACING"] = "false"

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize model
model = init_chat_model("gpt-4o-mini")

# Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Vector store
vector_store = InMemoryVectorStore(embeddings)

# Load PDF
loader = PyPDFLoader("Summary_PhD_thesis_Manco.pdf")
docs = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)

all_splits = text_splitter.split_documents(docs)

# Store embeddings
vector_store.add_documents(all_splits)

# Create retriever
retriever = vector_store.as_retriever()

# Query
query = "What is this document about?"

# Retrieve relevant chunks
retrieved_docs = retriever.invoke(query)

# Build context
context = "\n\n".join([doc.page_content for doc in retrieved_docs])

# Prompt
prompt = f"""
Answer the question based only on the context below:

{context}

Question: {query}
"""

# LLM response
response = model.invoke(prompt)

print("\nAnswer:")
print(response.content)
