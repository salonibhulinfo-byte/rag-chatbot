# AI Research Paper Chatbot

A chatbot that answers questions about AI research papers using RAG (Retrieval-Augmented Generation).

## What it does

I built this to learn about RAG and vector databases. You can ask it questions about AI papers like "What is attention mechanism?" or "How does LoRA work?" and it searches through research papers to give you answers.

## Papers it knows about

- Attention Is All You Need (Transformers)
- BERT
- GPT-3
- RAG paper
- LoRA
- Constitutional AI
- Chain-of-Thought Prompting
- LLaMA

## How I built it

1. Loaded 9 AI research papers (PDFs)
2. Split them into smaller chunks
3. Converted chunks into embeddings using sentence-transformers
4. Stored them in FAISS vector database
5. Built a RAG chain with LangChain
6. Added a Streamlit web interface

## Tech used

- Python
- LangChain
- HuggingFace (for embeddings and LLM)
- FAISS (vector database)
- Streamlit (web UI)

## How to run it
```bash
# Clone and setup
git clone https://github.com/salonibhulinfo-byte/rag-chatbot.git
cd rag-chatbot
python -m venv venv
venv\Scripts\activate

# Install packages
pip install -r requirements.txt

# Build vector database (first time only)
python build_vectorstore.py

# Run the chatbot
streamlit run app.py
```

## What I learned

- How RAG works in practice
- Vector databases and semantic search
- Working with LangChain
- Integrating LLMs into applications
- Building end-to-end AI projects

## Things I'd improve

- Use GPT-4 instead of the small local model for better answers
- Make it faster (currently uses CPU, would use GPU or API)
- Add source citations
- Deploy it online

Built as part of learning AI engineering after finishing my MSc in Data Science at University of Greenwich.