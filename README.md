# AI Research Paper RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about AI research papers using vector search and language models.

## 🎯 Project Overview

This chatbot demonstrates modern AI engineering techniques by combining:
- **Document Processing**: Loads and chunks academic PDFs
- **Vector Embeddings**: Converts text into searchable embeddings
- **Semantic Search**: Uses FAISS for efficient similarity search
- **LLM Integration**: Generates answers using retrieved context

## 📚 Research Papers Included

Attention Is All You Need (Transformers)
BERT: Pre-training of Deep Bidirectional Transformers
GPT-3: Language Models are Few-Shot Learners
RAG: Retrieval-Augmented Generation
LoRA: Low-Rank Adaptation
Constitutional AI
Chain-of-Thought Prompting
LLaMA and more...

## 🛠️ Tech Stack

- **LangChain**: RAG orchestration framework
- **HuggingFace**: Embeddings and language models
- **FAISS**: Vector database for similarity search
- **Streamlit**: Web interface
- **PyPDF**: PDF processing

## 🚀 How to Run

### Prerequisites
- Python 3.10+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/salonibhulinfo-byte/rag-chatbot.git
cd rag-chatbot
```

2. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Build the vector store (first time only):
```bash
python build_vectorstore.py
```

5. Run the chatbot:
```bash
streamlit run app.py
```

## 💡 How It Works

1. **Document Loading**: PDFs are loaded and split into chunks
2. **Embedding Creation**: Each chunk is converted to a vector using sentence-transformers
3. **Vector Storage**: Embeddings stored in FAISS for fast retrieval
4. **Query Processing**: User questions are embedded and matched against stored vectors
5. **Context Retrieval**: Top 3 most relevant chunks are retrieved
6. **Answer Generation**: LLM generates answer based on retrieved context

## 📊 Project Structure
```
rag-chatbot/
├── data/                    # Research papers (PDFs)
├── vectorstore/             # FAISS vector database
├── process_documents.py     # Document loading & chunking
├── build_vectorstore.py     # Create embeddings & vector store
├── rag_chain.py            # RAG chain implementation
├── app.py                  # Streamlit web interface
├── requirements.txt        # Python dependencies
└── README.md
```

## 🔧 Known Limitations

- Using flan-t5-small for cost efficiency - responses may be repetitive
- Local model runs on CPU, so inference is slower (~10-30 seconds per query)
- For production deployment, would use GPT-4 or Claude for better quality

## 🎓 Learning Outcomes

Built as part of my journey to become an AI Engineer. This project demonstrates:
- Understanding of RAG architecture
- Vector database implementation
- LLM integration and prompt engineering
- End-to-end ML application development

## 📝 Future Improvements

Integrate GPT-4 API for better answers
Add citation/source tracking
Implement conversation memory
Deploy to Streamlit Cloud
Add more research papers
Improve chunk sizing and overlap