import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Page config
st.set_page_config(page_title="AI Research Paper Assistant", page_icon="📚")

@st.cache_resource
def load_rag_chain():
    """Load vector store and create RAG chain (cached)"""
    # Load embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Load vector store
    vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    
    # Load language model
    model_id = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    
    # Create RAG chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    template = """Answer the question based on the context below. Keep it concise.

Context: {context}

Question: {question}

Answer: """
    
    prompt = PromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

# Title and description
st.title("📚 AI Research Paper Assistant")
st.markdown("Ask questions about AI research papers (Transformers, RAG, LoRA, etc.)")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Load RAG chain
with st.spinner("Loading AI model..."):
    chain = load_rag_chain()

# Chat input
if prompt := st.chat_input("Ask about AI research papers..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Searching papers..."):
            response = chain.invoke(prompt)
            st.markdown(response)
    
    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This chatbot uses RAG (Retrieval Augmented Generation) to answer questions about AI research papers.
    
    **Papers included:**
    - Attention Is All You Need (Transformers)
    - BERT
    - GPT-3
    - RAG
    - LoRA
    - And more!
    
    **Built with:**
    - LangChain
    - HuggingFace
    - FAISS
    - Streamlit
    """)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()