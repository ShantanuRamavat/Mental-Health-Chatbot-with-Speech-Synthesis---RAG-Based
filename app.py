import os
from flask import Flask, render_template, request, jsonify
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = Flask(__name__, static_folder='static')

# Groq API Key
groq_api_key = ""

def initialize_llm():
    llm = ChatGroq(
        temperature=0,
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile"
    )
    return llm

def create_vector_db():
    loader = DirectoryLoader("./data", glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory='./chroma_db')
    vector_db.persist()
    return vector_db

def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt_template = """You are an expert in reading human behaviour. Answer the questions accordingly
        {context}
        User: {question}
        Chatbot: """
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

# Initialize components
llm = initialize_llm()
db_path = "./chroma_db"
vector_db = create_vector_db() if not os.path.exists(db_path) else Chroma(
    persist_directory=db_path,
    embedding_function=HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
)
qa_chain = setup_qa_chain(vector_db, llm)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/ask', methods=['POST'])
def ask():
    try:
        response = qa_chain.run(request.form['query'])
        return jsonify({'answer': response})
    except Exception as e:
        return jsonify({'answer': f"Sorry, an error occurred: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)
