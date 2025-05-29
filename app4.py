from flask import Flask, render_template, request, session, redirect, url_for
from werkzeug.utils import secure_filename
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_ibm import WatsonxLLM,WatsonxEmbeddings
import tempfile

app = Flask(__name__)
app.secret_key = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()


os.environ["WATSONX_API_KEY"] = "VlVBKGHpDl9DOrHLm59KzDF5AqqYcTkk9-A2qkxbnm1-"

project_id="5e7fa573-9f0c-4215-8ccc-cbcfc488b893"

embedding_model = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr",  # your embedding model
    apikey=os.environ["WATSONX_API_KEY"],
    url="https://us-south.ml.cloud.ibm.com",  # your WatsonX instance URL
    project_id=project_id
)



llm = WatsonxLLM(
    apikey=os.environ["WATSONX_API_KEY"],
    model_id="ibm/granite-3-2-8b-instruct",
    project_id=project_id,
    url="https://us-south.ml.cloud.ibm.com",
    params={"max_new_tokens": 1024}
)


prompt_template1 = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful and knowledgeable assistant.

Use the context if it is relevant to answer the question. Otherwise, rely on your own knowledge to give a complete, fluent answer — but do not mention whether the context was used or not.

If the question is not relevant to context, don't merge with the context — rely on your own knowledge to give a complete, fluent answer.


Context:
{context}

Question:
{question}

Answer:"""
)


prompt_template2 = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful and knowledgeable ai chat bot.Max respond 10 lines only.

Use the context if it is relevant to answer the question. Otherwise, rely on your own knowledge to give a complete, fluent answer — but do not mention whether the context was used or not.

If the question is not relevant to context, don't merge with the context — rely on your own knowledge to give a complete, fluent answer.



Context:
{context}

Question:
{question}

Answer:"""
)




@app.route('/')
def home():
    session.clear()
    return render_template("homw2.html")


@app.route('/rag1')
def rag1():
    return render_template("doc_upload.html")


@app.route('/process_doc1', methods=['POST'])
def process_doc1():
    session.clear()
    global chat_chain
    if 'file' not in request.files:
        return "No file part", 400
        
    uploaded_file = request.files['file']
    
    # Check if a file was actually selected
    if uploaded_file.filename == '':
        return "No selected file", 400
        
    if uploaded_file:
        try:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(uploaded_file.filename))
            uploaded_file.save(filepath)

            if uploaded_file.filename.endswith('.pdf'):
                loader = PyPDFLoader(filepath)
            elif uploaded_file.filename.endswith('.docx') or uploaded_file.filename.endswith('.doc'):
                loader = Docx2txtLoader(filepath)
            else:
                return "Unsupported file format", 400

            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(documents)

            faiss_index = FAISS.from_documents(chunks, embedding_model)
            rag_retriever = faiss_index.as_retriever()
            chat_chain = RetrievalQA.from_chain_type(
                llm=llm, 
                retriever=rag_retriever, 
                chain_type="stuff", 
                chain_type_kwargs={"prompt": prompt_template2}
            )
            
            # Clean up the temporary file
            try:
                os.remove(filepath)
            except:
                pass
                
            return redirect(url_for('chat'))
            
        except Exception as e:
            return f"Error processing file: {str(e)}", 500
            
    return "File upload failed", 400

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    global chat_chain
    if chat_chain is None:
        return redirect(url_for('rag1'))
    
    if "history" not in session:
        session["history"] = []

    if request.method == "POST":
        user_input = request.form["que"]
        session["history"].append({"role": "user", "content": user_input})
        
        bot_response= chat_chain.invoke(user_input)
        response=bot_response["result"]
        session["history"].append({"role": "bot", "content": response})

        session.modified = True

    return render_template("chatbot.html", history=session.get("history", []))


@app.route('/rag')
def rag():
    return render_template("rag_upload.html")



@app.route('/process_doc', methods=['POST'])
def process_doc():
    session.clear()
    global rag_chain
    if 'file' not in request.files:
        return "No file part", 400
        
    uploaded_file = request.files['file']
    
    # Check if a file was actually selected
    if uploaded_file.filename == '':
        return "No selected file", 400
        
    if uploaded_file:
        try:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(uploaded_file.filename))
            uploaded_file.save(filepath)

            if uploaded_file.filename.endswith('.pdf'):
                loader = PyPDFLoader(filepath)
            elif uploaded_file.filename.endswith('.docx') or uploaded_file.filename.endswith('.doc'):
                loader = Docx2txtLoader(filepath)
            else:
                return "Unsupported file format", 400

            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(documents)

            faiss_index = FAISS.from_documents(chunks, embedding_model)
            rag_retriever = faiss_index.as_retriever()
            rag_chain = RetrievalQA.from_chain_type(
                llm=llm, 
                retriever=rag_retriever, 
                chain_type="stuff", 
                chain_type_kwargs={"prompt": prompt_template1}
            )
            

            # Clean up the temporary file
            try:
                os.remove(filepath)
            except:
                pass
                
            return redirect(url_for('rag_chat'))
            
        except Exception as e:
            return f"Error processing file: {str(e)}", 500
            
    return "File upload failed", 400


@app.route('/rag_chat', methods=['GET', 'POST'])
def rag_chat():
    global rag_chain
    if rag_chain is None:
        return redirect(url_for('rag'))

   
    response = None
    query=None
    if request.method == 'POST':
        question = request.form.get('que')
        result = rag_chain.invoke(question)
        response = result["result"]
        query= result["query"]
    return render_template('rag_chat.html', response=response,query=query)




