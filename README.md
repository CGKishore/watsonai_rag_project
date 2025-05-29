# watsonai_rag_project

App Link:https://watsonai-rag-project.onrender.com
Github repository link:https://github.com/CGKishore/watsonai_rag_project

What is WatsonDocBot?
WatsonDocBot is an AI-powered web application designed to help users interact with documents (such as PDFs and Word files) using natural language. It utilizes Retrieval-Augmented Generation (RAG) and IBM Watson AI to allow users to upload documents and then ask questions about the content, either through a direct Q&A interface or a chatbot-style conversation.

WatsonDocBot Aims to:
•	Save Time: Users get direct answers without reading long documents.
•	Improve Access to Information: Even if users don’t know where the information is located, they can still retrieve it easily.
•	Boost Productivity: Beneficial for professionals, students, support teams, and anyone who works with dense documentation regularly.
•	Enable Instant Script Support: Users can upload the latest chat scripts or service manuals for real-time support accuracy.

✅ Your Watson Doc Bot Setup (Summary)
It uses Watson AI (likely Watsonx or Watson Discovery) for document-based Q&A.

You’ve built a Flask app with a landing page having three options:

Q/A

Chatbot

RAG (Retrieval-Augmented Generation)

The RAG option allows uploading a PDF or DOC file, which then gets converted into vectors for document-based retrieval.

Now let me explain how your Watson Doc Bot works step-by-step based on your setup.

🔍 Your Watson Doc Bot – Detailed Working
🧱 Components in Your System
      Component	Purpose
      Flask App	Frontend + routing logic
      LLM (Watson AI)	Core language understanding model
      Vector Store	Stores document chunks as embeddings
      Document Uploader	Converts PDF/DOC into text and vector form
      Session Memory	Used in Chatbot mode to preserve Q&A history

🚀 Step-by-Step Workflow
🔹 1. Landing Page Selection
        When a user lands on your app:
        They choose Q/A, Chatbot, or RAG mode.
        Each redirects to a different route with a shared UI theme.

🔹 2. RAG Flow (PDF/DOC Upload)
        User uploads a file (PDF/DOC).
        Text Extraction: The document is parsed to extract clean text 
        (e.g., using PyMuPDF, pdfminer, or docx libraries).
        Text Chunking: The text is broken into smaller passages (e.g., 500 
        tokens).

Vectorization:
Each chunk is converted into a vector (embedding) using an embedding model (could be from Watson, SentenceTransformers, etc.).

Storage:
These vectors are stored in a vector store (e.g., FAISS, ChromaDB, or in-memory).

🔹 3. Question Answering (Q/A Page)
        User types a question.
        Your system:
        Converts the question into a vector.
        Performs semantic similarity search against the document vectors.
        Top-k results (passages) are retrieved.
        The LLM (Watson AI) is prompted:
        "Based on the following context, answer the question: [retrieved chunks] + [user question]"
        The answer is shown on the Q/A page.

🔹 4. Chatbot Flow
        User opens the Chatbot page.
        Each question goes through the same RAG process (retrieve + generate).
        However, unlike the Q/A page, here:
        Previous questions and answers are preserved (session memory).
        So it supports conversational context, e.g.:
        Q1: What’s the refund policy?
        Q2: What about international customers?
        Your chatbot understands that Q2 relates to Q1.

🧰 Technologies You Might Be Using
        Feature	Tools
        Document parsing	PyMuPDF, python-docx
        Vector embeddings	Watson Embedding API / SentenceTransformers
        Vector store	FAISS, ChromaDB
        Backend	Flask
        Frontend	HTML/CSS/JS (integrated in Flask)
        LLM	Watsonx.ai / OpenAI GPT / etc.
        Session memory	Flask session or in-memory dict

🔄 Reset Behavior
        In Chatbot mode, you maintain session state.
        When the app is closed or refreshed, it resets the chat history.
        In Q/A mode, each question is standalone.

✅ Summary
        Your Watson Doc Bot:
        Gives users three modes: Q/A, Chatbot, and RAG file upload.
        Uses document embeddings and semantic search to provide accurate answers.
        Maintains conversational context in Chatbot mode.
        Is built with Flask and can be enhanced further using caching, feedback, or multi-file upload support.
