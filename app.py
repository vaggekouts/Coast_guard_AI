import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings

# Φόρτωση κρυφών μεταβλητών (API Key)
load_dotenv()

st.set_page_config(page_title="Ψηφιακός Βοηθός Λ.Σ.", page_icon="⚓")
st.title("⚓ Ψηφιακός Βοηθός Λ.Σ. - ΕΛ.ΑΚΤ.")

# Φορτώνουμε το τοπικό μοντέλο ΜΟΝΟ μία φορά (Caching)
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# Συνάρτηση που διαβάζει ΟΛΑ τα PDF από τον φάκελο data και χτίζει τη βάση
def build_vector_db():
    if not os.path.exists("data"):
        st.error("Δεν βρέθηκε ο φάκελος 'data'.")
        return False
        
    pdf_files = [f for f in os.listdir("data") if f.endswith('.pdf')]
    if not pdf_files:
        st.error("Δεν βρέθηκαν αρχεία PDF στον φάκελο 'data'.")
        return False

    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    # Διάβασμα ΚΑΘΕ αρχείου PDF που υπάρχει στον φάκελο data
    for pdf_file in pdf_files:
        pdf_path = os.path.join("data", pdf_file)
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)

    embeddings = get_embeddings()
    Chroma.from_documents(all_chunks, embeddings, persist_directory="./chroma_db")
    return True


# --- 1. ΠΛΑΪΝΟ ΜΕΝΟΥ (SIDEBAR) - ΠΛΗΡΟΦΟΡΙΕΣ & ΔΙΑΧΕΙΡΙΣΗ ---
with st.sidebar:
    st.header("📚 Βάση Νομοθεσίας")
    
    # Εμφάνιση όλων των διαθέσιμων PDF στους πολίτες/αξιωματικούς
    if os.path.exists("data"):
        available_pdfs = [f for f in os.listdir("data") if f.endswith('.pdf')]
        if available_pdfs:
            st.markdown("Το σύστημα απαντά βάσει των εξής εγγράφων:")
            for pdf in available_pdfs:
                st.caption(f"📄 {pdf}")
        else:
            st.warning("Ο φάκελος 'data' είναι άδειος.")
            
    st.divider()
    st.header("⚙️ Διαχείριση")
    if st.button("🔄 Ανανέωση Βάσης PDF"):
        with st.spinner("Γίνεται ανάλυση όλων των εγγράφων του φακέλου data..."):
            if build_vector_db():
                st.success(f"Αναλύθηκαν επιτυχώς {len(available_pdfs)} έγγραφα!")


# --- 2. ΑΥΤΟΜΑΤΗ ΕΚΚΙΝΗΣΗ ΒΑΣΗΣ (ΑΝ Ο SERVER ΕΚΑΝΕ REBOOT) ---
if not os.path.exists("./chroma_db"):
    with st.spinner("🔄 Αρχικοποίηση βάσης νομοθεσίας Λ.Σ.-ΕΛ.ΑΚΤ. Παρακαλώ περιμένετε λίγο..."):
        build_vector_db()


# --- 3. ΣΥΣΤΗΜΑ ΕΡΩΤΑΠΟΚΡΙΣΕΩΝ ΜΕ ΜΝΗΜΗ ΚΑΙ ΠΗΓΕΣ (CHAT) ---
st.subheader("💬 Ρωτήστε τον Ψηφιακό Βοηθό")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_question := st.chat_input("Ρωτήστε για νομοθεσία, σωστικά μέσα σκαφών, ναυαγοσώστες κ.ά..."):
    
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    if not os.path.exists("./chroma_db"):
        with st.chat_message("assistant"):
            st.warning("Η βάση δεδομένων δεν είναι έτοιμη. Πατήστε 'Ανανέωση Βάσης PDF' αριστερά.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Αναζήτηση στη νομοθεσία..."):
                
                embeddings = get_embeddings()
                db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
                
                retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 30})
                
                llm = ChatGoogleGenerativeAI(
                    model="gemini-3.8-flash", 
                    temperature=0.3, 
                    google_api_key=os.environ["GOOGLE_API_KEY"]
                )
                
                chat_history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-5:]])
                
                prompt = ChatPromptTemplate.from_template("""
                Είσαι ένας επίσημος ψηφιακός βοηθός του Ελληνικού Λιμενικού Σώματος (Λ.Σ.-ΕΛ.ΑΚΤ.) και εξυπηρετείς τόσο στελέχη του Σώματος όσο και πολίτες.
                Χρησιμοποίησε ΜΟΝΟ το παρακάτω πλαίσιο (context) από τη νομοθεσία για να απαντήσεις με σαφήνεια και ευγένεια.
                
                Το πλαίσιο περιλαμβάνει την Πηγή (όνομα αρχείου) και τη Σελίδα από την οποία αντλήθηκε το κείμενο.
                ΣΗΜΑΝΤΙΚΟΣ ΚΑΝΟΝΑΣ: Στο τέλος της απάντησής σου, να αναφέρεις ΠΑΝΤΑ τις πηγές που χρησιμοποίησες, π.χ. "Πηγές: [Όνομα αρχείου], Σελίδα [X]". 
                
                Λάβε υπόψη το Ιστορικό Συζήτησης για να καταλάβεις σε τι αναφέρεται ο χρήστης αν η ερώτηση είναι σύντομη.
                Αν δεν ξέρεις την απάντηση βάσει του κειμένου, πες 'Δεν βρέθηκε σχετική πληροφορία στη διαθέσιμη νομοθεσία'.
                
                Ιστορικό Συζήτησης:
                {history}
                
                Πλαίσιο από Νομοθεσία: 
                {context}
                
                Τρέχουσα Ερώτηση: {question}
                """)
                
                def format_docs(docs):
                    formatted_texts = []
                    for doc in docs:
                        source = doc.metadata.get('source', 'Άγνωστο έγγραφο')
                        source_name = os.path.basename(source) 
                        page = doc.metadata.get('page', 0) + 1  
                        formatted_texts.append(f"[Πηγή: {source_name}, Σελίδα: {page}]\n{doc.page_content}")
                    return "\n\n---\n\n".join(formatted_texts)
                
                rag_chain = (
                    {
                        "context": retriever | format_docs, 
                        "history": lambda x: chat_history_text, 
                        "question": RunnablePassthrough()
                    }
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                
                response = rag_chain.invoke(user_question)
                st.markdown(response)
                
                st.session_state.messages.append({"role": "assistant", "content": response})

            # Παραθυράκι επαλήθευσης των αποσπασμάτων του PDF
            retrieved_docs = retriever.invoke(user_question)
            with st.expander("🔍 Δες ποια ακριβώς αποσπάσματα βρήκε στα PDF"):
                for i, doc in enumerate(retrieved_docs):
                    source_file = os.path.basename(doc.metadata.get('source', 'Άγνωστο'))
                    page_num = doc.metadata.get('page', 0) + 1
                    st.markdown(f"**Απόσπασμα {i+1}** (Από: `{source_file}`, Σελίδα: `{page_num}`)")
                    st.write(doc.page_content)
                    st.divider()