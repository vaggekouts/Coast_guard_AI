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

# Φορτώνουμε το δυνατό τοπικό μοντέλο ΜΟΝΟ μία φορά (Caching)
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# --- 1. ΕΚΜΑΘΗΣΗ ΝΟΜΟΘΕΣΙΑΣ (SIDEBAR) ---
with st.sidebar:
    st.header("⚙️ Διαχείριση Γνώσης")
    if st.button("📚 Διάβασμα PDF"):
        with st.spinner("Γίνεται αστραπιαία τοπική ανάλυση εγγράφου..."):
            pdf_files = [f for f in os.listdir("data") if f.endswith('.pdf')]
            if not pdf_files:
                st.error("Δεν βρέθηκε PDF στον φάκελο data.")
            else:
                loader = PyPDFLoader(os.path.join("data", pdf_files[0]))
                docs = loader.load()
                
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = splitter.split_documents(docs)
                
                embeddings = get_embeddings()
                
                # Δημιουργία και αποθήκευση της βάσης
                db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
                st.success("Τα έγγραφα αναλύθηκαν επιτυχώς!")


# --- 2. ΣΥΣΤΗΜΑ ΕΡΩΤΑΠΟΚΡΙΣΕΩΝ ΜΕ ΜΝΗΜΗ ΚΑΙ ΠΗΓΕΣ (CHAT) ---
st.subheader("💬 Ρωτήστε τον Ψηφιακό Βοηθό")

# Αρχικοποίηση της μνήμης της συνομιλίας
if "messages" not in st.session_state:
    st.session_state.messages = []

# Εμφάνιση όλου του ιστορικού της συζήτησης
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Πεδίο εισαγωγής κειμένου
if user_question := st.chat_input("Γράψτε την ερώτησή σας σχετικά με τη νομοθεσία..."):
    
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    if not os.path.exists("./chroma_db"):
        with st.chat_message("assistant"):
            st.warning("Παρακαλώ πατήστε 'Διάβασμα PDF' στα αριστερά πρώτα.")
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
                
                # ΑΛΛΑΓΗ 1: Νέο Prompt που ζητάει ρητά την αναφορά πηγών στο τέλος
                prompt = ChatPromptTemplate.from_template("""
                Είσαι ένας ψηφιακός βοηθός του Ελληνικού Λιμενικού Σώματος.
                Χρησιμοποίησε ΜΟΝΟ το παρακάτω πλαίσιο (context) από τη νομοθεσία για να απαντήσεις.
                
                Το πλαίσιο πλέον περιλαμβάνει την Πηγή (όνομα αρχείου) και τη Σελίδα από την οποία αντλήθηκε το κείμενο.
                ΣΗΜΑΝΤΙΚΟΣ ΚΑΝΟΝΑΣ: Στο τέλος της απάντησής σου, να αναφέρεις ΠΑΝΤΑ τις πηγές που χρησιμοποίησες, π.χ. "Πηγές: [Όνομα αρχείου], Σελίδα [X]". 
                
                Λάβε υπόψη το Ιστορικό Συζήτησης για να καταλάβεις σε τι αναφέρεται ο χρήστης αν η ερώτηση είναι σύντομη.
                Αν δεν ξέρεις την απάντηση βάσει του κειμένου, πες 'Δεν βρέθηκε σχετική πληροφορία'.
                
                Ιστορικό Συζήτησης:
                {history}
                
                Πλαίσιο από Νομοθεσία: 
                {context}
                
                Τρέχουσα Ερώτηση: {question}
                """)
                
                # ΑΛΛΑΓΗ 2: Η συνάρτηση πλέον "τραβάει" και τα Metadata (σελίδα, όνομα αρχείου)
                def format_docs(docs):
                    formatted_texts = []
                    for doc in docs:
                        # Ανάκτηση του ονόματος αρχείου (αφαιρώντας τη λέξη 'data/')
                        source = doc.metadata.get('source', 'Άγνωστο έγγραφο')
                        source_name = os.path.basename(source) 
                        
                        # Το PyPDFLoader μετράει τις σελίδες από το 0, οπότε προσθέτουμε 1
                        page = doc.metadata.get('page', 0) + 1  
                        
                        # Συνθέτουμε το τελικό κείμενο που θα διαβάσει το AI
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

            # ΑΛΛΑΓΗ 3: Αναβαθμισμένο παραθυράκι επαλήθευσης που δείχνει και αυτό τη σελίδα!
            retrieved_docs = retriever.invoke(user_question)
            with st.expander("🔍 Δες ποια ακριβώς αποσπάσματα βρήκε στο PDF"):
                for i, doc in enumerate(retrieved_docs):
                    source_file = os.path.basename(doc.metadata.get('source', 'Άγνωστο'))
                    page_num = doc.metadata.get('page', 0) + 1
                    st.markdown(f"**Απόσπασμα {i+1}** (Από: `{source_file}`, Σελίδα: `{page_num}`)")
                    st.write(doc.page_content)
                    st.divider()