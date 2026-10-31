import streamlit as st
import os
import tempfile
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

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# --- 1. ΕΚΜΑΘΗΣΗ ΝΟΜΟΘΕΣΙΑΣ ΜΕ DRAG & DROP (SIDEBAR) ---
with st.sidebar:
    st.header("⚙️ Διαχείριση Γνώσης")
    
    # ΝΕΟ: Εργαλείο μεταφόρτωσης (υποστηρίζει πολλαπλά αρχεία)
    uploaded_files = st.file_uploader(
        "Ανεβάστε ένα ή περισσότερα PDF με νομοθεσία:", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    if st.button("📚 Ανάλυση Εγγράφων"):
        if not uploaded_files:
            st.warning("Παρακαλώ ανεβάστε τουλάχιστον ένα αρχείο PDF πρώτα.")
        else:
            with st.spinner("Ανάγνωση και ανάλυση των εγγράφων..."):
                all_chunks = []
                
                # Επεξεργασία κάθε ανεβασμένου αρχείου
                for uploaded_file in uploaded_files:
                    # Δημιουργία προσωρινού φακέλου και αρχείου για να το διαβάσει το LangChain
                    temp_dir = tempfile.mkdtemp()
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    # Φόρτωση του PDF
                    loader = PyPDFLoader(temp_path)
                    docs = loader.load()
                    
                    # Ενημέρωση των metadata ώστε η πηγή να έχει το σωστό όνομα του αρχείου
                    for doc in docs:
                        doc.metadata['source'] = uploaded_file.name
                    
                    # Τεμαχισμός (Splitting)
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    chunks = splitter.split_documents(docs)
                    
                    # Προσθήκη των κομματιών στη συνολική λίστα
                    all_chunks.extend(chunks)
                
                embeddings = get_embeddings()
                
                # Δημιουργία της διανυσματικής βάσης από ΟΛΑ τα κομμάτια ΟΛΩΝ των PDF
                db = Chroma.from_documents(all_chunks, embeddings, persist_directory="./chroma_db")
                st.success(f"Επιτυχής ανάλυση {len(uploaded_files)} εγγράφου/ων!")


# --- 2. ΣΥΣΤΗΜΑ ΕΡΩΤΑΠΟΚΡΙΣΕΩΝ ΜΕ ΜΝΗΜΗ ΚΑΙ ΠΗΓΕΣ (CHAT) ---
st.subheader("💬 Ρωτήστε τον Ψηφιακό Βοηθό")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_question := st.chat_input("Γράψτε την ερώτησή σας σχετικά με τη νομοθεσία..."):
    
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    if not os.path.exists("./chroma_db"):
        with st.chat_message("assistant"):
            st.warning("Παρακαλώ ανεβάστε και αναλύστε τα PDF από το μενού στα αριστερά πρώτα.")
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
                Είσαι ένας ψηφιακός βοηθός του Ελληνικού Λιμενικού Σώματος.
                Χρησιμοποίησε ΜΟΝΟ το παρακάτω πλαίσιο (context) από τη νομοθεσία για να απαντήσεις.
                
                Το πλαίσιο περιλαμβάνει την Πηγή (όνομα αρχείου) και τη Σελίδα.
                ΣΗΜΑΝΤΙΚΟΣ ΚΑΝΟΝΑΣ: Στο τέλος της απάντησής σου, να αναφέρεις ΠΑΝΤΑ τις πηγές που χρησιμοποίησες, π.χ. "Πηγές: [Όνομα αρχείου], Σελίδα [X]". 
                
                Λάβε υπόψη το Ιστορικό Συζήτησης για να καταλάβεις σε τι αναφέρεται ο χρήστης αν η ερώτηση είναι σύντομη.
                Αν δεν ξέρεις την απάντηση βάσει του κειμένου, πες 'Δεν βρέθηκε σχετική πληροφορία'.
                
                Ιστορικό Συζήτησης:
                {history}
                
                Πλαίσιο από Νομοθεσία: 
                {context}
                
                Τρέχουσα Ερώτηση: {question}
                """)
                
                def format_docs(docs):
                    formatted_texts = []
                    for doc in docs:
                        source_name = doc.metadata.get('source', 'Άγνωστο έγγραφο')
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

            with st.expander("🔍 Δες ποια ακριβώς αποσπάσματα βρήκε στα PDF"):
                for i, doc in enumerate(retrieved_docs):
                    source_file = doc.metadata.get('source', 'Άγνωστο')
                    page_num = doc.metadata.get('page', 0) + 1
                    st.markdown(f"**Απόσπασμα {i+1}** (Από: `{source_file}`, Σελίδα: `{page_num}`)")
                    st.write(doc.page_content)
                    st.divider()