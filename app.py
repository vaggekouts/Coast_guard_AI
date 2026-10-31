import streamlit as st
import os
import shutil
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings

# Φόρτωση κρυφών μεταβλητών (API Key από το τοπικό .env)
load_dotenv()

st.set_page_config(
    page_title="Ψηφιακός Βοηθός Λ.Σ.-ΕΛ.ΑΚΤ.",
    page_icon="⚓",
    layout="centered"
)

# --- ΠΡΟΣΑΡΜΟΣΜΕΝΟ CSS (ΧΡΩΜΑΤΑ Λ.Σ. - ΕΛ.ΑΚΤ.) ---
st.markdown("""
<style>
    .ls-header {
        background: linear-gradient(135deg, #002347 0%, #004080 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-bottom: 4px solid #cca300;
        text-align: center;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .ls-header h1 {
        color: white !important;
        font-size: 1.75rem !important;
        margin: 0 !important;
        padding: 0 !important;
        font-weight: 700;
    }
    .ls-header p {
        color: #e0e8f0 !important;
        font-size: 0.95rem !important;
        margin-top: 0.4rem !important;
        margin-bottom: 0 !important;
    }
    .emergency-box {
        background-color: #fff5f5;
        border: 1px solid #feb2b2;
        border-left: 6px solid #d90429;
        padding: 12px 15px;
        border-radius: 6px;
        margin-bottom: 15px;
        color: #2b2d42;
    }
    .emergency-title {
        color: #d90429;
        font-weight: 800;
        font-size: 1.05rem;
        margin-bottom: 4px;
        display: block;
    }
    .disclaimer-box {
        background-color: #f8f9fa;
        border-left: 4px solid #003366;
        padding: 10px 14px;
        font-size: 0.82rem;
        color: #495057;
        border-radius: 4px;
        margin-top: 15px;
        line-height: 1.4;
    }
    div.stButton > button {
        border: 1px solid #cbd5e1;
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:hover {
        border-color: #003366;
        background-color: #f0f4f8;
        color: #002347;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="ls-header">
    <h1>⚓ Ψηφιακός Βοηθός Λ.Σ. - ΕΛ.ΑΚΤ.</h1>
    <p>Ευφυές Σύστημα Αναζήτησης Νομοθεσίας & Εξυπηρέτησης Πολιτών και Στελεχών</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

def build_vector_db():
    if not os.path.exists("data"):
        st.error("Δεν βρέθηκε ο φάκελος 'data'.")
        return False
        
    pdf_files = [f for f in os.listdir("data") if f.endswith('.pdf')]
    if not pdf_files:
        st.error("Δεν βρέθηκαν αρχεία PDF στον φάκελο 'data'.")
        return False

    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db", ignore_errors=True)

    all_chunks = []
    # Μεγαλύτερα κομμάτια (1500) με μεγάλη επικάλυψη (200) για να μην χάνονται ολόκληρες παράγραφοι δικαιολογητικών
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

    for pdf_file in pdf_files:
        pdf_path = os.path.join("data", pdf_file)
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        for doc in docs:
            doc.page_content = doc.page_content.replace("\x00", "").strip()
            
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)

    embeddings = get_embeddings()
    Chroma.from_documents(all_chunks, embeddings, persist_directory="./chroma_db")
    return True

def create_export_text(messages):
    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    lines = [
        "==========================================================",
        "       ΨΗΦΙΑΚΟΣ ΒΟΗΘΟΣ ΛΙΜΕΝΙΚΟΥ ΣΩΜΑΤΟΣ (Λ.Σ.-ΕΛ.ΑΚΤ.)   ",
        "       Ενημερωτικό Δελτίο Νομοθεσίας & Δικαιολογητικών    ",
        f"       Ημερομηνία Εξαγωγής: {now}                        ",
        "==========================================================\n"
    ]
    for msg in messages:
        role_label = "👤 ΕΡΩΤΗΣΗ ΠΟΛΙΤΗ / ΣΤΕΛΕΧΟΥΣ:" if msg["role"] == "user" else "⚓ ΑΠΑΝΤΗΣΗ ΨΗΦΙΑΚΟΥ ΒΟΗΘΟΥ:"
        lines.append(role_label)
        lines.append(msg["content"])
        lines.append("\n----------------------------------------------------------\n")
        
    lines.append("ΝΟΜΙΚΗ ΣΗΜΕΙΩΣΗ: Το παρόν αρχείο δημιουργήθηκε αυτόματα με χρήση Τεχνητής Νοημοσύνης")
    lines.append("βάσει της διαθέσιμης νομοθεσίας και έχει αποκλειστικά ενημερωτικό χαρακτήρα.")
    return "\n".join(lines)

with st.sidebar:
    st.markdown("""
    <div class="emergency-box">
        <span class="emergency-title">🚨 ΕΚΤΑΚΤΗ ΑΝΑΓΚΗ</span>
        📞 Άμεση Δράση Λ.Σ.: <b>108</b><br>
        📞 Ευρωπαϊκός Αριθμός: <b>112</b><br>
        📻 Κανάλι VHF Κινδύνου: <b>Ch. 16</b>
    </div>
    """, unsafe_allow_html=True)

    st.header("📚 Βάση Νομοθεσίας")
    
    if os.path.exists("data"):
        available_pdfs = [f for f in os.listdir("data") if f.endswith('.pdf')]
        if available_pdfs:
            st.markdown("Ενεργά νομοθετήματα στη βάση:")
            for pdf in available_pdfs:
                st.caption(f"📄 {pdf}")
        else:
            st.warning("Ο φάκελος 'data' είναι άδειος.")
            
    st.divider()
    st.header("⚙️ Διαχείριση")
    
    if st.button("🔄 Ανανέωση Βάσης PDF", use_container_width=True):
        with st.spinner("Γίνεται ανάλυση όλων των εγγράφων του φακέλου data..."):
            if build_vector_db():
                st.success(f"Αναλύθηκαν επιτυχώς {len(available_pdfs)} έγγραφα!")

    if st.button("🗑️ Καθαρισμός Συνομιλίας", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("🔗 **Χρήσιμοι Σύνδεσμοι:**")
    st.markdown("- [Επίσημος Ιστότοπος Λ.Σ.-ΕΛ.ΑΚΤ.](https://www.hcg.gr)")
    st.markdown("- [Ηλεκτρονικές Υπηρεσίες (e-ΔΛΑ)](https://edla.hcg.gr)")

if not os.path.exists("./chroma_db"):
    with st.spinner("🔄 Αρχικοποίηση βάσης νομοθεσίας Λ.Σ.-ΕΛ.ΑΚΤ. Παρακαλώ περιμένετε λίγο..."):
        build_vector_db()

st.markdown("##### 💡 Συχνές Ερωτήσεις Πολιτών (Επιλέξτε με ένα κλικ):")

col1, col2 = st.columns(2)
quick_question = None

with col1:
    if st.button("🚤 Τι χαρακτηρίζεται ως «Μικρό σκάφος»;", use_container_width=True):
        quick_question = "Τι χαρακτηρίζεται ως «Μικρό σκάφος» σύμφωνα με τη νομοθεσία;"
    if st.button("📋 Ποιοι έχουν υποχρέωση εγγραφής στο Β.Ε.Μ.Σ.;", use_container_width=True):
        quick_question = "Ποιοι έχουν υποχρέωση εγγραφής στο Βιβλίο Εγγραφής Μικρών Σκαφών (Β.Ε.Μ.Σ.);"

with col2:
    if st.button("📝 Διαδικασία & δικαιολογητικά εγγραφής στο Β.Ε.Μ.Σ.", use_container_width=True):
        quick_question = "Ποια είναι η διαδικασία εγγραφής στο Β.Ε.Μ.Σ. και ποια δικαιολογητικά χρειάζονται;"
    if st.button("📄 Έκδοση Άδειας εκτέλεσης πλόων & δικαιολογητικά", use_container_width=True):
        quick_question = "Ποια είναι η διαδικασία έκδοσης Άδειας εκτέλεσης πλόων και ποια δικαιολογητικά απαιτούνται;"

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

chat_input_question = st.chat_input("Γράψτε τη δική σας ερώτηση σχετικά με τη νομοθεσία...")
user_question = quick_question or chat_input_question

if user_question:
    
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    if not os.path.exists("./chroma_db"):
        with st.chat_message("assistant"):
            st.warning("Η βάση δεδομένων δεν είναι έτοιμη. Πατήστε 'Ανανέωση Βάσης PDF' αριστερά.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Αναζήτηση στη νομοθεσία..."):
                try:
                    embeddings = get_embeddings()
                    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
                    
                    # Αυξημένα k (15) και fetch_k (50) για να διαβάζει πολύ ευρύτερο φάσμα από όλα τα PDF
                    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 15, "fetch_k": 50})
                    
                    api_key = os.environ["GOOGLE_API_KEY"]
                    
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-3.8-flash", 
                        temperature=0.3, 
                        max_retries=3, 
                        google_api_key=api_key
                    )
                    
                    chat_history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-5:]])
                    
                    # Πιο ευέλικτο prompt που δίνει έμφαση στη σύνθεση και όχι στην άρνηση
                    prompt = ChatPromptTemplate.from_template("""
                    Είσαι ένας επίσημος ψηφιακός βοηθός του Ελληνικού Λιμενικού Σώματος (Λ.Σ.-ΕΛ.ΑΚΤ.) και εξυπηρετείς πολίτες και στελέχη.
                    Απάντησε στην ερώτηση του χρήστη βασιζόμενος στα παρακάτω αποσπάσματα από τη νομοθεσία. 
                    Δώσε μια κατατοπιστική, δομημένη απάντηση (με bullets για τα δικαιολογητικά και τη διαδικασία). 
                    Αν κάποια λεπτομέρεια δεν αναφέρεται ρητά, δώσε την πληρέστερη δυνατή απάντηση βάσει των σχετικών πληροφοριών που υπάρχουν στα κείμενα.
                    
                    ΣΗΜΑΝΤΙΚΟΣ ΚΑΝΟΝΑΣ: Στο τέλος της απάντησής σου, να αναφέρετε πάντα τις πηγές (όνομα αρχείου και σελίδα) που χρησιμοποίησες.
                    
                    Ιστορικό Συζήτησης:
                    {history}
                    
                    Αποσπάσματα από τη Νομοθεσία: 
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
                    
                    retrieved_docs = retriever.invoke(user_question)
                    formatted_context = format_docs(retrieved_docs)
                    
                    rag_chain = (
                        {
                            "context": lambda x: formatted_context, 
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
                            source_file = os.path.basename(doc.metadata.get('source', 'Άγνωστο'))
                            page_num = doc.metadata.get('page', 0) + 1
                            st.markdown(f"**Απόσπασμα {i+1}** (Από: `{source_file}`, Σελίδα: `{page_num}`)")
                            st.write(doc.page_content)
                            st.divider()

                except Exception as e:
                    st.error("⚠️ Παρουσιάστηκε σφάλμα επικοινωνίας με το μοντέλο AI. Παρακαλώ ξαναδοκιμάστε.")
                    st.caption(f"Τεχνική λεπτομέρεια: {e}")

if st.session_state.messages:
    export_data = create_export_text(st.session_state.messages).encode("utf-8-sig")
    st.download_button(
        label="📥 Λήψη Απαντήσεων & Δικαιολογητικών (.txt)",
        data=export_data,
        file_name="Ενημέρωση_ΛΣ_ΕΛΑΚΤ.txt",
        mime="text/plain",
        use_container_width=True,
        key="main_download_btn"
    )

st.markdown("""
<div class="disclaimer-box">
    ⚖️ <b>Νομική Σημείωση & Αποποίηση Ευθύνης:</b> Ο Ψηφιακός Βοηθός αναπτύχθηκε στο πλαίσιο ακαδημαϊκής έρευνας και χρησιμοποιεί μοντέλα Τεχνητής Νοημοσύνης (RAG) για την άντληση πληροφοριών από την κείμενη νομοθεσία. Οι απαντήσεις έχουν αποκλειστικά ενημερωτικό χαρακτήρα και <b>δεν υποκαθιστούν</b> τις επίσημες διοικητικές πράξεις, εγκυκλίους ή υποδείξεις των κατά τόπους Λιμενικών Αρχών.
</div>
""", unsafe_allow_html=True)