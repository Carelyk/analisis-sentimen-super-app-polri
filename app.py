import streamlit as st
import os
import re
import joblib
import numpy as np
from groq import Groq
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Konfigurasi Halaman
st.set_page_config(page_title="Analisis Sentimen Super App Polri", page_icon="🚔", layout="wide")

# Muat Model
@st.cache_resource
def load_models():
    model_svm = joblib.load('model_svm_best.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model_svm, tfidf_vectorizer

model_svm, tfidf_vectorizer = load_models()

# Setup Groq API dari Streamlit Secrets
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
client_groq = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Kamus Normalisasi (Sama persis dengan saat training)
KAMUS_NORMALISASI = {
    'gak': 'tidak', 'ga': 'tidak', 'nggak': 'tidak', 'enggak': 'tidak',
    'udah': 'sudah', 'yg': 'yang', 'dgn': 'dengan', 'utk': 'untuk',
    'dr': 'dari', 'krn': 'karena', 'bgt': 'sangat', 'banget': 'sangat',
    'lemot': 'lambat', 'lelet': 'lambat', 'mantap': 'bagus', 'mantul': 'bagus',
    'ngelag': 'lambat', 'lag': 'lambat', 'ribet': 'rumit', 'app': 'aplikasi'
}

KATA_NEGASI = {'tidak', 'tak', 'bukan', 'belum', 'tanpa', 'jangan', 'tiada', 'kurang'}
stopwords_id = set(stopwords.words('indonesian')) - KATA_NEGASI
stemmer = StemmerFactory().create_stemmer()

def preprocessing(teks):
    if not isinstance(teks, str): return ""
    teks = teks.lower()
    teks = re.sub(r'http\S+|www\S+|@\w+|#\w+|\d+|[^\w\s]', ' ', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    kata = [KAMUS_NORMALISASI.get(k, k) for k in teks.split()]
    tokens = word_tokenize(' '.join(kata))
    tokens = [t for t in tokens if t not in stopwords_id]
    tokens_stemmed = [t if t in KATA_NEGASI else stemmer.stem(t) for t in tokens]
    return ' '.join(tokens_stemmed)

def generate_ai_insight(teks_asli, prediksi, confidence):
    if not client_groq:
        return "⚠️ API Key Groq belum dikonfigurasi di Streamlit Secrets."
    
    prompt = f"""Anda adalah analis sentimen profesional untuk aplikasi Super App Presisi Polri.
    ULASAN PENGGUNA: "{teks_asli}"
    HASIL KLASIFIKASI: Sentimen {prediksi.upper()} (Keyakinan {confidence:.2f}%)
    TUGAS: Berikan analisis singkat (3-5 kalimat) berisi:
    1. Analisis Konteks
    2. Rekomendasi untuk Pengembang
    3. Tingkat Prioritas (1-5)
    Gunakan bahasa Indonesia yang profesional."""
    
    try:
        completion = client_groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "Anda adalah analis sentimen profesional."},
                      {"role": "user", "content": prompt}],
            temperature=0.7, max_tokens=500
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"⚠️ Gagal memuat AI Insight. Error: {str(e)}"

# UI Streamlit
st.title(" Sistem Analisis Sentimen Super App Presisi Polri")
st.markdown("**Model:** SVM + Grid Search | **AI:** Groq Cloud (LLaMA 3.3)")

teks_ulasan = st.text_area("Masukkan Ulasan Pengguna", height=100, placeholder="Contoh: Aplikasi bagus tapi sering error...")

if st.button(" Analisis Sentimen", type="primary"):
    if not teks_ulasan:
        st.warning("Silakan masukkan teks ulasan terlebih dahulu!")
    else:
        with st.spinner("Sedang menganalisis..."):
            teks_bersih = preprocessing(teks_ulasan)
            vektor = tfidf_vectorizer.transform([teks_bersih])
            pred = model_svm.predict(vektor)[0]
            
            distances = model_svm.decision_function(vektor)[0]
            max_distance = max(abs(distances))
            confidence = 1 / (1 + np.exp(-max_distance)) * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Prediksi Sentimen", f"{'😊' if pred=='positif' else '😐' if pred=='netral' else '😠'} {pred.upper()}")
            with col2:
                st.metric("Tingkat Keyakinan", f"{confidence:.2f}%")
            
            st.markdown("---")
            st.subheader(" AI Insight (Groq AI)")
            st.markdown(generate_ai_insight(teks_ulasan, pred, confidence))
