import os
import re
import joblib
import numpy as np
import gradio as gr
from groq import Groq
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk

# Download NLTK data saat aplikasi pertama kali dijalankan di server
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# ===== 1. MUAT MODEL DARI FOLDER =====
# Di Hugging Face, file .pkl harus berada di folder yang sama dengan app.py
model_svm = joblib.load('model_svm_best.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# ===== 2. SETUP GROQ API =====
# API Key akan diambil dari "Secrets" di pengaturan Hugging Face Space
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client_groq = Groq(api_key=GROQ_API_KEY)

# ===== 3. KAMUS & PREPROCESSING =====
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

# ===== 4. FUNGSI AI INSIGHT =====
def generate_ai_insight(teks_asli, prediksi, confidence):
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
        return f"⚠️ Gagal memuat AI Insight. Periksa API Key Groq di Secrets. Error: {str(e)}"

# ===== 5. FUNGSI UTAMA GRADIO =====
def prediksi_dengan_insight(teks_ulasan):
    if not teks_ulasan or len(teks_ulasan.strip()) == 0:
        return "⚠️ Silakan masukkan teks ulasan terlebih dahulu!"
    
    teks_bersih = preprocessing(teks_ulasan)
    vektor = tfidf_vectorizer.transform([teks_bersih])
    pred = model_svm.predict(vektor)[0]
    
    distances = model_svm.decision_function(vektor)[0]
    max_distance = max(abs(distances))
    confidence = 1 / (1 + np.exp(-max_distance)) * 100
    
    emoji_map = {'positif': '😊 POSITIF', 'netral': '😐 NETRAL', 'negatif': '😠 NEGATIF'}
    insight = generate_ai_insight(teks_ulasan, pred, confidence)
    
    return f"""
    ### 📊 Klasifikasi Model (SVM + Grid Search)
    | Informasi | Detail |
    |---|---|
    | **Teks Asli** | "{teks_ulasan}" |
    | **Prediksi Sentimen** | **{emoji_map[pred]}** |
    | **Tingkat Keyakinan** | {confidence:.2f}% |

    ---
    ###  AI Insight (Powered by Groq - LLaMA 3.3 70B)
    {insight}
    """

# ===== 6. JALANKAN INTERFACE =====
iface = gr.Interface(
    fn=prediksi_dengan_insight,
    inputs=gr.Textbox(lines=4, label="📝 Teks Ulasan"),
    outputs=gr.Markdown(label="📊 Hasil Analisis"),
    title="🚔 Sistem Analisis Sentimen Super App Presisi Polri",
    description="Model: SVM + Grid Search | AI: Groq Cloud (LLaMA 3.3)",
    examples=[
        ["Aplikasinya sangat bagus, perpanjangan SIM jadi lebih mudah!"],
        ["Sering error dan lemot, tidak bisa dibuka sama sekali."],
        ["Bagaimana cara perpanjang STNK di aplikasi ini?"]
    ]
)

if __name__ == "__main__":
    iface.launch()