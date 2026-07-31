import gradio as gr
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from groq import Groq
import os

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Load model dan vectorizer
model_svm = joblib.load('model_svm_best.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Setup Groq API
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client_groq = Groq(api_key=GROQ_API_KEY)

# Kamus normalisasi lengkap (217 kata)
KAMUS_NORMALISASI = {
    # === KATA NEGASI & MODAL ===
    'gk': 'tidak', 'nggak': 'tidak', 'gak': 'tidak', 'ga': 'tidak', 'tdk': 'tidak',
    'blm': 'belum', 'blum': 'belum', 'udh': 'sudah', 'sda': 'sudah', 'sdh': 'sudah',
    'dah': 'sudah', 'deh': 'sudah',
    'syg': 'sayang', 'aq': 'saya', 'aku': 'saya', 'gue': 'saya', 'gw': 'saya',
    'elo': 'kamu', 'lu': 'kamu', 'km': 'kamu', 'anda': 'anda',
    'yg': 'yang', 'dgn': 'dengan', 'utk': 'untuk', 'thd': 'terhadap',
    'dr': 'dari', 'krn': 'karena', 'karna': 'karena', 'spy': 'supaya',
    'biar': 'agar', 'klo': 'kalau', 'klu': 'kalau', 'kalo': 'kalau', 'kl': 'kalau',
    'dlm': 'dalam', 'ttg': 'tentang', 'ttng': 'tentang', 'slh': 'salah',
    'bgt': 'sangat', 'banget': 'sangat', 'sekali': 'sangat', 'bnget': 'sangat',
    
    # === KATA SIFAT POSITIF ===
    'bagus': 'bagus', 'baik': 'bagus', 'mantap': 'bagus', 'mantul': 'bagus',
    'keren': 'bagus', 'top': 'bagus', 'joss': 'bagus', 'ok': 'bagus', 'oke': 'bagus',
    'sip': 'bagus', 'good': 'bagus', 'great': 'bagus', 'nice': 'bagus',
    'puas': 'puas', 'senang': 'senang', 'seneng': 'senang', 'happy': 'senang',
    'mudah': 'mudah', 'gampang': 'mudah', 'simple': 'mudah', 'simpel': 'mudah',
    'cepat': 'cepat', 'cepet': 'cepat', 'fast': 'cepat', 'quick': 'cepat',
    'efisien': 'efisien', 'praktis': 'praktis', 'nyaman': 'nyaman',
    'membantu': 'bantu', 'helpful': 'bantu', 'berguna': 'bantu',
    'lengkap': 'lengkap', 'komplit': 'lengkap',
    
    # === KATA SIFAT NEGATIF ===
    'jelek': 'jelek', 'buruk': 'jelek', 'bad': 'jelek', 'worst': 'jelek',
    'parah': 'parah', 'teruk': 'parah', 'horrible': 'parah',
    'error': 'error', 'rusak': 'error', 'broken': 'error',
    'lemot': 'lambat', 'lelet': 'lambat', 'slow': 'lambat', 'lamban': 'lambat',
    'bingung': 'bingung', 'confused': 'bingung', 'membingungkan': 'bingung',
    'susah': 'sulit', 'sulit': 'sulit', 'ribet': 'sulit', 'difficult': 'sulit',
    'kesal': 'kesal', 'sebal': 'kesal', 'annoyed': 'kesal',
    'ganggu': 'ganggu', 'mengganggu': 'ganggu', 'annoying': 'ganggu',
    'crash': 'error', 'force close': 'error', 'fc': 'error',
    'ngelag': 'lambat', 'lag': 'lambat', 'laggy': 'lambat',
    'hang': 'error', 'freeze': 'error', 'not responding': 'error',
    'bug': 'error', 'buggy': 'error', 'glitch': 'error',
    'confusing': 'bingung', 'membingungkan': 'bingung',
    'complicated': 'rumit', 'complex': 'rumit',
    
    # === ISTILAH APLIKASI & LAYANAN ===
    'aplikasi': 'aplikasi', 'app': 'aplikasi', 'apps': 'aplikasi',
    'sim': 'sim', 'stnk': 'stnk', 'skck': 'skck', 'etle': 'etle',
    'tilang': 'tilang', 'e-tilang': 'tilang', 'etilang': 'tilang',
    'polri': 'polri', 'polisi': 'polisi', 'kepolisian': 'polisi',
    'pengaduan': 'aduan', 'adu': 'aduan', 'complain': 'aduan',
    'complaint': 'aduan', 'lapor': 'lapor', 'report': 'lapor',
    'layanan': 'layanan', 'service': 'layanan', 'pelayanan': 'layanan',
    'fitur': 'fitur', 'feature': 'fitur',
    'menu': 'menu', 'tombol': 'tombol', 'button': 'tombol',
    'halaman': 'halaman', 'page': 'halaman', 'tampilan': 'tampilan',
    'interface': 'tampilan', 'ui': 'tampilan', 'ux': 'tampilan',
    
    # === KATA KERJA ===
    'bisa': 'bisa', 'dapat': 'bisa', 'can': 'bisa',
    'tidak bisa': 'tidak bisa', 'gabisa': 'tidak bisa', 'gk bisa': 'tidak bisa',
    'pake': 'guna', 'pakai': 'guna', 'gunakan': 'guna',
    'download': 'unduh', 'unduh': 'unduh', 'install': 'instal',
    'update': 'perbarui', 'perbarui': 'perbarui', 'upgrade': 'perbarui',
    'login': 'masuk', 'masuk': 'masuk', 'sign in': 'masuk',
    'logout': 'keluar', 'keluar': 'keluar', 'sign out': 'keluar',
    'register': 'daftar', 'daftar': 'daftar', 'sign up': 'daftar',
    'verifikasi': 'verifikasi', 'verify': 'verifikasi',
    'upload': 'unggah', 'unggah': 'unggah', 'submit': 'kirim',
    'kirim': 'kirim', 'submit': 'kirim',
    'buka': 'buka', 'open': 'buka',
    'tutup': 'tutup', 'close': 'tutup',
    'cari': 'cari', 'search': 'cari',
    'pilih': 'pilih', 'select': 'pilih',
    'klik': 'klik', 'click': 'klik', 'tap': 'klik',
}

# Fungsi preprocessing lengkap
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Case folding
    text = text.lower()
    
    # Cleaning
    text = re.sub(r'http\S+|www\S+|@\S+|#\S+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Normalisasi
    words = text.split()
    normalized_words = [KAMUS_NORMALISASI.get(word, word) for word in words]
    text = ' '.join(normalized_words)
    
    # Tokenisasi
    tokens = word_tokenize(text)
    
    # Hapus stopwords (kecuali kata negasi)
    stop_words = set(stopwords.words('indonesian'))
    negation_words = {'tidak', 'tidak', 'bukan', 'belum', 'jangan', 'jangan'}
    tokens = [word for word in tokens if word not in stop_words or word in negation_words]
    
    # Stemming
    stemmer = StemmerFactory().create_stemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    
    return ' '.join(stemmed_tokens)

# Fungsi prediksi sentimen
def predict_sentiment(text):
    if not text.strip():
        return "⚠️ Silakan masukkan teks ulasan terlebih dahulu."
    
    # Preprocessing
    cleaned_text = preprocess_text(text)
    
    # Ekstraksi fitur
    features = tfidf_vectorizer.transform([cleaned_text])
    
    # Prediksi
    prediction = model_svm.predict(features)[0]
    probabilities = model_svm.predict_proba(features)[0]
    confidence = max(probabilities) * 100
    
    # Mapping label
    label_map = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
    sentiment_label = label_map.get(prediction, 'Tidak Diketahui')
    
    # Generate AI Insight
    ai_insight = generate_ai_insight(text, sentiment_label, confidence)
    
    return f"""
## 📊 Hasil Klasifikasi Sentimen

**Teks Asli:**
> {text}

**Teks Setelah Preprocessing:**
> {cleaned_text}

---

### 🎯 Prediksi Sentimen: **{sentiment_label}**
**Tingkat Keyakinan:** {confidence:.2f}%

---

{ai_insight}
"""

# Fungsi AI Insight dengan Groq
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
            messages=[
                {"role": "system", "content": "Anda adalah analis sentimen profesional."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return f"""### 🤖 AI Insight (Groq LLaMA 3.3)

{completion.choices[0].message.content}
"""
    except Exception as e:
        return f"""### ⚠️ AI Insight Tidak Tersedia

Gagal memuat AI Insight. Periksa API Key Groq di Secrets.

**Error:** {str(e)}
"""

# Interface Gradio
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Masukkan ulasan pengguna aplikasi Super App Presisi Polri...",
        label="📝 Ulasan Pengguna"
    ),
    outputs=gr.Markdown(label=" Hasil Analisis"),
    title="🚔 Analisis Sentimen Super App Presisi Polri",
    description="""
Sistem analisis sentimen berbasis **Support Vector Machine (SVM)** dengan optimasi **Grid Search** 
dan **AI Insight** menggunakan **Groq LLaMA 3.3 70B**.

**Fitur:**
- Klasifikasi 3 kelas: Positif, Netral, Negatif
- Preprocessing teks lengkap dengan 217 kata normalisasi
- AI Insight kontekstual dari Groq AI
- Tingkat keyakinan prediksi

**Dikembangkan oleh:** Carel Alberto Karma  
**Dataset:** Ulasan Google Play Store (2025–2026)
""",
    examples=[
        ["Aplikasinya sangat bagus, perpanjangan SIM jadi lebih mudah dan cepat!"],
        ["Sering error dan lemot, tidak bisa dibuka sama sekali saat mau bayar STNK."],
        ["Bagaimana cara perpanjang STNK di aplikasi ini? Masih bingung menu nya."],
        ["Fitur pengaduan masyarakat sangat membantu, tapi loadingnya agak lama."],
        ["Aplikasi crash terus, sudah coba install ulang tetap sama. Kecewa banget!"],
        ["Lumayan bagus tapi kadang ngelag saat upload foto SKCK."],
        ["Menu untuk cek tilang ETLE ada di mana ya? Tidak ketemu."]
    ],
    theme=gr.themes.Soft()
    # allow_flagging="never"  <-- BARIS INI SUDAH DIHAPUS
)

# Launch aplikasi
if __name__ == "__main__":
    iface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
