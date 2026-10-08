# Analisis Sentimen Ulasan Pengguna Super App Presisi Polri

Repo ini berisi notebook penelitian skripsi untuk analisis sentimen ulasan pengguna **Super App Presisi Polri** (aplikasi layanan digital Kepolisian Negara Republik Indonesia) dari Google Play Store, secara end-to-end menggunakan algoritma **Support Vector Machine (SVM)** yang dioptimasi dengan **Grid Search** dalam kerangka kerja **CRISP-DM**, serta antarmuka **Gradio** yang terintegrasi dengan **GPT-OSS 120B via Groq API** untuk analisis naratif real-time.

## Catatan Abstraksi (Skripsi)

> Super App Presisi Polri merupakan platform layanan digital terintegrasi yang dikembangkan oleh Kepolisian Negara Republik Indonesia untuk memberikan kemudahan akses layanan kepolisian secara daring. Volume ulasan pengguna pada Google Play Store yang terus bertambah menjadikan proses analisis sentimen secara manual tidak efektif dan berpotensi menghasilkan penilaian yang subjektif. Penelitian ini bertujuan membangun sistem analisis sentimen ulasan pengguna Super App Presisi Polri secara end-to-end menggunakan algoritma Support Vector Machine (SVM) yang dioptimasi melalui teknik Grid Search dalam kerangka kerja CRISP-DM. Data yang digunakan sebanyak 3.000 ulasan yang dikumpulkan melalui teknik web scraping pada periode Juni hingga Agustus 2026, dengan 2.673 ulasan berhasil diproses melalui enam tahap text preprocessing dengan pengamanan kata negasi. Ekstraksi fitur dilakukan menggunakan TF-IDF yang menghasilkan 886 fitur, sementara ketidakseimbangan kelas (positif 78,4%, negatif 18,1%, netral 3,5%) ditangani dengan SVMSMOTE di dalam pipeline untuk mencegah kebocoran data. Hasil optimasi Grid Search dari 20 kombinasi parameter menetapkan C=1,0, gamma=scale, kernel=rbf, SVMSMOTE sampling={'netral': 300} sebagai konfigurasi terbaik. Model final menghasilkan akurasi uji 90,09%, presisi tertimbang 90,52%, recall tertimbang 90,09%, dan F1-Score tertimbang 88,76%, yang terkonfirmasi stabil melalui validasi silang 5-fold (88,96% ± 0,74%). Analisis per kelas menunjukkan kelas positif dikenali baik (F1-Score 0,95), kelas negatif cukup baik (F1-Score 0,80), sedangkan kelas netral masih menjadi tantangan (F1-Score 0,10). Diagnosis overfitting dari tiga sumber gap (6,68%–7,95%) menunjukkan indikasi ringan yang wajar untuk data teks sparse. Model terbaik diimplementasikan ke dalam antarmuka Gradio terintegrasi GPT-OSS 120B via Groq API yang menghasilkan analisis naratif tujuh dimensi secara real-time, dengan pengujian tiga kasus fungsional yang membuktikan konsistensi antara keluaran SVM dan narasi LLM.

## Isi Repositori

| File | Keterangan |
| --- | --- |
| [`Skripsi_SVM_Modeling.ipynb`](Skripsi_SVM_Modeling.ipynb) | Notebook training end-to-end: web scraping ulasan Google Play Store → enam tahap text preprocessing (dengan pengamanan kata negasi) → TF-IDF → SVMSMOTE di dalam pipeline → GridSearchCV (5-fold stratified) → evaluasi akhir, confusion matrix, analisis per kelas, dan diagnosis overfitting. Menghasilkan artefak model `model_svm_best.pkl`, `tfidf_vectorizer.pkl`, dan `model_metadata.pkl`. |
| [`Gradio_AI_Insight.ipynb`](Gradio_AI_Insight.ipynb) | Antarmuka Gradio untuk inferensi: memuat artefak model dari notebook training, melakukan prediksi SVM, lalu menghasilkan analisis naratif tujuh dimensi secara real-time melalui GPT-OSS 120B via Groq API. |

## Cara Menjalankan

Kedua notebook dirancang untuk dijalankan di **Google Colab**.

1. **Training model** — jalankan `Skripsi_SVM_Modeling.ipynb` dari awal hingga akhir. Download ketiga artefak `.pkl` yang dihasilkan (`model_svm_best.pkl`, `tfidf_vectorizer.pkl`, `model_metadata.pkl`).
2. **Menjalankan aplikasi** — buka `Gradio_AI_Insight.ipynb`, upload ketiga artefak `.pkl` pada cell upload, lalu jalankan cell Gradio.

### Konfigurasi API Key (Groq)

API key **tidak disimpan di dalam repo**. Gunakan **Colab Secrets**:

1. Buka panel **Secrets** di Google Colab (ikon kunci 🔑 di panel kiri).
2. Tambahkan secret dengan nama `GROQ_API_KEY` (dapatkan dari [console.groq.com](https://console.groq.com)).
3. Nyalakan toggle **Notebook access** untuk kedua notebook.

## Keamanan

- Tidak ada API key, token, atau kredensial yang disimpan di dalam repo ini.
- API key hanya dibaca dari environment/Colab Secrets saat runtime.
- File artefak model (`*.pkl`) dan dependensi tidak dimasukkan ke repo; dihasilkan dan digunakan secara lokal.
