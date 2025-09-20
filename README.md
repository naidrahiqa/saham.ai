# saham.ai
📈 Program Analisis Volume Saham Indonesia
Program ini dirancang untuk menganalisis volume perdagangan saham Indonesia dan memberikan sinyal beli (buy) atau jual (sell) berdasarkan indikator teknikal yang komprehensif.

🚀 Fitur Utama
Analisis Volume: Menggunakan Volume Moving Average (VMA), Volume Rate of Change (VROC), On Balance Volume (OBV)

Indikator Teknikal: MACD, RSI, Moving Average (SMA 20 & 50)

Sinyal Trading: Rekomendasi beli/jual berdasarkan multiple konfirmasi indikator

Risk Management: Target price (+8%) dan stop loss (-5%) otomatis

Visualisasi: Grafik candlestick dengan penanda sinyal dan indikator teknikal

Export Data: Kemampuan menyimpan hasil analisis ke file CSV

📋 Prerequisites
Sebelum menjalankan program, pastikan Anda telah menginstall:

Python 3.6 atau lebih baru

Pip (Python package manager)

🔧 Instalasi
Clone atau download repository ini

Install dependencies yang diperlukan:

bash
pip install pandas numpy yfinance matplotlib mplfinance
🎯 Cara Menggunakan
Jalankan program di terminal:

bash
python saham.py
Masukkan kode saham Indonesia yang ingin dianalisis (contoh: BBCA, TLKM, BBRI)

Program akan menampilkan hasil analisis yang mencakup:

Rekomendasi trading (Beli/Jual/Tahan)

Target price dan stop loss

Level support dan resistance

Analisis indikator teknikal (RSI, MACD, Volume)

Sinyal historis 30 hari terakhir

Pilih opsi untuk melihat grafik analisis teknikal atau menyimpan hasil ke CSV

📊 Indikator yang Digunakan
1. Analisis Volume
VMA (Volume Moving Average): Rata-rata volume 20 hari

VROC (Volume Rate of Change): Perubahan volume 10 hari

OBV (On Balance Volume): Akumulasi volume positif/negatif

VPT (Volume Price Trend): Hubungan volume dan perubahan harga

2. Indikator Teknikal
MACD: Momentum trend (12, 26, 9 periode)

RSI: Relative Strength Index (14 periode)

SMA: Simple Moving Average (20 & 50 periode)

⚡ Kriteria Sinyal
🟢 Sinyal BELI (Buy):
Volume > 1.5x VMA dengan harga menguat

MACD bullish crossover dengan RSI < 70

OBV trending up dengan trend bullish (SMA 20 > SMA 50)

🔴 Sinyal JUAL (Sell):
Volume > 1.5x VMA dengan harga melemah

MACD bearish crossover dengan RSI > 30

OBV trending down dengan trend bearish (SMA 20 < SMA 50)

📁 Struktur Output
Program akan menghasilkan:

Analisis real-time di terminal

Grafik teknikal interaktif (opsional)

File CSV dengan data historis dan sinyal (opsional)

🎪 Contoh Saham Populer Indonesia
Beberapa saham populer yang dapat dianalisis:

BBCA (Bank BCA)

TLKM (Telkom Indonesia)

BBRI (Bank BRI)

ASII (Astra International)

UNVR (Unilever Indonesia)

ICBP (Indofood CBP)

EXCL (XL Axiata)

ADRO (Adaro Energy)

ANTM (Aneka Tambang)

BMRI (Bank Mandiri)

⚠️ Disclaimer
PENTING: Program ini hanya sebagai alat bantu analisis teknikal dan bukan merupakan saran finansial. Keputusan investasi sepenuhnya merupakan tanggung jawab pengguna. Selalu lakukan penelitian sendiri dan pertimbangkan faktor fundamental sebelum melakukan investasi.

🔄 Update Terbaru
Analisis volume dan indikator teknikal terintegrasi

Rekomendasi target price dan stop loss otomatis

Visualisasi grafik teknikal dengan mplfinance

Support untuk saham Indonesia (.JK)

Export hasil analisis ke CSV

📝 License
Program ini dibuat untuk tujuan edukasi dan analisis teknikal. Penggunaan untuk tujuan komersial memerlukan izin penulis.

🤝 Kontribusi
Untuk pertanyaan atau saran pengembangan, silakan buka issue atau pull request di repository ini.

Selamat menganalisis! Semoga program ini membantu dalam pengambilan keputusan trading Anda. 📊💹
