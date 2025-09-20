import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import mplfinance as mpf

class AnalisisVolumeSaham:
    def __init__(self):
        self.data_saham = None
        
    def unduh_data_saham(self, kode_saham, periode="6mo"):
        """
        Mengunduh data saham dari Yahoo Finance
        """
        try:
            # Untuk saham Indonesia, tambahkan .JK di akhir kode saham
            ticker = kode_saham + ".JK"
            print(f"Mengunduh data untuk {ticker}...")
            saham = yf.Ticker(ticker)
            self.data_saham = saham.history(period=periode)
            
            if self.data_saham.empty:
                print(f"Tidak dapat menemukan data untuk {kode_saham}")
                return False
                
            print(f"Berhasil mengunduh data untuk {kode_saham}")
            return True
            
        except Exception as e:
            print(f"Error mengunduh data: {e}")
            return False
    
    def hitung_indikator_volume(self):
        """
        Menghitung berbagai indikator berbasis volume
        """
        if self.data_saham is None or self.data_saham.empty:
            print("Tidak ada data saham yang tersedia")
            return
        
        df = self.data_saham.copy()
        
        # 1. Volume Moving Average (VMA)
        df['VMA_20'] = df['Volume'].rolling(window=20).mean()
        
        # 2. Volume Rate of Change (VROC)
        df['VROC_10'] = ((df['Volume'] - df['Volume'].shift(10)) / df['Volume'].shift(10)) * 100
        
        # 3. On Balance Volume (OBV)
        df['OBV'] = 0
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                df['OBV'].iloc[i] = df['OBV'].iloc[i-1] + df['Volume'].iloc[i]
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                df['OBV'].iloc[i] = df['OBV'].iloc[i-1] - df['Volume'].iloc[i]
            else:
                df['OBV'].iloc[i] = df['OBV'].iloc[i-1]
        
        # 4. Volume Price Trend (VPT)
        df['VPT'] = 0
        for i in range(1, len(df)):
            vpt_change = df['Volume'].iloc[i] * ((df['Close'].iloc[i] - df['Close'].iloc[i-1]) / df['Close'].iloc[i-1])
            df['VPT'].iloc[i] = df['VPT'].iloc[i-1] + vpt_change
        
        # 5. Moving Average Convergence Divergence (MACD)
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # 6. Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 7. Simple Moving Average (SMA)
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        self.data_saham = df
        return df
    
    def generate_sinyal(self):
        """
        Menghasilkan sinyal beli/jual berdasarkan analisis volume dan teknikal
        """
        if self.data_saham is None or self.data_saham.empty:
            print("Tidak ada data saham yang tersedia")
            return None
        
        df = self.data_saham.copy()
        
        # Inisialisasi kolom sinyal
        df['Sinyal'] = 'Tahan'
        df['Alasan'] = ''
        
        # Aturan untuk sinyal beli:
        kondisi_beli_1 = (
            (df['Volume'] > 1.5 * df['VMA_20']) & 
            (df['Close'] > df['Open']) &  # Hari hijau (harga naik)
            (df['VROC_10'] > 20)  # Volume meningkat signifikan
        )
        
        kondisi_beli_2 = (
            (df['MACD'] > df['MACD_Signal']) &  # MACD bullish crossover
            (df['RSI'] < 70) &  # RSI tidak overbought
            (df['Close'] > df['SMA_20'])  # Harga di atas SMA 20
        )
        
        kondisi_beli_3 = (
            (df['OBV'] > df['OBV'].shift(5)) &  # OBV trending up
            (df['SMA_20'] > df['SMA_50'])  # Trend jangka pendek bullish
        )
        
        # Aturan untuk sinyal jual:
        kondisi_jual_1 = (
            (df['Volume'] > 1.5 * df['VMA_20']) & 
            (df['Close'] < df['Open']) &  # Hari merah (harga turun)
            (df['VROC_10'] < -20)  # Volume menurun signifikan
        )
        
        kondisi_jual_2 = (
            (df['MACD'] < df['MACD_Signal']) &  # MACD bearish crossover
            (df['RSI'] > 30) &  # RSI tidak oversold
            (df['Close'] < df['SMA_20'])  # Harga di bawah SMA 20
        )
        
        kondisi_jual_3 = (
            (df['OBV'] < df['OBV'].shift(5)) &  # OBV trending down
            (df['SMA_20'] < df['SMA_50'])  # Trend jangka pendek bearish
        )
        
        # Terapkan kondisi beli
        df.loc[kondisi_beli_1, 'Sinyal'] = 'Beli'
        df.loc[kondisi_beli_1, 'Alasan'] = 'Volume tinggi dengan harga menguat'
        
        df.loc[kondisi_beli_2, 'Sinyal'] = 'Beli'
        df.loc[kondisi_beli_2, 'Alasan'] = 'Konfirmasi bullish dari MACD dan RSI'
        
        df.loc[kondisi_beli_3, 'Sinyal'] = 'Beli'
        df.loc[kondisi_beli_3, 'Alasan'] = 'Momentum positif dari OBV dan trend'
        
        # Terapkan kondisi jual
        df.loc[kondisi_jual_1, 'Sinyal'] = 'Jual'
        df.loc[kondisi_jual_1, 'Alasan'] = 'Volume tinggi dengan harga melemah'
        
        df.loc[kondisi_jual_2, 'Sinyal'] = 'Jual'
        df.loc[kondisi_jual_2, 'Alasan'] = 'Konfirmasi bearish dari MACD dan RSI'
        
        df.loc[kondisi_jual_3, 'Sinyal'] = 'Jual'
        df.loc[kondisi_jual_3, 'Alasan'] = 'Momentum negatif dari OBV dan trend'
        
        # Prioritaskan sinyal yang lebih kuat
        for i in range(len(df)):
            if df['Sinyal'].iloc[i] == 'Beli' and df['Alasan'].iloc[i] == 'Volume tinggi dengan harga menguat':
                if i > 0 and df['Sinyal'].iloc[i-1] == 'Beli':
                    df['Sinyal'].iloc[i] = 'Tahan'
                    df['Alasan'].iloc[i] = 'Sinyal beli sudah aktif'
        
        return df
    
    def rekomendasi_trading(self, df_sinyal, kode_saham):
        """
        Memberikan rekomendasi trading yang lebih detail
        """
        # Ambil data terbaru
        latest = df_sinyal.iloc[-1]
        
        print(f"\n{'='*70}")
        print(f"REKOMENDASI TRADING UNTUK {kode_saham}")
        print(f"{'='*70}")
        
        # Tampilkan informasi dasar
        print(f"Tanggal Analisis    : {df_sinyal.index[-1].strftime('%d %B %Y')}")
        print(f"Harga Terakhir      : Rp {latest['Close']:,.2f}")
        print(f"Volume Perdagangan  : {latest['Volume']:,.0f}")
        print(f"Rasio Volume/VMA    : {latest['Volume']/latest['VMA_20']:.2f}x")
        print(f"RSI (14)            : {latest['RSI']:.2f}")
        print(f"MACD                : {latest['MACD']:.2f}")
        print(f"Signal Line         : {latest['MACD_Signal']:.2f}")
        
        # Berikan rekomendasi berdasarkan kondisi
        if latest['Sinyal'] == 'Beli':
            print(f"\n🚀 REKOMENDASI: BELI (BUY)")
            print(f"Alasan: {latest['Alasan']}")
            
            # Hitung target dan stop loss
            resistance = df_sinyal['High'].rolling(20).max().iloc[-1]
            support = df_sinyal['Low'].rolling(20).min().iloc[-1]
            
            target_price = latest['Close'] * 1.08  # Target 8% kenaikan
            stop_loss = latest['Close'] * 0.95     # Stop loss 5%
            
            print(f"\n📈 Target Price     : Rp {target_price:,.2f} (+{((target_price/latest['Close'])-1)*100:.1f}%)")
            print(f"🛑 Stop Loss        : Rp {stop_loss:,.2f} (-{((1-stop_loss/latest['Close']))*100:.1f}%)")
            print(f"📊 Resistance Level : Rp {resistance:,.2f}")
            print(f"📉 Support Level    : Rp {support:,.2f}")
            
            print(f"\n💡 Saran Trading:")
            print("- Entry: Beli di harga saat ini atau pada pullback ke support")
            print("- Kelola risk-reward ratio minimal 1:2")
            print("- Pertimbangkan untuk averaging jika harga turun ke support")
            
        elif latest['Sinyal'] == 'Jual':
            print(f"\n🔻 REKOMENDASI: JUAL (SELL)")
            print(f"Alasan: {latest['Alasan']}")
            
            # Hitung target dan stop loss
            resistance = df_sinyal['High'].rolling(20).max().iloc[-1]
            support = df_sinyal['Low'].rolling(20).min().iloc[-1]
            
            target_price = latest['Close'] * 0.92  # Target 8% penurunan
            stop_loss = latest['Close'] * 1.05     # Stop loss 5%
            
            print(f"\n📈 Target Price     : Rp {target_price:,.2f} (-{((1-target_price/latest['Close']))*100:.1f}%)")
            print(f"🛑 Stop Loss        : Rp {stop_loss:,.2f} (+{((stop_loss/latest['Close'])-1)*100:.1f}%)")
            print(f"📊 Resistance Level : Rp {resistance:,.2f}")
            print(f"📉 Support Level    : Rp {support:,.2f}")
            
            print(f"\n💡 Saran Trading:")
            print("- Exit: Jual di harga saat ini atau pada bounce ke resistance")
            print("- Pertimbangkan untuk stop loss trailing jika trend bearish kuat")
            print("- Hindari averaging down dalam kondisi downtrend")
            
        else:
            print(f"\n⚪ REKOMENDASI: TAHAN (HOLD)")
            print(f"Alasan: {latest['Alasan'] if latest['Alasan'] else 'Tidak ada sinyal kuat'}")
            
            print(f"\n💡 Saran Trading:")
            print("- Tunggu konfirmasi breakout atau breakdown")
            print("- Pantau level support dan resistance")
            print("- Perhatikan volume untuk konfirmasi pergerakan")
        
        print(f"{'='*70}")
    
    def sinyal_historis(self, df_sinyal):
        """
        Menampilkan sinyal historis untuk analisis pola
        """
        sinyal_beli = df_sinyal[df_sinyal['Sinyal'] == 'Beli']
        sinyal_jual = df_sinyal[df_sinyal['Sinyal'] == 'Jual']
        
        print(f"\n📊 Sinyal Historis 30 Hari Terakhir:")
        print("-" * 80)
        print("Tanggal     | Harga Tutup | Volume     | Sinyal | Alasan")
        print("-" * 80)
        
        for i in range(-30, 0):
            if i >= -len(df_sinyal):
                idx = i
                data = df_sinyal.iloc[idx]
                tanggal = df_sinyal.index[idx].strftime('%d-%m')
                harga = data['Close']
                volume = data['Volume']
                sinyal = data['Sinyal']
                alasan = data['Alasan'][:20] + "..." if len(data['Alasan']) > 20 else data['Alasan']
                
                print(f"{tanggal} | Rp {harga:8,.2f} | {volume:10,.0f} | {sinyal:5} | {alasan}")
    
    def plot_analisis_teknikal(self, df_sinyal, kode_saham):
        """
        Membuat plot analisis teknikal
        """
        # Siapkan data untuk plotting
        apds = [
            mpf.make_addplot(df_sinyal['SMA_20'], color='blue', width=1),
            mpf.make_addplot(df_sinyal['SMA_50'], color='red', width=1),
            mpf.make_addplot(df_sinyal['VMA_20'], panel=1, color='orange', width=1),
            mpf.make_addplot(df_sinyal['RSI'], panel=2, color='purple', width=1, ylim=[0, 100]),
            mpf.make_addplot([70] * len(df_sinyal), panel=2, color='red', width=0.5, linestyle='--'),
            mpf.make_addplot([30] * len(df_sinyal), panel=2, color='green', width=0.5, linestyle='--'),
        ]
        
        # Tandai sinyal beli dan jual
        beli_titik = df_sinyal[df_sinyal['Sinyal'] == 'Beli']
        jual_titik = df_sinyal[df_sinyal['Sinyal'] == 'Jual']
        
        if not beli_titik.empty:
            apds.append(mpf.make_addplot(beli_titik['Low'] * 0.99, type='scatter', 
                                        markersize=50, marker='^', color='green', panel=0))
        
        if not jual_titik.empty:
            apds.append(mpf.make_addplot(jual_titik['High'] * 1.01, type='scatter', 
                                        markersize=50, marker='v', color='red', panel=0))
        
        # Buat plot
        fig, axes = mpf.plot(df_sinyal, 
                            type='candle', 
                            style='charles',
                            addplot=apds,
                            title=f'Analisis Teknikal {kode_saham}',
                            ylabel='Harga (Rp)',
                            volume=True,
                            ylabel_lower='Volume',
                            figratio=(12, 8),
                            returnfig=True)
        
        # Tambahkan garis overbought/oversold pada RSI
        axes[2].axhline(y=70, color='r', linestyle='--', alpha=0.5)
        axes[2].axhline(y=30, color='g', linestyle='--', alpha=0.5)
        axes[2].set_ylabel('RSI')
        
        plt.show()
    
    def analisis_saham(self, kode_saham):
        """
        Melakukan analisis lengkap untuk sebuah saham
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print(f"\n{'='*70}")
        print(f"Menganalisis {kode_saham}...")
        print(f"{'='*70}")
        
        # Unduh data saham
        if not self.unduh_data_saham(kode_saham):
            return
        
        # Hitung indikator volume
        self.hitung_indikator_volume()
        
        # Generate sinyal
        df_sinyal = self.generate_sinyal()
        
        # Tampilkan rekomendasi trading
        self.rekomendasi_trading(df_sinyal, kode_saham)
        
        # Tampilkan sinyal historis
        self.sinyal_historis(df_sinyal)
        
        # Tanyakan apakah ingin melihat plot
        plot = input("\nLihat grafik analisis teknikal? (y/n): ").strip().lower()
        if plot == 'y':
            self.plot_analisis_teknikal(df_sinyal, kode_saham)
        
        return df_sinyal

def main():
    # Inisialisasi analyzer
    analyzer = AnalisisVolumeSaham()
    
    # Header program
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"{'='*70}")
    print("ANALISIS VOLUME SAHAM INDONESIA")
    print("Program untuk menentukan waktu beli/jual berdasarkan analisis teknikal")
    print(f"{'='*70}")
    
    # Daftar saham populer Indonesia
    saham_populer = ['BBCA', 'TLKM', 'BBRI', 'ASII', 'UNVR', 'ICBP', 'EXCL', 'ADRO', 'ANTM', 'BMRI']
    
    while True:
        print(f"\nSaham populer: {', '.join(saham_populer)}")
        kode_saham = input("\nMasukkan kode saham (atau 'quit' untuk keluar): ").strip().upper()
        
        if kode_saham.lower() == 'quit':
            print("Terima kasih telah menggunakan program analisis saham!")
            break
        
        if not kode_saham:
            print("Kode saham tidak boleh kosong!")
            continue
        
        # Lakukan analisis
        try:
            df_sinyal = analyzer.analisis_saham(kode_saham)
            
            # Tanyakan apakah ingin menyimpan hasil
            if df_sinyal is not None:
                simpan = input("\nSimpan hasil analisis ke file CSV? (y/n): ").strip().lower()
                if simpan == 'y':
                    nama_file = f"analisis_{kode_saham}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
                    df_sinyal.to_csv(nama_file)
                    print(f"Hasil analisis disimpan sebagai {nama_file}")
                
        except Exception as e:
            print(f"Error menganalisis {kode_saham}: {e}")
            print("Pastikan kode saham benar dan terhubung ke internet")

if __name__ == "__main__":
    # Install library tambahan jika belum ada
    try:
        import mplfinance
    except:
        print("Menginstall library mplfinance...")
        os.system('pip install mplfinance')
    
    main()