import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import mplfinance as mpf
import warnings
warnings.filterwarnings('ignore')

# Library untuk sentiment analysis
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("TextBlob tidak tersedia. Menggunakan analisis sentimen sederhana.")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

class AnalisisBerita:
    """Kelas untuk menganalisis berita terkait saham"""
    
    def __init__(self):
        self.berita_data = []
        
    def ambil_berita(self, ticker, max_berita=10):
        """
        Mengambil berita terkait saham dari Yahoo Finance
        """
        try:
            saham = yf.Ticker(ticker)
            berita = saham.news
            
            if not berita:
                return []
            
            # Ambil berita terbaru
            berita_list = []
            for item in berita[:max_berita]:
                berita_info = {
                    'title': item.get('title', ''),
                    'publisher': item.get('publisher', ''),
                    'link': item.get('link', ''),
                    'datetime': datetime.fromtimestamp(item.get('providerPublishTime', 0)) if item.get('providerPublishTime') else None
                }
                berita_list.append(berita_info)
            
            self.berita_data = berita_list
            return berita_list
            
        except Exception as e:
            print(f"Error mengambil berita: {e}")
            return []
    
    def analisis_sentimen(self, teks):
        """
        Menganalisis sentimen dari teks berita
        """
        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(teks)
                polarity = blob.sentiment.polarity
                
                if polarity > 0.1:
                    return 'Positif', polarity
                elif polarity < -0.1:
                    return 'Negatif', polarity
                else:
                    return 'Netral', polarity
            except:
                pass
        
        # Analisis sederhana jika TextBlob tidak tersedia
        teks_lower = teks.lower()
        kata_positif = ['naik', 'meningkat', 'untung', 'profit', 'growth', 'baik', 'positif', 'bullish', 'buy']
        kata_negatif = ['turun', 'menurun', 'rugi', 'loss', 'buruk', 'negatif', 'bearish', 'sell', 'jatuh']
        
        skor = 0
        for kata in kata_positif:
            if kata in teks_lower:
                skor += 1
        for kata in kata_negatif:
            if kata in teks_lower:
                skor -= 1
        
        if skor > 0:
            return 'Positif', min(skor / 10, 1.0)
        elif skor < 0:
            return 'Negatif', max(skor / 10, -1.0)
        else:
            return 'Netral', 0.0
    
    def ringkasan_sentimen_berita(self, berita_list):
        """
        Memberikan ringkasan sentimen dari semua berita
        """
        if not berita_list:
            return None
        
        total_sentimen = 0
        jumlah_positif = 0
        jumlah_negatif = 0
        jumlah_netral = 0
        
        for berita in berita_list:
            sentimen, skor = self.analisis_sentimen(berita['title'])
            total_sentimen += skor
            
            if sentimen == 'Positif':
                jumlah_positif += 1
            elif sentimen == 'Negatif':
                jumlah_negatif += 1
            else:
                jumlah_netral += 1
        
        rata_sentimen = total_sentimen / len(berita_list)
        
        return {
            'rata_sentimen': rata_sentimen,
            'jumlah_positif': jumlah_positif,
            'jumlah_negatif': jumlah_negatif,
            'jumlah_netral': jumlah_netral,
            'total_berita': len(berita_list)
        }

class AnalisisTeknikalLengkap:
    """Kelas untuk analisis teknikal yang lebih lengkap"""
    
    @staticmethod
    def bollinger_bands(df, period=20, std_dev=2):
        """
        Menghitung Bollinger Bands
        """
        df['BB_Middle'] = df['Close'].rolling(window=period).mean()
        bb_std = df['Close'].rolling(window=period).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * std_dev)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * std_dev)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        # Hindari division by zero
        bb_range = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = np.where(
            bb_range != 0,
            (df['Close'] - df['BB_Lower']) / bb_range,
            0.5  # Default ke tengah jika tidak ada range
        )
        return df
    
    @staticmethod
    def stochastic_oscillator(df, k_period=14, d_period=3):
        """
        Menghitung Stochastic Oscillator
        """
        low_min = df['Low'].rolling(window=k_period).min()
        high_max = df['High'].rolling(window=k_period).max()
        # Hindari division by zero
        stoch_range = high_max - low_min
        df['%K'] = np.where(
            stoch_range != 0,
            100 * ((df['Close'] - low_min) / stoch_range),
            50  # Default ke tengah jika tidak ada range
        )
        df['%D'] = df['%K'].rolling(window=d_period).mean()
        return df
    
    @staticmethod
    def adx(df, period=14):
        """
        Menghitung Average Directional Index (ADX)
        """
        # True Range
        df['TR'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            )
        )
        
        # Directional Movement
        df['+DM'] = np.where(
            (df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
            np.maximum(df['High'] - df['High'].shift(1), 0),
            0
        )
        df['-DM'] = np.where(
            (df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
            np.maximum(df['Low'].shift(1) - df['Low'], 0),
            0
        )
        
        # Smoothed values
        df['TR_Smooth'] = df['TR'].rolling(window=period).sum()
        df['+DM_Smooth'] = df['+DM'].rolling(window=period).sum()
        df['-DM_Smooth'] = df['-DM'].rolling(window=period).sum()
        
        # Directional Indicators - hindari division by zero
        df['+DI'] = np.where(
            df['TR_Smooth'] != 0,
            100 * (df['+DM_Smooth'] / df['TR_Smooth']),
            0
        )
        df['-DI'] = np.where(
            df['TR_Smooth'] != 0,
            100 * (df['-DM_Smooth'] / df['TR_Smooth']),
            0
        )
        
        # ADX - hindari division by zero
        di_sum = df['+DI'] + df['-DI']
        df['DX'] = np.where(
            di_sum != 0,
            100 * abs(df['+DI'] - df['-DI']) / di_sum,
            0
        )
        df['ADX'] = df['DX'].rolling(window=period).mean()
        
        return df
    
    @staticmethod
    def williams_r(df, period=14):
        """
        Menghitung Williams %R
        """
        high_max = df['High'].rolling(window=period).max()
        low_min = df['Low'].rolling(window=period).min()
        # Hindari division by zero
        wr_range = high_max - low_min
        df['Williams_R'] = np.where(
            wr_range != 0,
            -100 * ((high_max - df['Close']) / wr_range),
            -50  # Default ke tengah jika tidak ada range
        )
        return df
    
    @staticmethod
    def cci(df, period=20):
        """
        Menghitung Commodity Channel Index (CCI)
        """
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        # Hindari division by zero
        df['CCI'] = np.where(
            mad != 0,
            (typical_price - sma_tp) / (0.015 * mad),
            0
        )
        return df
    
    @staticmethod
    def atr(df, period=14):
        """
        Menghitung Average True Range (ATR)
        """
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        df['ATR'] = true_range.rolling(window=period).mean()
        return df
    
    @staticmethod
    def fibonacci_retracement(df):
        """
        Menghitung level Fibonacci Retracement
        """
        recent_high = df['High'].rolling(window=50).max().iloc[-1]
        recent_low = df['Low'].rolling(window=50).min().iloc[-1]
        diff = recent_high - recent_low
        
        fib_levels = {
            '0%': recent_high,
            '23.6%': recent_high - (diff * 0.236),
            '38.2%': recent_high - (diff * 0.382),
            '50%': recent_high - (diff * 0.5),
            '61.8%': recent_high - (diff * 0.618),
            '78.6%': recent_high - (diff * 0.786),
            '100%': recent_low
        }
        
        return fib_levels

class AnalisisFundamental:
    """Kelas untuk analisis fundamental saham"""
    
    def __init__(self):
        self.info_saham = None
    
    def ambil_data_fundamental(self, ticker):
        """
        Mengambil data fundamental dari Yahoo Finance
        """
        try:
            saham = yf.Ticker(ticker)
            self.info_saham = saham.info
            return self.info_saham
        except Exception as e:
            print(f"Error mengambil data fundamental: {e}")
            return None
    
    def tampilkan_fundamental(self, kode_saham):
        """
        Menampilkan data fundamental saham
        """
        if not self.info_saham:
            return
        
        print(f"\n{'='*70}")
        print(f"ANALISIS FUNDAMENTAL - {kode_saham}")
        print(f"{'='*70}")
        
        # Data perusahaan
        print("\n📊 DATA PERUSAHAAN:")
        print(f"   Nama Perusahaan    : {self.info_saham.get('longName', 'N/A')}")
        print(f"   Sektor            : {self.info_saham.get('sector', 'N/A')}")
        print(f"   Industri          : {self.info_saham.get('industry', 'N/A')}")
        
        # Valuasi
        print("\n💰 VALUASI:")
        pe_ratio = self.info_saham.get('trailingPE', None)
        if pe_ratio:
            print(f"   P/E Ratio         : {pe_ratio:.2f}")
        
        pb_ratio = self.info_saham.get('priceToBook', None)
        if pb_ratio:
            print(f"   P/B Ratio         : {pb_ratio:.2f}")
        
        market_cap = self.info_saham.get('marketCap', None)
        if market_cap:
            print(f"   Market Cap        : Rp {market_cap:,.0f}")
        
        # Profitabilitas
        print("\n📈 PROFITABILITAS:")
        profit_margin = self.info_saham.get('profitMargins', None)
        if profit_margin:
            print(f"   Profit Margin     : {profit_margin*100:.2f}%")
        
        roe = self.info_saham.get('returnOnEquity', None)
        if roe:
            print(f"   ROE              : {roe*100:.2f}%")
        
        roa = self.info_saham.get('returnOnAssets', None)
        if roa:
            print(f"   ROA              : {roa*100:.2f}%")
        
        # Growth
        print("\n📊 PERTUMBUHAN:")
        revenue_growth = self.info_saham.get('revenueGrowth', None)
        if revenue_growth:
            print(f"   Revenue Growth    : {revenue_growth*100:.2f}%")
        
        earnings_growth = self.info_saham.get('earningsGrowth', None)
        if earnings_growth:
            print(f"   Earnings Growth   : {earnings_growth*100:.2f}%")
        
        # Dividen
        print("\n💵 DIVIDEN:")
        dividend_yield = self.info_saham.get('dividendYield', None)
        if dividend_yield:
            print(f"   Dividend Yield    : {dividend_yield*100:.2f}%")
        
        print(f"{'='*70}")

class AnalisisSahamLengkap:
    """Kelas utama untuk analisis saham yang lengkap"""
    
    def __init__(self):
        self.data_saham = None
        self.ticker = None
        self.analisis_berita = AnalisisBerita()
        self.analisis_fundamental = AnalisisFundamental()
        self.analisis_teknikal = AnalisisTeknikalLengkap()
        
    def unduh_data_saham(self, kode_saham, periode="6mo"):
        """
        Mengunduh data saham dari Yahoo Finance
        """
        try:
            # Untuk saham Indonesia, tambahkan .JK di akhir kode saham
            self.ticker = kode_saham + ".JK"
            print(f"Mengunduh data untuk {self.ticker}...")
            saham = yf.Ticker(self.ticker)
            self.data_saham = saham.history(period=periode)
            
            if self.data_saham.empty:
                print(f"Tidak dapat menemukan data untuk {kode_saham}")
                return False
                
            print(f"Berhasil mengunduh data untuk {kode_saham}")
            return True
            
        except Exception as e:
            print(f"Error mengunduh data: {e}")
            return False
    
    def hitung_indikator_teknikal(self):
        """
        Menghitung semua indikator teknikal
        """
        if self.data_saham is None or self.data_saham.empty:
            print("Tidak ada data saham yang tersedia")
            return
        
        df = self.data_saham.copy()
        
        # Indikator volume yang sudah ada
        df['VMA_20'] = df['Volume'].rolling(window=20).mean()
        # VROC_10 - hindari division by zero
        volume_shift = df['Volume'].shift(10)
        df['VROC_10'] = np.where(
            volume_shift != 0,
            ((df['Volume'] - volume_shift) / volume_shift) * 100,
            0
        )
        
        # OBV
        df['OBV'] = 0
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                df['OBV'].iloc[i] = df['OBV'].iloc[i-1] + df['Volume'].iloc[i]
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                df['OBV'].iloc[i] = df['OBV'].iloc[i-1] - df['Volume'].iloc[i]
            else:
                df['OBV'].iloc[i] = df['OBV'].iloc[i-1]
        
        # VPT
        df['VPT'] = 0
        for i in range(1, len(df)):
            vpt_change = df['Volume'].iloc[i] * ((df['Close'].iloc[i] - df['Close'].iloc[i-1]) / df['Close'].iloc[i-1])
            df['VPT'].iloc[i] = df['VPT'].iloc[i-1] + vpt_change
        
        # MACD
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI - hindari division by zero
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = np.where(loss != 0, gain / loss, 0)
        df['RSI'] = np.where(
            rs != 0,
            100 - (100 / (1 + rs)),
            50  # Default ke tengah jika tidak ada perhitungan
        )
        
        # SMA
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # EMA
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # Bollinger Bands
        df = self.analisis_teknikal.bollinger_bands(df)
        
        # Stochastic
        df = self.analisis_teknikal.stochastic_oscillator(df)
        
        # ADX
        df = self.analisis_teknikal.adx(df)
        
        # Williams %R
        df = self.analisis_teknikal.williams_r(df)
        
        # CCI
        df = self.analisis_teknikal.cci(df)
        
        # ATR
        df = self.analisis_teknikal.atr(df)
        
        self.data_saham = df
        return df
    
    def generate_sinyal_lengkap(self):
        """
        Menghasilkan sinyal beli/jual dengan analisis yang lebih lengkap
        """
        if self.data_saham is None or self.data_saham.empty:
            print("Tidak ada data saham yang tersedia")
            return None
        
        df = self.data_saham.copy()
        
        # Inisialisasi kolom sinyal
        df['Sinyal'] = 'Tahan'
        df['Alasan'] = ''
        df['Skor_Sinyal'] = 0  # Skor untuk mengukur kekuatan sinyal
        
        # Aturan untuk sinyal beli (dengan skor)
        skor_beli = 0
        
        # Kondisi 1: Volume tinggi dengan harga menguat
        kondisi_beli_1 = (
            (df['Volume'] > 1.5 * df['VMA_20']) & 
            (df['Close'] > df['Open']) &
            (df['VROC_10'] > 20)
        )
        df.loc[kondisi_beli_1, 'Skor_Sinyal'] += 2
        df.loc[kondisi_beli_1, 'Alasan'] += 'Volume tinggi dengan harga menguat. '
        
        # Kondisi 2: MACD bullish
        kondisi_beli_2 = (
            (df['MACD'] > df['MACD_Signal']) &
            (df['RSI'] < 70) &
            (df['RSI'] > 30) &
            (df['Close'] > df['SMA_20'])
        )
        df.loc[kondisi_beli_2, 'Skor_Sinyal'] += 2
        df.loc[kondisi_beli_2, 'Alasan'] += 'Konfirmasi bullish dari MACD dan RSI. '
        
        # Kondisi 3: OBV trending up
        kondisi_beli_3 = (
            (df['OBV'] > df['OBV'].shift(5)) &
            (df['SMA_20'] > df['SMA_50'])
        )
        df.loc[kondisi_beli_3, 'Skor_Sinyal'] += 1
        df.loc[kondisi_beli_3, 'Alasan'] += 'Momentum positif dari OBV. '
        
        # Kondisi 4: Bollinger Bands - harga mendekati lower band
        kondisi_beli_4 = (
            (df['BB_Position'] < 0.2) &
            (df['RSI'] < 50) &
            (df['Close'] > df['BB_Lower'])
        )
        df.loc[kondisi_beli_4, 'Skor_Sinyal'] += 1
        df.loc[kondisi_beli_4, 'Alasan'] += 'Harga mendekati support (BB Lower). '
        
        # Kondisi 5: Stochastic oversold
        kondisi_beli_5 = (
            (df['%K'] < 20) &
            (df['%D'] < 20) &
            (df['%K'] > df['%D'])
        )
        df.loc[kondisi_beli_5, 'Skor_Sinyal'] += 1
        df.loc[kondisi_beli_5, 'Alasan'] += 'Stochastic oversold dengan potensi reversal. '
        
        # Kondisi 6: ADX menunjukkan trend kuat
        kondisi_beli_6 = (
            (df['ADX'] > 25) &
            (df['+DI'] > df['-DI']) &
            (df['Close'] > df['SMA_20'])
        )
        df.loc[kondisi_beli_6, 'Skor_Sinyal'] += 1
        df.loc[kondisi_beli_6, 'Alasan'] += 'Trend bullish kuat (ADX). '
        
        # Aturan untuk sinyal jual
        # Kondisi 1: Volume tinggi dengan harga melemah
        kondisi_jual_1 = (
            (df['Volume'] > 1.5 * df['VMA_20']) & 
            (df['Close'] < df['Open']) &
            (df['VROC_10'] < -20)
        )
        df.loc[kondisi_jual_1, 'Skor_Sinyal'] -= 2
        df.loc[kondisi_jual_1, 'Alasan'] += 'Volume tinggi dengan harga melemah. '
        
        # Kondisi 2: MACD bearish
        kondisi_jual_2 = (
            (df['MACD'] < df['MACD_Signal']) &
            (df['RSI'] > 30) &
            (df['RSI'] < 70) &
            (df['Close'] < df['SMA_20'])
        )
        df.loc[kondisi_jual_2, 'Skor_Sinyal'] -= 2
        df.loc[kondisi_jual_2, 'Alasan'] += 'Konfirmasi bearish dari MACD dan RSI. '
        
        # Kondisi 3: OBV trending down
        kondisi_jual_3 = (
            (df['OBV'] < df['OBV'].shift(5)) &
            (df['SMA_20'] < df['SMA_50'])
        )
        df.loc[kondisi_jual_3, 'Skor_Sinyal'] -= 1
        df.loc[kondisi_jual_3, 'Alasan'] += 'Momentum negatif dari OBV. '
        
        # Kondisi 4: Bollinger Bands - harga mendekati upper band
        kondisi_jual_4 = (
            (df['BB_Position'] > 0.8) &
            (df['RSI'] > 50) &
            (df['Close'] < df['BB_Upper'])
        )
        df.loc[kondisi_jual_4, 'Skor_Sinyal'] -= 1
        df.loc[kondisi_jual_4, 'Alasan'] += 'Harga mendekati resistance (BB Upper). '
        
        # Kondisi 5: Stochastic overbought
        kondisi_jual_5 = (
            (df['%K'] > 80) &
            (df['%D'] > 80) &
            (df['%K'] < df['%D'])
        )
        df.loc[kondisi_jual_5, 'Skor_Sinyal'] -= 1
        df.loc[kondisi_jual_5, 'Alasan'] += 'Stochastic overbought dengan potensi reversal. '
        
        # Kondisi 6: ADX menunjukkan trend bearish kuat
        kondisi_jual_6 = (
            (df['ADX'] > 25) &
            (df['-DI'] > df['+DI']) &
            (df['Close'] < df['SMA_20'])
        )
        df.loc[kondisi_jual_6, 'Skor_Sinyal'] -= 1
        df.loc[kondisi_jual_6, 'Alasan'] += 'Trend bearish kuat (ADX). '
        
        # Tentukan sinyal berdasarkan skor
        df.loc[df['Skor_Sinyal'] >= 3, 'Sinyal'] = 'Beli'
        df.loc[df['Skor_Sinyal'] <= -3, 'Sinyal'] = 'Jual'
        df.loc[(df['Skor_Sinyal'] > -3) & (df['Skor_Sinyal'] < 3), 'Sinyal'] = 'Tahan'
        
        # Fill NaN values dengan nilai default yang aman
        df = df.fillna({
            'RSI': 50,
            'MACD': 0,
            'MACD_Signal': 0,
            '%K': 50,
            '%D': 50,
            'ADX': 0,
            '+DI': 0,
            '-DI': 0,
            'Williams_R': -50,
            'CCI': 0,
            'ATR': 0,
            'BB_Position': 0.5,
            'VROC_10': 0
        })
        
        return df
    
    def rekomendasi_trading_lengkap(self, df_sinyal, kode_saham, ringkasan_berita=None):
        """
        Memberikan rekomendasi trading yang lebih detail dengan integrasi berita
        """
        latest = df_sinyal.iloc[-1]
        
        print(f"\n{'='*70}")
        print(f"REKOMENDASI TRADING LENGKAP - {kode_saham}")
        print(f"{'='*70}")
        
        # Informasi dasar dengan error handling
        print(f"\n📊 INFORMASI DASAR:")
        try:
            print(f"   Tanggal Analisis    : {df_sinyal.index[-1].strftime('%d %B %Y')}")
        except:
            print(f"   Tanggal Analisis    : N/A")
        
        close_price = latest.get('Close', 0)
        volume = latest.get('Volume', 0)
        
        if pd.notna(close_price):
            print(f"   Harga Terakhir      : Rp {close_price:,.2f}")
        else:
            print(f"   Harga Terakhir      : N/A")
        
        if pd.notna(volume):
            print(f"   Volume Perdagangan  : {volume:,.0f}")
        else:
            print(f"   Volume Perdagangan  : N/A")
        
        vma_20 = latest.get('VMA_20', None)
        if pd.notna(vma_20) and vma_20 != 0 and pd.notna(volume):
            print(f"   Rasio Volume/VMA    : {volume/vma_20:.2f}x")
        else:
            print(f"   Rasio Volume/VMA    : N/A")
        
        # Indikator teknikal dengan error handling
        print(f"\n📈 INDIKATOR TEKNIKAL:")
        rsi = latest.get('RSI', 50)
        if pd.notna(rsi):
            rsi_status = '(Overbought)' if rsi > 70 else '(Oversold)' if rsi < 30 else '(Normal)'
            print(f"   RSI (14)            : {rsi:.2f} {rsi_status}")
        else:
            print(f"   RSI (14)            : N/A")
        
        macd = latest.get('MACD', 0)
        macd_signal = latest.get('MACD_Signal', 0)
        if pd.notna(macd) and pd.notna(macd_signal):
            print(f"   MACD                : {macd:.2f}")
            print(f"   Signal Line         : {macd_signal:.2f}")
        else:
            print(f"   MACD                : N/A")
            print(f"   Signal Line         : N/A")
        
        stoch_k = latest.get('%K', 50)
        stoch_d = latest.get('%D', 50)
        if pd.notna(stoch_k) and pd.notna(stoch_d):
            print(f"   Stochastic %K       : {stoch_k:.2f}")
            print(f"   Stochastic %D       : {stoch_d:.2f}")
        else:
            print(f"   Stochastic %K       : N/A")
            print(f"   Stochastic %D       : N/A")
        
        adx = latest.get('ADX', 0)
        if pd.notna(adx):
            adx_status = '(Trend Kuat)' if adx > 25 else '(Trend Lemah)'
            print(f"   ADX                 : {adx:.2f} {adx_status}")
        else:
            print(f"   ADX                 : N/A")
        
        williams_r = latest.get('Williams_R', -50)
        cci = latest.get('CCI', 0)
        atr = latest.get('ATR', 0)
        bb_pos = latest.get('BB_Position', 0.5)
        
        if pd.notna(williams_r):
            print(f"   Williams %R         : {williams_r:.2f}")
        else:
            print(f"   Williams %R         : N/A")
        
        if pd.notna(cci):
            print(f"   CCI                 : {cci:.2f}")
        else:
            print(f"   CCI                 : N/A")
        
        if pd.notna(atr):
            print(f"   ATR                 : {atr:.2f}")
        else:
            print(f"   ATR                 : N/A")
        
        if pd.notna(bb_pos):
            print(f"   BB Position         : {bb_pos:.2f} (0=Lower, 1=Upper)")
        else:
            print(f"   BB Position         : N/A")
        
        # Sentimen berita
        if ringkasan_berita:
            print(f"\n📰 SENTIMEN BERITA:")
            print(f"   Total Berita       : {ringkasan_berita['total_berita']}")
            print(f"   Berita Positif     : {ringkasan_berita['jumlah_positif']}")
            print(f"   Berita Negatif     : {ringkasan_berita['jumlah_negatif']}")
            print(f"   Berita Netral      : {ringkasan_berita['jumlah_netral']}")
            print(f"   Rata-rata Sentimen : {ringkasan_berita['rata_sentimen']:.2f}")
            
            if ringkasan_berita['rata_sentimen'] > 0.2:
                print(f"   ⚠️  Sentimen cenderung POSITIF")
            elif ringkasan_berita['rata_sentimen'] < -0.2:
                print(f"   ⚠️  Sentimen cenderung NEGATIF")
            else:
                print(f"   ⚠️  Sentimen cenderung NETRAL")
        
        # Rekomendasi dengan error handling
        print(f"\n🎯 REKOMENDASI:")
        sinyal = latest.get('Sinyal', 'Tahan')
        skor = latest.get('Skor_Sinyal', 0)
        alasan = latest.get('Alasan', '')
        
        print(f"   Sinyal              : {sinyal}")
        if pd.notna(skor):
            print(f"   Skor Sinyal         : {skor:.0f}")
        else:
            print(f"   Skor Sinyal         : 0")
        print(f"   Alasan              : {alasan if alasan else 'Tidak ada sinyal kuat'}")
        
        # Target dan stop loss dengan error handling
        try:
            resistance = df_sinyal['High'].rolling(20).max().iloc[-1]
            support = df_sinyal['Low'].rolling(20).min().iloc[-1]
            if pd.isna(resistance):
                resistance = df_sinyal['High'].max()
            if pd.isna(support):
                support = df_sinyal['Low'].min()
        except:
            resistance = df_sinyal['High'].max()
            support = df_sinyal['Low'].min()
        
        try:
            fib_levels = self.analisis_teknikal.fibonacci_retracement(df_sinyal)
        except:
            fib_levels = {}
        
        sinyal_value = latest.get('Sinyal', 'Tahan')
        
        if sinyal_value == 'Beli':
            close_val = latest.get('Close', 0)
            if pd.notna(close_val) and close_val > 0:
                target_price = close_val * 1.08
                stop_loss = close_val * 0.95
                
                print(f"\n💰 TARGET & RISK MANAGEMENT:")
                print(f"   Target Price       : Rp {target_price:,.2f} (+{((target_price/close_val)-1)*100:.1f}%)")
                print(f"   Stop Loss          : Rp {stop_loss:,.2f} (-{((1-stop_loss/close_val))*100:.1f}%)")
            else:
                print(f"\n💰 TARGET & RISK MANAGEMENT:")
                print(f"   Target Price       : N/A")
                print(f"   Stop Loss          : N/A")
            print(f"   Resistance Level   : Rp {resistance:,.2f}")
            print(f"   Support Level      : Rp {support:,.2f}")
            print(f"\n   Fibonacci Levels:")
            for level, price in fib_levels.items():
                print(f"      {level:6s} : Rp {price:,.2f}")
            
            print(f"\n💡 SARAN TRADING:")
            print("   - Entry: Beli di harga saat ini atau pada pullback ke support")
            print("   - Kelola risk-reward ratio minimal 1:2")
            print("   - Pertimbangkan untuk averaging jika harga turun ke support")
            if ringkasan_berita and ringkasan_berita['rata_sentimen'] > 0.2:
                print("   - ⚠️  Sentimen berita positif mendukung keputusan beli")
            
        elif sinyal_value == 'Jual':
            close_val = latest.get('Close', 0)
            if pd.notna(close_val) and close_val > 0:
                target_price = close_val * 0.92
                stop_loss = close_val * 1.05
                
                print(f"\n💰 TARGET & RISK MANAGEMENT:")
                print(f"   Target Price       : Rp {target_price:,.2f} (-{((1-target_price/close_val))*100:.1f}%)")
                print(f"   Stop Loss          : Rp {stop_loss:,.2f} (+{((stop_loss/close_val)-1)*100:.1f}%)")
            else:
                print(f"\n💰 TARGET & RISK MANAGEMENT:")
                print(f"   Target Price       : N/A")
                print(f"   Stop Loss          : N/A")
            print(f"   Resistance Level   : Rp {resistance:,.2f}")
            print(f"   Support Level      : Rp {support:,.2f}")
            print(f"\n   Fibonacci Levels:")
            for level, price in fib_levels.items():
                print(f"      {level:6s} : Rp {price:,.2f}")
            
            print(f"\n💡 SARAN TRADING:")
            print("   - Exit: Jual di harga saat ini atau pada bounce ke resistance")
            print("   - Pertimbangkan untuk stop loss trailing jika trend bearish kuat")
            print("   - Hindari averaging down dalam kondisi downtrend")
            if ringkasan_berita and ringkasan_berita['rata_sentimen'] < -0.2:
                print("   - ⚠️  Sentimen berita negatif mendukung keputusan jual")
        else:
            print(f"\n💰 LEVEL PENTING:")
            print(f"   Resistance Level   : Rp {resistance:,.2f}")
            print(f"   Support Level      : Rp {support:,.2f}")
            print(f"\n   Fibonacci Levels:")
            for level, price in fib_levels.items():
                print(f"      {level:6s} : Rp {price:,.2f}")
            
            print(f"\n💡 SARAN TRADING:")
            print("   - Tunggu konfirmasi breakout atau breakdown")
            print("   - Pantau level support dan resistance")
            print("   - Perhatikan volume untuk konfirmasi pergerakan")
            print("   - Awasi sentimen berita untuk trigger selanjutnya")
        
        print(f"{'='*70}")
    
    def tampilkan_berita(self, berita_list, max_tampil=5):
        """
        Menampilkan berita terkait saham
        """
        if not berita_list:
            print("\n📰 Tidak ada berita yang ditemukan")
            return
        
        print(f"\n{'='*70}")
        print(f"BERITA TERKINI ({len(berita_list)} berita)")
        print(f"{'='*70}")
        
        for i, berita in enumerate(berita_list[:max_tampil], 1):
            sentimen, skor = self.analisis_berita.analisis_sentimen(berita['title'])
            emoji = "📈" if sentimen == 'Positif' else "📉" if sentimen == 'Negatif' else "📊"
            
            print(f"\n{emoji} Berita #{i}:")
            print(f"   Judul     : {berita['title']}")
            print(f"   Publisher : {berita['publisher']}")
            if berita['datetime']:
                print(f"   Tanggal   : {berita['datetime'].strftime('%d %B %Y %H:%M')}")
            print(f"   Sentimen  : {sentimen} (skor: {skor:.2f})")
            if berita['link']:
                print(f"   Link      : {berita['link']}")
        
        print(f"{'='*70}")
    
    def plot_analisis_teknikal_lengkap(self, df_sinyal, kode_saham):
        """
        Membuat plot analisis teknikal yang lebih lengkap
        """
        try:
            # Siapkan data untuk plotting dengan error handling
            apds = []
            
            # Moving Averages
            if 'SMA_20' in df_sinyal.columns and df_sinyal['SMA_20'].notna().any():
                apds.append(mpf.make_addplot(df_sinyal['SMA_20'], color='blue', width=1, label='SMA 20'))
            if 'SMA_50' in df_sinyal.columns and df_sinyal['SMA_50'].notna().any():
                apds.append(mpf.make_addplot(df_sinyal['SMA_50'], color='red', width=1, label='SMA 50'))
            
            # Bollinger Bands
            if 'BB_Upper' in df_sinyal.columns and df_sinyal['BB_Upper'].notna().any():
                apds.append(mpf.make_addplot(df_sinyal['BB_Upper'], color='gray', width=0.5, linestyle='--', alpha=0.5))
            if 'BB_Lower' in df_sinyal.columns and df_sinyal['BB_Lower'].notna().any():
                apds.append(mpf.make_addplot(df_sinyal['BB_Lower'], color='gray', width=0.5, linestyle='--', alpha=0.5))
            
            # Volume
            if 'VMA_20' in df_sinyal.columns and df_sinyal['VMA_20'].notna().any():
                apds.append(mpf.make_addplot(df_sinyal['VMA_20'], panel=1, color='orange', width=1))
            
            # RSI
            if 'RSI' in df_sinyal.columns and df_sinyal['RSI'].notna().any():
                apds.append(mpf.make_addplot(df_sinyal['RSI'], panel=2, color='purple', width=1, ylim=[0, 100]))
                apds.append(mpf.make_addplot([70] * len(df_sinyal), panel=2, color='red', width=0.5, linestyle='--'))
                apds.append(mpf.make_addplot([30] * len(df_sinyal), panel=2, color='green', width=0.5, linestyle='--'))
            
            # MACD
            if 'MACD' in df_sinyal.columns and df_sinyal['MACD'].notna().any():
                apds.append(mpf.make_addplot(df_sinyal['MACD'], panel=3, color='blue', width=1, label='MACD'))
            if 'MACD_Signal' in df_sinyal.columns and df_sinyal['MACD_Signal'].notna().any():
                apds.append(mpf.make_addplot(df_sinyal['MACD_Signal'], panel=3, color='red', width=1, label='Signal'))
            
            # Stochastic
            if '%K' in df_sinyal.columns and df_sinyal['%K'].notna().any():
                apds.append(mpf.make_addplot(df_sinyal['%K'], panel=4, color='blue', width=1, label='%K'))
            if '%D' in df_sinyal.columns and df_sinyal['%D'].notna().any():
                apds.append(mpf.make_addplot(df_sinyal['%D'], panel=4, color='red', width=1, label='%D'))
                apds.append(mpf.make_addplot([80] * len(df_sinyal), panel=4, color='red', width=0.5, linestyle='--'))
                apds.append(mpf.make_addplot([20] * len(df_sinyal), panel=4, color='green', width=0.5, linestyle='--'))
            
            # Tandai sinyal beli dan jual
            if 'Sinyal' in df_sinyal.columns:
                beli_titik = df_sinyal[df_sinyal['Sinyal'] == 'Beli']
                jual_titik = df_sinyal[df_sinyal['Sinyal'] == 'Jual']
                
                if not beli_titik.empty and 'Low' in beli_titik.columns:
                    apds.append(mpf.make_addplot(beli_titik['Low'] * 0.99, type='scatter', 
                                                markersize=50, marker='^', color='green', panel=0))
                
                if not jual_titik.empty and 'High' in jual_titik.columns:
                    apds.append(mpf.make_addplot(jual_titik['High'] * 1.01, type='scatter', 
                                                markersize=50, marker='v', color='red', panel=0))
            
            # Buat plot
            fig, axes = mpf.plot(df_sinyal, 
                                type='candle', 
                                style='charles',
                                addplot=apds if apds else None,
                                title=f'Analisis Teknikal Lengkap - {kode_saham}',
                                ylabel='Harga (Rp)',
                                volume=True,
                                ylabel_lower='Volume',
                                figratio=(14, 10),
                                returnfig=True)
            
            # Tambahkan label dengan error handling
            try:
                if len(axes) > 2 and 'RSI' in df_sinyal.columns:
                    axes[2].axhline(y=70, color='r', linestyle='--', alpha=0.5)
                    axes[2].axhline(y=30, color='g', linestyle='--', alpha=0.5)
                    axes[2].set_ylabel('RSI')
            except:
                pass
            
            try:
                if len(axes) > 4 and '%K' in df_sinyal.columns:
                    axes[4].axhline(y=80, color='r', linestyle='--', alpha=0.5)
                    axes[4].axhline(y=20, color='g', linestyle='--', alpha=0.5)
                    axes[4].set_ylabel('Stochastic')
            except:
                pass
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error saat membuat plot: {e}")
            print("Mencoba membuat plot sederhana...")
            try:
                mpf.plot(df_sinyal, type='candle', volume=True, title=f'Analisis Teknikal - {kode_saham}')
                plt.show()
            except Exception as e2:
                print(f"Error membuat plot sederhana: {e2}")
    
    def analisis_saham_lengkap(self, kode_saham, tampilkan_berita=True, tampilkan_fundamental=True):
        """
        Melakukan analisis lengkap untuk sebuah saham
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print(f"\n{'='*70}")
        print(f"ANALISIS SAHAM LENGKAP - {kode_saham}")
        print(f"{'='*70}")
        
        # Unduh data saham
        if not self.unduh_data_saham(kode_saham):
            return None
        
        # Analisis fundamental
        if tampilkan_fundamental:
            try:
                print("\n📊 Mengambil data fundamental...")
                self.analisis_fundamental.ambil_data_fundamental(self.ticker)
                if self.analisis_fundamental.info_saham:
                    self.analisis_fundamental.tampilkan_fundamental(kode_saham)
                else:
                    print("⚠️  Data fundamental tidak tersedia untuk saham ini")
            except Exception as e:
                print(f"⚠️  Error mengambil data fundamental: {e}")
        
        # Analisis berita
        ringkasan_berita = None
        if tampilkan_berita:
            try:
                print("\n📰 Mengambil berita terkini...")
                berita_list = self.analisis_berita.ambil_berita(self.ticker, max_berita=10)
                if berita_list:
                    ringkasan_berita = self.analisis_berita.ringkasan_sentimen_berita(berita_list)
                    self.tampilkan_berita(berita_list, max_tampil=5)
                else:
                    print("⚠️  Tidak ada berita yang ditemukan untuk saham ini")
            except Exception as e:
                print(f"⚠️  Error mengambil berita: {e}")
        
        # Hitung indikator teknikal
        try:
            print("\n📈 Menghitung indikator teknikal...")
            self.hitung_indikator_teknikal()
            
            # Generate sinyal
            print("🎯 Menghasilkan sinyal trading...")
            df_sinyal = self.generate_sinyal_lengkap()
            
            if df_sinyal is None or df_sinyal.empty:
                print("⚠️  Error: Tidak dapat menghasilkan sinyal trading")
                return None
            
            # Tampilkan rekomendasi trading
            self.rekomendasi_trading_lengkap(df_sinyal, kode_saham, ringkasan_berita)
            
        except Exception as e:
            print(f"⚠️  Error dalam analisis teknikal: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        # Tanyakan apakah ingin melihat plot
        try:
            plot = input("\nLihat grafik analisis teknikal lengkap? (y/n): ").strip().lower()
            if plot == 'y':
                self.plot_analisis_teknikal_lengkap(df_sinyal, kode_saham)
        except Exception as e:
            print(f"⚠️  Error saat meminta input plot: {e}")
        
        return df_sinyal

def main():
    # Inisialisasi analyzer
    analyzer = AnalisisSahamLengkap()
    
    # Header program
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"{'='*70}")
    print("ANALISIS SAHAM INDONESIA - VERSI LENGKAP")
    print("Program untuk analisis saham dengan teknikal, fundamental, dan berita")
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
        
        # Tanyakan opsi analisis
        print("\nOpsi analisis:")
        print("1. Analisis lengkap (Teknikal + Fundamental + Berita)")
        print("2. Analisis teknikal saja")
        print("3. Analisis teknikal + berita")
        pilihan = input("Pilih opsi (1/2/3, default=1): ").strip() or "1"
        
        tampilkan_berita = pilihan in ['1', '3']
        tampilkan_fundamental = pilihan == '1'
        
        # Lakukan analisis
        try:
            df_sinyal = analyzer.analisis_saham_lengkap(
                kode_saham, 
                tampilkan_berita=tampilkan_berita,
                tampilkan_fundamental=tampilkan_fundamental
            )
            
            # Tanyakan apakah ingin menyimpan hasil
            if df_sinyal is not None:
                try:
                    simpan = input("\nSimpan hasil analisis ke file CSV? (y/n): ").strip().lower()
                    if simpan == 'y':
                        try:
                            nama_file = f"analisis_{kode_saham}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                            df_sinyal.to_csv(nama_file)
                            print(f"Hasil analisis disimpan sebagai {nama_file}")
                        except Exception as e:
                            print(f"⚠️  Error menyimpan file: {e}")
                except Exception as e:
                    print(f"⚠️  Error saat meminta input: {e}")
                
        except Exception as e:
            print(f"Error menganalisis {kode_saham}: {e}")
            print("Pastikan kode saham benar dan terhubung ke internet")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # Install library tambahan jika belum ada
    libraries = {
        'mplfinance': 'mplfinance',
        'textblob': 'textblob',
        'requests': 'requests'
    }
    
    for lib_name, pip_name in libraries.items():
        try:
            __import__(lib_name)
        except ImportError:
            print(f"Menginstall library {pip_name}...")
            os.system(f'pip install {pip_name}')
    
    main()
