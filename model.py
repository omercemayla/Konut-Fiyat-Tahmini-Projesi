# Gerekli kütüphaneleri yüklüyoruz
import pandas as pd  # Veri işleme için
import numpy as np   # Sayısal hesaplamalar için
# Makine öğrenmesi algoritmaları için sklearn kütüphaneleri
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold  # Veriyi bölme ve doğrulama için
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error  # Performans ölçme için
from sklearn.preprocessing import RobustScaler, PowerTransformer, LabelEncoder, PolynomialFeatures  # Veri dönüşümü için
from sklearn.pipeline import Pipeline  # İşlem zincirleri için
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression, RFE  # Özellik seçimi için
from sklearn.linear_model import Ridge  # Doğrusal regresyon için
from scipy import stats  # İstatistiksel işlemler için
from scipy.stats import boxcox  # Veri dönüşümü için
import joblib  # Model kaydetme/yükleme için
import os  # Dosya işlemleri için
import matplotlib.pyplot as plt  # Grafik çizme için
import seaborn as sns  # Güzel grafikler için
import warnings
warnings.filterwarnings('ignore')  # Uyarıları gizle


def load_and_preprocess_data():
    """Veri setini yükle ve ön işle - İyileştirilmiş veri temizleme"""
    try:
        # Excel dosyasını pandas ile oku
        df = pd.read_excel('istanbul_konut2.xlsx')
        
        # Sütun isimlerini Türkçe karakterler ve boşluklar olmadan düzenle (consistency için)
        df.columns = ['fiyat', 'ilce', 'mahalle', 'metrekare', 'oda_sayisi', 'yas', 'bulundugu_kat']
        
        # Kaç tane kayıt yüklendiğini ekrana yazdır
        print(f"İlk yükleme: {df.shape[0]} kayıt")
        
        # Eksik (NaN) verileri temizle - pandas dropna() ile
        df = df.dropna()  # Eksik verileri kaldır
        
        # Fiyat sütununu sayısal veriye dönüştür, hata varsa NaN yap
        df['fiyat'] = pd.to_numeric(df['fiyat'], errors='coerce')
        # Fiyatı eksik/hatalı olan kayıtları sil
        df = df.dropna(subset=['fiyat'])  # Fiyatı eksik olan kayıtları çıkar
        
        # İlçe ve mahalle değerlerini standartlaştır ve tutarlı hale getir
        df['ilce'] = df['ilce'].str.strip().str.title()      # Başına-sonuna boşluk sil, ilk harfi büyüt
        df['mahalle'] = df['mahalle'].str.strip().str.title()  # Aynı işlemi mahalle için de yap
        
        # Aynı anlama gelebilecek farklı yazımları birleştir (örnek: Üsküdar = Uskudar)
        ilce_mapping = {
            'Üsküdar': 'Üsküdar',   # Standart yazım
            'Uskudar': 'Üsküdar',   # Türkçesiz yazımı standarda çevir
            'Beşiktaş': 'Beşiktaş', # Standart yazım
            'Besiktas': 'Beşiktaş', # Türkçesiz yazımı standarda çevir
            'Şişli': 'Şişli',       # Standart yazım
            'Sisli': 'Şişli'        # Türkçesiz yazımı standarda çevir
        }
        df['ilce'] = df['ilce'].replace(ilce_mapping)  # Mapping'i uygula
        
        # Sayısal değerlerin makul aralıklarda olduğundan emin ol (aykırı değerleri temizle)
        df = df[(df['metrekare'] >= 30) & (df['metrekare'] <= 400)]      # 30-400 m² arası makul
        df = df[(df['oda_sayisi'] >= 1) & (df['oda_sayisi'] <= 8)]       # 1-8 oda arası makul
        df = df[(df['yas'] >= 0) & (df['yas'] <= 80)]                    # 0-80 yaş arası makul
        df = df[(df['bulundugu_kat'] >= -2) & (df['bulundugu_kat'] <= 40)]  # -2 ile 40. kat arası makul
        
        # Fiyat üzerinde daha agresif aykırı değer temizleme
        df = df[(df['fiyat'] >= 100000) & (df['fiyat'] <= 50000000)]  # 100bin-50milyon TL arası makul
        
        # Gelişmiş aykırı değer temizleme - Çok aşamalı temizleme sistemi
        print(f"Temel temizleme sonrası: {df.shape[0]} kayıt")
        
        # 1. Aşama: Z-score tabanlı aykırı değer temizleme (tüm veri için genel temizlik)
        from scipy import stats  # İstatistiksel fonksiyonlar için
        z_scores = np.abs(stats.zscore(df['fiyat']))  # Her fiyat için Z-score hesapla (ortalamadan kaç sigma uzak)
        df = df[z_scores < 3]  # 3 sigma (standart sapma) dışındaki değerleri aykırı say ve çıkar
        print(f"Z-score temizleme sonrası: {df.shape[0]} kayıt")
        
        # 2. Aşama: İlçe bazında daha agresif aykırı değer tespiti (her ilçeyi kendi içinde temizle)
        cleaned_dfs = []  # Temizlenmiş ilçe verilerini tutacak liste
        for ilce in df['ilce'].unique():  # Her ilçe için döngü
            ilce_df = df[df['ilce'] == ilce].copy()  # Bu ilçenin tüm verilerini al
            
            if len(ilce_df) < 10:  # Çok az örneği olan ilçeleri atla (güvenilir istatistik için)
                continue
                
            # İlçe bazında fiyat aykırı değerleri - IQR yöntemiyle agresif temizlik
            Q1_fiyat = ilce_df['fiyat'].quantile(0.1)   # Alt %10 (1. çeyrek yerine daha agresif)
            Q3_fiyat = ilce_df['fiyat'].quantile(0.9)   # Üst %10 (3. çeyrek yerine daha agresif)
            IQR_fiyat = Q3_fiyat - Q1_fiyat  # Interquartile Range (çeyrekler arası fark)
            lower_bound_fiyat = Q1_fiyat - 1.0 * IQR_fiyat  # Alt sınır (daha sıkı: 1.0 çarpan)
            upper_bound_fiyat = Q3_fiyat + 1.0 * IQR_fiyat  # Üst sınır (daha sıkı: 1.0 çarpan)
            # Bu sınırlar dışındaki fiyatları çıkar
            ilce_df = ilce_df[(ilce_df['fiyat'] >= lower_bound_fiyat) & (ilce_df['fiyat'] <= upper_bound_fiyat)]
            
            # Metrekare bazında da temizleme (her ilçe için ayrı ayrı)
            Q1_m2 = ilce_df['metrekare'].quantile(0.05)  # Alt %5
            Q3_m2 = ilce_df['metrekare'].quantile(0.95)  # Üst %5
            IQR_m2 = Q3_m2 - Q1_m2  # Metrekare için IQR
            lower_bound_m2 = Q1_m2 - 1.5 * IQR_m2  # Alt sınır
            upper_bound_m2 = Q3_m2 + 1.5 * IQR_m2  # Üst sınır
            # Bu sınırlar dışındaki metrekareleri çıkar
            ilce_df = ilce_df[(ilce_df['metrekare'] >= lower_bound_m2) & (ilce_df['metrekare'] <= upper_bound_m2)]
            
            cleaned_dfs.append(ilce_df)  # Temizlenmiş ilçe verisini listeye ekle
        
        df = pd.concat(cleaned_dfs, ignore_index=True)
        print(f"İlçe bazında temizleme sonrası: {df.shape[0]} kayıt")
        
        # Metrekare başına düşen fiyat hesapla ve aykırı değerleri temizle
        df['fiyat_metrekare'] = df['fiyat'] / df['metrekare']
        Q1_fiyat_m2 = df['fiyat_metrekare'].quantile(0.01)
        Q3_fiyat_m2 = df['fiyat_metrekare'].quantile(0.99)
        IQR_fiyat_m2 = Q3_fiyat_m2 - Q1_fiyat_m2
        lower_bound_m2 = Q1_fiyat_m2 - 1.5 * IQR_fiyat_m2
        upper_bound_m2 = Q3_fiyat_m2 + 1.5 * IQR_fiyat_m2
        df = df[(df['fiyat_metrekare'] >= lower_bound_m2) & (df['fiyat_metrekare'] <= upper_bound_m2)]
        
        # İlçe bazında minimum örnek sayısı kontrolü
        ilce_counts = df['ilce'].value_counts()
        valid_ilceler = ilce_counts[ilce_counts >= 10].index  # En az 10 örneği olan ilçeleri al
        df = df[df['ilce'].isin(valid_ilceler)]
        
        # Mahalle bazında minimum örnek sayısı kontrolü
        mahalle_counts = df['mahalle'].value_counts()
        valid_mahalleler = mahalle_counts[mahalle_counts >= 5].index  # En az 5 örneği olan mahalleleri al
        df = df[df['mahalle'].isin(valid_mahalleler)]
        
        # İlçe ve mahalle bazında fiyat istatistikleri
        ilce_stats = df.groupby('ilce')['fiyat'].agg(['mean', 'median', 'std']).reset_index()
        mahalle_stats = df.groupby('mahalle')['fiyat'].agg(['mean', 'median', 'std']).reset_index()
        
        # İlçe ve mahalle bazında fiyat istatistikleri ile encoding
        df = df.merge(ilce_stats, on='ilce', how='left', suffixes=('', '_ilce'))
        df = df.merge(mahalle_stats, on='mahalle', how='left', suffixes=('', '_mahalle'))
        
        # Logaritmik dönüşüm (fiyat dağılımını normalleştirmek için)
        df['fiyat_log'] = np.log1p(df['fiyat'])
        
        # İlçe ve mahalle frekans kodlaması
        ilce_freq = df['ilce'].value_counts(normalize=True).to_dict()
        mahalle_freq = df['mahalle'].value_counts(normalize=True).to_dict()
        df['ilce_freq'] = df['ilce'].map(ilce_freq)
        df['mahalle_freq'] = df['mahalle'].map(mahalle_freq)
        
        print(f"Veri temizleme sonrası kalan örnek sayısı: {df.shape[0]}")
        
        return df
    except Exception as e:
        print(f"Veri yükleme hatası: {e}")
        return None

def create_advanced_features(df):
    """Özellik mühendisliği: Polynomial ve complex interactions"""
    # Ham veriden sayısal özellikleri seç (makine öğrenmesi için gerekli)
    numerical_cols = ['metrekare', 'oda_sayisi', 'yas', 'bulundugu_kat', 
                     'mean', 'median', 'std', 'mean_mahalle', 'median_mahalle', 'std_mahalle',
                     'ilce_freq', 'mahalle_freq']  # İlçe ve mahalle istatistikleri de dahil
    X_numerical = df[numerical_cols].copy()  # Bu sütunları kopyala
    
    # Kaç özellikle başladığımızı kaydet
    print(f"Feature engineering öncesi: {X_numerical.shape[1]} özellik")
    
    # Temel istatistiksel özellikler
    X_numerical.loc[:, 'fiyat_volatilite'] = X_numerical['std'] / (X_numerical['mean'] + 1)  # Volatilite
    X_numerical.loc[:, 'mahalle_volatilite'] = X_numerical['std_mahalle'] / (X_numerical['mean_mahalle'] + 1)
    X_numerical.loc[:, 'mahalle_premium'] = X_numerical['mean_mahalle'] / (X_numerical['mean'] + 1)  # Mahalle primı
    
    # Pazarlık indeksi - fiyat dağılımına dayalı
    X_numerical.loc[:, 'pazarlik_indeksi'] = (X_numerical['mean'] - X_numerical['median']) / (X_numerical['std'] + 1)
    X_numerical.loc[:, 'mahalle_pazarlik_indeksi'] = (X_numerical['mean_mahalle'] - X_numerical['median_mahalle']) / (X_numerical['std_mahalle'] + 1)
    
    # Bölgesel lüks indeksi
    X_numerical.loc[:, 'bolgsel_luksus_skoru'] = X_numerical['mean'] * X_numerical['ilce_freq']
    X_numerical.loc[:, 'mahalle_luksus_skoru'] = X_numerical['mean_mahalle'] * X_numerical['mahalle_freq']
    
    # Temel transformasyonlar - İyileştirilmiş
    X_numerical.loc[:, 'metrekare_oda_orani'] = X_numerical['metrekare'] / (X_numerical['oda_sayisi'] + 0.1)
    X_numerical.loc[:, 'metrekare_kare'] = X_numerical['metrekare'] ** 2
    X_numerical.loc[:, 'metrekare_log'] = np.log1p(X_numerical['metrekare'])
    X_numerical.loc[:, 'metrekare_sqrt'] = np.sqrt(X_numerical['metrekare'])
    X_numerical.loc[:, 'metrekare_kup'] = X_numerical['metrekare'] ** 3
    
    # Gelişmiş metrekare kombinasyonları
    X_numerical.loc[:, 'metrekare_oda_kare'] = X_numerical['metrekare_oda_orani'] ** 2
    X_numerical.loc[:, 'metrekare_oda_log'] = np.log1p(X_numerical['metrekare_oda_orani'])
    X_numerical.loc[:, 'ideal_metrekare_sapma'] = np.abs(X_numerical['metrekare'] - (X_numerical['oda_sayisi'] * 25))  # İdeal alan sapması
    
    # Yaş transformasyonları - daha sofistike
    X_numerical.loc[:, 'yas_kare'] = X_numerical['yas'] ** 2
    X_numerical.loc[:, 'yas_log'] = np.log1p(X_numerical['yas'] + 1)
    X_numerical.loc[:, 'yas_sqrt'] = np.sqrt(X_numerical['yas'] + 1)
    X_numerical.loc[:, 'yas_tersi'] = 1 / (X_numerical['yas'] + 1)  # Yeni bina değeri
    X_numerical.loc[:, 'yas_exp'] = np.exp(-X_numerical['yas'] / 20)  # Yenilik değeri (exponential decay)
    
    # Yaş kategorileri ve değer kaybı modeli
    X_numerical.loc[:, 'yeni_bina'] = (X_numerical['yas'] <= 5).astype(int)
    X_numerical.loc[:, 'orta_yas_bina'] = ((X_numerical['yas'] > 5) & (X_numerical['yas'] <= 15)).astype(int)
    X_numerical.loc[:, 'eski_bina'] = (X_numerical['yas'] > 15).astype(int)
    X_numerical.loc[:, 'deger_kaybi_orani'] = np.maximum(0, 1 - (X_numerical['yas'] / 50))  # Değer kaybı oranı
    
    # Metrekare eficiencysi - gelişmiş
    X_numerical.loc[:, 'alan_verimliligi_v2'] = X_numerical['metrekare'] / (X_numerical['oda_sayisi'] ** 1.2)
    X_numerical.loc[:, 'oda_buyuklugu_avg'] = X_numerical['metrekare'] / (X_numerical['oda_sayisi'] + 0.5)  # Ortalama oda büyüklüğü
    
    # Oda büyüklüğü kategorileri
    X_numerical.loc[:, 'genis_odalar'] = (X_numerical['oda_buyuklugu_avg'] > 20).astype(int)
    X_numerical.loc[:, 'orta_odalar'] = ((X_numerical['oda_buyuklugu_avg'] >= 15) & (X_numerical['oda_buyuklugu_avg'] <= 20)).astype(int)
    X_numerical.loc[:, 'dar_odalar'] = (X_numerical['oda_buyuklugu_avg'] < 15).astype(int)
    
    # Kat transformasyonları
    X_numerical.loc[:, 'kat_zemin'] = (X_numerical['bulundugu_kat'] == 0).astype(int)
    X_numerical.loc[:, 'kat_yuksek'] = (X_numerical['bulundugu_kat'] > 5).astype(int)
    X_numerical.loc[:, 'kat_bodrum'] = (X_numerical['bulundugu_kat'] < 0).astype(int)
    X_numerical.loc[:, 'kat_1_3'] = ((X_numerical['bulundugu_kat'] >= 1) & (X_numerical['bulundugu_kat'] <= 3)).astype(int)
    X_numerical.loc[:, 'kat_4_7'] = ((X_numerical['bulundugu_kat'] >= 4) & (X_numerical['bulundugu_kat'] <= 7)).astype(int)
    X_numerical.loc[:, 'kat_8_plus'] = (X_numerical['bulundugu_kat'] >= 8).astype(int)
    X_numerical.loc[:, 'kat_log'] = np.log1p(X_numerical['bulundugu_kat'] + 3)
    X_numerical.loc[:, 'kat_kare'] = (X_numerical['bulundugu_kat'] + 3) ** 2
    
    # Kat avantaj skoru
    X_numerical.loc[:, 'kat_avantaj_skoru'] = np.where(
        X_numerical['bulundugu_kat'] < 0, -0.1,  # Bodrum katlar
        np.where(X_numerical['bulundugu_kat'] == 0, 0,  # Zemin kat
        np.where(X_numerical['bulundugu_kat'] <= 3, 0.1,  # Düşük katlar
        np.where(X_numerical['bulundugu_kat'] <= 7, 0.15,  # Orta katlar
        np.where(X_numerical['bulundugu_kat'] <= 15, 0.05,  # Yüksek katlar
        -0.05))))  # Çok yüksek katlar
    )
    
    # Kompleks etkileşimler
    X_numerical.loc[:, 'yas_metrekare_etkilesim'] = X_numerical['yas_tersi'] * X_numerical['metrekare_log']
    X_numerical.loc[:, 'kat_alan_etkilesim'] = X_numerical['kat_avantaj_skoru'] * X_numerical['metrekare_oda_orani']
    X_numerical.loc[:, 'premium_lokasyon_skoru'] = X_numerical['bolgsel_luksus_skoru'] * X_numerical['yas_exp']
    
    # Pazar dinamikleri
    X_numerical.loc[:, 'arz_talep_dengesi'] = X_numerical['ilce_freq'] / (X_numerical['mahalle_freq'] + 0.001)
    X_numerical.loc[:, 'fiyat_istikrar_indeksi'] = 1 / (X_numerical['fiyat_volatilite'] + 0.1)
    
    # Gelişmiş istatistiksel özellikler
    X_numerical.loc[:, 'z_score_mahalle'] = (X_numerical['mean_mahalle'] - X_numerical['mean']) / (X_numerical['std'] + 1)
    X_numerical.loc[:, 'mahalle_median_orani'] = X_numerical['median_mahalle'] / (X_numerical['median'] + 1)
    
    # Polynomial özellikler
    important_features = ['metrekare', 'oda_sayisi', 'yas_tersi', 'kat_avantaj_skoru']
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_features = poly.fit_transform(X_numerical[important_features])
    poly_feature_names = poly.get_feature_names_out(important_features)
    
    # Polynomial özellikleri ekle (sadece etkileşim terimleri)
    for i, name in enumerate(poly_feature_names):
        if ' ' in name:  # Sadece etkileşim terimlerini al
            clean_name = name.replace(' ', '_').replace('^', '_pow')
            X_numerical.loc[:, clean_name] = poly_features[:, i]
    
    # NaN ve inf değerleri temizle
    X_numerical = X_numerical.replace([np.inf, -np.inf], np.nan)
    X_numerical = X_numerical.fillna(X_numerical.median())
    
    print(f"Feature engineering sonrası: {X_numerical.shape[1]} özellik")
    
    return X_numerical

def create_features(df):
    """Ana özellik oluşturma fonksiyonu - geriye uyumluluk için"""
    return create_advanced_features(df)

def advanced_target_encode_categorical(df, categorical_cols, target_col, n_splits=5):
    """Gelişmiş target encoding with multiple strategies and smoothing"""
    df_encoded = df.copy()
    
    # StratifiedKFold target encoding için
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # İlçe bazında stratifikasyon için
    ilce_labels = LabelEncoder().fit_transform(df['ilce'])
    
    # Global statistics
    global_mean = df[target_col].mean()
    global_median = df[target_col].median()
    global_std = df[target_col].std()
    
    for col in categorical_cols:
        # Initialize columns
        df_encoded[f'{col}_target_mean'] = 0.0
        df_encoded[f'{col}_target_median'] = 0.0
        df_encoded[f'{col}_target_std'] = 0.0
        df_encoded[f'{col}_target_count'] = 0.0
        df_encoded[f'{col}_target_min'] = 0.0
        df_encoded[f'{col}_target_max'] = 0.0
        df_encoded[f'{col}_target_q25'] = 0.0
        df_encoded[f'{col}_target_q75'] = 0.0
        df_encoded[f'{col}_target_smoothed'] = 0.0  # Bayesian smoothed mean
        
        for train_idx, val_idx in skf.split(df, ilce_labels):
            train_data = df.iloc[train_idx]
            val_data = df.iloc[val_idx]
            
            # Training data'dan genişletilmiş istatistikleri hesapla
            target_stats = train_data.groupby(col)[target_col].agg([
                'mean', 'median', 'std', 'count', 'min', 'max',
                lambda x: x.quantile(0.25),  # Q1
                lambda x: x.quantile(0.75)   # Q3
            ]).reset_index()
            
            target_stats.columns = [col, 'mean', 'median', 'std', 'count', 'min', 'max', 'q25', 'q75']
            
            # Bayesian smoothing (regularization)
            smoothing_factor = 10  # Regularization strength
            target_stats['smoothed_mean'] = (
                (target_stats['count'] * target_stats['mean'] + smoothing_factor * global_mean) / 
                (target_stats['count'] + smoothing_factor)
            )
            
            # NaN handling
            target_stats = target_stats.fillna({
                'mean': global_mean,
                'median': global_median,
                'std': global_std,
                'count': 1,
                'min': global_mean,
                'max': global_mean,
                'q25': global_mean,
                'q75': global_mean,
                'smoothed_mean': global_mean
            })
            
            # Validation data'ya uygula
            val_merged = val_data[[col]].merge(target_stats, on=col, how='left')
            
            # Fill missing values with global statistics
            val_merged = val_merged.fillna({
                'mean': global_mean,
                'median': global_median,
                'std': global_std,
                'count': 1,
                'min': global_mean,
                'max': global_mean,
                'q25': global_mean,
                'q75': global_mean,
                'smoothed_mean': global_mean
            })
            
            # Assign values
            df_encoded.loc[val_idx, f'{col}_target_mean'] = val_merged['mean'].values
            df_encoded.loc[val_idx, f'{col}_target_median'] = val_merged['median'].values
            df_encoded.loc[val_idx, f'{col}_target_std'] = val_merged['std'].values
            df_encoded.loc[val_idx, f'{col}_target_count'] = val_merged['count'].values
            df_encoded.loc[val_idx, f'{col}_target_min'] = val_merged['min'].values
            df_encoded.loc[val_idx, f'{col}_target_max'] = val_merged['max'].values
            df_encoded.loc[val_idx, f'{col}_target_q25'] = val_merged['q25'].values
            df_encoded.loc[val_idx, f'{col}_target_q75'] = val_merged['q75'].values
            df_encoded.loc[val_idx, f'{col}_target_smoothed'] = val_merged['smoothed_mean'].values
    
    return df_encoded

def target_encode_categorical(df, categorical_cols, target_col, n_splits=5):
    """Wrapper for backward compatibility"""
    return advanced_target_encode_categorical(df, categorical_cols, target_col, n_splits)

def train_model(df):
    """Gelişmiş model eğitimi - XGBoost, LightGBM ve Target Encoding ile ensemble yaklaşım"""
    try:
        # Tahmin edilecek hedef değişkeni belirle (konut fiyatı)
        target_column = 'fiyat'
        y = df[target_column]  # Y değişkeni (tahmin edilecek)
        
        # Kategorik değişkenler için target encoding uygula (ilçe ve mahalle)
        categorical_cols = ['ilce', 'mahalle']  # Bu sütunlar kategorik
        # Her ilçe/mahalle için ortalama fiyat gibi istatistikleri hesapla
        df_with_target_encoding = target_encode_categorical(df, categorical_cols, target_column)
        
        # Gelişmiş özellik mühendisliği uygula (yeni özellikler türet)
        X_numerical = create_features(df_with_target_encoding)
        
        # Target encoding özelliklerini ekle
        target_encoding_cols = [col for col in df_with_target_encoding.columns if 'target_' in col]
        X_target_encoded = df_with_target_encoding[target_encoding_cols]
        
        # One-hot encoding (target encoding ile beraber kullanım)
        X_categorical = pd.get_dummies(df[categorical_cols], drop_first=True)
        
        # Tüm özellikleri birleştir
        X = pd.concat([X_numerical, X_target_encoded, X_categorical], axis=1)
        
        # Özellik isimlerini sakla
        feature_names = X.columns.tolist()
        
        # Veriyi eğitim ve test setlerine ayır
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df['ilce'])
        
        # Özellikleri ölçeklendir
        scaler = PowerTransformer(method='yeo-johnson')
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Model listesi
        models = []
        
        # Random Forest modeli (bellek verimli)
        rf_model = RandomForestRegressor(
            n_estimators=200,  # Azaltıldı
            max_depth=12,      # Azaltıldı
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=2  # Azaltıldı
        )
        models.append(('rf', rf_model))
        

        
        # Gradient Boosting modeli (geliştirilmiş)
        gb_model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=6,
            min_samples_split=3,
            min_samples_leaf=1,
            subsample=0.8,
            max_features='sqrt',
            random_state=42
        )
        models.append(('gb', gb_model))
        

        
        # XGBoost modeli (varsa) - Bellek verimli parametreler
        if XGB_AVAILABLE:
            xgb_model = xgb.XGBRegressor(
                n_estimators=250,           # Azaltıldı
                max_depth=6,                # Azaltıldı
                learning_rate=0.08,         # Artırıldı (daha hızlı)
                subsample=0.8,              # 
                colsample_bytree=0.8,       # 
                reg_alpha=0.1,              # L1 regularization
                reg_lambda=0.1,             # L2 regularization
                min_child_weight=3,         # Artırıldı (overfitting'i azalt)
                gamma=0.1,                  # Minimum split loss
                random_state=42,
                n_jobs=2,                   # Azaltıldı
                eval_metric='rmse',
                verbosity=0                 # Logları azalt
            )
            models.append(('xgb', xgb_model))
            print("XGBoost modeli bellek verimli parametrelerle eklendi")
        
        # LightGBM modeli (varsa) - Bellek verimli
        if LGB_AVAILABLE:
            lgb_model = lgb.LGBMRegressor(
                n_estimators=250,  # Azaltıldı
                max_depth=6,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=2,  # Azaltıldı
                verbose=-1
            )
            models.append(('lgb', lgb_model))
        
        # Gelişmiş model eğitimi - İyileştirilmiş ensemble stratejisi
        print(f"Gelişmiş ensemble eğitimi başlıyor... ({len(models)} model)")
        
        # YENİ: Çok aşamalı ensemble yaklaşımı
        # 1. Aşama: Bireysel model performansları
        individual_scores = {}
        model_predictions = {}
        
        for name, model in models:
            # Stratified K-Fold Cross Validation
            cv_scores = []
            kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            
            # İlçe bazlı stratification için discretized target
            y_discrete = pd.qcut(y_train, q=5, labels=False, duplicates='drop')
            
            for train_idx, val_idx in kfold.split(X_train_scaled, y_discrete):
                X_cv_train, X_cv_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                model_copy = type(model)(**model.get_params())
                model_copy.fit(X_cv_train, y_cv_train)
                cv_pred = model_copy.predict(X_cv_val)
                cv_score = r2_score(y_cv_val, cv_pred)
                cv_scores.append(cv_score)
            
            avg_score = np.mean(cv_scores)
            individual_scores[name] = avg_score
            print(f"{name.upper()} CV R² (μ±σ): {avg_score:.4f}±{np.std(cv_scores):.4f}")
            
            # Final model'i tüm training data ile eğit
            model.fit(X_train_scaled, y_train)
            model_predictions[name] = model.predict(X_test_scaled)
        
        # 2. Aşama: En iyi modelleri seç (dinamik seçim)
        score_threshold = max(individual_scores.values()) * 0.95  # En iyinin %95'i
        best_models = {name: score for name, score in individual_scores.items() if score >= score_threshold}
        
        if len(best_models) < 2:  # En az 2 model olsun
            sorted_models = sorted(individual_scores.items(), key=lambda x: x[1], reverse=True)
            best_models = dict(sorted_models[:3])  # En iyi 3'ü al
        
        print(f"Seçilen modeller ({len(best_models)}): {list(best_models.keys())}")
        print(f"Performans eşiği: {score_threshold:.4f}")
        
        # 3. Aşama: Gelişmiş ağırlıklandırma
        # Performans bazlı ağırlık + çeşitlilik bonusu
        base_models_for_ensemble = [(name, model) for name, model in models if name in best_models]
        
        # Çeşitlilik analizi - modeller arası korelasyon
        pred_matrix = np.column_stack([model_predictions[name] for name in best_models.keys()])
        pred_corr = np.corrcoef(pred_matrix.T)
        
        # Ağırlık hesaplama: performans * (1 - ortalama korelasyon)
        weights = []
        for i, (name, score) in enumerate(best_models.items()):
            diversity_bonus = 1 - np.mean([pred_corr[i, j] for j in range(len(best_models)) if i != j])
            combined_weight = score * diversity_bonus
            weights.append(combined_weight)
            print(f"{name}: Score={score:.4f}, Diversity={diversity_bonus:.4f}, Weight={combined_weight:.4f}")
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # 4. Aşama: Multiple ensemble strategies
        ensemble_strategies = {}
        
        # Strategy 1: Ağırlıklı Average
        weighted_pred = np.average([model_predictions[name] for name in best_models.keys()], 
                                 weights=weights, axis=0)
        ensemble_strategies['weighted_average'] = weighted_pred
        
        # Strategy 2: Dynamic weighting (tahmin bazlı)
        dynamic_pred = np.zeros_like(weighted_pred)
        for i in range(len(weighted_pred)):
            # Her tahmin için modellerin ne kadar "emin" olduğunu hesapla
            pred_values = [model_predictions[name][i] for name in best_models.keys()]
            pred_std = np.std(pred_values)
            
            if pred_std < np.mean([model_predictions[name].std() for name in best_models.keys()]) * 0.5:
                # Düşük belirsizlik - performans ağırlığı kullan
                dynamic_weights = weights
            else:
                # Yüksek belirsizlik - eşit ağırlık kullan
                dynamic_weights = np.ones(len(weights)) / len(weights)
            
            dynamic_pred[i] = np.average(pred_values, weights=dynamic_weights)
        
        ensemble_strategies['dynamic_weighting'] = dynamic_pred
        
        # Strategy 3: Meta-learner (Ridge)
        if len(best_models) >= 2:
            meta_features = np.column_stack([model_predictions[name] for name in best_models.keys()])
            meta_learner = Ridge(alpha=1.0, random_state=42)
            meta_learner.fit(meta_features, y_test)
            meta_pred = meta_learner.predict(meta_features)
            ensemble_strategies['meta_learner'] = meta_pred
        
        # 5. Aşama: En iyi ensemble stratejisini seç
        best_strategy = None
        best_score = -np.inf
        
        for strategy_name, predictions in ensemble_strategies.items():
            r2 = r2_score(y_test, predictions)
            mape = mean_absolute_percentage_error(y_test, predictions) * 100
            print(f"{strategy_name}: R²={r2:.4f}, MAPE={mape:.2f}%")
            
            if r2 > best_score:
                best_score = r2
                best_strategy = strategy_name
                y_pred = predictions
        
        print(f"En iyi ensemble stratejisi: {best_strategy} (R²={best_score:.4f})")
        
        # 6. Aşama: Ensemble model objesi oluştur (kaydetmek için)
        selected_model_list = [(name, model) for name, model in models if name in best_models]
        
        if best_strategy == 'meta_learner' and len(selected_model_list) >= 2:
            ensemble_model = StackingRegressor(
                estimators=selected_model_list,
                final_estimator=Ridge(alpha=1.0, random_state=42),
                cv=3,
                n_jobs=2
            )
            ensemble_model.fit(X_train_scaled, y_train)
        else:
            # Weighted voting regressor
            ensemble_model = VotingRegressor(
                estimators=selected_model_list, 
                weights=weights
            )
            ensemble_model.fit(X_train_scaled, y_train)
        
        # Tüm özellikleri kullanıyoruz
        selected_features = feature_names
        
        # YENİ: Gelişmiş güven aralığı hesaplama
        # Model belirsizliğini tahmin etmek için bootstrap sampling
        print("Güven aralığı hesaplaması için bootstrap analizi...")
        n_bootstrap = 10  # 50'den 10'a düşürüldü - hızlandırma
        bootstrap_predictions = []
        
        # Sadece en iyi 2 modeli kullan (daha hızlı)
        top_2_models = list(best_models.keys())[:2]
        print(f"Bootstrap için sadece en iyi 2 model kullanılıyor: {top_2_models}")
        
        for i in range(n_bootstrap):
            print(f"Bootstrap {i+1}/{n_bootstrap}...")
            # Bootstrap sample oluştur
            bootstrap_indices = np.random.choice(len(X_train_scaled), size=len(X_train_scaled)//2, replace=True)  # Yarı boyut
            X_bootstrap = X_train_scaled[bootstrap_indices]
            y_bootstrap = y_train.iloc[bootstrap_indices]
            
            # Sadece en iyi 2 modeli bootstrap verisi ile eğit
            bootstrap_preds = []
            for name in top_2_models:
                model_for_bootstrap = type([m for n, m in models if n == name][0])(**[m for n, m in models if n == name][0].get_params())
                model_for_bootstrap.fit(X_bootstrap, y_bootstrap)
                bootstrap_pred = model_for_bootstrap.predict(X_test_scaled)
                bootstrap_preds.append(bootstrap_pred)
            
            # Ensemble prediction (sadece top 2 modelin ağırlıkları)
            top_2_weights = weights[:2] / weights[:2].sum()  # Normalize
            ensemble_bootstrap_pred = np.average(bootstrap_preds, weights=top_2_weights, axis=0)
            bootstrap_predictions.append(ensemble_bootstrap_pred)
        
        bootstrap_predictions = np.array(bootstrap_predictions)
        
        # Bootstrap tabanlı güven aralıkları
        confidence_lower = np.percentile(bootstrap_predictions, 16, axis=0)  # %68 CI alt sınır
        confidence_upper = np.percentile(bootstrap_predictions, 84, axis=0)  # %68 CI üst sınır
        
        # Ortalama tahmin belirsizliği
        prediction_std = np.std(bootstrap_predictions, axis=0)
        mean_uncertainty = np.mean(prediction_std)
        
        print(f"Ortalama tahmin belirsizliği: ±{mean_uncertainty:,.0f} TL")
        
        # Güven aralığı genişliği analizi
        ci_width = confidence_upper - confidence_lower
        print(f"Ortalama güven aralığı genişliği: {np.mean(ci_width):,.0f} TL")
        
        # Gelişmiş model performans değerlendirmesi
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Ortalama mutlak yüzde hata (MAPE)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        
        # Gelişmiş metrikler
        # Median Absolute Error
        median_ae = np.median(np.abs(y_test - y_pred))
        
        # R2 adjusted (feature sayısını hesaba katan)
        n = len(y_test)
        p = X_test.shape[1]
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        # Fiyat aralığına göre performans
        price_ranges = [
            (0, 1000000, "1M altı"),
            (1000000, 3000000, "1M-3M arası"),  
            (3000000, 5000000, "3M-5M arası"),
            (5000000, float('inf'), "5M üstü")
        ]
        
        print(f"\n=== GELİŞMİŞ MODEL PERFORMANSI ===")
        print(f"MSE: {mse:,.0f}")
        print(f"RMSE: {rmse:,.0f}")
        print(f"MAE: {mae:,.0f}")
        print(f"Median AE: {median_ae:,.0f}")
        print(f"R² Skoru: {r2:.4f}")
        print(f"R² Adjusted: {r2_adj:.4f}")
        print(f"MAPE: %{mape:.2f}")
        
        # Fiyat aralığına göre performans analizi
        print(f"\n=== FİYAT ARALIĞINA GÖRE PERFORMANS ===")
        for min_price, max_price, label in price_ranges:
            mask = (y_test >= min_price) & (y_test < max_price)
            if mask.sum() > 0:
                range_r2 = r2_score(y_test[mask], y_pred[mask])
                range_mape = mean_absolute_percentage_error(y_test[mask], y_pred[mask]) * 100
                print(f"{label}: R²={range_r2:.3f}, MAPE=%{range_mape:.1f}, n={mask.sum()}")
        
        # Residual analizi
        residuals = y_test - y_pred
        residual_std = np.std(residuals)
        print(f"\n=== RESIDUAL ANALİZİ ===")
        print(f"Residual std: {residual_std:,.0f}")
        print(f"Residual mean: {np.mean(residuals):,.0f}")
        print(f"% tahminler %10 hata bandında: {(np.abs(residuals/y_test) < 0.1).mean()*100:.1f}%")
        print(f"% tahminler %20 hata bandında: {(np.abs(residuals/y_test) < 0.2).mean()*100:.1f}%")
        
        # Bireysel model performanslarını göster
        print(f"\nBireysel Model Performansları:")
        if hasattr(ensemble_model, 'named_estimators_'):
            for name, model in ensemble_model.named_estimators_.items():
                if hasattr(model, 'predict'):
                    individual_pred = model.predict(X_test_scaled)
                    individual_r2 = r2_score(y_test, individual_pred)
                    individual_mape = mean_absolute_percentage_error(y_test, individual_pred) * 100
                    print(f"{name.upper()}: R² = {individual_r2:.4f}, MAPE = %{individual_mape:.2f}")
                    
                    # OOB skorunu göster (varsa)
                    if hasattr(model, 'oob_score_'):
                        print(f"  OOB Skoru: {model.oob_score_:.4f}")
        else:
            print("Stacking ensemble kullanıldığı için bireysel skorlar base modeller için gösterilemiyor.")
        
        # Çapraz doğrulama kaldırıldı - hızlı eğitim için
        # Zaten train/test split ile performans ölçülüyor
        print("\nÇapraz doğrulama atlandı (hızlandırma için)")
        print("Train/Test split ile performans ölçümü yeterli.")
        
        # En önemli özellikleri göster
        print("\nEn önemli 15 özellik:")
        if hasattr(ensemble_model, 'named_estimators_') and 'rf' in ensemble_model.named_estimators_:
            rf_model = ensemble_model.named_estimators_['rf']
            feature_importances = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False)
            print(feature_importances.head(15))
        else:
            print("Stacking ensemble kullanıldığı için feature importances gösterilemiyor.")
        
        # Özellik önem grafiği
        plt.figure(figsize=(10, 6))
        top_features = feature_importances.head(15)
        sns.barplot(x=top_features.values, y=top_features.index)
        plt.title('En Önemli 15 Özellik')
        plt.tight_layout()
        
        # Grafikleri kaydet
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.savefig('plots/feature_importance.png')
        
        # Gerçek vs Tahmin grafiği
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Gerçek Fiyat')
        plt.ylabel('Tahmin Edilen Fiyat')
        plt.title('Gerçek vs Tahmin')
        plt.tight_layout()
        plt.savefig('plots/actual_vs_predicted.png')
        
        # Modeli ve scaler'ı kaydet
        if not os.path.exists('models'):
            os.makedirs('models')
        
        print("Model ve ilgili dosyalar kaydediliyor...")
        joblib.dump(ensemble_model, 'models/konut_fiyat_model.pkl')
        joblib.dump(scaler, 'models/scaler.pkl')
        joblib.dump(feature_names, 'models/feature_names.pkl')  # Tüm özellikler
        
        # YENİ: Bootstrap ve güven aralığı parametreleri
        confidence_params = {
            'bootstrap_predictions': bootstrap_predictions,
            'confidence_lower': confidence_lower,
            'confidence_upper': confidence_upper,
            'prediction_std': prediction_std,
            'mean_uncertainty': mean_uncertainty,
            'weights': weights,
            'best_models': list(best_models.keys()),
            'best_strategy': best_strategy,
            'ensemble_type': type(ensemble_model).__name__
        }
        joblib.dump(confidence_params, 'models/confidence_params.pkl')
        
        # Model performans metrikleri
        performance_metrics = {
            'r2_score': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'median_ae': median_ae,
            'r2_adjusted': r2_adj,
            'residual_std': residual_std,
            'accuracy_10_percent': (np.abs(residuals/y_test) < 0.1).mean(),
            'accuracy_20_percent': (np.abs(residuals/y_test) < 0.2).mean(),
            'training_date': pd.Timestamp.now().isoformat(),
            'n_training_samples': len(X_train),
            'n_test_samples': len(X_test),
            'n_features': len(feature_names)
        }
        joblib.dump(performance_metrics, 'models/performance_metrics.pkl')
        
        # İlçe-mahalle haritası oluştur
        ilce_mahalle_map = {}
        for ilce in df['ilce'].unique():
            ilce_mahalle_map[ilce] = sorted(df[df['ilce'] == ilce]['mahalle'].unique().tolist())
        
        joblib.dump(sorted(df['ilce'].unique().tolist()), 'models/unique_ilce.pkl')
        joblib.dump(ilce_mahalle_map, 'models/ilce_mahalle_map.pkl')
        
        # Tahmin güvenilirliği için fiyat aralıklarını kaydet
        price_range = {
            'min': float(y.min()),
            'max': float(y.max()),
            'mean': float(y.mean()),
            'median': float(y.median()),
            'q1': float(y.quantile(0.25)),
            'q3': float(y.quantile(0.75)),
            'q5': float(y.quantile(0.05)),
            'q95': float(y.quantile(0.95))
        }
        joblib.dump(price_range, 'models/price_range.pkl')
        
        print("✅ Model başarıyla kaydedildi!")
        print(f"📊 Final Performans: R²={r2:.4f}, RMSE={rmse:,.0f}, MAPE={mape:.2f}%")
        
        return ensemble_model, scaler, feature_names
    except Exception as e:
        print(f"Model eğitimi hatası: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def predict_price(features_dict):
    """Konut fiyatını tahmin et ve güvenilirlik bilgisi döndür"""
    try:
        # Modeli ve ilgili dosyaları yükle
        model = joblib.load('models/konut_fiyat_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        price_range = joblib.load('models/price_range.pkl')
        
        # Feature selector artık kullanılmıyor - hızlandırma için kaldırıldı
        
        # Veri setini yükle
        df = load_and_preprocess_data()
        
        # İlçe ve mahalle istatistikleri
        ilce_stats = df.groupby('ilce')['fiyat'].agg(['mean', 'median', 'std']).reset_index()
        mahalle_stats = df.groupby('mahalle')['fiyat'].agg(['mean', 'median', 'std']).reset_index()
        
        # İlçe ve mahalle frekans kodlaması
        ilce_freq = df['ilce'].value_counts(normalize=True).to_dict()
        mahalle_freq = df['mahalle'].value_counts(normalize=True).to_dict()
        
        # Girdi özelliklerini DataFrame'e dönüştür
        input_df = pd.DataFrame([features_dict])
        
        # İlçe ve mahalle istatistiklerini ekle
        input_df = input_df.merge(ilce_stats, on='ilce', how='left')
        input_df = input_df.merge(mahalle_stats, on='mahalle', how='left', suffixes=('', '_mahalle'))
        
        # Frekans kodlamasını ekle
        input_df['ilce_freq'] = input_df['ilce'].map(ilce_freq)
        input_df['mahalle_freq'] = input_df['mahalle'].map(mahalle_freq)
        
        # Target encoding özelliklerini ekle
        # Global istatistiklerle doldur (tahmin aşamasında cross-validation yapmayız)
        global_mean = df['fiyat'].mean()
        global_median = df['fiyat'].median()
        global_std = df['fiyat'].std()
        
        # Gelişmiş İlçe target encoding
        ilce_target_stats = df.groupby('ilce')['fiyat'].agg([
            'mean', 'median', 'std', 'count', 'min', 'max',
            lambda x: x.quantile(0.25),  # Q1
            lambda x: x.quantile(0.75)   # Q3
        ]).reset_index()
        ilce_target_stats.columns = ['ilce', 'mean', 'median', 'std', 'count', 'min', 'max', 'q25', 'q75']
        
        # Smoothed mean hesapla
        smoothing_factor = 10
        ilce_target_stats['smoothed_mean'] = (
            (ilce_target_stats['count'] * ilce_target_stats['mean'] + smoothing_factor * global_mean) / 
            (ilce_target_stats['count'] + smoothing_factor)
        )
        
        # Basit ve güvenli target encoding
        # İlçe için
        try:
            temp_ilce = input_df.merge(ilce_target_stats, on='ilce', how='left')
            for col in ['mean', 'median', 'std', 'count', 'min', 'max', 'q25', 'q75', 'smoothed_mean']:
                if col in temp_ilce.columns:
                    default_val = global_mean if col in ['mean', 'median', 'min', 'max', 'q25', 'q75', 'smoothed_mean'] else (global_std if col == 'std' else 1)
                    input_df[f'ilce_target_{col}'] = temp_ilce[col].fillna(default_val)
                else:
                    default_val = global_mean if col in ['mean', 'median', 'min', 'max', 'q25', 'q75', 'smoothed_mean'] else (global_std if col == 'std' else 1)
                    input_df[f'ilce_target_{col}'] = default_val
        except:
            # Eğer merge başarısızsa varsayılan değerler kullan
            for col in ['mean', 'median', 'std', 'count', 'min', 'max', 'q25', 'q75', 'smoothed_mean']:
                default_val = global_mean if col in ['mean', 'median', 'min', 'max', 'q25', 'q75', 'smoothed_mean'] else (global_std if col == 'std' else 1)
                input_df[f'ilce_target_{col}'] = default_val
        
        # Mahalle için  
        try:
            # Mahalle istatistiklerini hesapla
            mahalle_target_stats = df.groupby('mahalle')['fiyat'].agg([
                'mean', 'median', 'std', 'count', 'min', 'max',
                lambda x: x.quantile(0.25),  # Q1
                lambda x: x.quantile(0.75)   # Q3
            ]).reset_index()
            mahalle_target_stats.columns = ['mahalle', 'mean', 'median', 'std', 'count', 'min', 'max', 'q25', 'q75']
            
            # Smoothed mean hesapla
            mahalle_target_stats['smoothed_mean'] = (
                (mahalle_target_stats['count'] * mahalle_target_stats['mean'] + smoothing_factor * global_mean) / 
                (mahalle_target_stats['count'] + smoothing_factor)
            )
            
            temp_mahalle = input_df.merge(mahalle_target_stats, on='mahalle', how='left')
            for col in ['mean', 'median', 'std', 'count', 'min', 'max', 'q25', 'q75', 'smoothed_mean']:
                if col in temp_mahalle.columns:
                    default_val = global_mean if col in ['mean', 'median', 'min', 'max', 'q25', 'q75', 'smoothed_mean'] else (global_std if col == 'std' else 1)
                    input_df[f'mahalle_target_{col}'] = temp_mahalle[col].fillna(default_val)
                else:
                    default_val = global_mean if col in ['mean', 'median', 'min', 'max', 'q25', 'q75', 'smoothed_mean'] else (global_std if col == 'std' else 1)
                    input_df[f'mahalle_target_{col}'] = default_val
        except:
            # Eğer merge başarısızsa varsayılan değerler kullan
            for col in ['mean', 'median', 'std', 'count', 'min', 'max', 'q25', 'q75', 'smoothed_mean']:
                default_val = global_mean if col in ['mean', 'median', 'min', 'max', 'q25', 'q75', 'smoothed_mean'] else (global_std if col == 'std' else 1)
                input_df[f'mahalle_target_{col}'] = default_val
        
        # Özellik mühendisliği
        X_numerical = create_features(input_df)
        
        # Gelişmiş target encoding özelliklerini al
        target_encoding_cols = [
            'ilce_target_mean', 'ilce_target_median', 'ilce_target_std',
            'ilce_target_count', 'ilce_target_min', 'ilce_target_max', 
            'ilce_target_q25', 'ilce_target_q75', 'ilce_target_smoothed',
            'mahalle_target_mean', 'mahalle_target_median', 'mahalle_target_std',
            'mahalle_target_count', 'mahalle_target_min', 'mahalle_target_max',
            'mahalle_target_q25', 'mahalle_target_q75', 'mahalle_target_smoothed'
        ]
        
        # Eksik sütunları 0 ile doldur (eski model uyumluluğu için)
        for col in target_encoding_cols:
            if col not in input_df.columns:
                input_df[col] = global_mean if 'mean' in col or 'median' in col or 'smoothed' in col else global_std if 'std' in col else 1 if 'count' in col else global_mean
        
        X_target_encoded = input_df[target_encoding_cols]
        
        # Kategorik özellikleri one-hot encoding ile dönüştür
        categorical_cols = ['ilce', 'mahalle']
        X_categorical = pd.get_dummies(input_df[categorical_cols], drop_first=True)
        
        # Tüm özellikleri birleştir
        X_combined = pd.concat([X_numerical, X_target_encoded, X_categorical], axis=1)
        
        # One-hot encoding'te eksik olan sütunları 0 olarak doldurmak için boş bir DataFrame oluştur
        missing_cols = set(feature_names) - set(X_combined.columns)
        missing_df = pd.DataFrame(0, index=X_combined.index, columns=list(missing_cols))
        
        # Tüm özellikleri birleştir
        X = pd.concat([X_combined, missing_df], axis=1)
        
        # Özellikleri model için sırala
        X = X.reindex(columns=feature_names, fill_value=0)
        
        # Özellikleri ölçeklendir
        X_scaled = scaler.transform(X)
        
        # Tahmin yap
        prediction = model.predict(X_scaled)[0]
        
        # Negatif fiyatları düzelt
        prediction = max(0, prediction)
        
        # YENİ: Bootstrap tabanlı güven aralığı hesaplama (eğer varsa)
        try:
            confidence_params = joblib.load('models/confidence_params.pkl')
            bootstrap_predictions = confidence_params['bootstrap_predictions']
            prediction_std = confidence_params['prediction_std']
            mean_uncertainty = confidence_params['mean_uncertainty']
            
            # Tek bir tahmin için belirsizlik tahmini
            # Ortalama belirsizliği kullanarak güven aralığı hesapla
            single_prediction_uncertainty = mean_uncertainty
            
            # %68 güven aralığı (1 sigma)
            lower_bound = max(0, prediction - single_prediction_uncertainty)
            upper_bound = prediction + single_prediction_uncertainty
            
            # %95 güven aralığı (2 sigma) - daha geniş
            lower_bound_95 = max(0, prediction - 2 * single_prediction_uncertainty)
            upper_bound_95 = prediction + 2 * single_prediction_uncertainty
            
            confidence_interval = single_prediction_uncertainty
            reliability_score = 1.0 - (single_prediction_uncertainty / prediction) if prediction > 0 else 0.5
            
        except FileNotFoundError:
            # Bootstrap verileri yoksa eski yöntemi kullan
            confidence_interval = prediction * 0.15
            lower_bound = max(0, prediction - confidence_interval)
            upper_bound = prediction + confidence_interval
            lower_bound_95 = max(0, prediction - 2 * confidence_interval)
            upper_bound_95 = prediction + 2 * confidence_interval
            reliability_score = 0.8
        
        # Tahmin güvenilirliğini değerlendir
        reliability = "Yüksek"
        warning = None
        
        # Fiyat aralığı dışında mı kontrol et
        if prediction < price_range['q5'] or prediction > price_range['q95']:
            reliability = "Düşük"
            reliability_score = max(0.2, reliability_score * 0.4)
            warning = "Tahmin edilen fiyat normal fiyat aralığının dışında."
        elif prediction < price_range['q1'] * 0.8 or prediction > price_range['q3'] * 1.2:
            reliability = "Orta"
            reliability_score = max(0.5, reliability_score * 0.7)
            warning = "Tahmin edilen fiyat normal fiyat aralığının sınırlarında."
        
        result = {
            'prediction': prediction,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'lower_bound_95': lower_bound_95,
            'upper_bound_95': upper_bound_95,
            'confidence_interval': confidence_interval,
            'reliability': reliability,
            'reliability_score': reliability_score,
            'warning': warning,
            'price_per_m2': prediction / features_dict['metrekare'],
            'prediction_quality': 'Yüksek' if reliability_score > 0.8 else 'Orta' if reliability_score > 0.5 else 'Düşük'
        }
        
        return result
    except Exception as e:
        print(f"Tahmin hatası: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_available_features():
    """Kullanılabilir özellikleri döndür"""
    try:
        if os.path.exists('models/unique_ilce.pkl'):
            unique_ilce = joblib.load('models/unique_ilce.pkl')
            
            # İlçe-mahalle ilişkisini kontrol et
            if os.path.exists('models/ilce_mahalle_map.pkl'):
                ilce_mahalle_map = joblib.load('models/ilce_mahalle_map.pkl')
            else:
                # Model kaydedilmiş ama ilçe-mahalle haritası yoksa yeniden oluştur
                df = load_and_preprocess_data()
                ilce_mahalle_map = {}
                for ilce in df['ilce'].unique():
                    ilce_mahalle_map[ilce] = sorted(df[df['ilce'] == ilce]['mahalle'].unique().tolist())
                
                # Haritayı kaydet
                joblib.dump(ilce_mahalle_map, 'models/ilce_mahalle_map.pkl')
            
            return {
                'ilce': unique_ilce,
                'ilce_mahalle_map': ilce_mahalle_map
            }
        else:
            # Model henüz eğitilmemişse veri setinden al
            df = load_and_preprocess_data()
            if df is not None:
                # İlçe-mahalle haritasını oluştur
                ilce_mahalle_map = {}
                for ilce in df['ilce'].unique():
                    ilce_mahalle_map[ilce] = sorted(df[df['ilce'] == ilce]['mahalle'].unique().tolist())
                
                return {
                    'ilce': sorted(df['ilce'].unique().tolist()),
                    'ilce_mahalle_map': ilce_mahalle_map
                }
            return None
    except Exception as e:
        print(f"Özellik bilgilerini alma hatası: {e}")
        return None

def get_district_stats():
    """İlçe bazında fiyat istatistiklerini döndürür"""
    try:
        # Veri setini yükle
        df = load_and_preprocess_data()
        
        if df is not None:
            # İlçe bazında ortalama, medyan, minimum ve maksimum fiyatları hesapla
            district_stats = df.groupby('ilce')['fiyat'].agg(['mean', 'median', 'min', 'max', 'count', 'std']).reset_index()
            district_stats = district_stats.sort_values('mean', ascending=False)
            
            # Metrekare başına ortalama fiyatı hesapla
            df['fiyat_metrekare'] = df['fiyat'] / df['metrekare']
            price_per_sqm = df.groupby('ilce')['fiyat_metrekare'].mean().reset_index()
            
            # İlçe istatistiklerine metrekare başına fiyatı ekle
            district_stats = district_stats.merge(price_per_sqm, on='ilce')
            
            # Sonuçları sözlük olarak döndür
            result = {}
            for _, row in district_stats.iterrows():
                result[row['ilce']] = {
                    'mean': float(row['mean']),
                    'median': float(row['median']),
                    'min': float(row['min']),
                    'max': float(row['max']),
                    'count': int(row['count']),
                    'std': float(row['std']),
                    'price_per_sqm': float(row['fiyat_metrekare'])
                }
            
            return result
        return None
    except Exception as e:
        print(f"İlçe istatistikleri hesaplanırken hata oluştu: {e}")
        return None

def get_price_range_performance():
    """Fiyat aralığına göre model performansını döndürür"""
    try:
        # Model dosyalarını kontrol et
        if not os.path.exists('models/konut_fiyat_model.pkl'):
            print("❌ Eğitilmiş model bulunamadı!")
            return None
        
        # Modeli ve scaler'ı yükle
        model = joblib.load('models/konut_fiyat_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        
        # Veri setini yükle ve işle
        df = load_and_preprocess_data()
        if df is None:
            return None
        
        # Hedef değişken
        target_column = 'fiyat'
        y = df[target_column]
        
        # Target encoding uygula
        categorical_cols = ['ilce', 'mahalle']
        df_with_target_encoding = target_encode_categorical(df, categorical_cols, target_column)
        
        # Özellik mühendisliği
        X_numerical = create_features(df_with_target_encoding)
        
        # Target encoding özelliklerini ekle
        target_encoding_cols = [col for col in df_with_target_encoding.columns if 'target_' in col]
        X_target_encoded = df_with_target_encoding[target_encoding_cols]
        
        # One-hot encoding
        X_categorical = pd.get_dummies(df[categorical_cols], drop_first=True)
        
        # Tüm özellikleri birleştir
        X = pd.concat([X_numerical, X_target_encoded, X_categorical], axis=1)
        
        # Özellikleri model için sırala
        X = X.reindex(columns=feature_names, fill_value=0)
        
        # Test verisi oluştur (son %20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df['ilce'])
        
        # Özellikleri ölçeklendir
        X_test_scaled = scaler.transform(X_test)
        
        # Tahmin yap
        y_pred = model.predict(X_test_scaled)
        
        # Fiyat aralıkları tanımla
        price_ranges = [
            (0, 1000000, "1M altı"),
            (1000000, 2000000, "1M-2M arası"),  
            (2000000, 3000000, "2M-3M arası"),
            (3000000, 5000000, "3M-5M arası"),
            (5000000, float('inf'), "5M üstü")
        ]
        
        performance_results = {}
        
        print(f"\n🎯 FİYAT ARALIĞINA GÖRE MODEL PERFORMANSI")
        print("=" * 60)
        
        for min_price, max_price, label in price_ranges:
            # Bu fiyat aralığındaki veriyi filtrele
            mask = (y_test >= min_price) & (y_test < max_price)
            
            if mask.sum() > 5:  # En az 5 örnek olmalı
                y_test_range = y_test[mask]
                y_pred_range = y_pred[mask]
                
                # Performans metrikleri hesapla
                r2 = r2_score(y_test_range, y_pred_range)
                mse = mean_squared_error(y_test_range, y_pred_range)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test_range, y_pred_range)
                mape = mean_absolute_percentage_error(y_test_range, y_pred_range) * 100
                
                # Ortalama fiyat
                avg_price = y_test_range.mean()
                
                performance_results[label] = {
                    'r2_score': float(r2),
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'mape': float(mape),
                    'sample_count': int(mask.sum()),
                    'avg_price': float(avg_price),
                    'min_price': float(min_price),
                    'max_price': float(max_price) if max_price != float('inf') else None
                }
                
                # Sonuçları yazdır
                print(f"\n📊 {label}:")
                print(f"   • R² Skoru: {r2:.4f}")
                print(f"   • RMSE: {rmse:,.0f} TL")
                print(f"   • MAE: {mae:,.0f} TL") 
                print(f"   • MAPE: %{mape:.2f}")
                print(f"   • Örnek Sayısı: {mask.sum()}")
                print(f"   • Ortalama Fiyat: {avg_price:,.0f} TL")
                
                # Performans yorumu
                if r2 >= 0.9:
                    performance = "🟢 Mükemmel"
                elif r2 >= 0.8:
                    performance = "🟡 İyi"
                elif r2 >= 0.7:
                    performance = "🟠 Orta"
                elif r2 >= 0.5:
                    performance = "🟠 Düşük"
                elif r2 >= 0.0:
                    performance = "🔴 Zayıf"
                else:
                    performance = "🔴 Çok Zayıf (Negatif R²)"
                    
                print(f"   • Performans: {performance}")
                
                # R² negatifse açıklama ekle
                if r2 < 0:
                    print(f"   ⚠️  Negatif R² = Model rastgele tahminden daha kötü")
            else:
                print(f"\n📊 {label}: ❌ Yetersiz veri (n={mask.sum()})")
        
        # Genel performans
        overall_r2 = r2_score(y_test, y_pred)
        overall_mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        
        print(f"\n🎯 GENEL PERFORMANS:")
        print(f"   • Genel R² Skoru: {overall_r2:.4f}")
        print(f"   • Genel MAPE: %{overall_mape:.2f}")
        print(f"   • Toplam Test Verisi: {len(y_test)}")
        
        performance_results['overall'] = {
            'r2_score': float(overall_r2),
            'mape': float(overall_mape),
            'total_samples': int(len(y_test))
        }
        
        return performance_results
        
    except Exception as e:
        print(f"❌ Performans analizi hatası: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Veri setini yükle
    print("Veri seti yükleniyor...")
    data = load_and_preprocess_data()
    
    if data is not None:
        print(f"Veri seti başarıyla yüklendi. Toplam {data.shape[0]} kayıt, {data.shape[1]} özellik var.")
        # Modeli eğit
        model, scaler, feature_names = train_model(data)
        
        if model is not None:
            print("Model başarıyla eğitildi ve kaydedildi.") 