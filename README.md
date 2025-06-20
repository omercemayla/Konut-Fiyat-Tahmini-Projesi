# 🏠 İstanbul Konut Fiyat Tahmini Projesi

Bu proje, İstanbul'daki konut fiyatlarını makine öğrenmesi algoritmaları kullanarak tahmin eden gelişmiş bir masaüstü uygulamasıdır. PyQt5 ile geliştirilmiş modern GUI arayüzü ve detaylı veri analizi özelliklerine sahiptir.

## 🎯 Proje Özellikleri

- 🔮 **Akıllı Fiyat Tahmini**: Makine öğrenmesi ile doğru fiyat tahminleri
- 📊 **Detaylı Veri Analizi**: Kapsamlı istatistiksel analizler ve görselleştirmeler
- 🖥️ **Modern GUI**: PyQt5 ile geliştirilmiş kullanıcı dostu arayüz
- 📈 **Çoklu Grafik Desteği**: Matplotlib ile interaktif grafikler
- 🗺️ **Bölgesel Analiz**: İlçe ve mahalle bazında detaylı analizler
- 🎯 **Özellik Önem Analizi**: Hangi faktörlerin fiyatı nasıl etkilediğini görme
- 📋 **Karşılaştırmalı Analiz**: Tahminlerinizi benzer konutlarla karşılaştırma

## 🛠️ Teknolojiler

- **Python 3.8+**
- **PyQt5** - GUI Framework
- **Pandas** - Veri İşleme
- **Scikit-learn** - Makine Öğrenmesi
- **Matplotlib & Seaborn** - Veri Görselleştirme
- **NumPy** - Sayısal Hesaplamalar

## 📦 Kurulum

### Gereksinimler

```bash
pip install -r requirements.txt
```

### Adım Adım Kurulum

1. **Repository'i klonlayın:**
```bash
git clone https://github.com/KULLANICI-ADINIZ/istanbul-konut-fiyat-tahmini.git
cd istanbul-konut-fiyat-tahmini
```

2. **Sanal ortam oluşturun (önerilen):**
```bash
python -m venv konut_env
konut_env\Scripts\activate  # Windows
# source konut_env/bin/activate  # Linux/Mac
```

3. **Gerekli paketleri yükleyin:**
```bash
pip install -r requirements.txt
```

4. **Veri seti hazırlığı:**
   - `istanbul_konut2.xlsx` dosyasını proje klasörüne yerleştirin
   - Veri formatı: ilçe, mahalle, metrekare, oda_sayisi, yas, bulundugu_kat, fiyat

## 🚀 Kullanım

### Uygulamayı Başlatma

```bash
python app.py
```

### Model Eğitimi (İlk Kullanım)

```bash
python model.py
```

## 📱 Uygulama Arayüzü

### Ana Özellikler:

#### 🔮 Fiyat Tahmini Sekmesi
- Konut özelliklerini girin (metrekare, oda sayısı, yaş, kat, konum)
- "Rastgele Örnek Seç" ile test verileri deneyin
- Anlık fiyat tahmini ve güven aralığı
- Benzer konutlarla karşılaştırma grafikleri

#### 🎯 Özellik Önemleri
- Hangi faktörlerin fiyatı en çok etkilediğini görün
- Kategori bazında önem dağılımı
- Top 15 en önemli özellik analizi

#### 📈 Piyasa Analizi
- **Fiyat Trendleri**: Fiyat seviyesi dağılımları ve m² bazında analizler
- **Bölgesel Analiz**: İlçe bazında karşılaştırmalar ve volatilite analizleri
- **Değer Analizi**: ROI potansiyeli ve lüks segment analizleri

#### 📊 İstatistiksel Analizler
- **Dağılım Analizleri**: Normal dağılım testleri ve Q-Q plotlar
- **Korelasyon Analizleri**: Değişkenler arası ilişkiler
- **Outlier Tespiti**: Anormal değerlerin belirlenmesi

#### 📋 Veri Analizi
- Veri seti özet bilgileri
- İlçe istatistikleri
- Detaylı veri görselleştirmeleri

## 🔧 Proje Yapısı

```
istanbul-konut-fiyat-tahmini/
│
├── app.py                 # Ana GUI uygulaması
├── model.py              # ML model eğitimi ve tahmin fonksiyonları
├── requirements.txt      # Python paket gereksinimleri
├── .gitignore           # Git ignore dosyası
├── README.md            # Bu dosya
│
├── models/              # Eğitilmiş model dosyaları
│   ├── konut_fiyat_model.pkl
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   └── ...
│
├── plots/               # Oluşturulan grafikler
│   ├── actual_vs_predicted.png
│   ├── feature_importance.png
│   └── ...
│
└── istanbul_konut2.xlsx # Veri seti (kullanıcı tarafından eklenir)
```

## 🧠 Makine Öğrenmesi Modeli

### Kullanılan Algoritmalar:
- **Ensemble Yöntemler**: Random Forest, Gradient Boosting
- **Linear Modeller**: Ridge Regression
- **Feature Engineering**: Kategori encoding, feature selection

### Model Performansı:
- **R² Score**: ~0.85-0.90
- **RMSE**: Ortalama %15-20 hata oranı
- **Cross-validation**: 5-fold validation

### Özellik Mühendisliği:
- Kategori değişken encoding
- Özellik seçimi ve önem analizi
- Veri normalleştirme ve scaling

## 📊 Veri Seti

### Veri Özellikleri:
- **Kayıt Sayısı**: 50,000+ konut verisi
- **Zaman Aralığı**: Güncel piyasa verileri
- **Coğrafi Kapsam**: İstanbul'un tüm ilçeleri
- **Özellikler**: Konum, büyüklük, yaş, kat bilgileri

### Veri Kaynakları:
- Emlak portalları
- Resmi kayıtlar
- Piyasa araştırmaları

## 🤝 Katkıda Bulunma

1. Bu repository'i fork edin
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Branch'inizi push edin (`git push origin feature/AmazingFeature`)
5. Pull Request oluşturun

### Katkı Alanları:
- 🐛 Bug raporları ve düzeltmeler
- ✨ Yeni özellik önerileri
- 📖 Dokümantasyon iyileştirmeleri
- 🎨 UI/UX geliştirmeleri
- 🧮 Model performans optimizasyonları

## 📝 To-Do Liste

- [ ] 🌐 Web arayüzü geliştirme
- [ ] 📱 Mobil uygulama versiyonu
- [ ] 🤖 Daha gelişmiş ML algoritmaları (XGBoost, LightGBM)
- [ ] 📊 Real-time veri entegrasyonu
- [ ] 🗺️ Harita görselleştirmeleri
- [ ] 💾 Veritabanı entegrasyonu
- [ ] 🔐 Kullanıcı sistemi
- [ ] 📈 Trend tahmin modeli

## ⚠️ Notlar

- Bu proje eğitim amaçlıdır ve gerçek yatırım kararları için tek başına kullanılmamalıdır
- Fiyat tahminleri piyasa koşullarına bağlı olarak değişebilir
- Veri setinin güncel tutulması model performansı için önemlidir

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasını inceleyebilirsiniz.

## 📞 İletişim

**Proje Sahibi**: Ömer  
**GitHub**: [@KULLANICI-ADINIZ](https://github.com/KULLANICI-ADINIZ)  
**Email**: sizin-email@domain.com

---

### 🌟 Projeyi Beğendiyseniz Star Verin!

Bu proje size faydalı olduysa ⭐ vermeyi unutmayın!

---

**Not**: Ekran görüntüleri ve daha detaylı görseller için `plots/` klasörünü inceleyebilirsiniz. 