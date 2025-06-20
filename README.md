# 🏠 İstanbul Konut Fiyat Tahmini Projesi - Proje Sunumu

## 📋 **Proje Açıklaması**

İstanbul'da ev alacaktım, fiyatlar çok karışık görünüyordu. Emlak sitelerinde aynı özellikte evler farklı fiyatlarda satılıyordu. Bu yüzden makine öğrenmesi ile gerçekçi fiyat hesaplama sistemi yaptım.

---

## 🔧 **VERİ TEMİZLEME İŞLEMLERİM**

### **İlk Aşama - Veriyi Yükledim:**
```python
df = pd.read_excel('istanbul_konut2.xlsx')
```
*"Excel dosyasını açtım, sütun isimlerini düzenli hale getirdim."*

### **İkinci Aşama - Saçma Verileri Temizledim:**
```python
# Çok pahalı veya çok ucuz evleri çıkardım
df = df[(df['fiyat'] >= 100000) & (df['fiyat'] <= 50000000)]
# Çok küçük veya çok büyük evleri çıkardım  
df = df[(df['metrekare'] >= 30) & (df['metrekare'] <= 400)]
```

**Neden böyle yaptım:**
*"Veri setinde 10 TL'ye ev, 100 milyon TL'ye ev gibi mantıksız değerler vardı. Bunları temizledim."*

### **Üçüncü Aşama - İlçe Bazında Temizlik:**
```python
# Her ilçeyi ayrı ayrı kontrol ettim
for ilce in df['ilce'].unique():
    ilce_df = df[df['ilce'] == ilce].copy()
    # Bu ilçede çok pahalı olan evleri çıkardım
```

**Mantığım:**
*"Beşiktaş'ta 5 milyon normal, ama Esenyurt'ta 5 milyon saçma. O yüzden her ilçeyi kendi içinde temizledim."*

---

## 🛠️ **YENİ ÖZELLİKLER OLUŞTURDUM**

### **Basit Özellikler:**
```python
# Metrekare başına oda sayısı
metrekare_oda_orani = metrekare / oda_sayisi

# Bina ne kadar yeni (0=çok eski, 1=çok yeni)
yenilik_skoru = 1 / (yas + 1)

# Kat ne kadar iyi
if kat < 0: kat_skoru = -0.1  # Bodrum kötü
elif kat <= 3: kat_skoru = 0.1  # Alt katlar iyi
elif kat <= 7: kat_skoru = 0.15  # Orta katlar en iyi
```

**Neden bunları yaptım:**
*"Sadece 'metrekare' değil, 'metrekare başına kaç oda düşüyor' daha önemli. Yeni bina daha değerli. 4-7. katlar genelde en çok tercih ediliyor."*

### **İlçe ve Mahalle Puanları:**
```python
# Her ilçenin ortalama fiyatını hesapladım
ilce_ortalama_fiyat = df.groupby('ilce')['fiyat'].mean()

# Her mahallenin o ilçeye göre ne kadar pahalı olduğunu hesapladım
mahalle_premium = mahalle_ortalama / ilce_ortalama
```

**Açıklamam:**
*"Nişantaşı'nda olmak bile fiyatı artırıyor. Ama Nişantaşı'nın içinde de iyi mahalle-kötü mahalle var."*

---

## 🤖 **MODEL SEÇİMİM**

### **Neden Birden Fazla Model Kullandım:**
```python
# 4 farklı model eğittim:
model1 = RandomForest()      # Kararlı ve güvenilir
model2 = GradientBoosting()  # Hatalarından öğrenen
model3 = XGBoost()          # Hızlı ve başarılı
model4 = LightGBM()         # Hafif ve etkili

# Sonra hepsini birleştirdim
final_model = (model1 + model2 + model3 + model4) / 4
```

**Mantığım:**
*"Bir doktor değil, 4 doktora danışıp ortalama alıyorsun. Daha güvenilir oluyor."*

### **Hangi Modeli Ne Kadar Dinledim:**
```python
# En başarılı modellere daha çok ağırlık verdim
if model_basari > %95: agirlik = yüksek
else: agirlik = düşük

# Birbirine benzer modelleri cezalandırdım
if model1_tahmini ≈ model2_tahmini: agirlik = azalt
```

**Açıklamam:**
*"4 doktor aynı şeyi söylüyorsa fazla bilgi yok. Farklı görüş veren doktorları da dinlemek lazım."*

---

## 🎯 **GÜVENİLİRLİK ÖLÇÜMÜ**

### **Tahminin Ne Kadar Doğru Olduğunu Nasıl Buldum:**
```python
# Veriyi 10 kez karıştırıp model eğittim
for i in range(10):
    veri_karistir = rastgele_sec(veri, yarim_boyut)
    model_egit(veri_karistir)
    tahmin_yap = model.predict(test_veri)
    tahminler.append(tahmin_yap)

# 10 tahminin ortalamasını ve sapmasını hesapladım
ortalama_tahmin = tahminler.mean()
belirsizlik = tahminler.std()
```

**Basit Açıklama:**
*"Aynı soruyu 10 kez farklı şekilde sordum. Hep aynı cevabı verirse güvenilir, farklı cevaplar verirse şüpheli."*

---

## 🎨 **KULLANICI ARAYÜZÜ**

### **Neden 5 Sekme Yaptım:**
1. **Ana Tahmin Sekmesi:** Kullanıcı bilgi girer, fiyat öğrenir
2. **Özellik Önemleri:** Hangi faktörler fiyatı etkiliyor
3. **Piyasa Analizi:** İlçe karşılaştırmaları, trendler
4. **İstatistik:** Grafikler, analizler
5. **Veri:** Ham veriye bakma

**Düşüncem:**
*"Herkes farklı şey merak ediyor. Kimisi sadece fiyat öğrenmek istiyor, kimisi detaylı analiz istiyor."*

### **Rastgele Örnek Özelliği:**
```python
def rastgele_ornek_sec():
    rastgele_ev = veri.sample(1)
    formu_doldur(rastgele_ev.bilgileri)
    gercek_fiyat = rastgele_ev.fiyat
    "Bu evin gerçek fiyatı X TL, hadi bakalım ne tahmin edeceksin"
```

**Neden yaptım:**
*"İnsanlar 'acaba doğru tahmin ediyor mu' diye merak ediyor. Gerçek bir evi gösterip tahmin ettiriyorum."*

---

## 📊 **KARŞILAŞTIRMA GRAFİKLERİ**

### **Benzer Evlerle Karşılaştırma:**
```python
# Senin girdiğin eve benzer evleri buluyorum
benzer_evler = veri[
    (ilce == senin_ilce) &              # Aynı ilçe
    (metrekare ± 20 == senin_metrekare) & # Benzer büyüklük  
    (oda_sayisi ± 1 == senin_oda)       # Benzer oda sayısı
]

# Bu evlerin fiyatlarını grafik yapıyorum
# Senin tahminin bu aralıkta mı diye bakıyorum
```

**Amacım:**
*"'2 milyon TL' diyorum ama sen 'çok mu pahalı çok mu ucuz' bilmiyorsun. Benzer evleri gösterince anlıyorsun."*

### **İlçe Ortalamasıyla Karşılaştırma:**
```python
ilce_ortalama = veri[veri.ilce == senin_ilce].fiyat.mean()
senin_tahmin = model.predict(senin_ev)

if senin_tahmin > ilce_ortalama:
    print("Senin evin ilçe ortalamasından pahalı")
else:
    print("Senin evin ilçe ortalamasından ucuz")
```

---

## 📈 **PERFORMANS SONUÇLARIM**

### **Modelin Başarı Oranı:**
- **R² Skoru:** 0.85-0.90 (100 üzerinden 85-90 puan)
- **Ortalama Hata:** %15-20 (5'te 1 oranında yanılıyor)
- **Parasal Hata:** ±200-300 bin TL

**Basit Açıklama:**
*"10 evden 8-9 tanesini doğru tahmin ediyorum. Yanıldığım zaman da çok uzak değil, yaklaşık %20 hata yapıyorum."*

### **Hangi Fiyat Aralığında Daha Başarılı:**
- **1-3 milyon arası:** Çok başarılı (%90+)
- **1 milyon altı:** İyi (%80-85)
- **5 milyon üstü:** Orta (%70-75)

**Sebebi:**
*"Çoğu ev 1-3 milyon arasında, o yüzden en çok bu aralıktan öğrenmiş. Çok ucuz veya çok pahalı evler az olduğu için daha zor."*

---

## 🎯 **SUNUMDA SÖYLEYECEKLERİM**

### **Proje Hikayesi:**
1. *"İstanbul'da ev alacaktım, fiyatlar çok karışık"*
2. *"Emlak sitelerinde aynı özellikte evler farklı fiyatlarda"*
3. *"Makine öğrenmesi ile gerçekçi fiyat hesaplama sistemi yaptım"*
4. *"50+ farklı faktörü hesaba katıyor"*
5. *"4 farklı algoritmanın ortalamasını alıyor"*

### **Teknik Zorluklar ve Çözümlerim:**
**Zorluk:** *"Verinin %30'u saçmaydı"*
**Çözümüm:** *"3 aşamalı temizlik sistemi"*

**Zorluk:** *"İlçe isimleri tutarsızdı"*  
**Çözümüm:** *"Otomatik düzeltme sistemi"*

**Zorluk:** *"Model çok ezberliyordu"*
**Çözümüm:** *"4 farklı model birleştirdim"*

### **Demo Esnasında:**
1. *"Şimdi rastgele bir ev seçelim"* → Butona tık
2. *"Bu evin gerçek fiyatı X TL"* → Fiyatı göster
3. *"Bakalım model ne tahmin edecek"* → Tahmin butonu
4. *"Fark sadece %Y, oldukça başarılı"* → Sonucu yorumla
5. *"Benzer evlere bakalım"* → Grafikleri açıkla

---

## 💻 **Kurulum ve Çalıştırma**

```bash
# Gerekli kütüphaneler:
pip install pandas numpy scikit-learn matplotlib seaborn PyQt5

# Model eğitimi (ilk sefer):
python model.py

# Uygulamayı başlat:
python app.py
```

---

*Bu şekilde hem anlaşılır hem profesyonel bir sunum yapabilirsin! 🎯* 