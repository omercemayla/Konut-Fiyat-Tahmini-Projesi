# Sistem ve işletim sistemi fonksiyonları için
import sys
import os
# Veri işleme ve manipülasyon için
import pandas as pd
import random  # Rastgele sayı üretimi için
# PyQt5 arayüz kütüphaneleri - GUI oluşturmak için
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QFormLayout, QLineEdit, QPushButton,
                            QLabel, QComboBox, QMessageBox, QSpinBox, QDoubleSpinBox,
                            QGroupBox, QTabWidget, QGridLayout, QScrollArea, QProgressBar,
                            QSplitter, QFrame, QTextEdit)
# PyQt5 temel sınıfları ve thread yönetimi için
from PyQt5.QtCore import Qt, QRegExp, QThread, pyqtSignal
# PyQt5 görsel öğeler (font, ikon, renkler) için
from PyQt5.QtGui import QFont, QRegExpValidator, QIcon, QPixmap, QLinearGradient, QColor, QPalette
# Grafik çizme kütüphaneleri
import matplotlib.pyplot as plt  # Grafik çizme için
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # PyQt5 ile matplotlib entegrasyonu
from matplotlib.figure import Figure  # Grafik figürleri için
import numpy as np  # Sayısal hesaplamalar için
import seaborn as sns  # Güzel istatistiksel grafikler için
import matplotlib.patches as patches  # Grafik üzerinde şekiller için
from matplotlib.patches import Rectangle  # Dikdörtgen şekiller için

# Kendi model.py dosyamızdan fonksiyonları import et
from model import load_and_preprocess_data, train_model, predict_price, get_available_features, get_district_stats

# Grafiklerin görünümü için stil ayarları
plt.style.use('seaborn-v0_8')  # Güzel seaborn stili
sns.set_palette("husl")  # Renkli palet ayarla
plt.rcParams['font.size'] = 8  # Varsayılan font boyutu
plt.rcParams['axes.titlesize'] = 10  # Grafik başlık font boyutu
plt.rcParams['axes.labelsize'] = 8  # Eksen etiket font boyutu
plt.rcParams['xtick.labelsize'] = 7  # X ekseni değer font boyutu
plt.rcParams['ytick.labelsize'] = 7  # Y ekseni değer font boyutu

# Ana uygulama sınıfı - QMainWindow'dan miras alır (PyQt5 ana pencere)
class KonutFiyatTahmini(QMainWindow):
    def __init__(self):
        super().__init__()  # Üst sınıfın (QMainWindow) __init__ metodunu çağır
        
        # Ana pencere ayarları
        self.setWindowTitle("🏡 İstanbul Konut Fiyat Tahmini")  # Pencere başlığını ayarla
        self.setGeometry(50, 50, 1400, 900)  # Pencere pozisyonu (x, y) ve boyutu (genişlik, yükseklik)
        self.setMinimumSize(1200, 800)  # Minimum pencere boyutunu ayarla
        
        # Modern tema ayarları
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
            QTabWidget::pane {
                border: 1px solid #ddd;
                background-color: white;
                border-radius: 8px;
            }
            QTabBar::tab {
                background-color: #e9ecef;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #007bff;
                color: white;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
                color: #495057;
            }
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #004085;
            }
        """)
        
        # Veri seti ve model bilgilerini içeren sınıf değişkenleri (instance variables)
        self.df = None  # Pandas DataFrame - konut verileri
        self.model = None  # Eğitilmiş makine öğrenmesi modeli
        self.scaler = None  # Veri normalleştirme objesi
        self.feature_names = None  # Model özelliklerinin isimleri listesi
        self.feature_importance = None  # Özelliklerin önem skorları
        self.current_real_price = None  # Rastgele seçilen örneğin gerçek fiyatı (karşılaştırma için)
        self.district_stats = None  # İlçe bazında istatistikler sözlüğü
        
        # Model.py'den ilçe ve mahalle listelerini al
        self.available_features = get_available_features()
        
        # Veri setini uygulama başlangıcında yükle (tek seferlik işlem)
        print("📊 Veri seti yükleniyor...")
        self.df = load_and_preprocess_data()  # Excel dosyasından veriyi oku ve temizle
        print("📈 İlçe istatistikleri hesaplanıyor...")
        self.district_stats = get_district_stats()  # Her ilçe için ortalama, min, max fiyatları hesapla
        print("✅ Veri yükleme tamamlandı!")
        
        # Arayüz elemanlarını oluştur
        self.create_ui()
        
        # Model bilgilerini yükle (daha önce eğitilmişse)
        self.load_model_info()
        
    def create_ui(self):
        """Kullanıcı arayüzünü oluşturur - ana UI metodu"""
        # Ana widget ve layout oluştur
        main_widget = QWidget()  # Ana pencereye yerleşecek widget
        main_layout = QVBoxLayout()  # Dikey düzen (vertical layout)
        
        # Sekme widget'ını oluştur (farklı fonksiyonlar için ayrı sekmeler)
        self.tab_widget = QTabWidget()
        
        # Her sekmeyi oluşturan metodları çağır
        self.create_prediction_tab()  # Fiyat tahmin sekmesi
        
        self.create_feature_importance_tab()  # Özellik önemleri analiz sekmesi
        
        self.create_market_analysis_tab()  # Piyasa trendleri sekmesi
        
        self.create_statistical_analysis_tab()  # İstatistiksel analizler sekmesi
        
        self.create_data_tab()  # Ham veri analizi sekmesi
        
        # Sekme widget'ını ana layout'a ekle
        main_layout.addWidget(self.tab_widget)
        
        # Ana widget'ın layout'unu ayarla ve pencereye yerleştir
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)  # QMainWindow'un merkez widget'ını ayarla
        
    def create_prediction_tab(self):
        """Fiyat tahmin sekmesini oluşturur - ana sekme"""
        prediction_tab = QWidget()  # Sekme için widget oluştur
        main_layout = QHBoxLayout()  # Yatay düzen (sol-sağ panel)
        
        # Sol panel: Kullanıcı input formu ve mini grafikler
        left_panel = QWidget()  # Sol taraf için widget
        left_layout = QVBoxLayout()  # Sol panelde dikey düzen
        
        # Konut özellikleri input formu grup kutusu
        form_group = QGroupBox("🏠 Konut Özellikleri")  # Başlıklı grup kutusu
        form_layout = QFormLayout()  # Form düzeni (etiket: input şeklinde)
        
        # Kullanıcının konut özelliklerini gireceği input alanları sözlüğü
        self.input_fields = {}  # Boş sözlük - sonra doldurulacak
        
        # Rastgele konut örneği seçen buton (test etmek için)
        random_sample_button = QPushButton("🎲 Rastgele Örnek Seç")  # Buton metni
        random_sample_button.setFont(QFont("Arial", 10, QFont.Bold))  # Font ayarları
        # Buton için CSS stil tanımlaması (mavi gradyan)
        random_sample_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                          stop:0 #4a90e2, stop:1 #357abd);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                          stop:0 #357abd, stop:1 #2868a0);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                          stop:0 #2868a0, stop:1 #1e5288);
            }
        """)
        # Butona tıklandığında load_random_sample metodunu çağır
        random_sample_button.clicked.connect(self.load_random_sample)
        # Butonu form layout'una ekle (etiket boş, buton sağda)
        form_layout.addRow("", random_sample_button)
        
        # Metrekare input alanı (m²)
        self.input_fields['metrekare'] = QSpinBox()  # Sayı giriş kutusu oluştur
        self.input_fields['metrekare'].setRange(30, 500)  # 30-500 m² arası kabul et
        self.input_fields['metrekare'].setValue(100)  # Varsayılan değer 100 m²
        form_layout.addRow("Metrekare (m²):", self.input_fields['metrekare'])  # Forma ekle
        
        # Oda sayısı input alanı
        self.input_fields['oda_sayisi'] = QSpinBox()  # Sayı giriş kutusu
        self.input_fields['oda_sayisi'].setRange(1, 10)  # 1-10 oda arası kabul et
        self.input_fields['oda_sayisi'].setValue(2)  # Varsayılan 2 oda
        form_layout.addRow("Oda Sayısı:", self.input_fields['oda_sayisi'])  # Forma ekle
        
        # Bina yaşı input alanı
        self.input_fields['yas'] = QSpinBox()  # Sayı giriş kutusu
        self.input_fields['yas'].setRange(0, 100)  # 0-100 yaş arası kabul et
        self.input_fields['yas'].setValue(5)  # Varsayılan 5 yaşında bina
        form_layout.addRow("Bina Yaşı:", self.input_fields['yas'])  # Forma ekle
        
        # Bulunduğu kat input alanı
        self.input_fields['bulundugu_kat'] = QSpinBox()  # Sayı giriş kutusu
        self.input_fields['bulundugu_kat'].setRange(0, 50)  # 0-50. kat arası kabul et
        self.input_fields['bulundugu_kat'].setValue(2)  # Varsayılan 2. kat
        form_layout.addRow("Bulunduğu Kat:", self.input_fields['bulundugu_kat'])  # Forma ekle
        
        # İlçe
        self.input_fields['ilce'] = QComboBox()
        if self.available_features is not None and 'ilce' in self.available_features:
            self.input_fields['ilce'].addItems(sorted(self.available_features['ilce']))
        else:
            # Örnek İlçeler
            self.input_fields['ilce'].addItems([
                "Adalar", "Arnavutköy", "Ataşehir", "Avcılar", "Bağcılar", "Bahçelievler",
                "Bakırköy", "Başakşehir", "Bayrampaşa", "Beşiktaş", "Beykoz", "Beylikdüzü",
                "Beyoğlu", "Büyükçekmece", "Çatalca", "Çekmeköy", "Esenler", "Esenyurt",
                "Eyüpsultan", "Fatih", "Gaziosmanpaşa", "Güngören", "Kadıköy", "Kağıthane",
                "Kartal", "Küçükçekmece", "Maltepe", "Pendik", "Sancaktepe", "Sarıyer",
                "Şile", "Silivri", "Şişli", "Sultanbeyli", "Sultangazi", "Tuzla",
                "Ümraniye", "Üsküdar", "Zeytinburnu"
            ])
        form_layout.addRow("İlçe:", self.input_fields['ilce'])
        
        # Mahalle
        self.input_fields['mahalle'] = QComboBox()
        # İlçe seçimine göre mahalleler güncellenecek
        self.input_fields['ilce'].currentTextChanged.connect(self.update_mahalle_list)
        form_layout.addRow("Mahalle:", self.input_fields['mahalle'])
        
        # İlk ilçe için mahalleleri yükle
        self.update_mahalle_list(self.input_fields['ilce'].currentText())
        
        form_group.setLayout(form_layout)
        
        # Tahmin butonu
        predict_button = QPushButton("🔮 Fiyat Tahmini Yap")
        predict_button.setFont(QFont("Arial", 12, QFont.Bold))
        predict_button.setMinimumHeight(50)
        predict_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                          stop:0 #28a745, stop:1 #20c997);
                color: white;
                border: none;
                border-radius: 25px;
                font-size: 14px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                          stop:0 #20c997, stop:1 #28a745);
            }
        """)
        predict_button.clicked.connect(self.predict)
        
        # Mini grafikler için canvas
        self.prediction_mini_canvas = FigureCanvas(Figure(figsize=(8, 6)))
        self.prediction_mini_fig = self.prediction_mini_canvas.figure
        self.prediction_mini_fig.patch.set_facecolor('white')
        
        # Sol paneli düzenle
        left_layout.addWidget(form_group)
        left_layout.addWidget(predict_button)
        left_layout.addWidget(self.prediction_mini_canvas)
        left_panel.setLayout(left_layout)
        
        # Sağ panel: Sonuçlar ve görselleştirme
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        
        # Sonuç grup kutusu
        result_group = QGroupBox("📊 Tahmin Sonuçları")
        result_group_layout = QVBoxLayout()
        
        # Ana sonuç etiketi
        self.result_label = QLabel("Tahmin yapmak için sol taraftaki formu doldurun ve 'Fiyat Tahmini Yap' butonuna tıklayın")
        self.result_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("""
            QLabel {
                background-color: #e3f2fd;
                padding: 20px;
                border-radius: 15px;
                border: 2px solid #2196f3;
                color: #1565c0;
            }
        """)
        
        # Detaylı sonuç ve güven aralığı
        self.confidence_label = QLabel("")
        self.confidence_label.setFont(QFont("Arial", 11))
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setWordWrap(True)
        self.confidence_label.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 10px;
                border: 1px solid #dee2e6;
                color: #495057;
            }
        """)
        
        # Karşılaştırma grafiği için canvas
        self.comparison_canvas = FigureCanvas(Figure(figsize=(10, 7)))
        self.comparison_fig = self.comparison_canvas.figure
        self.comparison_fig.patch.set_facecolor('white')
        
        result_group_layout.addWidget(self.result_label)
        result_group_layout.addWidget(self.confidence_label)
        result_group_layout.addWidget(self.comparison_canvas)
        result_group.setLayout(result_group_layout)
        
        right_layout.addWidget(result_group)
        right_panel.setLayout(right_layout)
        
        # Splitter ile bölünmüş düzen
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])  # Sol panel daha dar, sağ panel daha geniş
        
        main_layout.addWidget(splitter)
        prediction_tab.setLayout(main_layout)
        self.tab_widget.addTab(prediction_tab, "🔮 Fiyat Tahmini")
        
        # İlk yüklemede mini grafikleri çiz
        self.plot_prediction_mini_charts()
        
    def create_feature_importance_tab(self):
        """Özellik önemleri sekmesini oluşturur"""
        importance_tab = QWidget()
        main_layout = QVBoxLayout()
        
        # Özellik önemleri grup kutusu
        importance_group = QGroupBox("🎯 Özellik Önemleri Analizi")
        importance_layout = QVBoxLayout()
        
        # Özellik önemleri canvas
        self.importance_canvas = FigureCanvas(Figure(figsize=(12, 8)))
        self.importance_fig = self.importance_canvas.figure
        self.importance_fig.patch.set_facecolor('white')
        
        importance_layout.addWidget(self.importance_canvas)
        importance_group.setLayout(importance_layout)
        
        main_layout.addWidget(importance_group)
        importance_tab.setLayout(main_layout)
        self.tab_widget.addTab(importance_tab, "🎯 Özellik Önemleri")
        
    def create_market_analysis_tab(self):
        """Piyasa analizi sekmesini oluşturur"""
        market_tab = QWidget()
        main_layout = QVBoxLayout()
        
        # Piyasa analizi grup kutusu
        market_group = QGroupBox("📈 Piyasa Analizi ve Trend Çalışmaları")
        market_layout = QVBoxLayout()
        
        # Alt sekmeler
        self.market_tabs = QTabWidget()
        
        # Fiyat trend analizi tab
        price_trend_tab = QWidget()
        price_trend_layout = QVBoxLayout()
        self.price_trend_canvas = FigureCanvas(Figure(figsize=(12, 8)))
        self.price_trend_fig = self.price_trend_canvas.figure
        self.price_trend_fig.patch.set_facecolor('white')
        price_trend_layout.addWidget(self.price_trend_canvas)
        price_trend_tab.setLayout(price_trend_layout)
        self.market_tabs.addTab(price_trend_tab, "💰 Fiyat Trendleri")
        
        # Bölgesel analiz tab
        regional_tab = QWidget()
        regional_layout = QVBoxLayout()
        self.regional_canvas = FigureCanvas(Figure(figsize=(12, 8)))
        self.regional_fig = self.regional_canvas.figure
        self.regional_fig.patch.set_facecolor('white')
        regional_layout.addWidget(self.regional_canvas)
        regional_tab.setLayout(regional_layout)
        self.market_tabs.addTab(regional_tab, "🗺️ Bölgesel Analiz")
        
        # Değer analizi tab
        value_tab = QWidget()
        value_layout = QVBoxLayout()
        self.value_canvas = FigureCanvas(Figure(figsize=(12, 8)))
        self.value_fig = self.value_canvas.figure
        self.value_fig.patch.set_facecolor('white')
        value_layout.addWidget(self.value_canvas)
        value_tab.setLayout(value_layout)
        self.market_tabs.addTab(value_tab, "💎 Değer Analizi")
        
        market_layout.addWidget(self.market_tabs)
        market_group.setLayout(market_layout)
        main_layout.addWidget(market_group)
        market_tab.setLayout(main_layout)
        self.tab_widget.addTab(market_tab, "📈 Piyasa Analizi")
        
    def create_statistical_analysis_tab(self):
        """İstatistiksel analiz sekmesini oluşturur"""
        stats_tab = QWidget()
        main_layout = QVBoxLayout()
        
        # İstatistiksel analiz grup kutusu
        stats_group = QGroupBox("📊 Gelişmiş İstatistiksel Analizler")
        stats_layout = QVBoxLayout()
        
        # Alt sekmeler
        self.stats_tabs = QTabWidget()
        
        # Dağılım analizi tab
        distribution_tab = QWidget()
        distribution_layout = QVBoxLayout()
        self.distribution_canvas = FigureCanvas(Figure(figsize=(12, 8)))
        self.distribution_fig = self.distribution_canvas.figure
        self.distribution_fig.patch.set_facecolor('white')
        distribution_layout.addWidget(self.distribution_canvas)
        distribution_tab.setLayout(distribution_layout)
        self.stats_tabs.addTab(distribution_tab, "📈 Dağılım Analizi")
        
        # Korelasyon detay tab
        correlation_tab = QWidget()
        correlation_layout = QVBoxLayout()
        self.correlation_detail_canvas = FigureCanvas(Figure(figsize=(12, 8)))
        self.correlation_detail_fig = self.correlation_detail_canvas.figure
        self.correlation_detail_fig.patch.set_facecolor('white')
        correlation_layout.addWidget(self.correlation_detail_canvas)
        correlation_tab.setLayout(correlation_layout)
        self.stats_tabs.addTab(correlation_tab, "🔗 Detaylı Korelasyon")
        
        # Outlier analizi tab
        outlier_tab = QWidget()
        outlier_layout = QVBoxLayout()
        self.outlier_canvas = FigureCanvas(Figure(figsize=(12, 8)))
        self.outlier_fig = self.outlier_canvas.figure
        self.outlier_fig.patch.set_facecolor('white')
        outlier_layout.addWidget(self.outlier_canvas)
        outlier_tab.setLayout(outlier_layout)
        self.stats_tabs.addTab(outlier_tab, "🎯 Outlier Analizi")
        
        stats_layout.addWidget(self.stats_tabs)
        stats_group.setLayout(stats_layout)
        main_layout.addWidget(stats_group)
        stats_tab.setLayout(main_layout)
        self.tab_widget.addTab(stats_tab, "📊 İstatistiksel Analiz")
        
    def create_data_tab(self):
        """Veri sekmesini oluşturur"""
        data_tab = QWidget()
        main_layout = QHBoxLayout()
        
        # Sol panel: İstatistikler
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        
        # Veri seti özet bilgileri
        info_group = QGroupBox("📈 Veri Seti Özet Bilgileri")
        info_layout = QVBoxLayout()
        
        self.data_info_text = QLabel("Veri seti yükleniyor...")
        self.data_info_text.setWordWrap(True)
        self.data_info_text.setFont(QFont("Arial", 10))
        self.data_info_text.setStyleSheet("""
            QLabel {
                background-color: white;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #dee2e6;
            }
        """)
        
        info_layout.addWidget(self.data_info_text)
        info_group.setLayout(info_layout)
        
        # İlçe istatistikleri
        district_group = QGroupBox("🗺️ İlçe İstatistikleri")
        district_layout = QVBoxLayout()
        
        self.district_canvas = FigureCanvas(Figure(figsize=(7, 5)))
        self.district_fig = self.district_canvas.figure
        self.district_fig.patch.set_facecolor('white')
        
        district_layout.addWidget(self.district_canvas)
        district_group.setLayout(district_layout)
        
        left_layout.addWidget(info_group)
        left_layout.addWidget(district_group)
        left_panel.setLayout(left_layout)
        
        # Sağ panel: Detaylı analizler
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        
        # Ana veri analiz grafikleri
        analysis_group = QGroupBox("📊 Detaylı Veri Analizi")
        analysis_layout = QVBoxLayout()
        
        # Tab widget for different charts
        self.analysis_tabs = QTabWidget()
        
        # Fiyat dağılım tab
        price_tab = QWidget()
        price_layout = QVBoxLayout()
        self.price_canvas = FigureCanvas(Figure(figsize=(10, 7)))
        self.price_fig = self.price_canvas.figure
        self.price_fig.patch.set_facecolor('white')
        price_layout.addWidget(self.price_canvas)
        price_tab.setLayout(price_layout)
        self.analysis_tabs.addTab(price_tab, "💰 Fiyat Dağılımı")
        
        # Korelasyon tab
        corr_tab = QWidget()
        corr_layout = QVBoxLayout()
        self.corr_canvas = FigureCanvas(Figure(figsize=(10, 7)))
        self.corr_fig = self.corr_canvas.figure
        self.corr_fig.patch.set_facecolor('white')
        corr_layout.addWidget(self.corr_canvas)
        corr_tab.setLayout(corr_layout)
        self.analysis_tabs.addTab(corr_tab, "🔗 Korelasyon Analizi")
        
        # Trend analizi tab
        trend_tab = QWidget()
        trend_layout = QVBoxLayout()
        self.trend_canvas = FigureCanvas(Figure(figsize=(10, 7)))
        self.trend_fig = self.trend_canvas.figure
        self.trend_fig.patch.set_facecolor('white')
        trend_layout.addWidget(self.trend_canvas)
        trend_tab.setLayout(trend_layout)
        self.analysis_tabs.addTab(trend_tab, "📈 Trend Analizi")
        
        analysis_layout.addWidget(self.analysis_tabs)
        analysis_group.setLayout(analysis_layout)
        
        right_layout.addWidget(analysis_group)
        right_panel.setLayout(right_layout)
        
        # Splitter düzeni
        data_splitter = QSplitter(Qt.Horizontal)
        data_splitter.addWidget(left_panel)
        data_splitter.addWidget(right_panel)
        data_splitter.setSizes([400, 800])
        
        main_layout.addWidget(data_splitter)
        data_tab.setLayout(main_layout)
        self.tab_widget.addTab(data_tab, "📊 Veri Analizi")
    
    def update_mahalle_list(self, selected_ilce):
        """Seçilen ilçeye göre mahalle listesini güncelle"""
        self.input_fields['mahalle'].clear()
        
        if not selected_ilce:
            return
        
        if self.available_features is not None and 'ilce_mahalle_map' in self.available_features:
            # Seçilen ilçeye ait mahalleleri göster
            ilce_mahalle_map = self.available_features['ilce_mahalle_map']
            if selected_ilce in ilce_mahalle_map:
                mahalle_list = ilce_mahalle_map[selected_ilce]
                self.input_fields['mahalle'].addItems(mahalle_list)
        else:
            # Veri yoksa örnek mahalle isimleri
            self.input_fields['mahalle'].addItems(['Merkez Mahalle', 'Yeni Mahalle', 'Cumhuriyet Mahalle'])
    
    def load_model_info(self):
        """Daha önce eğitilmiş model varsa bilgilerini yükler"""
        self.load_model_files()
        self.update_data_info()
        
    def load_model_files(self):
        """Model dosyalarını yükler"""
        try:
            if os.path.exists('models/konut_fiyat_model.pkl'):
                # Model detaylarını göster
                import joblib
                self.model = joblib.load('models/konut_fiyat_model.pkl')
                self.scaler = joblib.load('models/scaler.pkl')
                self.feature_names = joblib.load('models/feature_names.pkl')
                
                # Özellik önemini çizdir
                self.plot_feature_importance()
                
                # Input alanlarını güncelle
                self.update_input_fields()
                
                print("✅ Model başarıyla yüklendi!")
            else:
                print("⚠️ Eğitilmiş model bulunamadı!")
        except Exception as e:
            print(f"❌ Model yükleme hatası: {e}")
    
    def update_input_fields(self):
        """Modele göre input alanlarını günceller"""
        # Modelden ilçe ve mahalle bilgilerini al
        self.available_features = get_available_features()
        
        if self.available_features is not None:
            # İlçe listesini güncelle
            if 'ilce' in self.available_features:
                current_ilce = self.input_fields['ilce'].currentText()
                self.input_fields['ilce'].clear()
                self.input_fields['ilce'].addItems(sorted(self.available_features['ilce']))
                
                # Önceki seçili ilçeyi koru
                index = self.input_fields['ilce'].findText(current_ilce)
                if index >= 0:
                    self.input_fields['ilce'].setCurrentIndex(index)
                else:
                    # Eğer önceki ilçe bulunamazsa ilk ilçeyi seç
                    if self.input_fields['ilce'].count() > 0:
                        self.input_fields['ilce'].setCurrentIndex(0)
            
            # Seçili ilçeye göre mahalle listesini güncelle
            self.update_mahalle_list(self.input_fields['ilce'].currentText())
    
    def update_data_info(self):
        """Veri seti bilgilerini günceller"""
        try:
            # Veri zaten yüklüyse yeniden yükleme
            if self.df is None:
                self.df = load_and_preprocess_data()
            if self.district_stats is None:
                self.district_stats = get_district_stats()
            
            if self.df is not None:
                # Detaylı veri seti özeti
                unique_districts = self.df['ilce'].nunique()
                unique_neighborhoods = self.df['mahalle'].nunique()
                
                info_text = f"📊 GENEL BİLGİLER\n"
                info_text += f"• Toplam Kayıt: {self.df.shape[0]:,} konut\n"
                info_text += f"• Toplam Özellik: {self.df.shape[1]} adet\n"
                info_text += f"• İlçe Sayısı: {unique_districts} adet\n"
                info_text += f"• Mahalle Sayısı: {unique_neighborhoods} adet\n\n"
                
                info_text += f"💰 FİYAT İSTATİSTİKLERİ\n"
                info_text += f"• Ortalama: {self.df['fiyat'].mean():,.0f} TL\n"
                info_text += f"• Medyan: {self.df['fiyat'].median():,.0f} TL\n"
                info_text += f"• Minimum: {self.df['fiyat'].min():,.0f} TL\n"
                info_text += f"• Maksimum: {self.df['fiyat'].max():,.0f} TL\n"
                info_text += f"• Std. Sapma: {self.df['fiyat'].std():,.0f} TL\n\n"
                
                info_text += f"🏠 KONUT ÖZELLİKLERİ\n"
                info_text += f"• Ort. Metrekare: {self.df['metrekare'].mean():.0f} m²\n"
                info_text += f"• Ort. Oda Sayısı: {self.df['oda_sayisi'].mean():.1f}\n"
                info_text += f"• Ort. Bina Yaşı: {self.df['yas'].mean():.1f} yıl\n"
                info_text += f"• Ort. Kat: {self.df['bulundugu_kat'].mean():.1f}\n\n"
                
                # M² başına fiyat
                price_per_m2 = (self.df['fiyat'] / self.df['metrekare']).mean()
                info_text += f"💡 DİĞER İSTATİSTİKLER\n"
                info_text += f"• Ort. m² Fiyatı: {price_per_m2:,.0f} TL/m²\n"
                
                # En pahalı ve en ucuz ilçe
                district_avg = self.df.groupby('ilce')['fiyat'].mean()
                most_expensive = district_avg.idxmax()
                cheapest = district_avg.idxmin()
                info_text += f"• En Pahalı İlçe: {most_expensive}\n"
                info_text += f"• En Uygun İlçe: {cheapest}\n"
                
                self.data_info_text.setText(info_text)
                
                # Tüm grafikleri çiz
                self.plot_price_distribution()
                self.plot_correlation_analysis()
                self.plot_trend_analysis()
                self.plot_district_statistics()
                
                # Yeni analiz grafiklerini çiz
                self.plot_market_analysis()
                self.plot_statistical_analysis()
        except Exception as e:
            self.data_info_text.setText(f"Veri yükleme hatası: {e}")
    
    def plot_feature_importance(self):
        """Gelişmiş özellik önem grafiği çizer"""
        try:
            if self.model is not None and hasattr(self.model, 'named_estimators_'):
                # Ensemble model için Random Forest'ten özellik önemini al
                if 'rf' in self.model.named_estimators_:
                    rf_model = self.model.named_estimators_['rf']
                    if hasattr(rf_model, 'feature_importances_'):
                        importances = rf_model.feature_importances_
                    else:
                        return
                else:
                    return
            elif hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
            else:
                return
                
            self.importance_fig.clear()
            gs = self.importance_fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
            
            # 1. Top 15 features
            ax1 = self.importance_fig.add_subplot(gs[0, :])
            indices = np.argsort(importances)[-15:]
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))
            bars = ax1.barh(range(len(indices)), importances[indices], color=colors)
            ax1.set_yticks(range(len(indices)))
            ax1.set_yticklabels([self.feature_names[i] for i in indices], fontsize=8)
            ax1.set_xlabel('Özellik Önemi', fontsize=10)
            ax1.set_title('🎯 En Önemli 15 Özellik', fontweight='bold', fontsize=12)
            ax1.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax1.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                        f'{width:.4f}', ha='left', va='center', fontsize=8)
            
            # 2. Feature importance by category
            ax2 = self.importance_fig.add_subplot(gs[1, 0])
            
            # Kategorilere göre grupla
            categories = {
                'Konum': ['ilce', 'mahalle', 'target', 'freq'],
                'Alan': ['metrekare', 'oda', 'alan'],
                'Yaş': ['yas', 'yeni', 'eski', 'tersi', 'exp'],
                'Kat': ['kat', 'zemin', 'yuksek', 'bodrum'],
                'Diğer': []
            }
            
            category_importance = {}
            for category, keywords in categories.items():
                total_importance = 0
                for i, name in enumerate(self.feature_names):
                    if any(keyword in name.lower() for keyword in keywords):
                        total_importance += importances[i]
                    elif category == 'Diğer' and not any(
                        any(kw in name.lower() for kw in kws) 
                        for kws in categories.values() if kws != []
                    ):
                        total_importance += importances[i]
                category_importance[category] = total_importance
            
            # Remove empty categories
            category_importance = {k: v for k, v in category_importance.items() if v > 0}
            
            if category_importance:
                categories_names = list(category_importance.keys())
                category_values = list(category_importance.values())
                
                wedges, texts, autotexts = ax2.pie(category_values, labels=categories_names, 
                                                  autopct='%1.1f%%', startangle=90,
                                                  colors=plt.cm.Set3(np.linspace(0, 1, len(categories_names))))
                ax2.set_title('Kategori Bazında Önem', fontweight='bold')
            
            # 3. Feature count by importance level
            ax3 = self.importance_fig.add_subplot(gs[1, 1])
            
            # Önem seviyelerine göre grupla
            high_imp = np.sum(importances > 0.02)
            medium_imp = np.sum((importances > 0.01) & (importances <= 0.02))
            low_imp = np.sum((importances > 0.005) & (importances <= 0.01))
            very_low_imp = np.sum(importances <= 0.005)
            
            levels = ['Çok Yüksek\n(>0.02)', 'Yüksek\n(0.01-0.02)', 'Orta\n(0.005-0.01)', 'Düşük\n(≤0.005)']
            counts = [high_imp, medium_imp, low_imp, very_low_imp]
            colors = ['red', 'orange', 'yellow', 'lightgray']
            
            bars = ax3.bar(levels, counts, color=colors, alpha=0.8)
            ax3.set_title('Önem Seviyesi Dağılımı', fontweight='bold')
            ax3.set_ylabel('Özellik Sayısı')
            
            # Add count labels
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{count}', ha='center', va='bottom', fontweight='bold')
            
            # Layout düzenlemesi
            self.importance_fig.tight_layout(pad=1.5)
            self.importance_canvas.draw()
            
        except Exception as e:
            print(f"Özellik önem grafiği hatası: {e}")
    
    def plot_prediction_mini_charts(self):
        """Tahmin sekmesinde mini grafikler çizer"""
        if self.df is None:
            return
            
        self.prediction_mini_fig.clear()
        
        # 2x2 subplot grid
        gs = self.prediction_mini_fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
        
        try:
            # 1. İlçe bazında ortalama fiyat (top 10)
            ax1 = self.prediction_mini_fig.add_subplot(gs[0, 0])
            district_prices = self.df.groupby('ilce')['fiyat'].mean().sort_values(ascending=False).head(10)
            district_prices.plot(kind='bar', ax=ax1, color='skyblue')
            ax1.set_title('En Pahalı 10 İlçe', fontsize=9, fontweight='bold')
            ax1.set_xlabel('')
            ax1.set_ylabel('Ortalama Fiyat (TL)', fontsize=8)
            ax1.tick_params(axis='x', rotation=45, labelsize=7)
            ax1.tick_params(axis='y', labelsize=7)
            
            # 2. Metrekare vs Fiyat scatter
            ax2 = self.prediction_mini_fig.add_subplot(gs[0, 1])
            sample_data = self.df.sample(min(1000, len(self.df)))  # Performance için sample
            ax2.scatter(sample_data['metrekare'], sample_data['fiyat'], alpha=0.5, s=1, c='coral')
            ax2.set_title('Metrekare vs Fiyat', fontsize=9, fontweight='bold')
            ax2.set_xlabel('Metrekare', fontsize=8)
            ax2.set_ylabel('Fiyat (TL)', fontsize=8)
            ax2.tick_params(labelsize=7)
            
            # 3. Oda sayısına göre fiyat dağılımı
            ax3 = self.prediction_mini_fig.add_subplot(gs[1, 0])
            room_prices = self.df.groupby('oda_sayisi')['fiyat'].mean()
            room_prices.plot(kind='bar', ax=ax3, color='lightgreen')
            ax3.set_title('Oda Sayısı vs Fiyat', fontsize=9, fontweight='bold')
            ax3.set_xlabel('Oda Sayısı', fontsize=8)
            ax3.set_ylabel('Ortalama Fiyat (TL)', fontsize=8)
            ax3.tick_params(axis='x', rotation=0, labelsize=7)
            ax3.tick_params(axis='y', labelsize=7)
            
            # 4. Yaş vs Fiyat
            ax4 = self.prediction_mini_fig.add_subplot(gs[1, 1])
            age_bins = pd.cut(self.df['yas'], bins=[0, 5, 10, 20, 30, 100], labels=['0-5', '6-10', '11-20', '21-30', '30+'])
            age_prices = self.df.groupby(age_bins)['fiyat'].mean()
            age_prices.plot(kind='bar', ax=ax4, color='orange')
            ax4.set_title('Bina Yaşı vs Fiyat', fontsize=9, fontweight='bold')
            ax4.set_xlabel('Bina Yaşı', fontsize=8)
            ax4.set_ylabel('Ortalama Fiyat (TL)', fontsize=8)
            ax4.tick_params(axis='x', rotation=45, labelsize=7)
            ax4.tick_params(axis='y', labelsize=7)
            
            # Layout düzenlemesi
            self.prediction_mini_fig.tight_layout(pad=1.0)
            self.prediction_mini_canvas.draw()
        except Exception as e:
            print(f"Mini grafik çizim hatası: {e}")
    
    def plot_price_distribution(self):
        """Gelişmiş fiyat dağılımı analizi"""
        if self.df is None:
            return
            
        self.price_fig.clear()
        gs = self.price_fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
        
        try:
            # 1. Histogram + KDE
            ax1 = self.price_fig.add_subplot(gs[0, :])
            sns.histplot(data=self.df, x='fiyat', kde=True, ax=ax1, color='skyblue', alpha=0.7)
            ax1.set_title('💰 Fiyat Dağılımı - Histogram & KDE', fontweight='bold', fontsize=12)
            ax1.set_xlabel('Fiyat (TL)')
            ax1.set_ylabel('Frekans')
            
            # 2. Box plot by district (top 10)
            ax2 = self.price_fig.add_subplot(gs[1, 0])
            top_districts = self.df['ilce'].value_counts().head(10).index
            df_top = self.df[self.df['ilce'].isin(top_districts)]
            sns.boxplot(data=df_top, y='ilce', x='fiyat', ax=ax2)
            ax2.set_title('İlçe Bazında Fiyat Dağılımı', fontweight='bold')
            ax2.set_xlabel('Fiyat (TL)')
            ax2.set_ylabel('')
            
            # 3. Violin plot by room count
            ax3 = self.price_fig.add_subplot(gs[1, 1])
            sns.violinplot(data=self.df, x='oda_sayisi', y='fiyat', ax=ax3)
            ax3.set_title('Oda Sayısına Göre Fiyat', fontweight='bold')
            ax3.set_xlabel('Oda Sayısı')
            ax3.set_ylabel('Fiyat (TL)')
            
            # Layout düzenlemesi
            self.price_fig.tight_layout(pad=1.5)
            self.price_canvas.draw()
        except Exception as e:
            print(f"Fiyat dağılım grafiği hatası: {e}")
    
    def plot_correlation_analysis(self):
        """Korelasyon analizi"""
        if self.df is None:
            return
            
        self.corr_fig.clear()
        
        try:
            # Sayısal sütunları seç
            numeric_cols = ['fiyat', 'metrekare', 'oda_sayisi', 'yas', 'bulundugu_kat']
            corr_data = self.df[numeric_cols].corr()
            
            ax = self.corr_fig.add_subplot(111)
            
            # Heatmap
            sns.heatmap(corr_data, annot=True, cmap='RdYlBu_r', center=0, 
                       square=True, ax=ax, cbar_kws={'shrink': 0.8})
            ax.set_title('🔗 Değişkenler Arası Korelasyon Matrisi', fontweight='bold', fontsize=14, pad=20)
            
            # Layout düzenlemesi
            self.corr_fig.tight_layout(pad=1.5)
            self.corr_canvas.draw()
        except Exception as e:
            print(f"Korelasyon analizi hatası: {e}")
    
    def plot_trend_analysis(self):
        """Trend analizi"""
        if self.df is None:
            return
            
        self.trend_fig.clear()
        gs = self.trend_fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
        
        try:
            # 1. Metrekare vs Fiyat trend
            ax1 = self.trend_fig.add_subplot(gs[0, 0])
            sample_df = self.df.sample(min(2000, len(self.df)))
            sns.scatterplot(data=sample_df, x='metrekare', y='fiyat', alpha=0.6, ax=ax1)
            sns.regplot(data=sample_df, x='metrekare', y='fiyat', scatter=False, color='red', ax=ax1)
            ax1.set_title('Metrekare vs Fiyat Trendi', fontweight='bold')
            
            # 2. Yaş vs Fiyat trend
            ax2 = self.trend_fig.add_subplot(gs[0, 1])
            sns.scatterplot(data=sample_df, x='yas', y='fiyat', alpha=0.6, ax=ax2)
            sns.regplot(data=sample_df, x='yas', y='fiyat', scatter=False, color='red', ax=ax2)
            ax2.set_title('Bina Yaşı vs Fiyat Trendi', fontweight='bold')
            
            # 3. Kat vs Fiyat
            ax3 = self.trend_fig.add_subplot(gs[1, 0])
            floor_price = self.df.groupby('bulundugu_kat')['fiyat'].mean().reset_index()
            floor_price = floor_price[(floor_price['bulundugu_kat'] >= 0) & (floor_price['bulundugu_kat'] <= 20)]
            ax3.plot(floor_price['bulundugu_kat'], floor_price['fiyat'], marker='o', linewidth=2, markersize=4)
            ax3.set_title('Kat vs Ortalama Fiyat', fontweight='bold')
            ax3.set_xlabel('Kat')
            ax3.set_ylabel('Ortalama Fiyat (TL)')
            ax3.grid(True, alpha=0.3)
            
            # 4. Fiyat/m² dağılımı
            ax4 = self.trend_fig.add_subplot(gs[1, 1])
            self.df['fiyat_per_m2'] = self.df['fiyat'] / self.df['metrekare']
            price_per_m2_by_district = self.df.groupby('ilce')['fiyat_per_m2'].mean().sort_values(ascending=False).head(10)
            price_per_m2_by_district.plot(kind='bar', ax=ax4, color='lightcoral')
            ax4.set_title('İlçe Bazında m² Fiyatı', fontweight='bold')
            ax4.set_xlabel('')
            ax4.set_ylabel('TL/m²')
            ax4.tick_params(axis='x', rotation=45)
            
            # Layout düzenlemesi
            self.trend_fig.tight_layout(pad=1.5)
            self.trend_canvas.draw()
        except Exception as e:
            print(f"Trend analizi hatası: {e}")
    
    def plot_district_statistics(self):
        """İlçe istatistikleri grafiği"""
        if self.district_stats is None:
            return
            
        self.district_fig.clear()
        
        try:
            # Top 15 ilçe
            sorted_districts = sorted(self.district_stats.items(), 
                                    key=lambda x: x[1]['mean'], reverse=True)[:15]
            
            districts = [item[0] for item in sorted_districts]
            means = [item[1]['mean'] for item in sorted_districts]
            counts = [item[1]['count'] for item in sorted_districts]
            
            ax = self.district_fig.add_subplot(111)
            
            # Bar plot
            bars = ax.bar(range(len(districts)), means, color='lightblue', alpha=0.8, edgecolor='navy')
            
            # Örnek sayısını barların üzerine yazdır
            for i, (bar, count) in enumerate(zip(bars, counts)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'n={count}', ha='center', va='bottom', fontsize=7)
            
            ax.set_title('🗺️ En Pahalı 15 İlçe (Ortalama Fiyat)', fontweight='bold', fontsize=11)
            ax.set_xlabel('İlçeler')
            ax.set_ylabel('Ortalama Fiyat (TL)')
            ax.set_xticks(range(len(districts)))
            ax.set_xticklabels(districts, rotation=45, ha='right', fontsize=7)
            ax.tick_params(axis='y', labelsize=8)
            ax.grid(axis='y', alpha=0.3)
            
            # Layout düzenlemesi
            self.district_fig.tight_layout(pad=1.0)
            self.district_canvas.draw()
        except Exception as e:
            print(f"İlçe istatistikleri hatası: {e}")
    
    def plot_market_analysis(self):
        """Piyasa analizi grafiklerini çizer"""
        if self.df is None:
            return
            
        # Fiyat Trendleri
        self.plot_price_trends()
        
        # Bölgesel Analiz
        self.plot_regional_analysis()
        
        # Değer Analizi
        self.plot_value_analysis()
    
    def plot_price_trends(self):
        """Fiyat trend grafiklerini çizer"""
        try:
            self.price_trend_fig.clear()
            gs = self.price_trend_fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)
            
            # 1. Fiyat seviyelerine göre dağılım
            ax1 = self.price_trend_fig.add_subplot(gs[0, 0])
            price_bins = pd.cut(self.df['fiyat'], bins=[0, 1000000, 2000000, 3000000, 5000000, float('inf')], 
                              labels=['<1M', '1M-2M', '2M-3M', '3M-5M', '>5M'])
            price_dist = price_bins.value_counts()
            price_dist.plot(kind='pie', ax=ax1, autopct='%1.1f%%', startangle=90)
            ax1.set_title('💰 Fiyat Seviyesi Dağılımı', fontweight='bold')
            ax1.set_ylabel('')
            
            # 2. Metrekare başına fiyat dağılımı
            ax2 = self.price_trend_fig.add_subplot(gs[0, 1])
            self.df['price_per_m2'] = self.df['fiyat'] / self.df['metrekare']
            m2_price = self.df['price_per_m2']
            ax2.hist(m2_price, bins=50, alpha=0.7, color='lightblue', edgecolor='navy')
            ax2.axvline(m2_price.mean(), color='red', linestyle='--', linewidth=2, label=f'Ortalama: {m2_price.mean():,.0f} TL/m²')
            ax2.axvline(m2_price.median(), color='green', linestyle='--', linewidth=2, label=f'Medyan: {m2_price.median():,.0f} TL/m²')
            ax2.set_title('📏 m² Başına Fiyat Dağılımı', fontweight='bold')
            ax2.set_xlabel('TL/m²')
            ax2.set_ylabel('Frekans')
            ax2.legend()
            
            # 3. Yaş gruplarına göre fiyat boxplot
            ax3 = self.price_trend_fig.add_subplot(gs[1, 0])
            age_groups = pd.cut(self.df['yas'], bins=[0, 5, 10, 20, 30, 100], labels=['0-5', '6-10', '11-20', '21-30', '30+'])
            df_with_age_groups = self.df.copy()
            df_with_age_groups['age_group'] = age_groups
            sns.boxplot(data=df_with_age_groups, x='age_group', y='fiyat', ax=ax3)
            ax3.set_title('🏗️ Yaş Gruplarına Göre Fiyat Dağılımı', fontweight='bold')
            ax3.set_xlabel('Yaş Grubu')
            ax3.set_ylabel('Fiyat (TL)')
            ax3.tick_params(axis='x', rotation=0)
            
            # 4. Oda sayısına göre ortalama fiyat ve adet
            ax4 = self.price_trend_fig.add_subplot(gs[1, 1])
            room_stats = self.df.groupby('oda_sayisi').agg({'fiyat': ['mean', 'count']}).round(0)
            room_stats.columns = ['Ortalama Fiyat', 'Adet']
            
            ax4_twin = ax4.twinx()
            bars1 = ax4.bar(room_stats.index, room_stats['Ortalama Fiyat'], alpha=0.7, color='lightcoral', label='Ortalama Fiyat')
            line1 = ax4_twin.plot(room_stats.index, room_stats['Adet'], color='navy', marker='o', linewidth=2, label='Emlak Adedi')
            
            ax4.set_title('🚪 Oda Sayısı Analizi', fontweight='bold')
            ax4.set_xlabel('Oda Sayısı')
            ax4.set_ylabel('Ortalama Fiyat (TL)', color='red')
            ax4_twin.set_ylabel('Emlak Adedi', color='navy')
            
            # 5. Kat seviyelerine göre fiyat analizi
            ax5 = self.price_trend_fig.add_subplot(gs[2, 0])
            floor_groups = pd.cut(self.df['bulundugu_kat'], bins=[-1, 0, 3, 7, 15, 100], 
                                labels=['Bodrum/Zemin', '1-3. Kat', '4-7. Kat', '8-15. Kat', '15+ Kat'])
            df_with_floor_groups = self.df.copy()
            df_with_floor_groups['floor_group'] = floor_groups
            floor_avg = df_with_floor_groups.groupby('floor_group')['fiyat'].mean()
            floor_avg.plot(kind='bar', ax=ax5, color='lightgreen', alpha=0.8)
            ax5.set_title('🏢 Kat Seviyesine Göre Ortalama Fiyat', fontweight='bold')
            ax5.set_xlabel('Kat Seviyesi')
            ax5.set_ylabel('Ortalama Fiyat (TL)')
            ax5.tick_params(axis='x', rotation=45)
            
            # 6. Fiyat-Metrekare scatter plot ile yoğunluk
            ax6 = self.price_trend_fig.add_subplot(gs[2, 1])
            sample_df = self.df.sample(min(3000, len(self.df)))
            scatter = ax6.scatter(sample_df['metrekare'], sample_df['fiyat'], 
                                alpha=0.6, c=sample_df['yas'], cmap='viridis', s=20)
            ax6.set_title('🎯 Metrekare-Fiyat-Yaş İlişkisi', fontweight='bold')
            ax6.set_xlabel('Metrekare')
            ax6.set_ylabel('Fiyat (TL)')
            cbar = self.price_trend_fig.colorbar(scatter, ax=ax6)
            cbar.set_label('Bina Yaşı')
            
            # Layout düzenlemesi
            self.price_trend_fig.tight_layout(pad=1.5)
            self.price_trend_canvas.draw()
            
        except Exception as e:
            print(f"Fiyat trend grafiği hatası: {e}")
    
    def plot_regional_analysis(self):
        """Bölgesel analiz grafiklerini çizer"""
        try:
            self.regional_fig.clear()
            gs = self.regional_fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)
            
            # 1. En pahalı 15 ilçe - horizontal bar
            ax1 = self.regional_fig.add_subplot(gs[0, :])
            district_avg = self.df.groupby('ilce')['fiyat'].mean().sort_values(ascending=True).tail(15)
            colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(district_avg)))
            bars = ax1.barh(range(len(district_avg)), district_avg.values, color=colors)
            ax1.set_yticks(range(len(district_avg)))
            ax1.set_yticklabels(district_avg.index)
            ax1.set_title('🏆 En Pahalı 15 İlçe (Ortalama Fiyat)', fontweight='bold', fontsize=14)
            ax1.set_xlabel('Ortalama Fiyat (TL)')
            
            # Değerleri barların sonuna yazdır
            for i, (bar, value) in enumerate(zip(bars, district_avg.values)):
                ax1.text(value + value*0.01, bar.get_y() + bar.get_height()/2,
                        f'{value:,.0f}', ha='left', va='center', fontsize=9)
            
            # 2. İlçe bazında toplam emlak sayısı
            ax2 = self.regional_fig.add_subplot(gs[1, 0])
            district_count = self.df['ilce'].value_counts().head(10)
            district_count.plot(kind='bar', ax=ax2, color='lightblue', alpha=0.8)
            ax2.set_title('📊 En Çok Emlak Bulunan İlçeler', fontweight='bold')
            ax2.set_xlabel('İlçe')
            ax2.set_ylabel('Emlak Sayısı')
            ax2.tick_params(axis='x', rotation=45)
            
            # 3. İlçe bazında fiyat volatilitesi (std dev)
            ax3 = self.regional_fig.add_subplot(gs[1, 1])
            district_volatility = self.df.groupby('ilce')['fiyat'].std().sort_values(ascending=False).head(10)
            district_volatility.plot(kind='bar', ax=ax3, color='orange', alpha=0.8)
            ax3.set_title('📈 En Volatil 10 İlçe (Fiyat Std. Sapması)', fontweight='bold')
            ax3.set_xlabel('İlçe')
            ax3.set_ylabel('Standart Sapma (TL)')
            ax3.tick_params(axis='x', rotation=45)
            
            # 4. İlçe bazında minimum-maksimum fiyat aralığı
            ax4 = self.regional_fig.add_subplot(gs[2, 0])
            district_range = self.df.groupby('ilce')['fiyat'].agg(['min', 'max']).head(10)
            district_range['range'] = district_range['max'] - district_range['min']
            district_range = district_range.sort_values('range', ascending=False)
            
            x_pos = range(len(district_range))
            ax4.bar(x_pos, district_range['max'], alpha=0.7, label='Maksimum', color='red')
            ax4.bar(x_pos, district_range['min'], alpha=0.7, label='Minimum', color='green')
            ax4.set_title('💹 İlçe Fiyat Aralıkları (Min-Max)', fontweight='bold')
            ax4.set_xlabel('İlçe')
            ax4.set_ylabel('Fiyat (TL)')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(district_range.index, rotation=45)
            ax4.legend()
            
            # 5. Mahalle yoğunluk analizi
            ax5 = self.regional_fig.add_subplot(gs[2, 1])
            neighborhood_density = self.df.groupby('ilce')['mahalle'].nunique().sort_values(ascending=False).head(10)
            neighborhood_density.plot(kind='bar', ax=ax5, color='purple', alpha=0.8)
            ax5.set_title('🏘️ En Çok Mahalleye Sahip İlçeler', fontweight='bold')
            ax5.set_xlabel('İlçe')
            ax5.set_ylabel('Mahalle Sayısı')
            ax5.tick_params(axis='x', rotation=45)
            
            # Layout düzenlemesi
            self.regional_fig.tight_layout(pad=1.5)
            self.regional_canvas.draw()
            
        except Exception as e:
            print(f"Bölgesel analiz grafiği hatası: {e}")
    
    def plot_value_analysis(self):
        """Değer analizi grafiklerini çizer"""
        try:
            self.value_fig.clear()
            gs = self.value_fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)
            
            # 1. Fiyat/Metrekare efficiency scatter
            ax1 = self.value_fig.add_subplot(gs[0, 0])
            sample_df = self.df.sample(min(2000, len(self.df)))
            efficiency = sample_df['fiyat'] / (sample_df['metrekare'] * sample_df['oda_sayisi'])
            ax1.scatter(sample_df['metrekare'], efficiency, alpha=0.6, c=sample_df['yas'], cmap='coolwarm')
            ax1.set_title('💡 Fiyat Verimliliği Analizi', fontweight='bold')
            ax1.set_xlabel('Metrekare')
            ax1.set_ylabel('Fiyat/(m² × Oda)')
            
            # 2. En değerli mahalleler (m² fiyatına göre)
            ax2 = self.value_fig.add_subplot(gs[0, 1])
            neighborhood_value = self.df.groupby('mahalle').agg({
                'fiyat': 'mean',
                'metrekare': 'mean'
            })
            neighborhood_value['price_per_m2'] = neighborhood_value['fiyat'] / neighborhood_value['metrekare']
            top_neighborhoods = neighborhood_value.sort_values('price_per_m2', ascending=False).head(15)
            
            bars = ax2.barh(range(len(top_neighborhoods)), top_neighborhoods['price_per_m2'].values, 
                          color=plt.cm.Reds(np.linspace(0.4, 0.9, len(top_neighborhoods))))
            ax2.set_yticks(range(len(top_neighborhoods)))
            ax2.set_yticklabels(top_neighborhoods.index, fontsize=8)
            ax2.set_title('🏆 En Değerli 15 Mahalle (TL/m²)', fontweight='bold')
            ax2.set_xlabel('TL/m²')
            
            # 3. Yaş-Fiyat optimizasyon analizi
            ax3 = self.value_fig.add_subplot(gs[1, 0])
            age_value = self.df.groupby('yas').agg({
                'fiyat': ['mean', 'count']
            }).reset_index()
            age_value.columns = ['yas', 'avg_price', 'count']
            age_value = age_value[age_value['count'] >= 10]  # En az 10 örnek olan yaşlar
            
            ax3.scatter(age_value['yas'], age_value['avg_price'], s=age_value['count']*2, 
                       alpha=0.6, c=age_value['avg_price'], cmap='viridis')
            ax3.set_title('⏰ Yaş-Fiyat Optimizasyon Haritası', fontweight='bold')
            ax3.set_xlabel('Bina Yaşı')
            ax3.set_ylabel('Ortalama Fiyat (TL)')
            
            # 4. ROI potansiyel analizi (fiyat/oda oranı)
            ax4 = self.value_fig.add_subplot(gs[1, 1])
            roi_analysis = self.df.groupby(['oda_sayisi', 'yas']).agg({
                'fiyat': 'mean'
            }).reset_index()
            
            pivot_roi = roi_analysis.pivot(index='yas', columns='oda_sayisi', values='fiyat')
            sns.heatmap(pivot_roi.iloc[:30, :6], ax=ax4, cmap='RdYlBu_r', 
                       annot=False, fmt='.0f', cbar_kws={'label': 'Ortalama Fiyat (TL)'})
            ax4.set_title('🎯 ROI Isı Haritası (Yaş × Oda)', fontweight='bold')
            ax4.set_xlabel('Oda Sayısı')
            ax4.set_ylabel('Bina Yaşı')
            
            # 5. Lüks segment analizi
            ax5 = self.value_fig.add_subplot(gs[2, 0])
            luxury_threshold = self.df['fiyat'].quantile(0.9)  # Top 10%
            luxury_df = self.df[self.df['fiyat'] >= luxury_threshold]
            
            luxury_features = luxury_df.groupby('ilce').agg({
                'metrekare': 'mean',
                'oda_sayisi': 'mean',
                'fiyat': 'mean'
            }).sort_values('fiyat', ascending=False).head(8)
            
            x = range(len(luxury_features))
            width = 0.25
            
            ax5.bar([i - width for i in x], luxury_features['metrekare'], width, label='Ortalama m²', alpha=0.8)
            ax5.bar(x, luxury_features['oda_sayisi']*20, width, label='Oda Sayısı×20', alpha=0.8)
            ax5.bar([i + width for i in x], luxury_features['fiyat']/100000, width, label='Fiyat÷100K', alpha=0.8)
            
            ax5.set_title('💎 Lüks Segment Analizi (Top 10%)', fontweight='bold')
            ax5.set_xlabel('İlçe')
            ax5.set_xticks(x)
            ax5.set_xticklabels(luxury_features.index, rotation=45)
            ax5.legend()
            
            # 6. Değer-Büyüklük matrisi
            ax6 = self.value_fig.add_subplot(gs[2, 1])
            value_size_matrix = self.df.groupby(['ilce']).agg({
                'fiyat': ['mean', 'count'],
                'metrekare': 'mean'
            }).reset_index()
            value_size_matrix.columns = ['ilce', 'avg_price', 'count', 'avg_size']
            value_size_matrix = value_size_matrix[value_size_matrix['count'] >= 20].head(15)
            
            bubble = ax6.scatter(value_size_matrix['avg_size'], value_size_matrix['avg_price'],
                               s=value_size_matrix['count'], alpha=0.6, c=value_size_matrix['avg_price'],
                               cmap='plasma')
            ax6.set_title('🎪 Değer-Büyüklük Matrisi', fontweight='bold')
            ax6.set_xlabel('Ortalama Metrekare')
            ax6.set_ylabel('Ortalama Fiyat (TL)')
            
            # En önemli noktaları etiketle
            for i, row in value_size_matrix.head(5).iterrows():
                ax6.annotate(row['ilce'], (row['avg_size'], row['avg_price']), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # Layout düzenlemesi
            self.value_fig.tight_layout(pad=1.5)
            self.value_canvas.draw()
            
        except Exception as e:
            print(f"Değer analizi grafiği hatası: {e}")
    
    def plot_statistical_analysis(self):
        """İstatistiksel analiz grafiklerini çizer"""
        if self.df is None:
            return
            
        # Dağılım Analizi
        self.plot_distribution_analysis()
        
        # Detaylı Korelasyon
        self.plot_detailed_correlation()
        
        # Outlier Analizi
        self.plot_outlier_analysis()
    
    def plot_distribution_analysis(self):
        """Dağılım analizi grafiklerini çizer"""
        try:
            self.distribution_fig.clear()
            gs = self.distribution_fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)
            
            # 1. Fiyat dağılımı - Normal dağılıma göre
            ax1 = self.distribution_fig.add_subplot(gs[0, 0])
            from scipy import stats
            
            # Log normal dağılım kontrolü
            log_prices = np.log(self.df['fiyat'])
            ax1.hist(log_prices, bins=50, alpha=0.7, density=True, color='lightblue', label='Gerçek Dağılım')
            
            # Normal dağılım fit
            mu, sigma = stats.norm.fit(log_prices)
            x = np.linspace(log_prices.min(), log_prices.max(), 100)
            normal_fit = stats.norm.pdf(x, mu, sigma)
            ax1.plot(x, normal_fit, 'r-', linewidth=2, label=f'Normal Fit (μ={mu:.2f}, σ={sigma:.2f})')
            
            ax1.set_title('📊 Log-Fiyat Dağılımı vs Normal', fontweight='bold')
            ax1.set_xlabel('Log(Fiyat)')
            ax1.set_ylabel('Yoğunluk')
            ax1.legend()
            
            # 2. Q-Q Plot
            ax2 = self.distribution_fig.add_subplot(gs[0, 1])
            stats.probplot(log_prices, dist="norm", plot=ax2)
            ax2.set_title('📈 Q-Q Plot (Normallik Testi)', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # 3. Metrekare dağılımı analizi
            ax3 = self.distribution_fig.add_subplot(gs[1, 0])
            metrekare_data = self.df['metrekare']
            
            # Histogram ve KDE
            ax3.hist(metrekare_data, bins=50, alpha=0.7, density=True, color='lightgreen')
            
            # KDE curve
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(metrekare_data)
            x_kde = np.linspace(metrekare_data.min(), metrekare_data.max(), 100)
            ax3.plot(x_kde, kde(x_kde), 'r-', linewidth=2, label='KDE')
            
            ax3.axvline(metrekare_data.mean(), color='blue', linestyle='--', label=f'Ortalama: {metrekare_data.mean():.0f}')
            ax3.axvline(metrekare_data.median(), color='green', linestyle='--', label=f'Medyan: {metrekare_data.median():.0f}')
            ax3.set_title('🏠 Metrekare Dağılımı', fontweight='bold')
            ax3.set_xlabel('Metrekare')
            ax3.set_ylabel('Yoğunluk')
            ax3.legend()
            
            # 4. Çok değişkenli dağılım - Yaş vs Fiyat
            ax4 = self.distribution_fig.add_subplot(gs[1, 1])
            sample_df = self.df.sample(min(2000, len(self.df)))
            
            # 2D histogram
            hist, xedges, yedges = np.histogram2d(sample_df['yas'], sample_df['fiyat'], bins=30)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            
            im = ax4.imshow(hist.T, extent=extent, origin='lower', cmap='Blues', aspect='auto')
            ax4.set_title('🎯 2D Dağılım: Yaş vs Fiyat', fontweight='bold')
            ax4.set_xlabel('Bina Yaşı')
            ax4.set_ylabel('Fiyat (TL)')
            self.distribution_fig.colorbar(im, ax=ax4, label='Frekans')
            
            # 5. Oda sayısı dağılımı (discrete)
            ax5 = self.distribution_fig.add_subplot(gs[2, 0])
            room_dist = self.df['oda_sayisi'].value_counts().sort_index()
            
            # Bar plot
            bars = ax5.bar(room_dist.index, room_dist.values, alpha=0.8, color='orange')
            
            # Percentages on bars
            total = room_dist.sum()
            for bar, count in zip(bars, room_dist.values):
                percentage = count / total * 100
                ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + bar.get_height()*0.01,
                        f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            ax5.set_title('🚪 Oda Sayısı Dağılımı', fontweight='bold')
            ax5.set_xlabel('Oda Sayısı')
            ax5.set_ylabel('Frekans')
            
            # 6. Kat dağılımı analizi
            ax6 = self.distribution_fig.add_subplot(gs[2, 1])
            
            # Kat bilgilerini grupla
            floor_data = self.df['bulundugu_kat']
            floor_bins = [-2, 0, 5, 10, 15, 25, 50]
            floor_labels = ['Bodrum', '1-5', '6-10', '11-15', '16-25', '25+']
            floor_groups = pd.cut(floor_data, bins=floor_bins, labels=floor_labels)
            floor_dist = floor_groups.value_counts()
            
            # Donut chart
            wedges, texts, autotexts = ax6.pie(floor_dist.values, labels=floor_dist.index, 
                                              autopct='%1.1f%%', startangle=90,
                                              colors=plt.cm.Set3(range(len(floor_dist))))
            
            # Hollow center for donut
            centre_circle = plt.Circle((0,0), 0.70, fc='white')
            ax6.add_artist(centre_circle)
            ax6.set_title('🏢 Kat Dağılımı', fontweight='bold')
            
            # Layout düzenlemesi
            self.distribution_fig.tight_layout(pad=1.5)
            self.distribution_canvas.draw()
            
        except Exception as e:
            print(f"Dağılım analizi grafiği hatası: {e}")
    
    def plot_detailed_correlation(self):
        """Detaylı korelasyon analizi grafiklerini çizer"""
        try:
            self.correlation_detail_fig.clear()
            gs = self.correlation_detail_fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
            
            # Numeric columns for correlation
            numeric_cols = ['fiyat', 'metrekare', 'oda_sayisi', 'yas', 'bulundugu_kat']
            correlation_matrix = self.df[numeric_cols].corr()
            
            # 1. Enhanced correlation heatmap
            ax1 = self.correlation_detail_fig.add_subplot(gs[0, :])
            
            # Triangular heatmap
            mask = np.triu(np.ones_like(correlation_matrix))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                       square=True, ax=ax1, cbar_kws={'shrink': 0.8, 'label': 'Korelasyon Katsayısı'})
            ax1.set_title('🔗 Gelişmiş Korelasyon Matrisi', fontweight='bold', fontsize=14)
            
            # 2. Scatter matrix (pairwise relationships)
            ax2 = self.correlation_detail_fig.add_subplot(gs[1, 0])
            
            # Sample for performance
            sample_df = self.df[numeric_cols].sample(min(1000, len(self.df)))
            
            # Focus on strongest correlations
            strongest_corr_pairs = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr_val = abs(correlation_matrix.iloc[i, j])
                    if corr_val > 0.3:  # Only strong correlations
                        strongest_corr_pairs.append((numeric_cols[i], numeric_cols[j], corr_val))
            
            if strongest_corr_pairs:
                # Plot strongest correlation
                strongest_pair = max(strongest_corr_pairs, key=lambda x: x[2])
                x_col, y_col, corr_val = strongest_pair
                
                ax2.scatter(sample_df[x_col], sample_df[y_col], alpha=0.6, s=20)
                
                # Add regression line
                z = np.polyfit(sample_df[x_col], sample_df[y_col], 1)
                p = np.poly1d(z)
                ax2.plot(sample_df[x_col].sort_values(), p(sample_df[x_col].sort_values()), "r--", linewidth=2)
                
                ax2.set_title(f'💫 En Güçlü İlişki: {x_col} vs {y_col}\n(r={corr_val:.3f})', fontweight='bold')
                ax2.set_xlabel(x_col)
                ax2.set_ylabel(y_col)
                ax2.grid(True, alpha=0.3)
            
            # 3. Correlation strength distribution
            ax3 = self.correlation_detail_fig.add_subplot(gs[1, 1])
            
            # Get all correlation values (excluding diagonal)
            corr_values = []
            for i in range(len(correlation_matrix)):
                for j in range(i+1, len(correlation_matrix)):
                    corr_values.append(abs(correlation_matrix.iloc[i, j]))
            
            # Histogram of correlation strengths
            ax3.hist(corr_values, bins=20, alpha=0.7, color='lightcoral', edgecolor='darkred')
            ax3.axvline(np.mean(corr_values), color='blue', linestyle='--', linewidth=2, 
                       label=f'Ortalama: {np.mean(corr_values):.3f}')
            ax3.set_title('📊 Korelasyon Gücü Dağılımı', fontweight='bold')
            ax3.set_xlabel('|Korelasyon Katsayısı|')
            ax3.set_ylabel('Frekans')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Layout düzenlemesi
            self.correlation_detail_fig.tight_layout(pad=1.5)
            self.correlation_detail_canvas.draw()
            
        except Exception as e:
            print(f"Detaylı korelasyon grafiği hatası: {e}")
    
    def plot_outlier_analysis(self):
        """Outlier analizi grafiklerini çizer"""
        try:
            from scipy import stats
            
            self.outlier_fig.clear()
            gs = self.outlier_fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)
            
            # 1. Fiyat outliers - Box plot
            ax1 = self.outlier_fig.add_subplot(gs[0, 0])
            
            # Box plot with outlier detection
            bp = ax1.boxplot(self.df['fiyat'], vert=True, patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][0].set_alpha(0.7)
            
            # Calculate outliers manually
            Q1 = self.df['fiyat'].quantile(0.25)
            Q3 = self.df['fiyat'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df['fiyat'] < lower_bound) | (self.df['fiyat'] > upper_bound)]
            outlier_count = len(outliers)
            outlier_percentage = outlier_count / len(self.df) * 100
            
            ax1.set_title(f'📦 Fiyat Outliers\n({outlier_count} adet, %{outlier_percentage:.1f})', fontweight='bold')
            ax1.set_ylabel('Fiyat (TL)')
            ax1.grid(True, alpha=0.3)
            
            # 2. Metrekare outliers
            ax2 = self.outlier_fig.add_subplot(gs[0, 1])
            
            bp2 = ax2.boxplot(self.df['metrekare'], vert=True, patch_artist=True)
            bp2['boxes'][0].set_facecolor('lightgreen')
            bp2['boxes'][0].set_alpha(0.7)
            
            # Metrekare outliers
            Q1_m = self.df['metrekare'].quantile(0.25)
            Q3_m = self.df['metrekare'].quantile(0.75)
            IQR_m = Q3_m - Q1_m
            outliers_m = self.df[(self.df['metrekare'] < Q1_m - 1.5 * IQR_m) | 
                               (self.df['metrekare'] > Q3_m + 1.5 * IQR_m)]
            
            ax2.set_title(f'📏 Metrekare Outliers\n({len(outliers_m)} adet, %{len(outliers_m)/len(self.df)*100:.1f})', 
                         fontweight='bold')
            ax2.set_ylabel('Metrekare')
            ax2.grid(True, alpha=0.3)
            
            # 3. Multivariate outliers - Isolation Forest
            ax3 = self.outlier_fig.add_subplot(gs[1, 0])
            
            try:
                from sklearn.ensemble import IsolationForest
                
                # Features for outlier detection
                features_for_outliers = self.df[['fiyat', 'metrekare', 'oda_sayisi', 'yas']].copy()
                
                # Normalize features
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features_for_outliers)
                
                # Isolation Forest
                iso_forest = IsolationForest(contamination=0.05, random_state=42)
                outlier_labels = iso_forest.fit_predict(features_scaled)
                
                # Plot results
                outlier_indices = outlier_labels == -1
                normal_indices = outlier_labels == 1
                
                ax3.scatter(features_for_outliers.loc[normal_indices, 'metrekare'], 
                          features_for_outliers.loc[normal_indices, 'fiyat'],
                          alpha=0.6, s=20, c='blue', label='Normal')
                ax3.scatter(features_for_outliers.loc[outlier_indices, 'metrekare'], 
                          features_for_outliers.loc[outlier_indices, 'fiyat'],
                          alpha=0.8, s=40, c='red', label='Outlier', marker='x')
                
                ax3.set_title(f'🎯 Çok Değişkenli Outliers\n({np.sum(outlier_indices)} adet)', fontweight='bold')
                ax3.set_xlabel('Metrekare')
                ax3.set_ylabel('Fiyat (TL)')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                
            except ImportError:
                ax3.text(0.5, 0.5, 'Isolation Forest\nkütüphanesi bulunamadı', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('🎯 Çok Değişkenli Outliers', fontweight='bold')
            
            # 4. Price per m² outliers
            ax4 = self.outlier_fig.add_subplot(gs[1, 1])
            
            price_per_m2 = self.df['fiyat'] / self.df['metrekare']
            
            # Z-score based outlier detection
            z_scores = np.abs(stats.zscore(price_per_m2))
            threshold = 3
            z_outliers = z_scores > threshold
            
            ax4.scatter(range(len(price_per_m2)), price_per_m2, alpha=0.6, s=10, c='blue', label='Normal')
            ax4.scatter(np.where(z_outliers)[0], price_per_m2[z_outliers], 
                       alpha=0.8, s=30, c='red', label='Z-score Outlier', marker='^')
            
            ax4.axhline(price_per_m2.mean() + 3*price_per_m2.std(), color='red', linestyle='--', alpha=0.7)
            ax4.axhline(price_per_m2.mean() - 3*price_per_m2.std(), color='red', linestyle='--', alpha=0.7)
            
            ax4.set_title(f'💰 m² Fiyat Outliers (Z-score)\n({np.sum(z_outliers)} adet)', fontweight='bold')
            ax4.set_xlabel('Index')
            ax4.set_ylabel('TL/m²')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # 5. Outlier summary by district
            ax5 = self.outlier_fig.add_subplot(gs[2, 0])
            
            # Count outliers by district
            outlier_districts = outliers.groupby('ilce').size().sort_values(ascending=False).head(10)
            
            bars = ax5.bar(range(len(outlier_districts)), outlier_districts.values, 
                          color='lightcoral', alpha=0.8)
            ax5.set_title('🗺️ İlçe Bazında Fiyat Outliers', fontweight='bold')
            ax5.set_xlabel('İlçe')
            ax5.set_ylabel('Outlier Sayısı')
            ax5.set_xticks(range(len(outlier_districts)))
            ax5.set_xticklabels(outlier_districts.index, rotation=45)
            
            # Add count labels
            for bar, count in zip(bars, outlier_districts.values):
                ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                        str(count), ha='center', va='bottom', fontweight='bold')
            
            # 6. Outlier detection methods comparison
            ax6 = self.outlier_fig.add_subplot(gs[2, 1])
            
            # Different outlier detection methods
            methods = ['IQR\n(Box Plot)', 'Z-Score\n(σ>3)', 'Modified Z-Score\n(MAD)', 'Percentile\n(95-99%)']
            
            # IQR method
            iqr_outliers = len(outliers)
            
            # Z-score method
            z_outliers_count = np.sum(np.abs(stats.zscore(self.df['fiyat'])) > 3)
            
            # Modified Z-score (using MAD)
            median = np.median(self.df['fiyat'])
            mad = np.median(np.abs(self.df['fiyat'] - median))
            modified_z_scores = 0.6745 * (self.df['fiyat'] - median) / mad
            mad_outliers_count = np.sum(np.abs(modified_z_scores) > 3.5)
            
            # Percentile method
            p1, p99 = np.percentile(self.df['fiyat'], [1, 99])
            percentile_outliers = len(self.df[(self.df['fiyat'] < p1) | (self.df['fiyat'] > p99)])
            
            counts = [iqr_outliers, z_outliers_count, mad_outliers_count, percentile_outliers]
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
            
            bars = ax6.bar(methods, counts, color=colors, alpha=0.8)
            ax6.set_title('📊 Outlier Yöntemleri Karşılaştırması', fontweight='bold')
            ax6.set_ylabel('Outlier Sayısı')
            
            # Add count labels
            for bar, count in zip(bars, counts):
                ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(counts)*0.01,
                        str(count), ha='center', va='bottom', fontweight='bold')
            
            # Layout düzenlemesi
            self.outlier_fig.tight_layout(pad=1.5)
            self.outlier_canvas.draw()
            
        except Exception as e:
            print(f"Outlier analizi grafiği hatası: {e}")
    

    
    def load_random_sample(self):
        """Veri setinden rastgele bir örnek seçer ve form alanlarını doldurur"""
        try:
            # Veri seti yüklü değilse yükle
            if self.df is None:
                print("📊 Veri seti yükleniyor...")
                self.df = load_and_preprocess_data()
            
            if self.df is not None and not self.df.empty:
                # Rastgele bir konut seç
                random_index = random.randint(0, len(self.df) - 1)
                random_sample = self.df.iloc[random_index]
                
                # Form alanlarını doldur
                self.input_fields['metrekare'].setValue(int(random_sample['metrekare']))
                self.input_fields['oda_sayisi'].setValue(int(random_sample['oda_sayisi']))
                self.input_fields['yas'].setValue(int(random_sample['yas']))
                self.input_fields['bulundugu_kat'].setValue(int(random_sample['bulundugu_kat']))
                
                # İlçe ve mahalle seçimini ayarla
                ilce = random_sample['ilce']
                mahalle = random_sample['mahalle']
                
                # İlçeyi seç
                ilce_index = self.input_fields['ilce'].findText(ilce)
                if ilce_index >= 0:
                    self.input_fields['ilce'].setCurrentIndex(ilce_index)
                
                # Mahalle listesi güncellendikten sonra mahalleyi seç
                QApplication.processEvents()  # UI güncellemesini bekle
                mahalle_index = self.input_fields['mahalle'].findText(mahalle)
                if mahalle_index >= 0:
                    self.input_fields['mahalle'].setCurrentIndex(mahalle_index)
                
                # Gerçek fiyatı sakla
                self.current_real_price = random_sample['fiyat']
                
                # Kullanıcıya bilgi ver
                QMessageBox.information(self, "Örnek Seçildi", 
                                     f"Rastgele bir konut örneği seçildi.\n"
                                     f"Gerçek fiyat: {self.current_real_price:,.0f} TL\n\n"
                                     f"Şimdi 'Fiyat Tahmini Yap' butonuna tıklayarak modelin tahminini görebilirsiniz.")
            else:
                QMessageBox.warning(self, "Veri Bulunamadı", "Veri seti yüklenemedi veya boş.")
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Örnek seçilirken bir hata oluştu: {e}")
            
    def predict(self):
        """Kullanıcının girdiği değerlere göre makine öğrenmesi ile fiyat tahmini yapar"""
        try:
            # Form alanlarından kullanıcının girdiği değerleri topla
            features = {}  # Boş sözlük oluştur
            # Her input alanını gez ve değeri al
            for key, widget in self.input_fields.items():
                if isinstance(widget, QComboBox):  # Dropdown menü ise
                    features[key] = widget.currentText()  # Seçili metni al
                elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):  # Sayı kutusu ise
                    features[key] = widget.value()  # Sayısal değeri al
                elif isinstance(widget, QLineEdit):  # Metin kutusu ise
                    features[key] = widget.text()  # Metni al
            
            # Model.py'deki predict_price fonksiyonunu çağır
            result = predict_price(features)  # Makine öğrenmesi ile tahmin yap
            
            if result is not None:
                # Sonuçları göster
                prediction = result['prediction']
                lower_bound = result.get('lower_bound', prediction * 0.9)
                upper_bound = result.get('upper_bound', prediction * 1.1)
                reliability = result.get('reliability', 'Orta')
                
                # Ana sonuç etiketi
                formatted_price = f"{prediction:,.0f} TL"
                confidence_range = f"{lower_bound:,.0f} - {upper_bound:,.0f} TL"
                
                # Eğer gerçek fiyat varsa karşılaştırma yap
                comparison_text = ""
                accuracy_info = ""
                if hasattr(self, 'current_real_price') and self.current_real_price:
                    real_price = self.current_real_price
                    formatted_real_price = f"{real_price:,.0f} TL"
                    
                    # Hata yüzdesi hesapla
                    error_percent = abs(prediction - real_price) / real_price * 100
                    error_amount = abs(prediction - real_price)
                    
                    comparison_text = f"\n\n🎯 Gerçek Fiyat: {formatted_real_price}"
                    comparison_text += f"\n📊 Tahmin Hatası: {error_amount:,.0f} TL (%{error_percent:.1f})"
                    
                    if error_percent < 10:
                        accuracy_info = "🎉 Mükemmel tahmin!"
                    elif error_percent < 20:
                        accuracy_info = "✅ İyi tahmin!"
                    elif error_percent < 30:
                        accuracy_info = "⚠️ Orta tahmin"
                    else:
                        accuracy_info = "❌ Zayıf tahmin"
                
                main_text = f"🏠 Tahmini Fiyat: {formatted_price}"
                # Güvenilirlik gösterilmiyor ama hesaplanıyor
                main_text += comparison_text
                if accuracy_info:
                    main_text += f"\n{accuracy_info}"
                
                self.result_label.setText(main_text)
                
                # Güven aralığı ve detaylar
                detail_text = f"📏 Güven Aralığı: {confidence_range}\n"
                detail_text += f"🏘️ İlçe: {features['ilce']}\n"
                detail_text += f"🏠 Mahalle: {features['mahalle']}\n"
                detail_text += f"📐 Metrekare: {features['metrekare']} m²\n"
                detail_text += f"🚪 Oda Sayısı: {features['oda_sayisi']}\n"
                detail_text += f"📅 Bina Yaşı: {features['yas']} yıl\n"
                detail_text += f"🏢 Kat: {features['bulundugu_kat']}\n"
                
                # M² başına fiyat
                price_per_m2 = prediction / features['metrekare']
                detail_text += f"💰 m² Fiyatı: {price_per_m2:,.0f} TL/m²"
                
                self.confidence_label.setText(detail_text)
                
                # Karşılaştırma grafiği çiz
                self.plot_prediction_comparison(features, prediction, lower_bound, upper_bound)
                
                # Başarılı tahmin stili
                self.result_label.setStyleSheet("""
                    QLabel {
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                                  stop:0 #d4edda, stop:1 #c3e6cb);
                        color: #155724;
                        padding: 20px;
                        border-radius: 15px;
                        border: 2px solid #28a745;
                        font-weight: bold;
                    }
                """)
            else:
                self.result_label.setText("❌ Tahmin yapılamadı!\nLütfen tüm alanları doldurun ve tekrar deneyin.")
                self.result_label.setStyleSheet("""
                    QLabel {
                        background-color: #f8d7da;
                        color: #721c24;
                        padding: 20px;
                        border-radius: 15px;
                        border: 2px solid #dc3545;
                    }
                """)
                self.confidence_label.setText("")
        except Exception as e:
            self.result_label.setText(f"❌ Tahmin hatası: {e}")
            self.result_label.setStyleSheet("""
                QLabel {
                    background-color: #f8d7da;
                    color: #721c24;
                    padding: 20px;
                    border-radius: 15px;
                    border: 2px solid #dc3545;
                }
            """)
            self.confidence_label.setText("")
    
    def plot_prediction_comparison(self, features, prediction, lower_bound, upper_bound):
        """Tahmin karşılaştırma grafiği çizer"""
        try:
            self.comparison_fig.clear()
            gs = self.comparison_fig.add_gridspec(2, 2, hspace=0.5, wspace=0.25)
            
            # 1. Tahmin vs. Benzer konutlar
            ax1 = self.comparison_fig.add_subplot(gs[0, 0])
            
            # Benzer konutları bul
            similar_houses = self.df[
                (self.df['ilce'] == features['ilce']) &
                (abs(self.df['metrekare'] - features['metrekare']) <= 20) &
                (abs(self.df['oda_sayisi'] - features['oda_sayisi']) <= 1)
            ]['fiyat']
            
            if len(similar_houses) > 0:
                ax1.hist(similar_houses, bins=20, alpha=0.7, color='lightblue', label='Benzer Konutlar')
                ax1.axvline(prediction, color='red', linestyle='--', linewidth=2, label='Tahmininiz')
                ax1.axvspan(lower_bound, upper_bound, alpha=0.3, color='red', label='Güven Aralığı')
                ax1.legend()
                ax1.set_title('Benzer Konutlarla Karşılaştırma', fontweight='bold', fontsize=10)
                ax1.set_xlabel('Fiyat (TL)')
                ax1.set_ylabel('Frekans')
            
            # 2. İlçe ortalaması ile karşılaştırma
            ax2 = self.comparison_fig.add_subplot(gs[0, 1])
            if self.district_stats and features['ilce'] in self.district_stats:
                district_avg = self.district_stats[features['ilce']]['mean']
                district_median = self.district_stats[features['ilce']]['median']
                
                categories = ['İlçe\nOrtalama', 'İlçe\nMedian', 'Tahmininiz']
                values = [district_avg, district_median, prediction]
                colors = ['lightcoral', 'lightgreen', 'gold']
                
                bars = ax2.bar(categories, values, color=colors, alpha=0.8)
                ax2.set_title(f'{features["ilce"]} İlçesi Karşılaştırması', fontweight='bold', fontsize=10)
                ax2.set_ylabel('Fiyat (TL)')
                
                # Değerleri barların üzerine yazdır
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:,.0f}',
                           ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            # 3. M² fiyatı karşılaştırması
            ax3 = self.comparison_fig.add_subplot(gs[1, 0])
            user_price_per_m2 = prediction / features['metrekare']
            
            # İlçedeki m² fiyatlarını al
            district_m2_prices = (self.df[self.df['ilce'] == features['ilce']]['fiyat'] / 
                                 self.df[self.df['ilce'] == features['ilce']]['metrekare'])
            
            if len(district_m2_prices) > 0:
                ax3.hist(district_m2_prices, bins=15, alpha=0.7, color='lightyellow', label='İlçe m² Fiyatları')
                ax3.axvline(user_price_per_m2, color='purple', linestyle='--', linewidth=2, label='Tahmininiz')
                ax3.legend()
                ax3.set_title('m² Fiyatı Karşılaştırması', fontweight='bold', fontsize=10)
                ax3.set_xlabel('TL/m²')
                ax3.set_ylabel('Frekans')
            
            # 4. Özellik tablosu
            ax4 = self.comparison_fig.add_subplot(gs[1, 1])
            ax4.axis('off')
            
            # Tablo verileri
            table_data = [
                ['Özellik', 'Değer', 'İlçe Ort.'],
                ['Metrekare', f"{features['metrekare']} m²", f"{self.df[self.df['ilce']==features['ilce']]['metrekare'].mean():.0f} m²"],
                ['Oda Sayısı', f"{features['oda_sayisi']}", f"{self.df[self.df['ilce']==features['ilce']]['oda_sayisi'].mean():.1f}"],
                ['Bina Yaşı', f"{features['yas']} yıl", f"{self.df[self.df['ilce']==features['ilce']]['yas'].mean():.0f} yıl"],
                ['Kat', f"{features['bulundugu_kat']}", f"{self.df[self.df['ilce']==features['ilce']]['bulundugu_kat'].mean():.0f}"]
            ]
            
            table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                            colWidths=[0.4, 0.3, 0.3])
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(0.9, 1.1)
            
            # Başlık satırını renklendir
            for i in range(3):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Başlık eklendi - minimal tasarım
            ax4.set_title('Özellik Karşılaştırması', fontweight='bold', fontsize=8, pad=5)
            
            # Layout düzenlemesi
            self.comparison_fig.tight_layout(pad=2.0)
            self.comparison_canvas.draw()
            
        except Exception as e:
            print(f"Karşılaştırma grafiği hatası: {e}")
    

    



# Ana program çalıştırma bloğu (eğer bu dosya direkt çalıştırılırsa)
if __name__ == "__main__":
    # PyQt5 uygulaması oluştur (sistem argümanlarını parametre olarak ver)
    app = QApplication(sys.argv)
    # Ana pencere sınıfından bir nesne oluştur
    window = KonutFiyatTahmini()
    # Pencereyi ekranda göster
    window.show()
    
    # Eğitilmiş model dosyasının varlığını kontrol et
    if not os.path.exists('models/konut_fiyat_model.pkl'):
        # Model yoksa kullanıcıya bilgi ver
        QMessageBox.information(window, "Model Bulunamadı", 
                              "Eğitilmiş model bulunamadı. Model eğitimi için 'model.py' dosyasını çalıştırın.")
    
    # Uygulamayı başlat ve kapanana kadar çalıştır (event loop)
    sys.exit(app.exec_()) 