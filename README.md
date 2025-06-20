# 🏠 Istanbul Real Estate Price Prediction Project

This project is an advanced desktop application that predicts real estate prices in Istanbul using machine learning algorithms. It features a modern GUI interface developed with PyQt5 and comprehensive data analysis capabilities.

## 🎯 Project Features

- 🔮 **Smart Price Prediction**: Accurate price predictions using machine learning
- 📊 **Comprehensive Data Analysis**: Extensive statistical analysis and visualizations
- 🖥️ **Modern GUI**: User-friendly interface developed with PyQt5
- 📈 **Multi-Chart Support**: Interactive charts with Matplotlib
- 🗺️ **Regional Analysis**: Detailed analysis by districts and neighborhoods
- 🎯 **Feature Importance Analysis**: Understanding which factors affect prices and how
- 📋 **Comparative Analysis**: Compare your predictions with similar properties

## 🛠️ Technologies

- **Python 3.8+**
- **PyQt5** - GUI Framework
- **Pandas** - Data Processing
- **Scikit-learn** - Machine Learning
- **Matplotlib & Seaborn** - Data Visualization
- **NumPy** - Numerical Computations

## 📦 Installation

### Requirements

```bash
pip install -r requirements.txt
```

### Step-by-Step Installation

1. **Clone the repository:**
```bash
git clone https://github.com/omercemayla/Konut-Fiyat-Tahmini-Projesi.git
cd Konut-Fiyat-Tahmini-Projesi
```

2. **Create virtual environment (recommended):**
```bash
python -m venv konut_env
konut_env\Scripts\activate  # Windows
# source konut_env/bin/activate  # Linux/Mac
```

3. **Install required packages:**
```bash
pip install -r requirements.txt
```

4. **Dataset preparation:**
   - Place `istanbul_konut2.xlsx` file in the project folder
   - Data format: ilce, mahalle, metrekare, oda_sayisi, yas, bulundugu_kat, fiyat

## 🚀 Usage

### Running the Application

```bash
python app.py
```

### Model Training (First Use)

```bash
python model.py
```

## 📱 Application Interface

### Main Features:

#### 🔮 Price Prediction Tab
- Enter property features (square meters, room count, age, floor, location)
- Try "Random Sample" for test data
- Real-time price prediction with confidence intervals
- Comparison charts with similar properties

#### 🎯 Feature Importance
- See which factors most influence prices
- Category-based importance distribution
- Top 15 most important features analysis

#### 📈 Market Analysis
- **Price Trends**: Price level distributions and per square meter analysis
- **Regional Analysis**: District-based comparisons and volatility analysis
- **Value Analysis**: ROI potential and luxury segment analysis

#### 📊 Statistical Analysis
- **Distribution Analysis**: Normal distribution tests and Q-Q plots
- **Correlation Analysis**: Relationships between variables
- **Outlier Detection**: Identification of anomalous values

#### 📋 Data Analysis
- Dataset summary information
- District statistics
- Detailed data visualizations

## 🔧 Project Structure

```
istanbul-konut-fiyat-tahmini/
│
├── app.py                 # Main GUI application
├── model.py              # ML model training and prediction functions
├── requirements.txt      # Python package requirements
├── .gitignore           # Git ignore file
├── README.md            # This file
│
├── models/              # Trained model files
│   ├── konut_fiyat_model.pkl
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   └── ...
│
├── plots/               # Generated charts
│   ├── actual_vs_predicted.png
│   ├── feature_importance.png
│   └── ...
│
└── istanbul_konut2.xlsx # Dataset (added by user)
```

## 🧠 Machine Learning Model

### Algorithms Used:
- **Ensemble Methods**: Random Forest, Gradient Boosting
- **Linear Models**: Ridge Regression
- **Feature Engineering**: Category encoding, feature selection

### Model Performance:
- **R² Score**: ~0.85-0.90
- **RMSE**: Average 15-20% error rate
- **Cross-validation**: 5-fold validation

### Feature Engineering:
- Categorical variable encoding
- Feature selection and importance analysis
- Data normalization and scaling

## 📊 Dataset

### Data Characteristics:
- **Record Count**: 50,000+ property data
- **Time Range**: Current market data
- **Geographic Coverage**: All districts of Istanbul
- **Features**: Location, size, age, floor information

### Data Sources:
- Real estate portals
- Official records
- Market research

## 🤝 Contributing

1. Fork this repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Create Pull Request

### Contribution Areas:
- 🐛 Bug reports and fixes
- ✨ New feature suggestions
- 📖 Documentation improvements
- 🎨 UI/UX enhancements
- 🧮 Model performance optimizations

## 📝 To-Do List

- [ ] 🌐 Web interface development
- [ ] 📱 Mobile application version
- [ ] 🤖 Advanced ML algorithms (XGBoost, LightGBM)
- [ ] 📊 Real-time data integration
- [ ] 🗺️ Map visualizations
- [ ] 💾 Database integration
- [ ] 🔐 User authentication system
- [ ] 📈 Trend prediction model

## ⚠️ Important Notes

- This project is for educational purposes and should not be used alone for real investment decisions
- Price predictions may vary depending on market conditions
- Keeping the dataset up-to-date is important for model performance

## 📞 Contact

**Project Owner**: Ömer Cem Ayla
**GitHub**: [@omercemayla](https://github.com/omercemayla)  
**Email**: omer_cem3@hotmail.com

**Note**: For screenshots and detailed visuals, please check the `plots/` folder. 