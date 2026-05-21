# Stock Movement Prediction Dashboard 📈

An interactive Streamlit dashboard for visualizing and predicting stock movements using BERT embeddings and deep learning models (LSTM, GRU, Bidirectional).

## 🔬 Technical Approach

### Data Alignment Strategy

This project implements a robust data alignment pipeline to ensure temporal consistency between tweets and stock prices:

1. **Timestamp-to-Trading-Day Mapping**: Each tweet is aligned to its corresponding trading day
   - If tweet timestamp falls on a trading day → use that day
   - If tweet is on a non-trading day (weekend/holiday) → use the most recent previous trading day
   - This ensures tweets are paired with relevant stock price data

2. **Label Generation**: Stock movement labels are derived from actual price changes
   - Label = 1 if Close[t] > Close[t-1] (price increased)
   - Label = 0 if Close[t] ≤ Close[t-1] (price decreased or stayed same)

3. **Feature Engineering**: 
   - BERT embeddings (768 dimensions) capture tweet sentiment
   - Stock features (Open, High, Low, Close) provide price context
   - Combined features undergo proper normalization to prevent data leakage

### Model Input Shape Strategy

The models use a specific input shape strategy for compatibility:
- Raw features: 768 (BERT) + 4 (stock) = 772 dimensions
- After normalization: Skip first 2 features → 770 dimensions
- Reshape to (35 time steps, 22 features per step) for LSTM/GRU input
- This preserves compatibility with the trained model architecture

### Preventing Data Leakage

Key measures to ensure training/inference consistency:
- **Scaler fitted on training data only**: MinMaxScaler parameters are computed from training set
- **Scaler saved and reused**: The fitted scaler (model/scaler.pkl) is loaded during inference
- **No per-sample normalization**: Test/inference data uses the training distribution
- This ensures predictions use the same feature distribution as training

## 🚀 Features

### 1. **Home Page** 🏠
- Project overview and architecture
- Technology stack information
- System workflow diagram
- Quick dataset statistics

### 2. **Data Exploration** 📈
- Interactive stock price charts with zoom and pan
- Tweet samples with sentiment analysis
- Label distribution visualization
- Statistical summaries
- Feature correlation heatmaps
- Date range filtering

### 3. **Model Performance** 🤖
- Three model architectures comparison:
  - LSTM (Baseline)
  - LSTM + GRU (Hybrid)
  - Bidirectional LSTM + GRU (Best)
- Training history visualization (accuracy/loss curves)
- Performance metrics (accuracy, precision, recall, F1-score)
- Model comparison charts

### 4. **Live Prediction** 🔮
- Custom tweet input
- Stock price context selection
- Model selection (choose from 3 models)
- Real-time prediction with confidence scores
- Visual prediction result display
- Sample tweet templates

### 5. **Insights & Conclusion** 💡
- Key findings from the analysis
- Model comparison summary
- Potential applications
- Limitations and considerations
- Future enhancement suggestions

## 📋 Prerequisites

- Python 3.7 or higher
- pip package manager
- At least 4GB RAM (for loading models)

## 🔧 Installation

### Important Security Notice

⚠️ **Security Update**: The requirements.txt has been updated with secure versions of all dependencies to address multiple known vulnerabilities in the original versions:

- **TensorFlow**: Updated from 1.14.0 (2019) to 2.12.0+ (fixes 200+ vulnerabilities)
- **Keras**: Now included in TensorFlow 2.x (fixes path traversal, deserialization, code injection)
- **PyTorch**: Updated from 1.6.0 to 2.2.0+ (fixes RCE, heap overflow, use-after-free)
- **Transformers**: Updated from 3.4.0 to 4.48.0+ (fixes deserialization vulnerabilities)
- **nltk**: Updated from 3.4.5 to 3.9+ (fixes ReDoS vulnerabilities)
- **sentencepiece**: Updated from 0.1.96 to 0.2.1+ (fixes heap overflow)
- **scikit-learn**: Updated from 0.22.2 to 1.0.0+ (fixes deserialization issues)

**Note**: The pre-trained models in the `model/` directory were trained with older library versions. With the updated dependencies:
1. Models may need to be retrained using the notebook with new library versions
2. Alternatively, use TensorFlow's compatibility mode to load old models
3. For the dashboard to work with live predictions, ensure models are compatible

### Installation Steps

1. **Clone the repository** (if not already done):
```bash
git clone https://github.com/Pranavvv08/Stock-Movement.git
cd Stock-Movement
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

**Note**: The installation may take several minutes as it includes TensorFlow, PyTorch, and Sentence Transformers.

### Model Compatibility

If you encounter model loading errors with the updated libraries:

**Option 1: Retrain Models (Recommended)**
```bash
# Open and run the training notebook with new library versions
jupyter notebook StockMovement.ipynb
```

**Option 2: Use Compatibility Mode**
The dashboard will attempt to load models even if there are library mismatches, but functionality may be limited.

### Installation Tips

- **Python Version**: Requires Python 3.8-3.11
- **For Windows**: Use `python` instead of `python3` in commands
- **Virtual Environment** (recommended):
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  pip install -r requirements.txt
  ```

## 🎯 Running the Dashboard

### Prerequisites

Before running the dashboard, you must:

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the models**:
   ```bash
   python train.py
   ```
   
   This step is **required** and will:
   - Load and align tweet data to AAPL trading days
   - Compute BERT embeddings for tweets
   - Train three deep learning models (LSTM, LSTM+GRU, Bidirectional)
   - Save all models and the fitted scaler to prevent data leakage
   - Generate performance metrics
   
   **Training options:**
   - `--epochs N` - Number of training epochs (default: 50)
   - `--batch-size N` - Batch size for training (default: 32)
   - `--force-bert` - Force recomputation of BERT embeddings
   
   Example:
   ```bash
   python train.py --epochs 30 --batch-size 64
   ```

### Running the Dashboard

Once training is complete, launch the dashboard:

### Option 1: Using the batch file (Windows)
```bash
run_dashboard.bat
```

### Option 2: Using command line
```bash
streamlit run app.py
```

### Option 3: Using Python
```bash
python -m streamlit run app.py
```

The dashboard will automatically open in your default web browser at `http://localhost:8501`

## 📊 Data Files Required

The dashboard expects the following files:

### Dataset Files (in `Dataset/` directory):
- `AAPL.csv` - Apple stock historical data with columns: Date, Open, High, Low, Close, Adj Close, Volume, Label
- `tweets.csv` - Financial tweets with columns: Tweets, Label

### Model Files (in `model/` directory):

**Important:** Before running the dashboard, you must train the models first using the training pipeline:

```bash
python train.py
```

This will generate all required model files:
- `bert.npy` - Pre-computed BERT embeddings (768-dimensional vectors)
- `scaler.pkl` - Fitted MinMaxScaler for feature normalization (prevents data leakage)
- `lstm_model.h5` - LSTM baseline model
- `lstm_history.pckl` - LSTM training history
- `propose_model.h5` - LSTM+GRU hybrid model
- `propose_history.pckl` - LSTM+GRU training history
- `extension_model.h5` - Bidirectional LSTM+GRU model
- `extension_history.pckl` - Bidirectional training history
- `metrics.json` - Performance metrics for all models

**Note:** The scaler.pkl file is critical for accurate predictions. It contains the normalization parameters fitted on the training data only, ensuring consistency between training and inference without data leakage.

## 🎨 Dashboard Pages

### Navigation
Use the sidebar to navigate between different pages:
- 🏠 **Home**: Overview and introduction
- 📈 **Data Exploration**: Visualize datasets
- 🤖 **Model Performance**: Compare model metrics
- 🔮 **Live Prediction**: Make predictions on custom inputs
- 💡 **Insights & Conclusion**: Key findings and takeaways

### Interactive Features

#### Stock Price Visualization
- **Zoom**: Click and drag on the chart
- **Pan**: Hold shift and drag
- **Reset**: Double-click on the chart
- **Hover**: View detailed information

#### Date Range Selection
- Use the date pickers to filter data
- View statistics for specific time periods

#### Live Predictions
1. Enter a tweet about Apple stock
2. Select a date for stock price context
3. Choose a prediction model
4. Click "Predict Stock Movement"
5. View the prediction result with confidence score

## 🛠️ Troubleshooting

### Common Issues

**1. Module not found errors**
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

**2. TensorFlow compatibility issues**
```bash
# Try upgrading to TensorFlow 2.x
pip install tensorflow>=2.10.0
```

**3. BERT model download issues**
```bash
# The first run will download the BERT model (~500MB)
# Ensure you have internet connection
```

**4. Port already in use**
```bash
# Use a different port
streamlit run app.py --server.port 8502
```

**5. Models not loading**
- Ensure you've run the training notebook (`StockMovement.ipynb`) first
- Check that model files exist in the `model/` directory

### Performance Tips

- **First load is slow**: The dashboard caches data and models after the first load
- **Memory issues**: Close other applications if you run out of RAM
- **Slow predictions**: The BERT encoding may take a few seconds for the first prediction

## 📸 Screenshots

### Home Page
*(Screenshot placeholder - The home page shows project overview and architecture)*

### Data Exploration
*(Screenshot placeholder - Interactive stock price charts and tweet analysis)*

### Model Performance
*(Screenshot placeholder - Training history and metrics comparison)*

### Live Prediction
*(Screenshot placeholder - Real-time prediction interface)*

## 🎓 Educational Value

This dashboard is excellent for learning:
- **Multi-modal Deep Learning**: Combining text and numerical data
- **LSTM/GRU Architectures**: Sequential data processing
- **BERT Embeddings**: Transfer learning for NLP
- **Streamlit Development**: Interactive web applications
- **Data Visualization**: Plotly charts and graphs
- **Model Deployment**: From notebook to application

## 🔐 Security

### Dependency Updates

This dashboard uses updated, secure versions of all dependencies. The original project used outdated libraries with known vulnerabilities:

| Library | Original | Updated | Vulnerabilities Fixed |
|---------|----------|---------|----------------------|
| TensorFlow | 1.14.0 | ≥2.12.0 | 200+ (buffer overflows, code injection, DoS) |
| Keras | 2.2.4 | Included in TF2 | Path traversal, deserialization, code injection |
| PyTorch | 1.6.0 | ≥2.2.0 | RCE, heap overflow, use-after-free |
| Transformers | 3.4.0 | ≥4.48.0 | Deserialization vulnerabilities |
| nltk | 3.4.5 | ≥3.9 | ReDoS vulnerabilities |
| sentencepiece | 0.1.96 | ≥0.2.1 | Heap overflow |

### Security Best Practices

1. **No Model Pickling**: Dashboard validates inputs before processing
2. **Input Sanitization**: Tweet inputs are limited to 500 characters and stripped
3. **No Code Execution**: No `eval()` or `exec()` used
4. **Safe File Operations**: All file paths validated
5. **Error Handling**: Graceful error messages without exposing internals

### Reporting Security Issues

If you discover a security vulnerability, please:
1. **Do not** open a public issue
2. Contact the repository maintainer directly
3. Provide details of the vulnerability
4. Allow time for a fix before public disclosure

## ⚠️ Disclaimer

**IMPORTANT**: This dashboard and its predictions are for **educational purposes only**. 

- Do NOT use this as the sole basis for investment decisions
- Stock markets are complex and unpredictable
- Always consult with qualified financial professionals
- Past performance does not guarantee future results
- The creators are not responsible for any financial losses

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Add more stock symbols
- Implement real-time data streaming
- Add more technical indicators
- Improve model architectures
- Add more visualizations

## 📝 License

This project is for educational purposes. Please check the repository for license information.

## 📧 Contact

For questions or issues:
- Open an issue on GitHub
- Check existing documentation
- Review the code comments in `app.py`

## 🙏 Acknowledgments

- **Streamlit**: For the amazing dashboard framework
- **Plotly**: For interactive visualizations
- **Hugging Face**: For Sentence Transformers and BERT models
- **TensorFlow/Keras**: For deep learning capabilities
- **Financial community**: For open financial data

---

**Happy Exploring! 📊📈**

Remember: This is a learning tool, not a trading system. Use responsibly and continue learning!
