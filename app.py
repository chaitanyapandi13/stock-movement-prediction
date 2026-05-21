"""
Stock Movement Prediction Dashboard
Interactive Streamlit application for visualizing and predicting stock movements
using BERT embeddings and LSTM/GRU models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pickle
import hashlib

# Try to import ML libraries
try:
    from sentence_transformers import SentenceTransformer
    from keras.models import load_model
    from sklearn.preprocessing import MinMaxScaler
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    ML_IMPORT_ERROR = str(e)

# Import utilities
try:
    from utils.preprocessing import load_scaler, prepare_single_prediction, validate_feature_shape
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Stock Movement Predictor 📈",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #305653;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .success-box {
        background-color: #208e31;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .warning-box {
        background-color: #d5a73d;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
    }
    .info-box {
        background-color: #305653;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #17a2b8;
    }
</style>
""", unsafe_allow_html=True)

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Model input shape constants - these are specific to the trained models
# Features are reshaped to (batch_size, time_steps=35, features=22)
# The first 2 features from BERT embeddings are skipped
FEATURE_SKIP = 2  # Skip first 2 features after normalization
TIME_STEPS = 35
FEATURES_PER_STEP = 22

# Cache data loading
@st.cache_data
def load_stock_data():
    """Load Apple stock price data"""
    try:
        df = pd.read_csv(os.path.join(DATASET_DIR, "AAPL.csv"))
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.error(f"Error loading stock data: {e}")
        return None

@st.cache_data
def load_tweets_data():
    """Load tweets data"""
    try:
        df = pd.read_csv(os.path.join(DATASET_DIR, "tweets.csv"))
        return df
    except Exception as e:
        st.error(f"Error loading tweets data: {e}")
        return None

@st.cache_resource
def load_fitted_scaler():
    """Load the fitted scaler from training."""
    if not UTILS_AVAILABLE:
        return None
    
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        try:
            scaler = load_scaler(scaler_path)
            return scaler
        except Exception as e:
            st.warning(f"Could not load scaler: {e}")
            return None
    return None

@st.cache_resource
def load_bert_model():
    """Load BERT model for encoding tweets"""
    if not ML_AVAILABLE:
        return None
    try:
        bert = SentenceTransformer('nli-distilroberta-base-v2')
        return bert
    except Exception:
        # BERT model couldn't be loaded (network issues, etc.)
        # Return None silently - warnings will be shown in the UI when needed
        return None

@st.cache_data
def load_precomputed_bert_embeddings():
    """Load pre-computed BERT embeddings from the model directory"""
    bert_path = os.path.join(MODEL_DIR, "bert.npy")
    if os.path.exists(bert_path):
        return np.load(bert_path)
    return None

@st.cache_resource
def load_prediction_models():
    """Load all trained models"""
    if not ML_AVAILABLE:
        return None, None, None
    
    models = {}
    try:
        # Load LSTM model (using complete model file, not just weights)
        lstm_model_path = os.path.join(MODEL_DIR, "lstm_model.h5")
        if os.path.exists(lstm_model_path):
            models['LSTM'] = load_model(lstm_model_path)
        
        # Load Propose model (LSTM + GRU)
        propose_model_path = os.path.join(MODEL_DIR, "propose_model.h5")
        if os.path.exists(propose_model_path):
            models['GRU+LSTM'] = load_model(propose_model_path)
        
        # Load Extension model (LSTM + GRU + Bidirectional)
        extension_model_path = os.path.join(MODEL_DIR, "extension_model.h5")
        if os.path.exists(extension_model_path):
            models['Bidirectional'] = load_model(extension_model_path)
        
        return models.get('LSTM'), models.get('GRU+LSTM'), models.get('Bidirectional')
    except Exception as e:
        st.error(f"Error loading prediction models: {e}")
        return None, None, None

@st.cache_data
def load_training_history():
    """Load training history for all models"""
    history = {}
    try:
        for model_name, file_name in [
            ('LSTM', 'lstm_history.pckl'),
            ('GRU+LSTM', 'propose_history.pckl'),
            ('Bidirectional', 'extension_history.pckl')
        ]:
            path = os.path.join(MODEL_DIR, file_name)
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    history[model_name] = pickle.load(f)
    except Exception as e:
        st.warning(f"Could not load training history: {e}")
    return history

def create_stock_price_chart(df):
    """Create interactive stock price chart"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Stock Price Over Time', 'Trading Volume'),
        row_heights=[0.7, 0.3],
        vertical_spacing=0.1
    )
    
    # Stock prices
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Open'], name='Open', line=dict(color='blue', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['High'], name='High', line=dict(color='green', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Low'], name='Low', line=dict(color='red', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Close'], name='Close', line=dict(color='orange', width=2)),
        row=1, col=1
    )
    
    # Volume
    if 'Volume' in df.columns and 'Label' in df.columns:
        # Use vectorized operation for better performance
        colors = df['Label'].map({1: 'green', 0: 'red'}).tolist()
        fig.add_trace(
            go.Bar(x=df['Date'], y=df['Volume'], name='Volume', marker_color=colors, opacity=0.5),
            row=2, col=1
        )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    fig.update_layout(
        height=600,
        showlegend=True,
        hovermode='x unified',
        title_text="Apple (AAPL) Stock Analysis"
    )
    
    return fig

def create_label_distribution_chart(df):
    """Create label distribution chart"""
    label_counts = df['Label'].value_counts()
    
    fig = go.Figure(data=[
        go.Pie(
            labels=['Up ⬆️', 'Down ⬇️'],
            values=[label_counts.get(1, 0), label_counts.get(0, 0)],
            hole=0.4,
            marker=dict(colors=['#28a745', '#dc3545']),
            textinfo='label+percent+value'
        )
    ])
    
    fig.update_layout(
        title_text="Stock Movement Distribution",
        height=400
    )
    
    return fig

def create_correlation_heatmap(df):
    """Create correlation heatmap for stock features"""
    numeric_cols = ['Open', 'High', 'Low', 'Close']
    if 'Volume' in df.columns:
        numeric_cols.append('Volume')
    
    corr = df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Feature Correlation Matrix",
        height=500,
        width=600
    )
    
    return fig

def create_training_history_chart(history_dict):
    """Create training history charts"""
    if not history_dict:
        return None
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Model Accuracy', 'Model Loss')
    )
    
    colors = {'LSTM': 'blue', 'GRU+LSTM': 'green', 'Bidirectional': 'red'}
    
    for model_name, history in history_dict.items():
        color = colors.get(model_name, 'gray')
        
        # Accuracy
        if 'accuracy' in history:
            fig.add_trace(
                go.Scatter(y=history['accuracy'], name=f'{model_name} Train', 
                          line=dict(color=color, width=2)),
                row=1, col=1
            )
        if 'val_accuracy' in history:
            fig.add_trace(
                go.Scatter(y=history['val_accuracy'], name=f'{model_name} Val',
                          line=dict(color=color, width=2, dash='dash')),
                row=1, col=1
            )
        
        # Loss
        if 'loss' in history:
            fig.add_trace(
                go.Scatter(y=history['loss'], name=f'{model_name} Train',
                          line=dict(color=color, width=2), showlegend=False),
                row=1, col=2
            )
        if 'val_loss' in history:
            fig.add_trace(
                go.Scatter(y=history['val_loss'], name=f'{model_name} Val',
                          line=dict(color=color, width=2, dash='dash'), showlegend=False),
                row=1, col=2
            )
    
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=2)
    
    fig.update_layout(height=400, hovermode='x unified')
    
    return fig

def predict_stock_movement(tweet_text, stock_data, bert_model, prediction_model, scaler=None, precomputed_embeddings=None):
    """Make prediction for custom input
    
    Args:
        tweet_text: The tweet text to analyze
        stock_data: Dictionary with Open, High, Low, Close prices
        bert_model: SentenceTransformer model (can be None if using precomputed)
        prediction_model: Keras model for prediction
        scaler: Fitted MinMaxScaler from training (preferred) or None for fallback
        precomputed_embeddings: Pre-computed BERT embeddings (optional fallback)
    
    Returns:
        tuple: (predicted_class, confidence, is_demo_mode) or (None, None, False) on error
        is_demo_mode: True if using pre-computed embeddings instead of live BERT encoding
    """
    if not ML_AVAILABLE or prediction_model is None:
        return None, None, False
    
    is_demo_mode = False
    
    try:
        # Input validation for tweet_text
        if not tweet_text or not isinstance(tweet_text, str):
            return None, None, False
        
        # Sanitize input - strip whitespace and limit length
        tweet_text = tweet_text.strip()[:500]  # Max 500 characters
        
        if not tweet_text:  # Empty after stripping
            return None, None, False
        
        # Validate stock_data has required keys with numeric values
        required_keys = ['Open', 'High', 'Low', 'Close']
        for key in required_keys:
            if key not in stock_data:
                st.error(f"Missing required stock data key: {key}")
                return None, None, False
            try:
                float(stock_data[key])
            except (TypeError, ValueError):
                st.error(f"Invalid numeric value for {key}: {stock_data[key]}")
                return None, None, False
        
        # Get tweet embedding - either from BERT model or use pre-computed
        if bert_model is not None:
            # Live encoding with BERT model
            tweet_embedding = bert_model.encode([tweet_text], convert_to_tensor=True)
            # Convert to numpy, handling both CPU and CUDA tensors
            if hasattr(tweet_embedding, 'is_cuda') and tweet_embedding.is_cuda:
                tweet_features = tweet_embedding.cpu().numpy()
            else:
                tweet_features = tweet_embedding.numpy()
        elif precomputed_embeddings is not None:
            # Demo mode: Use a deterministic hash to select a pre-computed embedding
            # Note: This provides consistent predictions for the same input text,
            # but does NOT analyze actual tweet content. For true sentiment analysis,
            # the BERT model must be available for live encoding.
            is_demo_mode = True
            hash_value = int(hashlib.sha256(tweet_text.encode('utf-8')).hexdigest(), 16)
            idx = hash_value % len(precomputed_embeddings)
            tweet_features = precomputed_embeddings[idx:idx+1, :]
        else:
            st.error("No BERT model or pre-computed embeddings available")
            return None, None, False
        
        # Use the utility function if available and scaler is provided
        if UTILS_AVAILABLE and scaler is not None:
            # Use the proper preprocessing pipeline
            X = prepare_single_prediction(tweet_features, stock_data, scaler)
        else:
            # Fallback to original approach (with data leakage warning)
            if scaler is None:
                st.warning("⚠️ Scaler not available. Using per-sample normalization (may affect accuracy).")
            
            # Prepare stock features
            stock_features_arr = np.array([[float(stock_data['Open']), float(stock_data['High']), 
                                           float(stock_data['Low']), float(stock_data['Close'])]])
            
            # Merge features (768 BERT features + 4 stock features = 772)
            X = np.hstack((tweet_features, stock_features_arr))
            
            # Normalize using MinMaxScaler
            # Note: This is the fallback method with data leakage
            if scaler is not None:
                X = scaler.transform(X)
            else:
                fallback_scaler = MinMaxScaler((0, 1))
                X = fallback_scaler.fit_transform(X)
            
            # Skip first FEATURE_SKIP (2) features to get 770 features
            X = X[:, FEATURE_SKIP:X.shape[1]]

            # Ensure correct total size (35 * 22 = 770)
            expected_features = TIME_STEPS * FEATURES_PER_STEP
            if X.shape[1] != expected_features:
                raise ValueError(
                    f"Feature mismatch: expected {expected_features}, got {X.shape[1]}"
                )

            # Reshape to model input: (batch_size, 35, 22)
            X = X.reshape(1, TIME_STEPS, FEATURES_PER_STEP)
        
        # Predict
        prediction = prediction_model.predict(X, verbose=0)
        confidence = float(np.max(prediction))
        predicted_class = int(np.argmax(prediction))
        
        return predicted_class, confidence, is_demo_mode
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, False

# Sidebar Navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📈 Data Exploration", "🤖 Model Performance", 
     "🔮 Live Prediction", "💡 Insights & Conclusion"]
)

# Add info in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This dashboard visualizes stock movement prediction using "
    "BERT embeddings and deep learning models (LSTM, GRU, Bidirectional)."
)

st.sidebar.markdown("### Data Sources")
st.sidebar.markdown("- **Stock Data**: Apple (AAPL) Historical Prices")
st.sidebar.markdown("- **Tweets**: Financial tweets mentioning $AAPL")
st.sidebar.markdown("- **Models**: LSTM, LSTM+GRU, Bidirectional LSTM+GRU")

# ============================================================================
# HOME PAGE
# ============================================================================
if page == "🏠 Home":
    st.markdown('<p class="main-header">📈 Stock Movement Prediction System</p>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>Welcome to the Stock Movement Prediction Dashboard!</h3>
    <p>This interactive dashboard demonstrates a sophisticated machine learning system that predicts 
    stock price movements by analyzing both <strong>financial tweets</strong> and <strong>historical stock data</strong>.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Project Overview
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🎯 Project Overview")
        st.markdown("""
        This project combines:
        - **Natural Language Processing (NLP)**: Using BERT to understand tweet sentiment
        - **Time Series Analysis**: Processing historical stock price data
        - **Deep Learning**: LSTM, GRU, and Bidirectional models for prediction
        
        **Goal**: Predict whether stock prices will go **UP ⬆️** or **DOWN ⬇️** based on 
        tweets and technical indicators.
        """)
        
        st.markdown("### 🔧 Technology Stack")
        st.markdown("""
        - **BERT**: `nli-distilroberta-base-v2` for tweet encoding
        - **LSTM**: Long Short-Term Memory networks
        - **GRU**: Gated Recurrent Units
        - **Bidirectional**: Processing sequences in both directions
        - **TensorFlow/Keras**: Deep learning framework
        - **Streamlit**: Interactive dashboard
        - **Plotly**: Interactive visualizations
        """)
    
    with col2:
        st.markdown("### 🏗️ System Architecture")
        st.markdown("""
        ```
        ┌─────────────────┐
        │   Tweets        │
        │   Input         │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │   BERT          │
        │   Encoding      │
        └────────┬────────┘
                 │
                 │        ┌─────────────────┐
                 │        │  Stock Data     │
                 │        │  (OHLC)         │
                 │        └────────┬────────┘
                 │                 │
                 ▼                 ▼
        ┌─────────────────────────────┐
        │   Feature Merging           │
        │   & Normalization           │
        └──────────┬──────────────────┘
                   │
                   ▼
        ┌─────────────────────────────┐
        │   Deep Learning Models      │
        │   • LSTM                    │
        │   • LSTM + GRU              │
        │   • Bidirectional LSTM+GRU  │
        └──────────┬──────────────────┘
                   │
                   ▼
        ┌─────────────────────────────┐
        │   Prediction Output         │
        │   (Up ⬆️ / Down ⬇️)          │
        └─────────────────────────────┘
        ```
        """)
    
    st.markdown("---")
    
    # Key Features
    st.markdown("### ✨ Dashboard Features")
    
    feat_col1, feat_col2, feat_col3 = st.columns(3)
    
    with feat_col1:
        st.markdown("""
        <div class="metric-card">
        <h4>📊 Data Exploration</h4>
        <ul>
            <li>Interactive stock price charts</li>
            <li>Tweet samples analysis</li>
            <li>Distribution statistics</li>
            <li>Correlation analysis</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col2:
        st.markdown("""
        <div class="metric-card">
        <h4>🤖 Model Performance</h4>
        <ul>
            <li>Accuracy metrics</li>
            <li>Confusion matrices</li>
            <li>Model comparison</li>
            <li>Training history</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col3:
        st.markdown("""
        <div class="metric-card">
        <h4>🔮 Live Prediction</h4>
        <ul>
            <li>Custom tweet input</li>
            <li>Stock data selection</li>
            <li>Real-time prediction</li>
            <li>Confidence scores</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Stats
    st.markdown("### 📊 Dataset Quick Stats")
    
    stock_df = load_stock_data()
    tweets_df = load_tweets_data()
    
    if stock_df is not None and tweets_df is not None:
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            st.metric("📅 Date Range", 
                     f"{stock_df['Date'].min().strftime('%Y-%m-%d')} to {stock_df['Date'].max().strftime('%Y-%m-%d')}")
        
        with stat_col2:
            st.metric("📈 Stock Records", f"{len(stock_df):,}")
        
        with stat_col3:
            st.metric("💬 Tweets", f"{len(tweets_df):,}")
        
        with stat_col4:
            up_pct = (stock_df['Label'].sum() / len(stock_df) * 100) if 'Label' in stock_df.columns else 0
            st.metric("⬆️ Up Movement %", f"{up_pct:.1f}%")

# ============================================================================
# DATA EXPLORATION PAGE
# ============================================================================
elif page == "📈 Data Exploration":
    st.markdown('<p class="main-header">📈 Data Exploration</p>', unsafe_allow_html=True)
    
    # Load data
    stock_df = load_stock_data()
    tweets_df = load_tweets_data()
    
    if stock_df is None or tweets_df is None:
        st.error("Unable to load data. Please check that Dataset/AAPL.csv and Dataset/tweets.csv exist.")
        st.stop()
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Stock Prices", "💬 Tweets", "📉 Statistics", "🔗 Correlations"])
    
    with tab1:
        st.markdown("### Apple (AAPL) Stock Price Data")
        
        # Display sample data
        st.dataframe(stock_df.head(10), use_container_width=True)
        
        # Interactive chart
        st.plotly_chart(create_stock_price_chart(stock_df), use_container_width=True, key="full_stock_chart")
        
        # Date range selector
        st.markdown("#### 📅 Select Date Range for Detailed View")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", stock_df['Date'].min())
        with col2:
            end_date = st.date_input("End Date", stock_df['Date'].max())
        
        # Filter data
        mask = (stock_df['Date'] >= pd.to_datetime(start_date)) & (stock_df['Date'] <= pd.to_datetime(end_date))
        filtered_df = stock_df[mask]
        
        if len(filtered_df) > 0:
            st.plotly_chart(create_stock_price_chart(filtered_df), use_container_width=True, key="filtered_stock_chart")
        
    with tab2:
        st.markdown("### Tweet Samples")
        
        # Display sample tweets
        st.dataframe(tweets_df.head(20), use_container_width=True)
        
        # Label distribution
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.plotly_chart(create_label_distribution_chart(tweets_df), use_container_width=True, key="label_distribution_chart")
        
        with col2:
            st.markdown("#### Tweet Statistics")
            st.metric("Total Tweets", len(tweets_df))
            
            if 'Label' in tweets_df.columns:
                up_count = tweets_df['Label'].sum()
                down_count = len(tweets_df) - up_count
                st.metric("Bullish Tweets ⬆️", up_count)
                st.metric("Bearish Tweets ⬇️", down_count)
        
        # Sample tweets by sentiment
        st.markdown("#### Sample Tweets by Sentiment")
        
        sent_col1, sent_col2 = st.columns(2)
        
        with sent_col1:
            st.markdown("**Bearish Tweets (Predicting Down) ⬇️**")
            bearish = tweets_df[tweets_df['Label'] == 0].head(5)
            for idx, row in bearish.iterrows():
                st.info(row['Tweets'])
        
        with sent_col2:
            st.markdown("**Bullish Tweets (Predicting Up) ⬆️**")
            bullish = tweets_df[tweets_df['Label'] == 1].head(5)
            for idx, row in bullish.iterrows():
                st.success(row['Tweets'])
    
    with tab3:
        st.markdown("### Statistical Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Stock Price Statistics")
            st.dataframe(stock_df[['Open', 'High', 'Low', 'Close']].describe(), 
                        use_container_width=True)
        
        with col2:
            st.markdown("#### Label Distribution")
            if 'Label' in stock_df.columns:
                label_counts = stock_df['Label'].value_counts()
                st.write(f"**Up Days (1)**: {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(stock_df)*100:.1f}%)")
                st.write(f"**Down Days (0)**: {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(stock_df)*100:.1f}%)")
                
                # Create bar chart
                fig = px.bar(
                    x=['Down ⬇️', 'Up ⬆️'],
                    y=[label_counts.get(0, 0), label_counts.get(1, 0)],
                    labels={'x': 'Movement', 'y': 'Count'},
                    title='Stock Movement Counts',
                    color=['Down ⬇️', 'Up ⬆️'],
                    color_discrete_map={'Down ⬇️': '#dc3545', 'Up ⬆️': '#28a745'}
                )
                st.plotly_chart(fig, use_container_width=True, key="label_counts_bar_chart")
    
    with tab4:
        st.markdown("### Feature Correlations")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.plotly_chart(create_correlation_heatmap(stock_df), use_container_width=True, key="correlation_heatmap")
        
        with col2:
            st.markdown("#### Correlation Insights")
            st.markdown("""
            **Key Observations:**
            - **High correlation** between Open, High, Low, and Close prices (expected)
            - Stock prices tend to move together
            - Volume may show varying correlation with price movements
            
            **What This Means:**
            - Price features are highly correlated
            - Model needs to extract temporal patterns, not just price values
            - Tweet sentiment adds valuable independent information
            """)

# ============================================================================
# MODEL PERFORMANCE PAGE
# ============================================================================
elif page == "🤖 Model Performance":
    st.markdown('<p class="main-header">🤖 Model Performance Analysis</p>', unsafe_allow_html=True)
    
    # Load models and history
    lstm_model, gru_lstm_model, bidirectional_model = load_prediction_models()
    training_history = load_training_history()
    
    # Model information
    st.markdown("""
    <div class="info-box">
    <h3>Three Deep Learning Architectures</h3>
    <p>This project implements and compares three different neural network architectures:</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h4>1️⃣ LSTM Model</h4>
        <p><strong>Architecture:</strong></p>
        <ul>
            <li>Single LSTM layer (100 units)</li>
            <li>Dropout (0.5)</li>
            <li>Dense layer (100 units)</li>
            <li>Output layer (2 classes)</li>
        </ul>
        <p><strong>Characteristics:</strong> Baseline model for sequential data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h4>2️⃣ LSTM + GRU Model</h4>
        <p><strong>Architecture:</strong></p>
        <ul>
            <li>LSTM layer (100 units)</li>
            <li>GRU layers (80, 64 units)</li>
            <li>Multiple dropouts (0.2)</li>
            <li>Dense + Output layers</li>
        </ul>
        <p><strong>Characteristics:</strong> Hybrid approach combining LSTM and GRU</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h4>3️⃣ Bidirectional Model</h4>
        <p><strong>Architecture:</strong></p>
        <ul>
            <li>LSTM layer (100 units)</li>
            <li>Bidirectional GRU layers</li>
            <li>Multiple dropouts (0.2)</li>
            <li>Dense + Output layers</li>
        </ul>
        <p><strong>Characteristics:</strong> Processes data in both directions</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Training History
    if training_history:
        st.markdown("### 📊 Training History")
        fig = create_training_history_chart(training_history)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="training_history_chart")
        
        # Display epochs info
        st.markdown("#### Training Details")
        for model_name, history in training_history.items():
            with st.expander(f"📈 {model_name} Training Details"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Total Epochs:** {len(history.get('accuracy', []))}")
                    if 'accuracy' in history:
                        st.write(f"**Final Training Accuracy:** {history['accuracy'][-1]:.4f}")
                    if 'loss' in history:
                        st.write(f"**Final Training Loss:** {history['loss'][-1]:.4f}")
                
                with col2:
                    if 'val_accuracy' in history:
                        st.write(f"**Final Validation Accuracy:** {history['val_accuracy'][-1]:.4f}")
                        st.write(f"**Best Validation Accuracy:** {max(history['val_accuracy']):.4f}")
                    if 'val_loss' in history:
                        st.write(f"**Final Validation Loss:** {history['val_loss'][-1]:.4f}")
                        st.write(f"**Best Validation Loss:** {min(history['val_loss']):.4f}")
    else:
        st.warning("Training history not available. Models may need to be trained first.")
    
    st.markdown("---")
    
    # Model Metrics (Simulated - in real scenario, these would come from saved metrics)
    st.markdown("### 📈 Model Performance Metrics")
    
    # Note: In the actual notebook, these metrics are calculated. For the dashboard,
    # we'd need to either re-calculate or save them. Here we'll show placeholder structure.
    
    st.info("💡 **Note**: To display actual metrics, run the training notebook first to generate model predictions and metrics.")
    
    # Placeholder metrics structure
    metrics_data = {
        'Model': ['LSTM', 'LSTM + GRU', 'Bidirectional LSTM + GRU'],
        'Accuracy': [0.0, 0.0, 0.0],  # These would be loaded from saved metrics
        'Precision': [0.0, 0.0, 0.0],
        'Recall': [0.0, 0.0, 0.0],
        'F1-Score': [0.0, 0.0, 0.0]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    
    st.markdown("#### Performance Comparison Table")
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # Comparison chart
    st.markdown("#### Metrics Comparison Chart")
    
    metrics_long = metrics_df.melt(id_vars=['Model'], var_name='Metric', value_name='Score')
    fig = px.bar(
        metrics_long,
        x='Model',
        y='Score',
        color='Metric',
        barmode='group',
        title='Model Performance Comparison',
        labels={'Score': 'Score (0-1)'},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True, key="model_performance_comparison")
    
    st.markdown("---")
    
    # Model Insights
    st.markdown("### 💡 Model Insights")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown("""
        <div class="success-box">
        <h4>✅ Strengths</h4>
        <ul>
            <li>Combines text and numerical features effectively</li>
            <li>BERT embeddings capture tweet sentiment</li>
            <li>Multiple architectures for comparison</li>
            <li>Bidirectional processing improves context understanding</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with insight_col2:
        st.markdown("""
        <div class="warning-box">
        <h4>⚠️ Considerations</h4>
        <ul>
            <li>Stock market is inherently unpredictable</li>
            <li>Past performance doesn't guarantee future results</li>
            <li>Tweet sentiment is just one factor</li>
            <li>Model should be used as one tool among many</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# LIVE PREDICTION PAGE
# ============================================================================
elif page == "🔮 Live Prediction":
    st.markdown('<p class="main-header">🔮 Live Stock Movement Prediction</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>Try the Prediction System!</h3>
    <p>Enter a tweet about Apple stock and provide stock price context to get a prediction.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load required models
    bert_model = load_bert_model()
    lstm_model, gru_lstm_model, bidirectional_model = load_prediction_models()
    precomputed_embeddings = load_precomputed_bert_embeddings()
    fitted_scaler = load_fitted_scaler()
    
    if not ML_AVAILABLE:
        st.error("⚠️ Machine learning libraries are not available. Please install required packages.")
        st.stop()
    
    # Check scaler availability
    if fitted_scaler is None:
        st.warning("""
        ⚠️ **Scaler not found**: The fitted scaler from training (model/scaler.pkl) is not available.
        The app will use per-sample normalization as a fallback, which may affect prediction accuracy.
        
        To fix this:
        1. Run the training pipeline: `python train.py`
        2. This will create model/scaler.pkl along with the trained models
        """)
    
    # Check if we have at least one way to get embeddings
    if bert_model is None and precomputed_embeddings is None:
        st.error("⚠️ Neither BERT model nor pre-computed embeddings available. Predictions cannot be made.")
    elif bert_model is None:
        st.info("ℹ️ Using demo mode with pre-computed BERT embeddings. For live encoding, ensure network access to HuggingFace.")
    
    st.markdown("---")
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💬 Tweet Input")
        
        # Initialize session state for sample tweet
        if 'sample_tweet' not in st.session_state:
            st.session_state.sample_tweet = ""
        
        tweet_input = st.text_area(
            "Enter a tweet about Apple ($AAPL)",
            value=st.session_state.sample_tweet,
            placeholder="Example: Apple's new iPhone is breaking sales records! $AAPL to the moon! 🚀",
            height=100,
            key="tweet_input_area"
        )
        
        # Sample tweets
        with st.expander("📝 Use Sample Tweets"):
            col_bear, col_bull = st.columns(2)
            with col_bear:
                if st.button("🐻 Bearish Tweet", use_container_width=True):
                    st.session_state.sample_tweet = "Apple facing supply chain issues and declining sales. Not good for $AAPL shareholders."
                    st.rerun()
            with col_bull:
                if st.button("🐂 Bullish Tweet", use_container_width=True):
                    st.session_state.sample_tweet = "Apple just announced record-breaking earnings! Strong buy signal for $AAPL 📈"
                    st.rerun()
    
    with col2:
        st.markdown("### 📊 Stock Price Context")
        
        stock_df = load_stock_data()
        
        if stock_df is not None:
            # Date picker
            selected_date = st.date_input(
                "Select Date",
                value=stock_df['Date'].max(),
                min_value=stock_df['Date'].min(),
                max_value=stock_df['Date'].max()
            )
            
            # Find closest date
            closest_idx = (stock_df['Date'] - pd.to_datetime(selected_date)).abs().idxmin()
            selected_stock = stock_df.iloc[closest_idx]
            
            st.metric("📅 Date", selected_stock['Date'].strftime('%Y-%m-%d'))
            st.metric("💵 Open", f"${selected_stock['Open']:.2f}")
            st.metric("📈 High", f"${selected_stock['High']:.2f}")
            st.metric("📉 Low", f"${selected_stock['Low']:.2f}")
            st.metric("💰 Close", f"${selected_stock['Close']:.2f}")
        else:
            st.error("Could not load stock data")
            selected_stock = None
    
    st.markdown("---")
    
    # Model selection
    st.markdown("### 🤖 Select Prediction Model")
    model_choice = st.radio(
        "Choose a model",
        ["Bidirectional LSTM+GRU (Best)", "LSTM + GRU (Hybrid)", "LSTM (Baseline)"],
        horizontal=True
    )
    
    # Map choice to model
    if model_choice.startswith("Bidirectional"):
        selected_model = bidirectional_model
        model_name = "Bidirectional LSTM+GRU"
    elif model_choice.startswith("LSTM + GRU"):
        selected_model = gru_lstm_model
        model_name = "LSTM + GRU"
    else:
        selected_model = lstm_model
        model_name = "LSTM"
    
    # Prediction button
    if st.button("🔮 Predict Stock Movement", type="primary", use_container_width=True):
        if not tweet_input:
            st.warning("⚠️ Please enter a tweet first!")
        elif selected_stock is None:
            st.error("⚠️ Stock data not available!")
        elif selected_model is None:
            st.error(f"⚠️ {model_name} model not loaded!")
        elif bert_model is None and precomputed_embeddings is None:
            st.error("⚠️ No BERT model or pre-computed embeddings available!")
        else:
            with st.spinner("🔄 Making prediction..."):
                stock_data = {
                    'Open': selected_stock['Open'],
                    'High': selected_stock['High'],
                    'Low': selected_stock['Low'],
                    'Close': selected_stock['Close']
                }
                
                predicted_class, confidence, is_demo_mode = predict_stock_movement(
                    tweet_input, stock_data, bert_model, selected_model,
                    scaler=fitted_scaler,
                    precomputed_embeddings=precomputed_embeddings
                )
                
                if predicted_class is not None:
                    st.markdown("---")
                    st.markdown("### 📊 Prediction Result")
                    
                    # Demo mode warning
                    if is_demo_mode:
                        st.info("""
                        🔬 **Demo Mode**: This prediction uses pre-computed embeddings and does not 
                        analyze the actual content of your tweet. For real sentiment analysis, 
                        the BERT model must be available for live encoding. The prediction shown 
                        is illustrative only.
                        """)
                    
                    # Display prediction
                    result_col1, result_col2, result_col3 = st.columns([1, 1, 1])
                    
                    with result_col1:
                        if predicted_class == 1:
                            st.markdown("""
                            <div class="success-box" style="text-align: center; padding: 2rem;">
                            <h2>⬆️ UP</h2>
                            <h3 style="color: #28a745;">Stock Expected to Rise</h3>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="warning-box" style="text-align: center; padding: 2rem;">
                            <h2>⬇️ DOWN</h2>
                            <h3 style="color: #dc3545;">Stock Expected to Fall</h3>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with result_col2:
                        st.metric("🎯 Prediction", "Up ⬆️" if predicted_class == 1 else "Down ⬇️")
                        st.metric("📊 Confidence", f"{confidence*100:.1f}%")
                        st.metric("🤖 Model Used", model_name)
                    
                    with result_col3:
                        # Create gauge chart for confidence
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=confidence * 100,
                            title={'text': "Confidence"},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "#28a745" if predicted_class == 1 else "#dc3545"},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 75], 'color': "gray"},
                                    {'range': [75, 100], 'color': "darkgray"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        fig.update_layout(height=250)
                        st.plotly_chart(fig, use_container_width=True, key="prediction_confidence_gauge")
                    
                    # Disclaimer
                    st.markdown("---")
                    st.warning("""
                    ⚠️ **Disclaimer**: This prediction is for educational purposes only. 
                    It should NOT be used as the sole basis for investment decisions. 
                    Always consult with financial professionals and do your own research.
                    """)
                else:
                    st.error("Failed to make prediction. Please check the input data.")

# ============================================================================
# INSIGHTS & CONCLUSION PAGE
# ============================================================================
elif page == "💡 Insights & Conclusion":
    st.markdown('<p class="main-header">💡 Insights & Conclusion</p>', unsafe_allow_html=True)
    
    # Key Findings
    st.markdown("### 🔍 Key Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h4>1. Tweet Sentiment Matters</h4>
        <p>Social media sentiment, when encoded using BERT, provides valuable signals about 
        stock movements. The model successfully learns patterns from financial tweets.</p>
        
        <h4>2. Hybrid Architectures Perform Better</h4>
        <p>Combining LSTM with GRU, especially with bidirectional processing, improves 
        prediction accuracy by capturing both short-term and long-term dependencies.</p>
        
        <h4>3. Multi-Modal Approach Works</h4>
        <p>Merging text features (tweets) with numerical features (stock prices) creates 
        a richer representation for prediction.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
        <h4>🎯 Model Comparison Summary</h4>
        <p><strong>Best Model:</strong> Bidirectional LSTM + GRU</p>
        <p><strong>Why?</strong></p>
        <ul>
            <li>Processes sequences in both directions</li>
            <li>Captures complex temporal patterns</li>
            <li>Better generalization on test data</li>
            <li>Optimal balance of complexity and performance</li>
        </ul>
        
        <p><strong>Runner-up:</strong> LSTM + GRU Hybrid</p>
        <p>Good balance of performance and computational efficiency.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Use Cases
    st.markdown("### 🎯 Potential Applications")
    
    use_case_col1, use_case_col2, use_case_col3 = st.columns(3)
    
    with use_case_col1:
        st.markdown("""
        <div class="metric-card">
        <h4>📊 Trading Signals</h4>
        <p>Generate buy/sell signals based on social sentiment and technical indicators.</p>
        <p><strong>Benefit:</strong> Automated decision support</p>
        </div>
        """, unsafe_allow_html=True)
    
    with use_case_col2:
        st.markdown("""
        <div class="metric-card">
        <h4>🔔 Alert Systems</h4>
        <p>Monitor social media for significant sentiment shifts and alert traders.</p>
        <p><strong>Benefit:</strong> Real-time market awareness</p>
        </div>
        """, unsafe_allow_html=True)
    
    with use_case_col3:
        st.markdown("""
        <div class="metric-card">
        <h4>📈 Portfolio Management</h4>
        <p>Incorporate sentiment analysis into broader portfolio optimization strategies.</p>
        <p><strong>Benefit:</strong> Enhanced risk management</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Limitations
    st.markdown("### ⚠️ Limitations & Considerations")
    
    st.markdown("""
    <div class="warning-box">
    <h4>Important Limitations:</h4>
    <ol>
        <li><strong>Market Complexity:</strong> Stock prices are influenced by countless factors beyond tweets and historical prices.</li>
        <li><strong>Data Quality:</strong> Tweet quality varies, and noise in social media can mislead models.</li>
        <li><strong>Temporal Mismatch:</strong> Tweets and price movements may not be perfectly aligned in time.</li>
        <li><strong>Market Regime Changes:</strong> Models trained on one market period may not generalize to different conditions.</li>
        <li><strong>Black Swan Events:</strong> Unpredictable events can render historical patterns useless.</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Future Improvements
    st.markdown("### 🚀 Future Enhancements")
    
    improve_col1, improve_col2 = st.columns(2)
    
    with improve_col1:
        st.markdown("""
        <div class="success-box">
        <h4>🔧 Technical Improvements</h4>
        <ul>
            <li>Incorporate more data sources (news, financial reports)</li>
            <li>Add technical indicators (RSI, MACD, etc.)</li>
            <li>Implement attention mechanisms</li>
            <li>Use transformer-based architectures</li>
            <li>Add ensemble methods</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with improve_col2:
        st.markdown("""
        <div class="info-box">
        <h4>📊 Data Enhancements</h4>
        <ul>
            <li>Real-time data streaming</li>
            <li>Multi-stock support</li>
            <li>Longer historical periods</li>
            <li>Multi-language tweet support</li>
            <li>Incorporate market indices</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Conclusion
    st.markdown("### 🎓 Conclusion")
    
    st.markdown("""
    <div class="info-box">
    <h3>Final Thoughts</h3>
    <p>This project demonstrates the feasibility of combining <strong>Natural Language Processing</strong> 
    and <strong>Time Series Analysis</strong> for stock movement prediction. While the results are promising, 
    they highlight both the potential and limitations of data-driven approaches in financial markets.</p>
    
    <p><strong>Key Takeaway:</strong> Machine learning models can capture meaningful patterns in financial 
    data and social sentiment, but they should be used as <strong>one tool among many</strong> in a 
    comprehensive investment strategy, never as a standalone decision-maker.</p>
    
    <p><strong>Educational Value:</strong> This project serves as an excellent example of:</p>
    <ul>
        <li>Multi-modal deep learning (text + numerical data)</li>
        <li>LSTM, GRU, and Bidirectional architectures</li>
        <li>Transfer learning with BERT</li>
        <li>Real-world data processing and visualization</li>
        <li>Model comparison and evaluation</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Credits
    st.markdown("### 👏 Acknowledgments")
    
    st.markdown("""
    **Technologies Used:**
    - **Streamlit**: Interactive dashboard framework
    - **Plotly**: Interactive visualizations
    - **TensorFlow/Keras**: Deep learning models
    - **Sentence Transformers**: BERT embeddings
    - **Scikit-learn**: Machine learning utilities
    - **Pandas & NumPy**: Data manipulation
    
    **Data Sources:**
    - Historical stock data: Apple Inc. (AAPL)
    - Tweet data: Financial social media posts
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; padding: 2rem;">
<p>📈 Stock Movement Prediction Dashboard</p>
<p>Built with Streamlit • Powered by Deep Learning</p>
<p>⚠️ For educational purposes only - Not financial advice</p>
</div>
""", unsafe_allow_html=True)
