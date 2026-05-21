#!/usr/bin/env python3
"""
Stock Movement Prediction - Training Pipeline

This script provides a clean, reproducible training pipeline that:
1. Loads and validates datasets with timestamp alignment
2. Computes BERT embeddings for tweets
3. Aligns tweets to AAPL trading days
4. Prepares features with proper normalization (scaler fitted on training set only)
5. Trains three models: LSTM, LSTM+GRU, Bidirectional LSTM+GRU
6. Saves models, scaler, and metrics

Usage:
    python train.py [--force-bert] [--epochs EPOCHS]
"""

import os
import sys
import argparse
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime

# ML imports
try:
    from sklearn.model_selection import train_test_split
    from sentence_transformers import SentenceTransformer
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout, Bidirectional
    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.utils import to_categorical
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
except ImportError as e:
    print(f"Error: Required ML libraries not installed: {e}")
    print("Install with: pip install -r requirements.txt")
    sys.exit(1)

# Import utilities
from utils.data_alignment import validate_datasets, prepare_aligned_dataset
from utils.preprocessing import prepare_features, save_scaler, validate_feature_shape


# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)


def load_datasets():
    """Load stock and tweet datasets."""
    print("\n" + "="*80)
    print("STEP 1: Loading Datasets")
    print("="*80)
    
    # Load stock data
    stock_path = os.path.join(DATASET_DIR, "AAPL.csv")
    print(f"Loading stock data from {stock_path}")
    stock_df = pd.read_csv(stock_path)
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    print(f"  - Loaded {len(stock_df)} stock records")
    print(f"  - Date range: {stock_df['Date'].min()} to {stock_df['Date'].max()}")
    
    # Load tweets data
    tweets_path = os.path.join(DATASET_DIR, "tweets.csv")
    print(f"\nLoading tweets from {tweets_path}")
    tweets_df = pd.read_csv(tweets_path)
    print(f"  - Loaded {len(tweets_df)} tweets")
    
    # Check for timestamp column
    timestamp_col = None
    for col in ['timestamp', 'Timestamp', 'Date', 'date', 'created_at']:
        if col in tweets_df.columns:
            timestamp_col = col
            print(f"  - Found timestamp column: {col}")
            break
    
    if timestamp_col is None:
        print("  - WARNING: No timestamp column found in tweets dataset")
        print("  - Will use index-based pairing (assumes pre-aligned data)")
    
    return stock_df, tweets_df, timestamp_col


def compute_or_load_bert_embeddings(tweets_df, force_compute=False):
    """Compute BERT embeddings or load from cache."""
    print("\n" + "="*80)
    print("STEP 2: BERT Embeddings")
    print("="*80)
    
    bert_path = os.path.join(MODEL_DIR, "bert.npy")
    
    if os.path.exists(bert_path) and not force_compute:
        print(f"Loading pre-computed BERT embeddings from {bert_path}")
        embeddings = np.load(bert_path)
        print(f"  - Loaded embeddings shape: {embeddings.shape}")
        
        if len(embeddings) != len(tweets_df):
            print(f"  - WARNING: Embedding count ({len(embeddings)}) doesn't match tweet count ({len(tweets_df)})")
            print("  - Recomputing embeddings...")
            force_compute = True
        else:
            return embeddings
    
    if force_compute or not os.path.exists(bert_path):
        print("Computing BERT embeddings (this may take a while)...")
        print("  - Loading BERT model: nli-distilroberta-base-v2")
        bert = SentenceTransformer('nli-distilroberta-base-v2')
        
        print(f"  - Encoding {len(tweets_df)} tweets...")
        tweets = tweets_df['Tweets'].tolist()
        embeddings = bert.encode(tweets, convert_to_tensor=True, show_progress_bar=True)
        embeddings = embeddings.cpu().numpy()
        
        print(f"  - Computed embeddings shape: {embeddings.shape}")
        print(f"  - Saving to {bert_path}")
        np.save(bert_path, embeddings)
    
    return embeddings


def prepare_dataset(stock_df, tweets_df, bert_embeddings, timestamp_col=None):
    """Align data and prepare features."""
    print("\n" + "="*80)
    print("STEP 3: Dataset Alignment and Feature Preparation")
    print("="*80)
    
    # Validate datasets
    print("Validating datasets...")
    is_valid, errors = validate_datasets(stock_df, tweets_df)
    for error in errors:
        print(f"  - {error}")
    
    if not is_valid:
        raise ValueError("Dataset validation failed")
    
    # Prepare aligned dataset
    print("\nAligning tweets to trading days...")
    merged_df, labels, alignment_info = prepare_aligned_dataset(
        tweets_df, stock_df, timestamp_col, use_existing_labels=True
    )
    
    print(f"  - Alignment method: {alignment_info['method']}")
    print(f"  - Aligned samples: {alignment_info['aligned_count']}")
    print(f"  - Label source: {alignment_info.get('label_source', 'existing')}")
    
    # Extract stock features
    stock_features = merged_df[['Open', 'High', 'Low', 'Close']].values
    
    # Handle BERT embeddings alignment
    # If using index-based alignment, truncate embeddings to match
    if len(bert_embeddings) > len(merged_df):
        bert_embeddings = bert_embeddings[:len(merged_df)]
    elif len(bert_embeddings) < len(merged_df):
        raise ValueError(f"Not enough BERT embeddings ({len(bert_embeddings)}) for aligned data ({len(merged_df)})")
    
    print(f"\nFeature shapes:")
    print(f"  - BERT embeddings: {bert_embeddings.shape}")
    print(f"  - Stock features: {stock_features.shape}")
    print(f"  - Labels: {labels.shape}")
    
    return bert_embeddings, stock_features, labels, merged_df


def split_and_prepare_features(bert_embeddings, stock_features, labels, test_size=0.2, random_state=42):
    """Split data and prepare features with proper scaling."""
    print("\n" + "="*80)
    print("STEP 4: Train/Test Split and Feature Engineering")
    print("="*80)
    
    # Split data
    print(f"Splitting dataset: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")
    X_bert_train, X_bert_test, X_stock_train, X_stock_test, y_train, y_test = train_test_split(
        bert_embeddings, stock_features, labels,
        test_size=test_size,
        random_state=random_state
    )
    
    print(f"  - Training samples: {len(X_bert_train)}")
    print(f"  - Testing samples: {len(X_bert_test)}")
    
    # Prepare features - FIT SCALER ON TRAINING DATA ONLY
    print("\nPreparing features (fitting scaler on training data only)...")
    X_train, scaler = prepare_features(
        X_bert_train, X_stock_train,
        normalize=True,
        scaler=None,
        fit_scaler=True
    )
    
    # Transform test data using the fitted scaler
    print("Transforming test data with fitted scaler...")
    X_test, _ = prepare_features(
        X_bert_test, X_stock_test,
        normalize=True,
        scaler=scaler,
        fit_scaler=False
    )
    
    # Save the scaler
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    save_scaler(scaler, scaler_path)
    
    # Convert labels to categorical
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    print(f"\nFinal shapes:")
    print(f"  - X_train: {X_train.shape}")
    print(f"  - X_test: {X_test.shape}")
    print(f"  - y_train: {y_train.shape}")
    print(f"  - y_test: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler


def build_lstm_model(input_shape, num_classes=2):
    """Build LSTM baseline model."""
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(100))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model


def build_lstm_gru_model(input_shape, num_classes=2):
    """Build LSTM + GRU hybrid model."""
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(80, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(64))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model


def build_bidirectional_model(input_shape, num_classes=2):
    """Build Bidirectional LSTM + GRU model."""
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Bidirectional(GRU(80, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(GRU(64)))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model


def train_model(model, model_name, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    """Train a model with checkpointing."""
    print(f"\n{'='*80}")
    print(f"Training {model_name}")
    print(f"{'='*80}")
    
    # Set up checkpoint
    model_path = os.path.join(MODEL_DIR, f"{model_name.lower()}_model.h5")
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint],
        verbose=1
    )
    
    # Save history
    history_path = os.path.join(MODEL_DIR, f"{model_name.lower()}_history.pckl")
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"  - History saved to {history_path}")
    
    # Evaluate
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='weighted')),
        'recall': float(recall_score(y_true, y_pred, average='weighted')),
        'f1': float(f1_score(y_true, y_pred, average='weighted'))
    }
    
    print(f"\n{model_name} Metrics:")
    for metric, value in metrics.items():
        print(f"  - {metric.capitalize()}: {value:.4f}")
    
    return history, metrics


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train Stock Movement Prediction models')
    parser.add_argument('--force-bert', action='store_true', 
                       help='Force recomputation of BERT embeddings')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("Stock Movement Prediction - Training Pipeline")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Load datasets
    stock_df, tweets_df, timestamp_col = load_datasets()
    
    # Step 2: Compute BERT embeddings
    bert_embeddings = compute_or_load_bert_embeddings(tweets_df, force_compute=args.force_bert)
    
    # Step 3: Align and prepare dataset
    bert_features, stock_features, labels, merged_df = prepare_dataset(
        stock_df, tweets_df, bert_embeddings, timestamp_col
    )
    
    # Step 4: Split and prepare features
    X_train, X_test, y_train, y_test, scaler = split_and_prepare_features(
        bert_features, stock_features, labels
    )
    
    # Step 5: Train models
    print("\n" + "="*80)
    print("STEP 5: Model Training")
    print("="*80)
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"Model input shape: {input_shape}")
    
    all_metrics = {}
    
    # Train LSTM
    lstm_model = build_lstm_model(input_shape)
    _, lstm_metrics = train_model(
        lstm_model, "lstm", X_train, y_train, X_test, y_test,
        epochs=args.epochs, batch_size=args.batch_size
    )
    all_metrics['lstm'] = lstm_metrics
    
    # Train LSTM + GRU
    lstm_gru_model = build_lstm_gru_model(input_shape)
    _, lstm_gru_metrics = train_model(
        lstm_gru_model, "propose", X_train, y_train, X_test, y_test,
        epochs=args.epochs, batch_size=args.batch_size
    )
    all_metrics['propose'] = lstm_gru_metrics
    
    # Train Bidirectional
    bidirectional_model = build_bidirectional_model(input_shape)
    _, extension_metrics = train_model(
        bidirectional_model, "extension", X_train, y_train, X_test, y_test,
        epochs=args.epochs, batch_size=args.batch_size
    )
    all_metrics['extension'] = extension_metrics
    
    # Save all metrics
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n{'='*80}")
    print(f"All metrics saved to {metrics_path}")
    
    # Summary
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print("\nModel Comparison:")
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 68)
    for model_name, metrics in all_metrics.items():
        print(f"{model_name:<20} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nGenerated artifacts:")
    print(f"  - model/scaler.pkl")
    print(f"  - model/bert.npy")
    print(f"  - model/lstm_model.h5, lstm_history.pckl")
    print(f"  - model/propose_model.h5, propose_history.pckl")
    print(f"  - model/extension_model.h5, extension_history.pckl")
    print(f"  - model/metrics.json")
    print("\nTo run the dashboard: streamlit run app.py")


if __name__ == "__main__":
    main()
