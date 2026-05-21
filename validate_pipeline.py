#!/usr/bin/env python3
"""
Quick validation script to test the training pipeline components
without running full training.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_alignment import validate_datasets, prepare_aligned_dataset
from utils.preprocessing import prepare_features, validate_feature_shape

def test_load_datasets():
    """Test loading datasets."""
    print("\n" + "="*60)
    print("Test 1: Loading Datasets")
    print("="*60)
    
    stock_path = "Dataset/AAPL.csv"
    tweets_path = "Dataset/tweets.csv"
    
    if not os.path.exists(stock_path):
        print(f"✗ Stock data not found: {stock_path}")
        return False
    
    if not os.path.exists(tweets_path):
        print(f"✗ Tweets data not found: {tweets_path}")
        return False
    
    stock_df = pd.read_csv(stock_path)
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    
    tweets_df = pd.read_csv(tweets_path)
    
    print(f"✓ Loaded {len(stock_df)} stock records")
    print(f"✓ Loaded {len(tweets_df)} tweets")
    
    # Validate
    is_valid, errors = validate_datasets(stock_df, tweets_df)
    
    for error in errors:
        print(f"  {error}")
    
    if not is_valid:
        print("✗ Dataset validation failed")
        return False
    
    print("✓ Dataset validation passed")
    return True

def test_feature_preparation():
    """Test feature preparation."""
    print("\n" + "="*60)
    print("Test 2: Feature Preparation")
    print("="*60)
    
    # Create fake data
    n_samples = 10
    bert_embeddings = np.random.randn(n_samples, 768)
    stock_features = np.random.randn(n_samples, 4)
    
    print(f"✓ Created test BERT embeddings: {bert_embeddings.shape}")
    print(f"✓ Created test stock features: {stock_features.shape}")
    
    # Test feature preparation
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler((0, 1))
    
    try:
        X_reshaped, fitted_scaler = prepare_features(
            bert_embeddings,
            stock_features,
            normalize=True,
            scaler=scaler,
            fit_scaler=True
        )
        
        print(f"✓ Features prepared successfully")
        print(f"  - Output shape: {X_reshaped.shape}")
        print(f"  - Expected: ({n_samples}, 35, 22)")
        
        if X_reshaped.shape == (n_samples, 35, 22):
            print("✓ Feature shape is correct")
        else:
            print("✗ Feature shape is incorrect")
            return False
        
        # Test value range (allow small floating point errors)
        if X_reshaped.min() >= -1e-10 and X_reshaped.max() <= 1.0 + 1e-10:
            print("✓ Features normalized to [0, 1]")
        else:
            print(f"✗ Features not properly normalized: min={X_reshaped.min()}, max={X_reshaped.max()}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Feature preparation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_scaler_persistence():
    """Test scaler save/load."""
    print("\n" + "="*60)
    print("Test 3: Scaler Persistence")
    print("="*60)
    
    from utils.preprocessing import save_scaler, load_scaler
    from sklearn.preprocessing import MinMaxScaler
    import tempfile
    
    # Create and fit a scaler
    scaler = MinMaxScaler((0, 1))
    X = np.random.randn(100, 772)
    scaler.fit(X)
    
    # Save to temp file (using secure temporary file creation)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        temp_file = tmp.name
    
    try:
        save_scaler(scaler, temp_file)
        print(f"✓ Scaler saved to {temp_file}")
        
        # Load it back
        loaded_scaler = load_scaler(temp_file)
        
        if loaded_scaler is None:
            print("✗ Failed to load scaler")
            return False
        
        print("✓ Scaler loaded successfully")
        
        # Test that it produces same results
        test_data = np.random.randn(5, 772)
        original_transform = scaler.transform(test_data)
        loaded_transform = loaded_scaler.transform(test_data)
        
        if np.allclose(original_transform, loaded_transform):
            print("✓ Loaded scaler produces identical results")
        else:
            print("✗ Loaded scaler produces different results")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Scaler persistence test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)

def main():
    """Run all validation tests."""
    print("\n" + "="*60)
    print("Stock Movement Pipeline Validation")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Load Datasets", test_load_datasets()))
    results.append(("Feature Preparation", test_feature_preparation()))
    results.append(("Scaler Persistence", test_scaler_persistence()))
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:<30} {status}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ All validation tests passed!")
        print("\nYou can now run the training pipeline:")
        print("  python train.py")
    else:
        print("✗ Some validation tests failed")
        print("\nPlease fix the issues before running training.")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
