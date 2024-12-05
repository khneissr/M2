import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import pickle
import time
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

# Control variables
USE_SUBSET = True
SUBSET_SIZE = 1000

def ensure_model_directory():
    """Create models directory if it doesn't exist"""
    if not os.path.exists('./models'):
        os.makedirs('./models')
        print("Created models directory")

def load_and_preprocess_data():
    """Load and preprocess the toxic comment data"""
    print("Loading and preprocessing data...")
    train_df = pd.read_csv('./datasets/train.csv')
    
    if USE_SUBSET:
        print(f"Using subset of {SUBSET_SIZE} samples")
        train_df = train_df.sample(n=SUBSET_SIZE, random_state=42)
    else:
        print(f"Using full dataset of {len(train_df)} samples")
    
    # Define target columns
    target_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Get features and labels
    X = train_df['comment_text']
    y = train_df[target_columns]
    
    # Create TF-IDF features
    vectorizer = TfidfVectorizer(
        max_features=10000, 
        strip_accents='unicode',
        lowercase=True,
        analyzer='word',
        stop_words='english',
        ngram_range=(1, 2)
    )
    X_tfidf = vectorizer.fit_transform(X)
    
    return X_tfidf, y, vectorizer

def train_lstm_model():
    """Train LSTM model"""
    print("\nStarting LSTM training...")
    
    # Load data
    X_tfidf, y, vectorizer = load_and_preprocess_data()
    
    # Convert sparse matrix to dense array for LSTM
    print("Converting features to dense array...")
    X_dense = X_tfidf.toarray()
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_dense, y, test_size=0.2, random_state=42
    )
    
    print("Building LSTM model...")
    # Build LSTM model
    model = Sequential([
        Embedding(input_dim=X_train.shape[1], output_dim=64, input_length=X_train.shape[1]),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(6, activation='sigmoid')  # 6 outputs for all toxicity types
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Add early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=2,
        restore_best_weights=True
    )
    
    # Train model
    print("\nStarting LSTM training...")
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=64,
        callbacks=[early_stopping],
        verbose=1
    )
    
    end_time = time.time()
    print(f"\nTotal training time: {end_time - start_time:.2f} seconds")
    
    # Evaluate final model
    print("\nEvaluating model on validation set:")
    y_pred = model.predict(X_val)
    target_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    for i, column in enumerate(target_names):
        print(f"\nMetrics for {column}:")
        # Convert predictions to binary using 0.5 threshold
        y_pred_binary = (y_pred[:, i] > 0.5).astype(int)
        print(classification_report(y_val.iloc[:, i], y_pred_binary))
    
    # Save model and vectorizer
    print("\nSaving model and vectorizer...")
    ensure_model_directory()
    model_suffix = 'subset' if USE_SUBSET else 'full'
    model.save(f'./models/toxic_comment_lstm_{model_suffix}.h5')
    with open(f'./models/vectorizer_{model_suffix}.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    return model, history, vectorizer

if __name__ == "__main__":
    print(f"Starting toxic comment classification training pipeline...")
    print(f"{'Using subset of data' if USE_SUBSET else 'Using full dataset'}")
    print(f"Sample size: {SUBSET_SIZE if USE_SUBSET else 'FULL'}")
    model, history, vectorizer = train_lstm_model()
    print("\nTraining complete! Models saved in ./models/")
