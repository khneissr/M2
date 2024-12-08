# Import required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import pickle
import time
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm

# Global configuration parameters
USE_SUBSET = True  # Whether to use a subset of data
SUBSET_SIZE = 10000  # Size of data subset if USE_SUBSET is True
MAX_LENGTH = 128  # Maximum sequence length for BERT tokenizer
BATCH_SIZE = 32  # Training batch size
VAL_BATCH_SIZE = 64  # Validation batch size
LEARNING_RATE = 2e-5  # Learning rate for BERT model
NUM_EPOCHS = 3  # Number of training epochs
VAL_SIZE = 0.2  # Validation split ratio
RANDOM_SEED = 42  # Random seed for reproducibility
SAVE_DIR = 'models'  # Directory to save model artifacts
DATA_DIR = 'datasets'  # Directory containing input data

def set_seeds():
    """Set random seeds for reproducibility across all libraries"""
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

def ensure_model_directory():
    """Create models directory if it doesn't exist"""
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print("Created models directory")

# LSTM Model Training
def train_lstm_model():
    """Train LSTM model with TF-IDF features for toxic comment classification"""
    print("\nStarting LSTM training...")
    
    # Load and preprocess data
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    
    if USE_SUBSET:
        print(f"Using subset of {SUBSET_SIZE} samples")
        train_df = train_df.sample(n=SUBSET_SIZE, random_state=RANDOM_SEED)
    
    # Define target categories for toxicity classification
    target_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    X = train_df['comment_text']
    y = train_df[target_columns]
    
    # Create TF-IDF features
    vectorizer = TfidfVectorizer(
        max_features=10000,
        strip_accents='unicode',
        lowercase=True,
        analyzer='word',
        stop_words='english',
        ngram_range=(1, 2)  # Use both unigrams and bigrams
    )
    X_tfidf = vectorizer.fit_transform(X)
    X_dense = X_tfidf.toarray()
    
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_dense, y, test_size=VAL_SIZE, random_state=RANDOM_SEED
    )
    
    # Define LSTM model architecture
    model = Sequential([
        Embedding(input_dim=X_train.shape[1], output_dim=64, input_length=X_train.shape[1]),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(6, activation='sigmoid')  # 6 outputs for toxicity categories
    ])
    
    # Compile model with binary cross-entropy for multi-label classification
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Add early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=2,
        restore_best_weights=True
    )
    
    # Train the model
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )
    
    print(f"\nLSTM Training time: {time.time() - start_time:.2f} seconds")
    
    # Evaluate model performance
    y_pred = model.predict(X_val)
    for i, column in enumerate(target_columns):
        print(f"\nLSTM Metrics for {column}:")
        y_pred_binary = (y_pred[:, i] > 0.5).astype(int)
        print(classification_report(y_val.iloc[:, i], y_pred_binary))
    
    # Save model artifacts
    model.save(os.path.join(SAVE_DIR, 'toxic_comment_lstm.h5'))
    with open(os.path.join(SAVE_DIR, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    
    return model, history, vectorizer

# BERT Model Components
class ToxicDataset(Dataset):
    """Custom Dataset class for toxic comment data with BERT tokenization"""
    def __init__(self, texts, labels=None, tokenizer=None, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = labels is None
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Tokenize text with BERT tokenizer
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }
        
        if not self.is_test:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        
        return item

def train_bert_epoch(model, train_loader, optimizer, criterion, device):
    """Train BERT model for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc='Training'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass and loss calculation
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate_bert(model, val_loader, criterion, device):
    """Validate BERT model performance"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            
            total_loss += loss.item()
            
            # Collect predictions and labels for accuracy calculation
            preds = torch.sigmoid(outputs.logits).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate accuracy for each toxicity category
    accuracies = ((all_preds > 0.5) == all_labels).mean(axis=0)
    
    return total_loss / len(val_loader), accuracies

def train_bert_model():
    """Train DistilBERT model for toxic comment classification"""
    print("\nStarting DistilBERT training...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and preprocess data
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    if USE_SUBSET:
        train_df = train_df.sample(n=SUBSET_SIZE, random_state=RANDOM_SEED)
    
    # Initialize BERT tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=6  # Six toxicity categories
    ).to(device)
    
    # Prepare data
    target_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    train_texts = train_df['comment_text'].values
    train_labels = train_df[target_columns].values
    
    # Split data into train and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=VAL_SIZE, random_state=RANDOM_SEED
    )
    
    # Create datasets and dataloaders
    train_dataset = ToxicDataset(train_texts, train_labels, tokenizer)
    val_dataset = ToxicDataset(val_texts, val_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE)
    
    # Initialize optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Training loop
    start_time = time.time()
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_accuracies': []}
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        
        # Train and validate for one epoch
        train_loss = train_bert_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, accuracies = validate_bert(model, val_loader, criterion, device)
        
        # Store metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracies'].append(accuracies)
        
        # Print progress
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print("Validation Accuracies:", 
              dict(zip(target_columns, accuracies)))
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(SAVE_DIR, 'bert_best_model.pt'))
    
    print(f"\nBERT Training time: {time.time() - start_time:.2f} seconds")
    
    # Save final model and artifacts
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'bert_final_model.pt'))
    tokenizer.save_pretrained(os.path.join(SAVE_DIR, 'bert_tokenizer'))
    with open(os.path.join(SAVE_DIR, 'bert_training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    
    return model, history, tokenizer

def main():
    """Main training pipeline for toxic comment classification"""
    print("Starting toxic comment classification training pipeline...")
    print(f"{'Using subset of data' if USE_SUBSET else 'Using full dataset'}")
    print(f"Sample size: {SUBSET_SIZE if USE_SUBSET else 'FULL'}")
    
    ensure_model_directory()
    set_seeds()
    
    # Train both models sequentially
    lstm_model, lstm_history, vectorizer = train_lstm_model()
    bert_model, bert_history, tokenizer = train_bert_model()
    
    print("\nTraining complete! All models saved in ./models/")

if __name__ == "__main__":
    main()
