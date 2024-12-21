import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau as TFReduceLROnPlateau
from torch.optim.lr_scheduler import ReduceLROnPlateau as TorchReduceLROnPlateau
import pickle
import time
import os
from tqdm import tqdm

# Configuration Parameters
USE_SUBSET = True
SUBSET_SIZE = 1000
MAX_LENGTH = 128
BATCH_SIZE = 32
VAL_BATCH_SIZE = 64
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
VAL_SIZE = 0.2
RANDOM_SEED = 42
SAVE_DIR = 'models'
DATA_DIR = 'datasets'
TARGET_COLUMNS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def set_seeds():
    """Set random seeds for reproducibility"""
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

def create_balanced_sample(df, target_columns, size):
    """Create a balanced sample from the dataset"""
    # Get toxic and non-toxic samples
    toxic_samples = df[df[target_columns].any(axis=1)]
    non_toxic_samples = df[~df[target_columns].any(axis=1)]
    
    # Calculate balanced sample sizes
    n_toxic = min(len(toxic_samples), size // 2)
    n_non_toxic = size - n_toxic
    
    # Sample and combine
    sampled_toxic = toxic_samples.sample(n=n_toxic, random_state=RANDOM_SEED)
    sampled_non_toxic = non_toxic_samples.sample(n=n_non_toxic, random_state=RANDOM_SEED)
    
    return pd.concat([sampled_toxic, sampled_non_toxic]).sample(frac=1)

class ToxicDataset(Dataset):
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
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc='Training'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate_bert(model, val_loader, criterion, device):
    """Enhanced validation with better metrics tracking"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            
            total_loss += loss.item()
            
            preds = torch.sigmoid(outputs.logits).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracies = ((all_preds > 0.5) == all_labels).mean(axis=0)
    
    return total_loss / len(val_loader), accuracies

def train_bert_model():
    """Enhanced BERT model with improved handling of class imbalance"""
    print("\nStarting DistilBERT training with improvements...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    if USE_SUBSET:
        train_df = create_balanced_sample(train_df, TARGET_COLUMNS, SUBSET_SIZE)
    
    # Initialize tokenizer and model with improved configuration
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=len(TARGET_COLUMNS),
        problem_type="multi_label_classification"
    ).to(device)
    
    # Prepare data
    train_texts = train_df['comment_text'].values
    train_labels = train_df[TARGET_COLUMNS].values
    
    # Calculate positive class weights
    pos_weights = []
    for i in range(len(TARGET_COLUMNS)):
        pos_count = np.sum(train_labels[:, i])
        neg_count = len(train_labels) - pos_count
        pos_weights.append(neg_count / pos_count)
    pos_weight = torch.tensor(pos_weights).to(device)
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, 
        test_size=VAL_SIZE, 
        random_state=RANDOM_SEED,
        stratify=train_labels.any(axis=1)
    )
    
    # Create datasets with improved handling
    train_dataset = ToxicDataset(train_texts, train_labels, tokenizer)
    val_dataset = ToxicDataset(val_texts, val_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE)
    
    # Enhanced training setup with fixed scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = TorchReduceLROnPlateau(  # Using PyTorch's ReduceLROnPlateau
        optimizer, 
        mode='min', 
        patience=2, 
        factor=0.5, 
        verbose=True
    )
    
    # Training loop with improved monitoring
    start_time = time.time()
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_accuracies': []}
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        
        train_loss = train_bert_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, accuracies = validate_bert(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracies'].append(accuracies)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print("Validation Accuracies:", dict(zip(TARGET_COLUMNS, accuracies)))
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(SAVE_DIR, 'bert_best_model.pt'))
    
    # Save final artifacts
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'bert_final_model.pt'))
    tokenizer.save_pretrained(os.path.join(SAVE_DIR, 'bert_tokenizer'))
    
    # Save training history
    with open(os.path.join(SAVE_DIR, 'bert_history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    
    print(f"\nBERT Training time: {time.time() - start_time:.2f} seconds")
    
    return model, history, tokenizer

def train_lstm_model():
    """Train LSTM model with improved architecture and handling of class imbalance"""
    print("\nStarting LSTM training...")
    
    # Load data
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    
    if USE_SUBSET:
        print(f"Using balanced subset of {SUBSET_SIZE} samples")
        train_df = create_balanced_sample(train_df, TARGET_COLUMNS, SUBSET_SIZE)
    
    # Prepare data
    X = train_df['comment_text']
    y = train_df[TARGET_COLUMNS]
    
    # Enhanced TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=15000,
        strip_accents='unicode',
        lowercase=True,
        analyzer='word',
        stop_words='english',
        ngram_range=(1, 3),
        min_df=3
    )
    
    X_tfidf = vectorizer.fit_transform(X)
    X_dense = X_tfidf.toarray()
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_dense, y, 
        test_size=VAL_SIZE, 
        random_state=RANDOM_SEED,
        stratify=y.any(axis=1)
    )
    
    # Calculate class weights with safety check
    class_weights = {}
    for i, column in enumerate(TARGET_COLUMNS):
        neg_count = len(y_train) - y_train[column].sum()
        pos_count = max(1, y_train[column].sum())  # Avoid division by zero
        class_weights[i] = {0: 1.0, 1: neg_count/pos_count}
    
    # Improved LSTM architecture
    model = Sequential([
        Embedding(input_dim=X_train.shape[1], output_dim=100),
        Bidirectional(LSTM(100, return_sequences=True)),
        Dropout(0.4),
        Bidirectional(LSTM(50)),
        Dropout(0.4),
        Dense(100, activation='relu'),
        Dropout(0.3),
        Dense(50, activation='relu'),
        Dense(len(TARGET_COLUMNS), activation='sigmoid')
    ])
    
    # Enhanced callbacks for better training
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        TFReduceLROnPlateau(  # Using TensorFlow's ReduceLROnPlateau
            monitor='val_loss',
            factor=0.5,
            patience=2,
            verbose=1,
            min_lr=1e-6
        )
    ]
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    
    # Train model
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights[0],
        verbose=1
    )
    
    print(f"\nLSTM Training time: {time.time() - start_time:.2f} seconds")
    
    # Evaluate and save model
    model.save(os.path.join(SAVE_DIR, 'lstm_model_improved.h5'))
    with open(os.path.join(SAVE_DIR, 'tfidf_vectorizer_improved.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    
    return model, history, vectorizer

def main():
    """Main training pipeline with improved logging and error handling"""
    print("Starting toxic comment classification training pipeline...")
    print(f"{'Using subset of data' if USE_SUBSET else 'Using full dataset'}")
    print(f"Sample size: {SUBSET_SIZE if USE_SUBSET else 'FULL'}")
    
    try:
        # Setup
        ensure_model_directory()
        set_seeds()
        
        # Train LSTM model
        lstm_model, lstm_history, vectorizer = train_lstm_model()
        
        # Train BERT model
        bert_model, bert_history, tokenizer = train_bert_model()
        
        print("\nTraining complete! All models and artifacts saved in ./models/")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
