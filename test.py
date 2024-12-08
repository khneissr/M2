# Import required libraries
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from tensorflow.keras.models import load_model
import pickle
from sklearn.metrics import classification_report
from tqdm import tqdm

# Global configuration parameters
MAX_LENGTH = 128  # Maximum sequence length for BERT tokenizer
BATCH_SIZE = 64   # Batch size for testing
SAVE_DIR = 'models'  # Directory containing saved models
DATA_DIR = 'datasets'  # Directory containing test data

class ToxicDataset(Dataset):
    """Custom Dataset class for toxic comment data with BERT tokenization"""
    def __init__(self, texts, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
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
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

def load_test_data():
    """Load test data and labels if available"""
    print("Loading test data...")
    test_df = pd.read_csv(f'{DATA_DIR}/test.csv')
    
    try:
        test_labels = pd.read_csv(f'{DATA_DIR}/test_labels.csv')
        print("Test labels found and loaded")
    except:
        test_labels = None
        print("No test labels found - will only generate predictions")
    
    return test_df['comment_text'], test_df['id'], test_labels

def evaluate_predictions(predictions_df, test_labels, model_name):
    """Evaluate model predictions against test labels"""
    if test_labels is None:
        return
    
    print(f"\nEvaluation Results for {model_name}:")
    columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    for column in columns:
        print(f"\nMetrics for {column}:")
        y_pred = (predictions_df[column] > 0.5).astype(int)
        y_true = test_labels[column]
        print(classification_report(y_true, y_pred))

def test_lstm_model():
    """Test LSTM model on test data"""
    print("\nTesting LSTM model...")
    
    # Load saved model and vectorizer
    print("Loading LSTM model and vectorizer...")
    model = load_model(f'{SAVE_DIR}/toxic_comment_lstm.h5')
    with open(f'{SAVE_DIR}/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Load test data
    X_test, test_ids, test_labels = load_test_data()
    
    # Transform test data using saved vectorizer
    print("Transforming test data...")
    X_test_tfidf = vectorizer.transform(X_test)
    X_test_dense = X_test_tfidf.toarray()
    
    # Make predictions
    print("Making LSTM predictions...")
    predictions = model.predict(X_test_dense)
    
    # Create predictions dataframe
    columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    predictions_df = pd.DataFrame(predictions, columns=columns)
    predictions_df['id'] = test_ids
    predictions_df = predictions_df[['id'] + columns]
    
    # Save predictions
    predictions_df.to_csv(f'{SAVE_DIR}/lstm_predictions.csv', index=False)
    print("LSTM Predictions saved to ./models/lstm_predictions.csv")
    
    # Print prediction statistics
    print("\nLSTM Prediction Statistics:")
    for column in columns:
        pos_preds = (predictions_df[column] > 0.5).sum()
        mean_pred = predictions_df[column].mean()
        print(f"{column}:")
        print(f"  Positive predictions: {pos_preds}")
        print(f"  Mean prediction: {mean_pred:.4f}")
    
    # Evaluate predictions if labels are available
    evaluate_predictions(predictions_df, test_labels, "LSTM")
    
    return predictions_df

def test_bert_model():
    """Test BERT model on test data"""
    print("\nTesting BERT model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load saved model and tokenizer
    print("Loading BERT model and tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained(f'{SAVE_DIR}/bert_tokenizer')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=6  # Six toxicity categories
    ).to(device)
    
    # Load best model weights
    checkpoint = torch.load(f'{SAVE_DIR}/bert_best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test data
    X_test, test_ids, test_labels = load_test_data()
    
    # Create test dataset and dataloader
    test_dataset = ToxicDataset(X_test, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Make predictions
    print("Making BERT predictions...")
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Predicting'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.sigmoid(outputs.logits).cpu().numpy()
            all_predictions.extend(predictions)
    
    predictions = np.array(all_predictions)
    
    # Create predictions dataframe
    columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    predictions_df = pd.DataFrame(predictions, columns=columns)
    predictions_df['id'] = test_ids
    predictions_df = predictions_df[['id'] + columns]
    
    # Save predictions
    predictions_df.to_csv(f'{SAVE_DIR}/bert_predictions.csv', index=False)
    print("BERT Predictions saved to ./models/bert_predictions.csv")
    
    # Print prediction statistics
    print("\nBERT Prediction Statistics:")
    for column in columns:
        pos_preds = (predictions_df[column] > 0.5).sum()
        mean_pred = predictions_df[column].mean()
        print(f"{column}:")
        print(f"  Positive predictions: {pos_preds}")
        print(f"  Mean prediction: {mean_pred:.4f}")
    
    # Evaluate predictions if labels are available
    evaluate_predictions(predictions_df, test_labels, "BERT")
    
    return predictions_df

def main():
    """Main testing pipeline for toxic comment classification"""
    print("Starting toxic comment classification testing pipeline...")
    
    # Test both models sequentially
    lstm_predictions = test_lstm_model()
    bert_predictions = test_bert_model()
    
    print("\nTesting complete! Predictions saved in ./models/")

if __name__ == "__main__":
    main()
