import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from tensorflow.keras.models import load_model
import pickle
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

# Configuration
CONFIG = {
    'BATCH_SIZE': 64,
    'MAX_LENGTH': 128,
    'SAVE_DIR': 'models',
    'DATA_DIR': 'datasets',
    'TARGET_COLUMNS': ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
}

class ToxicDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
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
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

def enhanced_text_preprocessing(text):
    """
    Improved text preprocessing for consistent test data handling
    """
    # Convert to lowercase
    text = text.lower()
    
    # Handle contractions
    contractions = {
        "won't": "will not", "can't": "cannot", "n't": " not",
        "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
        "'ve": " have", "'m": " am"
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    # Remove special characters but keep important punctuation
    text = ''.join([char if char.isalnum() or char.isspace() or char in '.,!?' else ' ' for char in text])
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def plot_roc_curves(y_true, y_pred, model_name):
    """
    Create and save ROC curves for each toxicity category
    """
    plt.figure(figsize=(12, 8))
    
    for i, category in enumerate(CONFIG['TARGET_COLUMNS']):
        # Convert to binary format
        y_true_binary = y_true[:, i].astype(int)
        y_pred_binary = y_pred[:, i]
        
        # Calculate ROC curve
        try:
            fpr, tpr, _ = roc_curve(y_true_binary, y_pred_binary)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{category} (AUC = {roc_auc:.2f})')
        except Exception as e:
            print(f"Warning: Could not calculate ROC curve for {category}: {str(e)}")
            continue
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curves')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(CONFIG['SAVE_DIR'], f'{model_name}_roc_curves.png'))
    plt.close()

def create_confusion_matrix(y_true, y_pred, model_name):
    """
    Create and save confusion matrix visualization for all categories
    """
    plt.figure(figsize=(15, 10))
    for i, category in enumerate(CONFIG['TARGET_COLUMNS']):
        plt.subplot(2, 3, i+1)
        cm = pd.crosstab(
            y_true[:, i], 
            (y_pred[:, i] > 0.5).astype(int),
            margins=True
        )
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{category} Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['SAVE_DIR'], f'{model_name}_confusion_matrices.png'))
    plt.close()

def analyze_error_cases(texts, y_true, y_pred, model_name):
    """
    Analyze and save examples of prediction errors for each category
    """
    error_analysis = []
    
    for i, category in enumerate(CONFIG['TARGET_COLUMNS']):
        # Convert predictions to binary
        y_pred_binary = (y_pred[:, i] > 0.5).astype(int)
        
        # Find false positives and false negatives
        false_positives = (y_pred_binary == 1) & (y_true[:, i] == 0)
        false_negatives = (y_pred_binary == 0) & (y_true[:, i] == 1)
        
        # Sample error cases
        fp_examples = pd.DataFrame({
            'category': category,
            'text': texts[false_positives],
            'true_label': y_true[false_positives, i],
            'predicted_score': y_pred[false_positives, i],
            'error_type': 'False Positive'
        }).head(3)
        
        fn_examples = pd.DataFrame({
            'category': category,
            'text': texts[false_negatives],
            'true_label': y_true[false_negatives, i],
            'predicted_score': y_pred[false_negatives, i],
            'error_type': 'False Negative'
        }).head(3)
        
        error_analysis.append(pd.concat([fp_examples, fn_examples]))
    
    # Combine and save error analysis
    all_errors = pd.concat(error_analysis, ignore_index=True)
    all_errors.to_csv(
        os.path.join(CONFIG['SAVE_DIR'], f'{model_name}_error_analysis.csv'),
        index=False
    )
    return all_errors

def test_lstm_model():
    """
    Test LSTM model with comprehensive evaluation
    """
    print("\nTesting LSTM model...")
    
    # Load model and vectorizer
    print("Loading LSTM model and vectorizer...")
    model = load_model(os.path.join(CONFIG['SAVE_DIR'], 'lstm_model_improved.h5'))
    with open(os.path.join(CONFIG['SAVE_DIR'], 'tfidf_vectorizer_improved.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Load and preprocess test data
    test_df = pd.read_csv(os.path.join(CONFIG['DATA_DIR'], 'test.csv'))
    test_texts = test_df['comment_text'].apply(enhanced_text_preprocessing)
    
    # Transform test data
    print("Transforming test data...")
    X_test = vectorizer.transform(test_texts).toarray()
    
    # Load test labels if available
    try:
        test_labels = pd.read_csv(os.path.join(CONFIG['DATA_DIR'], 'test_labels.csv'))
        y_test = test_labels[CONFIG['TARGET_COLUMNS']].values
        have_labels = True
        print("Test labels found and loaded")
    except:
        have_labels = False
        print("No test labels found - will only generate predictions")
    
    # Generate predictions
    print("Generating predictions...")
    predictions = model.predict(X_test, batch_size=CONFIG['BATCH_SIZE'])
    
    # Create predictions dataframe
    predictions_df = pd.DataFrame(predictions, columns=CONFIG['TARGET_COLUMNS'])
    predictions_df['id'] = test_df['id']
    predictions_df = predictions_df[['id'] + CONFIG['TARGET_COLUMNS']]
    
    # Save predictions
    predictions_df.to_csv(os.path.join(CONFIG['SAVE_DIR'], 'lstm_predictions.csv'), index=False)
    print("LSTM Predictions saved to ./models/lstm_predictions.csv")

def test_bert_model():
    """
    Test BERT model with comprehensive evaluation
    """
    print("\nTesting BERT model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model and tokenizer
    print("Loading BERT model and tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained(os.path.join(CONFIG['SAVE_DIR'], 'bert_tokenizer'))
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=len(CONFIG['TARGET_COLUMNS'])
    ).to(device)
    
    # Load best model weights
    checkpoint = torch.load(os.path.join(CONFIG['SAVE_DIR'], 'bert_best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load and preprocess test data
    test_df = pd.read_csv(os.path.join(CONFIG['DATA_DIR'], 'test.csv'))
    test_texts = test_df['comment_text'].apply(enhanced_text_preprocessing)
    
    # Create test dataset and dataloader
    test_dataset = ToxicDataset(test_texts, tokenizer, CONFIG['MAX_LENGTH'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['BATCH_SIZE'])
    
    # Load test labels if available
    try:
        test_labels = pd.read_csv(os.path.join(CONFIG['DATA_DIR'], 'test_labels.csv'))
        y_test = test_labels[CONFIG['TARGET_COLUMNS']].values
        have_labels = True
        print("Test labels found and loaded")
    except:
        have_labels = False
        print("No test labels found - will only generate predictions")
    
    # Generate predictions
    print("Generating predictions...")
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
    predictions_df = pd.DataFrame(predictions, columns=CONFIG['TARGET_COLUMNS'])
    predictions_df['id'] = test_df['id']
    predictions_df = predictions_df[['id'] + CONFIG['TARGET_COLUMNS']]
    
    # Save predictions
    predictions_df.to_csv(os.path.join(CONFIG['SAVE_DIR'], 'bert_predictions.csv'), index=False)
    print("BERT Predictions saved to ./models/bert_predictions.csv")
    
    # If we have labels, create evaluation visualizations
    if have_labels:
        print("\nGenerating evaluation metrics and visualizations...")
        plot_roc_curves(y_test, predictions, 'BERT')
        create_confusion_matrix(y_test, predictions, 'BERT')
        error_analysis = analyze_error_cases(test_texts.values, y_test, predictions, 'BERT')
        
        # Print classification report for each category
        for i, category in enumerate(CONFIG['TARGET_COLUMNS']):
            print(f"\nMetrics for {category}:")
            y_pred_binary = (predictions[:, i] > 0.5).astype(int)
            print(classification_report(y_test[:, i], y_pred_binary))
    
    return predictions_df

def main():
    """
    Main testing pipeline
    """
    print("Starting toxic comment classification testing pipeline...")
    
    try:
        # Test LSTM model
        lstm_predictions = test_lstm_model()
        
        # Test BERT model
        bert_predictions = test_bert_model()
        
        print("\nTesting complete! Results saved in ./models/")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
