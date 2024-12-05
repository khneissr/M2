import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
from sklearn.metrics import classification_report

def load_test_data():
    """Load test data and labels if available"""
    print("Loading test data...")
    test_df = pd.read_csv('./datasets/test.csv')
    
    try:
        test_labels = pd.read_csv('./datasets/test_labels.csv')
        print("Test labels found and loaded")
    except:
        test_labels = None
        print("No test labels found - will only generate predictions")
    
    return test_df['comment_text'], test_df['id'], test_labels

def evaluate_predictions(predictions_df, test_labels):
    """Evaluate predictions against test labels"""
    if test_labels is None:
        return
    
    print("\nEvaluation Results:")
    columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    for column in columns:
        print(f"\nMetrics for {column}:")
        y_pred = (predictions_df[column] > 0.5).astype(int)
        y_true = test_labels[column]
        print(classification_report(y_true, y_pred))

def main():
    print("Loading model and vectorizer...")
    model = load_model('./models/toxic_comment_lstm_subset.h5')
    with open('./models/vectorizer_subset.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    X_test, test_ids, test_labels = load_test_data()
    
    print("Transforming test data...")
    X_test_tfidf = vectorizer.transform(X_test)
    X_test_dense = X_test_tfidf.toarray()
    
    print("Making predictions...")
    predictions = model.predict(X_test_dense)
    
    columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    predictions_df = pd.DataFrame(predictions, columns=columns)
    predictions_df['id'] = test_ids
    
    predictions_df = predictions_df[['id'] + columns]
    
    print("Saving predictions...")
    predictions_df.to_csv('./models/predictions.csv', index=False)
    print("Predictions saved to ./models/predictions.csv")
    
    evaluate_predictions(predictions_df, test_labels)

if __name__ == "__main__":
    main()