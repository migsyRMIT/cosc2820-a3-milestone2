import pandas as pd
import numpy as np
from gensim.models.fasttext import FastText
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
from nltk.stem import WordNetLemmatizer

def lemmatize_text(text):
    """Lemmatize text and return lemmas"""
    lemmatizer = WordNetLemmatizer()
    # Convert to lowercase and split
    words = str(text).lower().split()
    # Lemmatize words
    lemmas = [lemmatizer.lemmatize(word) for word in words]
    return lemmas

def docvecs(embeddings, docs):
    vecs = np.zeros((len(docs), embeddings.vector_size))
    for i, doc in enumerate(docs):
        valid_keys = [term for term in doc if term in embeddings.key_to_index]
        if valid_keys:  # Only process if there are valid words
            docvec = np.vstack([embeddings[term] for term in valid_keys])
            docvec = np.sum(docvec, axis=0)
            vecs[i,:] = docvec
    return vecs

def main():
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv('assignment3_II.csv')
    
    # Combine title and review text
    print("Preprocessing text...")
    df['combined_text'] = df['Title'] + ' ' + df['Review Text']
    
    # Tokenize the text
    df['tokenized_text'] = df['combined_text'].apply(lemmatize_text)
    
    # Load the FastText model
    print("Loading FastText model...")
    ft_model = FastText.load("milestone2FT.model")
    ft_wv = ft_model.wv
    
    # Generate document vectors
    print("Generating document vectors...")
    doc_vectors = docvecs(ft_wv, df['tokenized_text'].tolist())
    
    # Get the target variable
    y = df['Recommended IND'].values
    
    # Split the data
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        doc_vectors, y, test_size=0.2, random_state=42
    )
    
    # Train the model
    print("Training logistic regression model...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model
    print("Saving model...")
    with open('milestone2FT_LR.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model saved as 'milestone2FT_LR.pkl'")

if __name__ == "__main__":
    main() 