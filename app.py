from flask import Flask, render_template, request, jsonify
import pandas as pd
import urllib.parse
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.fasttext import FastText
import os
import pickle

app = Flask(__name__)

def docvecs(embeddings, docs):
    vecs = np.zeros((len(docs), embeddings.vector_size))
    for i, doc in enumerate(docs):
        valid_keys = [term for term in doc if term in embeddings.key_to_index]
        docvec = np.vstack([embeddings[term] for term in valid_keys])
        docvec = np.sum(docvec, axis=0)
        vecs[i,:] = docvec
    return vecs

# def load_glove_model(file_path='glove_doc_vectors_combined.txt'):
#     """Load GloVe model from file"""
#     try:
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"GloVe model file not found at {file_path}")
            
#         word_vectors = {}
#         with open(file_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 # Split by comma and convert to float
#                 values = line.strip().split(',')
#                 # Use the first value as the word
#                 word = values[0]
#                 # Convert the rest to float array
#                 vector = np.array([float(x) for x in values[1:]])
#                 word_vectors[word] = vector
#         return word_vectors
#     except Exception as e:
#         print(f"Error loading GloVe model: {str(e)}")
#         return None

# # Load GloVe model at startup
# glove_model = load_glove_model()

def lemmatize_text(text):
    """Lemmatize text and return both lemmas and sentiment words"""
    lemmatizer = WordNetLemmatizer()
    # Convert to lowercase and split
    words = text.lower().split()
    # Lemmatize words
    lemmas = [lemmatizer.lemmatize(word) for word in words]
    return lemmas

# def get_text_embedding(text, model):
#     """Generate embedding for input text using GloVe model with enhanced processing"""
#     if model is None:
#         return np.zeros(300)  # Return zero vector if model is not loaded
        
#     # Tokenize and lemmatize text
#     tokens = lemmatize_text(text)
    
#     # Get word vectors for each token
#     word_vectors = []
#     for token in tokens:
#         if token in model:
#             # Convert string vector to numpy array
#             vector_str = model[token]
#             if isinstance(vector_str, str):
#                 vector = np.array([float(x) for x in vector_str.split(',')])
#             else:
#                 vector = vector_str
#             word_vectors.append(vector)
    
#     if not word_vectors:
#         # If no words found in model, return zero vector
#         return np.zeros(300)
    
#     # Average the word vectors
#     return np.mean(word_vectors, axis=0)

# def gen_docVecs_combined(wv, processed_file_path):
#     """Generate document vectors from processed reviews with enhanced processing"""
#     try:
#         # Read processed reviews
#         df = pd.read_csv(processed_file_path, header=None)
        
#         # Initialize empty list for document vectors
#         doc_vectors = []
        
#         # Process each review
#         for _, row in df.iterrows():
#             # Get the review text
#             review_text = str(row[0])
            
#             # Get embedding using enhanced processing
#             doc_vector = get_text_embedding(review_text, wv)
#             doc_vectors.append(doc_vector)
        
#         return np.array(doc_vectors)
#     except Exception as e:
#         print(f"Error in gen_docVecs_combined: {str(e)}")
#         return np.array([])

# def predict_recommendation(review_title, review_text, model):
#     """Predict Recommended IND based on review text using enhanced similarity analysis"""
#     try:
#         # Combine title and text
#         combined_text = f"{review_title} {review_text}"
        
#         # Create a temporary DataFrame with the review text
#         temp_df = pd.DataFrame([combined_text])
#         temp_df.to_csv('temp_review.csv', index=False, header=False)
        
#         # Generate document vectors for the review
#         review_vectors = gen_docVecs_combined(model, 'temp_review.csv')
        
#         if len(review_vectors) == 0:
#             return 1  # Default to recommending if vector generation fails
        
#         # Load training data
#         df = pd.read_csv('assignment3_II.csv')
        
#         # Generate document vectors for all training reviews
#         training_texts = df.apply(lambda row: f"{row['Title']} {row['Review Text']}", axis=1)
#         training_df = pd.DataFrame(training_texts)
#         training_df.to_csv('temp_training.csv', index=False, header=False)
#         training_vectors = gen_docVecs_combined(model, 'temp_training.csv')
        
#         if len(training_vectors) == 0:
#             return 1  # Default to recommending if vector generation fails
        
#         # Calculate similarity with all training reviews
#         similarities = cosine_similarity(review_vectors, training_vectors)[0]
        
#         # Get top 20 most similar reviews
#         top_k = 20
#         top_indices = np.argsort(similarities)[-top_k:]
        
#         # Get the similarity scores for top reviews
#         top_similarities = similarities[top_indices]
        
#         # Normalize similarity scores to [0, 1] range
#         top_similarities = (top_similarities - top_similarities.min()) / (top_similarities.max() - top_similarities.min() + 1e-10)
        
#         # Calculate weighted scores with exponential weighting
#         weights = np.exp(top_similarities) / np.sum(np.exp(top_similarities))
        
#         # Weight the recommendations by similarity scores
#         weighted_positive = sum(weight for idx, weight in zip(top_indices, weights) 
#                               if df.iloc[idx]['Recommended IND'] == 1)
#         weighted_negative = sum(weight for idx, weight in zip(top_indices, weights) 
#                               if df.iloc[idx]['Recommended IND'] == 0)
        
#         # Calculate the ratio of positive to negative weights
#         total_weight = weighted_positive + weighted_negative
#         if total_weight > 0:
#             positive_ratio = weighted_positive / total_weight
#             # Use a dynamic threshold based on the distribution of weights
#             threshold = 0.5
#             # Adjust threshold based on the confidence of the prediction
#             confidence = abs(positive_ratio - 0.5) * 2  # Scale to [0, 1]
#             if confidence > 0.8:  # High confidence
#                 threshold = 0.4  # More lenient for positive predictions
#             elif confidence < 0.2:  # Low confidence
#                 threshold = 0.6  # More strict for positive predictions
            
#             predicted_recommendation = 1 if positive_ratio > threshold else 0
#         else:
#             # If no weights, use the most similar review's recommendation
#             most_similar_idx = top_indices[-1]
#             predicted_recommendation = df.iloc[most_similar_idx]['Recommended IND']
        
#         # Clean up temporary files
#         if os.path.exists('temp_review.csv'):
#             os.remove('temp_review.csv')
#         if os.path.exists('temp_training.csv'):
#             os.remove('temp_training.csv')
        
#         return predicted_recommendation
#     except Exception as e:
#         print(f"Error predicting recommendation: {str(e)}")
#         return 1  # Default to recommending if there's an error

def search_items(df, query):
    if not query:
        return df[['Clothing ID', 'Clothes Title']].drop_duplicates()
    
    # Lemmatize the search query
    query_lemmas = set(lemmatize_text(query))
    
    # Create a mask for matching items
    mask = pd.Series(False, index=df.index)
    
    # Search in both title and ID
    for idx, row in df.iterrows():
        # Lemmatize the title
        title_lemmas = set(lemmatize_text(str(row['Clothes Title'])))
        # Check if any query lemma matches any title lemma
        if query_lemmas.intersection(title_lemmas):
            mask[idx] = True
            continue
            
        # Check if the query matches the ID
        if query.lower() in str(row['Clothing ID']).lower():
            mask[idx] = True
    
    return df[mask][['Clothing ID', 'Clothes Title']].drop_duplicates()

def predict_recommendation_fasttext(review_title, review_text):
    """Predict Recommended IND based on review text using FastText and Logistic Regression"""
    try:
        # Combine title and text
        combined_text = f"{review_title} {review_text}"
        
        # Tokenize the text
        tokenized_data = lemmatize_text(combined_text)
        
        # Load the FastText model
        ft_model = FastText.load("milestone2FT.model")
        ft_wv = ft_model.wv
        
        # Generate vector representation of the tokenized data
        doc_vectors = docvecs(ft_wv, [tokenized_data])
        
        # Load the LR model
        pkl_filename = "milestone2FT_LR.pkl"
        with open(pkl_filename, 'rb') as file:
            model = pickle.load(file)
        
        # Predict the label
        y_pred = model.predict(doc_vectors)
        return y_pred[0]
        
    except Exception as e:
        print(f"Error predicting recommendation: {str(e)}")
        return 1  # Default to recommending if there's an error

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/shop')
def shop():
    # Read the CSV file
    df = pd.read_csv('assignment3_II.csv')
    
    # Get search query
    search_query = request.args.get('search', '').strip()
    
    # Get filtered items based on search
    clothing_items = search_items(df, search_query)
    
    # Get total number of matched items
    total_matches = len(clothing_items)
    
    # Pagination
    page = request.args.get('page', 1, type=int)
    items_per_page = 20
    total_items = total_matches
    total_pages = (total_items + items_per_page - 1) // items_per_page
    
    # Get items for current page
    start_idx = (page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    current_items = clothing_items.iloc[start_idx:end_idx].to_dict('records')
    
    return render_template('shop.html', 
                         clothing_items=current_items,
                         current_page=page,
                         total_pages=total_pages,
                         search_query=search_query,
                         total_matches=total_matches)

@app.route('/item/<int:item_id>/<path:title>', methods=['GET', 'POST'])
def item(item_id, title):
    try:
        # Read the CSV file
        df = pd.read_csv('assignment3_II.csv')
        
        # Get the specific item's details using both ID and title
        item_details = df[(df['Clothing ID'] == item_id) & (df['Clothes Title'] == title)].iloc[0]
        
        # Get all reviews for this specific item and sort in reverse order (newest first)
        item_reviews = df[(df['Clothing ID'] == item_id) & (df['Clothes Title'] == title)]
        all_reviews = item_reviews.iloc[::-1][['Review Text', 'Rating', 'Title', 'Recommended IND']].to_dict('records')
        
        if request.method == 'POST':
            try:
                # Get form data
                review_title = request.form.get('reviewTitle')
                review_text = request.form.get('reviewDescription')
                rating = int(request.form.get('rating'))
                age = int(request.form.get('userAge'))
                
                # Check if this is the final submission
                final_submission = request.form.get('final_submission')
                
                # If not final submission, just return the prediction
                if not final_submission:
                    predicted_recommendation = predict_recommendation_fasttext(review_title, review_text)
                    return jsonify({
                        'success': True,
                        'recommended': bool(predicted_recommendation)
                    })
                
                # For final submission, check if there's an override recommendation
                override_recommendation = request.form.get('override_recommendation')
                if override_recommendation is not None:
                    # Convert string 'true'/'false' to boolean
                    predicted_recommendation = 1 if override_recommendation.lower() == 'true' else 0
                else:
                    # Predict recommendation if no override
                    predicted_recommendation = predict_recommendation_fasttext(review_title, review_text)
                
                # Create new review row with all required fields
                new_review = {
                    'Clothing ID': item_id,
                    'Clothes Title': title,
                    'Clothes Description': item_details['Clothes Description'],
                    'Division Name': item_details['Division Name'],
                    'Department Name': item_details['Department Name'],
                    'Class Name': item_details['Class Name'],
                    'Title': review_title,
                    'Review Text': review_text,
                    'Rating': rating,
                    'Age': age,
                    'Recommended IND': predicted_recommendation,
                    'Positive Feedback Count': 0  # Initialize positive feedback count to 0
                }
                
                # Add new review to DataFrame
                df = pd.concat([df, pd.DataFrame([new_review])], ignore_index=True)
                
                # Save updated DataFrame to CSV
                df.to_csv('assignment3_II.csv', index=False)
                
                # Add new review to the list for immediate display
                all_reviews.insert(0, {
                    'Title': review_title,
                    'Review Text': review_text,
                    'Rating': rating,
                    'Recommended IND': predicted_recommendation
                })
                
                # Return JSON response for AJAX request
                return jsonify({
                    'success': True,
                    'recommended': bool(predicted_recommendation)
                })
            except Exception as e:
                print(f"Error processing review: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': 'Error processing review'
                }), 500
        
        # Review pagination
        review_page = request.args.get('review_page', 1, type=int)
        reviews_per_page = 5
        total_reviews = len(all_reviews)
        total_review_pages = (total_reviews + reviews_per_page - 1) // reviews_per_page
        
        # Get reviews for current page
        start_idx = (review_page - 1) * reviews_per_page
        end_idx = start_idx + reviews_per_page
        current_reviews = all_reviews[start_idx:end_idx]
        
        return render_template('item.html',
                             item={
                                 'id': item_id,
                                 'title': title,
                                 'description': item_details['Clothes Description'],
                                 'division': item_details['Division Name'],
                                 'department': item_details['Department Name'],
                                 'class': item_details['Class Name'],
                                 'reviews': current_reviews,
                                 'total_reviews': total_reviews,
                                 'current_review_page': review_page,
                                 'total_review_pages': total_review_pages
                             })
    except Exception as e:
        print(f"Error in item route: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500