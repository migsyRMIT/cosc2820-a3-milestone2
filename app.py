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

def lemmatize_text(text):
    """Lemmatize text and return both lemmas and sentiment words"""
    lemmatizer = WordNetLemmatizer()
    # Convert to lowercase and split
    words = text.lower().split()
    # Lemmatize words
    lemmas = [lemmatizer.lemmatize(word) for word in words]
    return lemmas

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