from flask import Flask, render_template, request
import pandas as pd
import urllib.parse
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    # Simple word splitting instead of using punkt tokenizer
    words = text.lower().split()
    return [lemmatizer.lemmatize(word) for word in words]

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

@app.route('/item/<int:item_id>/<path:title>')
def item(item_id, title):
    # Read the CSV file
    df = pd.read_csv('assignment3_II.csv')
    
    # Get the specific item's details using both ID and title
    item_details = df[(df['Clothing ID'] == item_id) & (df['Clothes Title'] == title)].iloc[0]
    
    # Get all reviews for this specific item
    all_reviews = df[(df['Clothing ID'] == item_id) & (df['Clothes Title'] == title)][['Review Text', 'Rating', 'Title', 'Recommended IND']].to_dict('records')
    
    # Review pagination
    review_page = request.args.get('review_page', 1, type=int)
    reviews_per_page = 5
    total_reviews = len(all_reviews)
    total_review_pages = (total_reviews + reviews_per_page - 1) // reviews_per_page
    
    # Get reviews for current page
    start_idx = (review_page - 1) * reviews_per_page
    end_idx = start_idx + reviews_per_page
    current_reviews = all_reviews[start_idx:end_idx]
    
    # Prepare item data
    item_data = {
        'id': item_id,
        'title': item_details['Clothes Title'],
        'description': item_details['Clothes Description'],
        'division': item_details['Division Name'],
        'department': item_details['Department Name'],
        'class': item_details['Class Name'],
        'reviews': current_reviews,
        'current_review_page': review_page,
        'total_review_pages': total_review_pages,
        'total_reviews': total_reviews
    }
    
    return render_template('item.html', item=item_data)