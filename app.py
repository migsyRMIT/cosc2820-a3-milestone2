from flask import Flask, render_template, request
import pandas as pd
import urllib.parse

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/shop')
def shop():
    # Read the CSV file
    df = pd.read_csv('assignment3_II.csv')
    # Get unique clothing items with their IDs and titles
    clothing_items = df[['Clothing ID', 'Clothes Title']].drop_duplicates().to_dict('records')
    
    # Pagination
    page = request.args.get('page', 1, type=int)
    items_per_page = 20
    total_items = len(clothing_items)
    total_pages = (total_items + items_per_page - 1) // items_per_page
    
    # Get items for current page
    start_idx = (page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    current_items = clothing_items[start_idx:end_idx]
    
    return render_template('shop.html', 
                         clothing_items=current_items,
                         current_page=page,
                         total_pages=total_pages)

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