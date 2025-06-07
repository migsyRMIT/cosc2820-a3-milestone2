# COSC2820-a3-milestone2
# Renato Miguel Alvarez - s3969740
# Clothing Review System

This is a Flask-based web application for a clothing review system that includes machine learning capabilities for predicting whether a review would recommend a product. The system uses FastText embeddings and Logistic Regression for predictions.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Extract the zip file to your desired location:
```bash
unzip cosc2820-a3-milestone2.zip
cd cosc2820-a3-milestone2
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Download required NLTK data:
```bash
python download_nltk_data.py
```

## Project Structure

- `app.py`: Main Flask application
- `train_lr_model.py`: Script for training the Logistic Regression model
- `milestone2FT.model`: Pre-trained FastText model
- `milestone2FT_LR.pkl`: Trained Logistic Regression model
- `assignment3_II.csv`: Dataset containing clothing reviews
- `templates/`: Directory containing HTML templates
- `requirements.txt`: List of Python dependencies

## Running the Application

1. Make sure all the required files are present:
   - `milestone2FT.model`
   - `milestone2FT_LR.pkl`
   - `assignment3_II.csv`

2. Start the Flask application:
```bash
python app.py
```

3. Open your web browser and navigate to:
```
http://localhost:5000
```

## Features

- Browse clothing items with search functionality
- View detailed item information and reviews
- Submit new reviews with automatic recommendation prediction
- Pagination for both items and reviews
- Machine learning-based recommendation prediction

## Dependencies

The project uses the following main dependencies:
- Flask 3.0.2
- pandas 2.2.1
- nltk 3.8.1
- numpy 1.26.4
- scikit-learn 1.4.1.post1

## Notes

- The application uses FastText embeddings for text processing
- The recommendation system is based on a Logistic Regression model
- All reviews are stored in the CSV file
- The system includes lemmatization for better text processing