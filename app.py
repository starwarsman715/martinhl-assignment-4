# app.py

from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords
import string
import re

# Initialize Flask app
app = Flask(__name__)

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')  # Required for tokenization if needed
nltk.download('wordnet')  # Required for lemmatization if implemented
nltk.download('averaged_perceptron_tagger')  # Required for POS tagging if implemented

# Fetch the 20 Newsgroups dataset with all parts included
newsgroups = fetch_20newsgroups(subset='all')  # No 'remove' parameter to include headers

# Preprocessing function: Remove punctuation and lowercase the text
def preprocess(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.lower().translate(translator)

processed_documents = [preprocess(doc) for doc in newsgroups.data]

# Extract 'From' and 'Subject' fields
def extract_headers(text):
    """
    Extracts 'From' and 'Subject' from a document's headers.
    Returns a tuple (from_field, subject_field).
    """
    from_match = re.search(r'^From:\s*(.*)', text, re.MULTILINE | re.IGNORECASE)
    subject_match = re.search(r'^Subject:\s*(.*)', text, re.MULTILINE | re.IGNORECASE)
    
    from_field = from_match.group(1).strip() if from_match else 'Unknown'
    subject_field = subject_match.group(1).strip() if subject_match else 'No Subject'
    
    return from_field, subject_field

# Create lists to store headers
from_fields = []
subject_fields = []

for doc in newsgroups.data:
    from_f, subject_f = extract_headers(doc)
    from_fields.append(from_f)
    subject_fields.append(subject_f)

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), max_features=10000)
tfidf_matrix = vectorizer.fit_transform(processed_documents)

# Apply Truncated SVD for LSA
n_components = 100  # Number of latent topics
svd = TruncatedSVD(n_components=n_components, random_state=42)
lsa_matrix = svd.fit_transform(tfidf_matrix)

# Compute norms for LSA matrix
lsa_norms = np.linalg.norm(lsa_matrix, axis=1, keepdims=True)

# Handle zero norms by setting them to one to avoid division by zero
lsa_norms[lsa_norms == 0] = 1

# Normalize LSA matrix
normalized_lsa = lsa_matrix / lsa_norms

def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list), from_fields (list), subject_fields (list)
    """
    # Preprocess the query
    processed_query = preprocess(query)
    
    # Transform the query using the same vectorizer
    query_tfidf = vectorizer.transform([processed_query])
    
    # Project the query into the LSA space
    query_lsa = svd.transform(query_tfidf)
    
    # Compute the norm of the query vector
    query_norm = np.linalg.norm(query_lsa)
    
    if query_norm == 0:
        # If the query vector has zero norm, return empty results
        return [], [], [], [], []
    
    # Normalize the query vector
    query_normalized = query_lsa / query_norm
    
    # Compute cosine similarity
    similarities = cosine_similarity(query_normalized, normalized_lsa)[0]
    
    # Handle any potential NaN values in similarities
    similarities = np.nan_to_num(similarities)
    
    # Get top 5 documents
    top_indices = similarities.argsort()[-5:][::-1]
    top_documents = [newsgroups.data[i] for i in top_indices]
    top_similarities = [round(similarities[i], 4) for i in top_indices]
    top_from_fields = [from_fields[i] for i in top_indices]
    top_subject_fields = [subject_fields[i] for i in top_indices]
    
    return top_documents, top_similarities, top_indices.tolist(), top_from_fields, top_subject_fields

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices, from_fields_result, subject_fields_result = search_engine(query)
    
    if not documents:
        return jsonify({
            'documents': [], 
            'similarities': [], 
            'indices': [], 
            'error': 'No relevant documents found.'
        })
    
    return jsonify({
        'documents': documents,
        'similarities': similarities,
        'indices': indices,
        'from_fields': from_fields_result,
        'subject_fields': subject_fields_result
    }) 

if __name__ == '__main__':
    app.run(debug=True)
