import os
import json
import docx
import pandas as pd
from PyPDF2 import PdfReader
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Preprocessing Function
def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Load Scraped Data
scraped_data_path = "New_enhanced_scraped_websites.json"

try:
    with open(scraped_data_path, "r", encoding="utf-8") as f:
        scraped_data = json.load(f)
    scraped_texts = []
    for site in scraped_data:
        scraped_texts.extend([preprocess_text(text) for text in site.get("main_content", [])])
        scraped_texts.extend([preprocess_text(text) for text in site.get("detailed_content", [])])
    print(f"Loaded {len(scraped_texts)} preprocessed paragraphs.")
except FileNotFoundError:
    print("Error: 'New_enhanced_scraped_websites.json' not found!")
    scraped_texts = []

# Load Sentence-BERT Model
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/check_plagiarism", methods=["POST"])
def check_plagiarism():
    input_text = ""

    if 'file' in request.files:
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(file_path)

            # Extract text based on file type
            file_ext = uploaded_file.filename.split('.')[-1].lower()

            if file_ext == "txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    input_text = f.read()
            elif file_ext == "pdf":
                pdf_reader = PdfReader(file_path)
                input_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
            elif file_ext == "docx":
                doc = docx.Document(file_path)
                input_text = "\n".join([para.text for para in doc.paragraphs])
            elif file_ext == "csv":
                df = pd.read_csv(file_path)
                input_text = df.to_string()
            else:
                return jsonify({"error": "Unsupported file format"}), 400

    if 'text' in request.form and request.form['text'].strip():
        input_text = request.form['text']

    if not input_text:
        return jsonify({"error": "No input text provided"}), 400

    input_text_preprocessed = preprocess_text(input_text)

    # Compute TF-IDF Similarity
    tfidf_vectorizer = TfidfVectorizer()
    corpus = scraped_texts + [input_text_preprocessed]
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    input_tfidf_vector = tfidf_matrix[-1]
    scraped_tfidf_matrix = tfidf_matrix[:-1]

    tfidf_similarities = cosine_similarity(input_tfidf_vector, scraped_tfidf_matrix)[0]

    # Compute Sentence Embeddings Similarity
    input_embedding = model.encode(input_text, convert_to_tensor=True)
    scraped_embeddings = model.encode(scraped_texts, convert_to_tensor=True)
    embedding_similarities = util.pytorch_cos_sim(input_embedding, scraped_embeddings)[0].cpu().numpy()

    # Compute Hybrid Similarity
    hybrid_similarity_scores = (tfidf_similarities + embedding_similarities) / 2

    threshold = 0.5
    plagiarized = any(score > threshold for score in hybrid_similarity_scores)

    return jsonify({"plagiarism_detected": plagiarized, "similarity_scores": hybrid_similarity_scores.tolist()})

if __name__ == "__main__":
    app.run(debug=True)
