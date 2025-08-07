from django.shortcuts import render
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


# Preprocessing function
def preprocess_text(text):
    # Lowercase text
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stem the tokens
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    # Join tokens back to a single string
    return ' '.join(tokens)

def user(request):
    return render(request, 'userinput.html')

def viewdata(request):
    # Load your dataset and train the model
    df_fake = pd.read_csv(r'C:\Users\PMLS\Documents\ML\Natural Language Processing\fake_job_postings_5000.csv')  
    df_true = pd.read_csv(r'C:\Users\PMLS\Documents\ML\Natural Language Processing\separated_true_job_postings.csv')  
    

    # Combine the data
    combined_data = pd.concat([df_fake, df_true], ignore_index=True)
    
    # Preprocess the text columns
    for col in ['title', 'description']:
        if col in combined_data.columns:
            combined_data[col] = combined_data[col].fillna("").apply(preprocess_text)

    # Combine title and description into one column
    combined_data['text'] = combined_data['title'] + ' ' + combined_data['description']

    # Define features and target
    X = combined_data['text']
    y = combined_data['fraudulent'] 
    
    # Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize the text data using TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)  

    # Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100,    
    max_depth=16,           
    max_features='sqrt',   
    min_samples_split=10, 
    min_samples_leaf=9,      
    max_leaf_nodes=35,)
    model.fit(X_train_tfidf, y_train)  

    if request.method == 'GET' and 'title' in request.GET and 'description' in request.GET:
        job_title = request.GET['title']
        job_description = request.GET['description']
        
        # Combine title and description into one text
        job_text = job_title + " " + job_description
        
        # Preprocess and vectorize the input text
        job_text_processed = preprocess_text(job_text)
        job_text_tfidf = vectorizer.transform([job_text_processed])
        
        # Make prediction
        prediction = model.predict(job_text_tfidf)
        
        # Map prediction to labels
        prediction_label = 'Fake' if prediction == 1 else 'True'
        
        # Prepare the result to display
        data = {
            'message': 'Your Posting Job is',
            'prediction': prediction_label
        }
        
        return render(request, 'viewdata.html', data)
