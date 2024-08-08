from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the dataset and split into input and output arrays
df = pd.read_json('C:/Users/suren/OneDrive/Desktop/Cyberbullying Detection Project/Cyberbullying-Detection-using-Machine-Learning-main/Dataset.json')

x = np.array(df["content"])
y = np.array(df["annotation"])

# Preprocess the data by removing stopwords and performing stemming
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
x_preprocessed = []
for doc in x:
    doc = re.sub(r'\W', ' ', doc)  # remove special characters
    doc = doc.lower()  # convert to lowercase
    words = doc.split()
    words = [w for w in words if w not in stop_words]  # remove stopwords
    words = [stemmer.stem(w) for w in words]  # perform stemming
    doc = ' '.join(words)
    x_preprocessed.append(doc)
x = np.array(x_preprocessed)

# Convert text data to numerical representation using CountVectorizer
cv = CountVectorizer()
x = cv.fit_transform(x)

# Convert the label data to string or integer format
y = np.array([str(yi) if type(yi) == dict else int(yi) for yi in y])

# Encode the label data to numeric format if it is not already in one
if not np.issubdtype(y.dtype, np.number) and len(set(y)) > 2:
    le = LabelEncoder()
    y = le.fit_transform(y)

# Train a random forest classifier on the full dataset
clf = RandomForestClassifier()
clf.fit(x.toarray(), y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']

    if not text:
        result = "Please enter text to predict."
        return render_template('index.html', text=text, result=result)

    # Preprocess the input text data
    doc = re.sub(r'\W', ' ', text)  # remove special characters
    doc = doc.lower()  # convert to lowercase
    words = doc.split()
    words = [w for w in words if w not in stop_words]  # remove stopwords
    words = [stemmer.stem(w) for w in words]  # perform stemming
    doc = ' '.join(words)
    x_test = cv.transform([doc])

    # Make predictions on the preprocessed input text
    predicted_label = clf.predict(x_test.toarray())[0]

    if float(predicted_label[25]) == 1:
        result = "Bullying detected"
    else:
        result = "Non-Bullying"

    # Pass back the input text along with the prediction result
    return render_template('index.html', text=text, result=result)

if __name__ == '__main__':
    app.run(debug=True)

