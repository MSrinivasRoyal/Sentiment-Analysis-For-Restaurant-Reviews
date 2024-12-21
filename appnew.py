
#Problem statement
#Develop a sentiment analysis model to classify restaurant reviews as positve or negative

#Description
# with the rapid growth of online platforms for sharing opinions and reviews,restarunts often rely 
#on the customer feedback to imporve their services and attract   a new customers.
# Analyzing the sentiment of these reviews can provide valuable insights into customer satisfaction.

import pandas as pd

data = pd.read_csv('Reviews.csv')

#checking the value counts
value_counts = data['Liked'].value_counts()

import matplotlib.pyplot as plt
import seaborn as sns

value_counts.plot(kind = 'bar' , color = ['blue', 'green'])
plt.title("Sentiment value counts")
plt.xlabel('Liked')
plt.ylabel('Count')
plt.xticks(ticks=[0,1] , labels=['Postive','Negative'],rotation=0)
plt.show()

from wordcloud import WordCloud

combined_text = " ".join(data['Review'])
wordcloud = WordCloud(width = 800 , height = 400 ,background_color = 'white').generate(combined_text)
plt.figure(figsize=(10,6))
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.title('Word Cloud of Reviews')
plt.show()

from collections import Counter

target_words = ['food','place','restaurant']
all_words = " ".join(data['Review']).lower().split()
word_counts = Counter(all_words)
target_word_counts = {word:word_counts[word] for word in target_words}
plt.figure(figsize=(8,6))
plt.bar(target_word_counts.keys(),target_word_counts.values() , color = ['blue','green','orange'])
plt.xlabel('words')
plt.ylabel('Frequenecy')
plt.title('Frequency of specific words in Reviews')
plt.show()

#Text preprocessing

#convert a data set into lower case
lowercased_text = data['Review'].str.lower()


#tokinization
from nltk.tokenize import word_tokenize

data['Tokens'] = data['Review'].apply(word_tokenize)

data['Review'].value_counts()

import string

data['Review'] = data['Review'].str.replace(f"[{string.punctuation}]"," ",regex = True)


data['Review'].value_counts()

#Removing the stop words like this, is , are ,was 
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

data['Tokens'] = data['Review'].apply(lambda x: [word for word in word_tokenize(x) if word not in stop_words])


#stemming
#stemming is the process of reducing the a word into root or base word form by removig suffix 
#example : driving stemmed is drive

#Stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()

data['stemmed'] = data['Review'].apply(lambda x: ' '.join([stemmer.stem(word) for word in word_tokenize(x)]))

#Lemmatization
#Lemmatization is the process transforming a word into its base or dictionary form
#example is better is lemmtized to good 

import nltk
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

data['Lemmatized'] = data['Review'].apply(lambda x :' '.join([lemmatizer.lemmatize(word , pos = wordnet.VERB) for word in word_tokenize(x)]))

#Removing the numbers from reviews
import re
data['No_Numbers'] = data['Review'].apply(lambda x : re.sub(r'\d+',' ' ,x))

#removing special characters like @ # %,*
data['cleaned'] = data['Review'].apply(lambda x: re.sub(r'[^A-Za-z0-9\s]','' ,x))
#expanding method
# don't eat food in this hotel , when we apply expanted text it will convert into do not eat food in this hotel
import contractions
data['Expanded'] = data['Review'].apply(contractions.fix)

#Removing emojis

import emoji
data['emoji'] = data['Review'].apply(emoji.demojize)

# removing links from review_ text
# food is good vist www.abchotel.in

from bs4 import BeautifulSoup

data['cleaned'] = data['Review'].apply(lambda x: BeautifulSoup(x,"html.parser").get_text())

#TF IDF VECTORIZER
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data['Review'])

#bulding a machine learning model
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report


vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data['Review'])
y = data['Liked']

X_train,X_test,y_train,y_test = train_test_split(X,y , test_size = 0.2 , random_state = 42)

model = MultinomialNB() 
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

report = classification_report(y_test,y_pred)

#prediction of new review
def preprocess_review(review):
    review = review.lower()
    review = BeautifulSoup(review,"html.parser").get_text()
    review = re.sub(f"[{string.punctuation}]"," ",review)
    review = contractions.fix(review)
    review = emoji.demojize(review)
    tokens = word_tokenize(review)
    stop_words =set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word, pos = 'v') for word in tokens]
    cleaned_review = ' '.join(lemmatized_tokens)
    return cleaned_review
import requests

import requests

def fetch_food_image(food_name):
    api_key = "46722190-a612c79ca90f15a07c520cb9d"  # Replace with your Pixabay API key
    url = f"https://pixabay.com/api/?key={api_key}&q={food_name}&image_type=photo&per_page=3"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        try:
            data = response.json()
            if data.get("hits"):
                return data["hits"][0]["webformatURL"]
            else:
                return "https://via.placeholder.com/600x400?text=No+Image+Available"
        except ValueError:
            print("JSON decoding error. Raw response:", response.text)
            return "https://via.placeholder.com/600x400?text=Error+Fetching+Image"
    else:
        print("Error:", response.status_code, response.text)
        return "https://via.placeholder.com/600x400?text=Error+Fetching+Image"

import streamlit as st
import re

st.title("Feedback/Review Page")

# Collect user input for food and review
food_item = st.text_input("What food did you eat?")
review = st.text_area("Please provide your feedback:")

# Display an image of the food (You can integrate an API like Unsplash here if you wish)
if food_item:
    image_url = fetch_food_image(food_item)
    st.image(image_url, caption=f"{food_item.capitalize()}")

# Preprocess and classify the review
if review:
    cleaned_review = preprocess_review(review)  # Use the preprocess function from your code
    review_vectorized = vectorizer.transform([cleaned_review])
    prediction = model.predict(review_vectorized)

    # Display results based on sentiment
    if prediction[0] == 1:
        st.write("### Positive Review")
        st.success("Thank you for your positive feedback!")
    else:
        st.write("### Negative Review")
        st.error("We're sorry to hear about your experience.")

    # Suggest other food items (Example: fixed recommendations or content-based filtering)
    st.write("### You may also enjoy:")
    if food_item in ['biriyani', 'chapathi', 'parotha']:
        st.write(" - Butter Naan")
        st.write(" - Chicken Pulao")
    else:
        st.write(" - French Fries")
        st.write(" - Chicken Wings")

		

# Generate evaluation metrics
if st.button("Generate Evaluation Metrics"):
    st.write(f"Accuracy: {accuracy}")
    st.text(f"Classification Report:\n{report}")

