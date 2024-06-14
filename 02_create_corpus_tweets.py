import os
import re
import pandas as pd
from gensim import corpora
import nltk
from nltk.corpus import stopwords

artifacts_dir = r"C:\Users\DELL\Desktop\CSC501\OneDrive-2024-06-06\Data of tweets"

input_path = os.path.join(artifacts_dir, "all_tweets_delimiter_cleaned.csv")

nltk.download('stopwords')  # Need to use NLTK stopwords because it includes turkish
turkish_stop_words = set(stopwords.words('turkish'))  # Select “turkish” sh stopwords
print(stopwords.words('turkish'))

# Read in tweets and drop columns that won't help with topic modeling
df = pd.read_csv(input_path)
df.drop(columns=['url', 'Hashtags', 'mentions', 'content_urls'], inplace=True)

# Convert the 'date' column to datetime
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')

# Extract year, month, and day
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day


def preprocess_text(text):  # New preprocessing code to handle turkish stopwords and other characters
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\S+', '', text)  # Remove mentions
    text = re.sub(r'#\S+', '', text)  # Remove hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = ' '.join(word for word in text.split() if word not in turkish_stop_words)  # Remove stop words
    return text


# Apply the preprocessing function to the text column
df['cleaned_text'] = df['content'].apply(preprocess_text)

# Prepare documents for Bag of Words
documents = [doc.split() for doc in df['cleaned_text']]  # Split text into words

# Create a dictionary and corpus for topic modeling
dictionary = corpora.Dictionary(documents)
BoW_corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in documents]

# Serialize the dictionary and corpus
dictionary.save(os.path.join(artifacts_dir, "dictionary.mm"))  # Save the dictionary
corpora.MmCorpus.serialize(os.path.join(artifacts_dir, "BoW_corpus_tweets.mm"), BoW_corpus)  # Save the BoW corpus

# Save the processed DataFrame if needed
output_csv_path = os.path.join(artifacts_dir, "corpus_preproc_tweets.csv")
df.to_csv(output_csv_path, index=False)

print(f"Processed data saved to {output_csv_path}")
print(f"Dictionary and BoW corpus saved in {artifacts_dir}")
