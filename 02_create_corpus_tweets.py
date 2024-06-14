import os
import re
import pandas as pd
from gensim import corpora
import nltk
from nltk.corpus import stopwords
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
artifacts_dir = r"C:\Users\DELL\Desktop\CSC501\OneDrive-2024-06-06\Data of tweets"
input_path = os.path.join(artifacts_dir, "all_tweets_delimiter_cleaned.csv")

# Download stopwords
nltk.download('stopwords')
turkish_stop_words = set(stopwords.words('turkish'))
logging.info(f'Turkish stopwords: {turkish_stop_words}')


def load_data(input_path, encoding='utf-8'):
    try:
        df = pd.read_csv(input_path, encoding=encoding)
        df.drop(columns=['url', 'Hashtags', 'mentions', 'content_urls'], inplace=True)
        df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y %H:%M')
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        return df
    except UnicodeDecodeError:
        logging.error(f"UnicodeDecodeError: Trying another encoding for {input_path}")
        return load_data(input_path, encoding='ISO-8859-1')
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise


def preprocess_text(text):
    try:
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\S+', '', text)
        text = re.sub(r'#\S+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = ' '.join(word for word in text.split() if word not in turkish_stop_words)
        return text
    except Exception as e:
        logging.error(f"Error preprocessing text: {e}")
        raise


def main():
    try:
        df = load_data(input_path)
        df['cleaned_text'] = df['content'].apply(preprocess_text)

        documents = [doc.split() for doc in df['cleaned_text']]
        dictionary = corpora.Dictionary(documents)
        BoW_corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in documents]

        dictionary_path = os.path.join(artifacts_dir, "dictionary.mm")
        corpus_path = os.path.join(artifacts_dir, "BoW_corpus_tweets.mm")
        output_csv_path = os.path.join(artifacts_dir, "corpus_preproc_tweets.csv")

        dictionary.save(dictionary_path)
        corpora.MmCorpus.serialize(corpus_path, BoW_corpus)
        df.to_csv(output_csv_path, index=False)

        logging.info(f"Processed data saved to {output_csv_path}")
        logging.info(f"Dictionary saved to {dictionary_path}")
        logging.info(f"BoW corpus saved to {corpus_path}")
    except Exception as e:
        logging.error(f"Error in main: {e}")


if __name__ == "__main__":
    main()
