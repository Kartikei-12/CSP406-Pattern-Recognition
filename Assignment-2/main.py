# Author: Kartikei Mittal
# Roll No: 2017KUCP1032
# Pattern Recognition Assignment

# Program Parameters
TOP_N = 6
NGRAM_MAX_LEN = 3
DATASET_FILE = "dataset.txt"

# Importing Libraries
import os
import re
import numpy as np
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Run Following download first time when running
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

STOP_WORDS = set(nltk.corpus.stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Processed Ngram Data
data_ = None

# Original sentence
data2_ = None

# Processed data
data3_ = None

num_docs = 0

def preprocess_sentence(sentence):
    """"""
    # Removing Puntuactions and lowering case
    sentence = re.sub(r'[^A-Za-z. ]', '', sentence.lower())

    # Removing Extra white spaces
    sentence = re.sub('\s+', ' ', sentence)
    sentence = sentence.strip(' ')
    
    # Word Tokenization
    sentence = nltk.word_tokenize(sentence)
    
    # Lemmatization
    sentence = list([lemmatizer.lemmatize(w) for w in sentence])
     
    # Stop Word Removal
    sentence = list([w for w in sentence if w not in STOP_WORDS])
    
    return sentence, list(nltk.everygrams(sentence, max_len = NGRAM_MAX_LEN))


def preprocess_dataset(dataset):
    """"""
    global data_
    global data2_
    global data3_
    global num_docs
    try:
        with open(dataset, 'r') as file_:
            data_ = file_.read()
    except FileNotFoundError:
        print("Dataset File Not Found. EXITING")
        exit()
    
    # Sentence Tokenizer
    data_ = data_.splitlines()
    num_docs = len(data_)

    # Preprocessing Sentences
    data2_ = list()
    data3_ = list()
    for i in range(num_docs):
        data2_.append(data_[i])
        sen, egram = preprocess_sentence(data_[i])
        data_[i] = egram
        data3_.append(' '.join(sen))
    
    # data_scores = tfidf_vectorizer.fit_transform(data2_)
    # features = tfidf_vectorizer.get_feature_names() 
    # word2tfidf = dict(zip(features, tfidf_vectorizer.idf_))
    # word2tfidf = pd.DataFrame(list(zip(features, tfidf_vectorizer.idf_)), columns = ['Term', 'Score'])
    # word2tfidf = word2tfidf.sort_values('Score', ascending = False)
    # word2tfidf.reset_index(drop = True, inplace = True)
    # print(word2tfidf.head(20))

def calculate_tfidf(term, document):
    """"""
    count_in_documents = np.sum([1 for docs in data_ if term in docs])
    return float(
            # Term Frequency
            document.count(term) / num_docs
        ) * float(
            # Inverse Document Frequency
            0 if count_in_documents == 0 else \
            np.log(num_docs / count_in_documents)
        ) 

def calculate_score(query, document, score_type = 'or'):
    """"""
    score = 0

    if score_type == 'and':
        if len(set(query).intersection(set(document))) == len(set(query)):
            score = np.sum([ calculate_tfidf(term, document) for term in query ])
        else:
            score = 0
    elif score_type == 'or':
        score = np.sum([ calculate_tfidf(term, document) for term in query ])
    else:
        score = 0
    return score

def main():
    """"""
    print("***************** Query String Mactcher *****************")
    # input_string = "Driven through midwicket for a couple of runs"
    input_string = input("Enter Query string or q to quit: ")
    if input_string == 'q':
        print("***************** Thank you for using Query String Mactcher *****************")
        exit()    
    print("Your Query String:", input_string)
    sentence, egram = preprocess_sentence(input_string)
    print("Pre-processed query string: ", sentence, end = '\n\n')

    scores = list()
    for doc in data_:
        scores.append(calculate_score(egram, doc, score_type = 'and' if input_string[0] == "\"" else 'or'))
    
    scores = np.array(scores)
    score_index = scores.argsort()[-TOP_N:][::-1]

    for i, index in enumerate(score_index):
        print("Rank:", i + 1, "Score: ", scores[index])
        print("Processed String:", data3_[i])
        print(data2_[index], end = '\n\n\n')

    print("\n\n\n\n")
    # exit()

if __name__ == "__main__":
    preprocess_dataset(dataset = DATASET_FILE)
    while True:
        main()
