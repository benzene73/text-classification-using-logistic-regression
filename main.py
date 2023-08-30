# Program for text classification using logistic regression and tf-idf vectorizer
import pickle

import numpy as np
import scipy
from scipy.sparse import csr_matrix, hstack
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from tqdm import tqdm
import re
import time
import contractions
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import warnings
import nltk
from nltk.corpus import wordnet
import argparse

warnings.filterwarnings("ignore")


def get_paths():
    # Get arguments from command line using argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='test', help="Choose train or test")
    parser.add_argument('--train_path', default=r'./train.csv', help="path to train.csv")
    parser.add_argument('--model_name', default=r'model', help="Save trained model as given model_name")
    parser.add_argument('--out_name', default=r'output', help="Save predictions in given out_name file.csv")
    parser.add_argument('--test_path', default=r'./input_test.csv', help="Load test data from test_path")
    return parser.parse_args()


# Preprocessing train data, which is one review text
def preprocess(review):
    # replace url with URl
    review = re.sub(r'http\S+', 'URL', review)
    # convert to lower case by looping through review and convert only those words which are not upper case
    for word in review.split():
        if not word.isupper():
            review = review.replace(word, word.lower())
    # expand contractions like "didn't" to "did not"
    review = contractions.fix(review)
    # remove punctuations
    review = re.sub(r'[^\w\s]', ' ', review)
    # reduce words of more than 3 consecutive same letters, like 4,5,6..., to two by regular expressions
    review = re.sub(r'(.)\1{2,}', r'\1\1', review)
    # remove stop words
    stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you'll",
                  'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
                  'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
                  'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is',
                  'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
                  'doing', 'a', 'an', 'the', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
                  'by', 'for', 'with', 'about', 'between', 'into', 'through', 'during', 'before', 'after',
                  'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
                  'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
                  'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'than', 'can', 'will',
                  'just', 'should', 'now', 'o', 'and', "dress", "s"}
    review = ' '.join([word for word in review.split() if word not in stop_words])
    return review


def tokenizer(text):
    text = preprocess(text)
    tokens = nltk.word_tokenize(text)
    # Stemming using snowball stemmer
    stemmer = nltk.stem.SnowballStemmer('english')
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens


def train():
    # Reading data from csv file
    # data_path = r".\val.csv"
    start = time.time()
    print("Training started...")

    data = pd.read_csv(args.train_path, header=None, encoding='latin-1')

    # Preprocessing data
    data = data.dropna()
    data = data.drop_duplicates()

    # Split data into X_train and y_train
    X_train = data.iloc[:, 0]
    y_train = data.iloc[:, 1]

    # Print shape of train data
    print("Shape of train data is " + str(X_train.shape))
    print("Shape of train labels is " + str(y_train.shape))

    # Vectorize the text using tf-idf
    vectorizer = TfidfVectorizer(tokenizer=tokenizer, min_df=5, max_df=0.8, ngram_range=(1, 3))
    # 0.503, 120s with bi-gram # 0.508, 117s with tri-gram [logreg]
    X_train = vectorizer.fit_transform(tqdm(X_train, desc="Vectorizing train data"))

    # Train the model
    model = LogisticRegression(solver='sag', n_jobs=-1, verbose=1)
    model.fit(X_train, y_train)
    # Save the model as a pickle file
    pickle.dump(model, open(args.model_name + ".pkl", 'wb'))
    pickle.dump(vectorizer, open("vectorizer.pkl", 'wb'))

    # Measure training time
    end = time.time()
    print("Time taken to train the model is " + str(end - start) + " seconds")

    # # Save X_val into the first column and y_val into second column of test.csv. Comment out later. Only for testing.
    # df = pd.DataFrame()
    # df['text'] = X_val
    # df['label'] = y_val
    # df.to_csv(r".\test.csv", index=False, header=False)


def test():
    # Time the test function
    start = time.time()
    print("Test started")

    # Read the test data
    test_data = pd.read_csv(args.test_path, header=None, encoding='latin-1')
    # test_data = test_data.dropna()
    # test_data = test_data.drop_duplicates()
    X_test = test_data.iloc[:, 0]
    # y_test = test_data.iloc[:, 1]

    # Print shape of test data
    print("Shape of test data is " + str(X_test.shape))

    # # Transform the test data by loading the vectorizer
    # vectorizer = pickle.load(open("vectorizer.pkl", 'rb'))
    # X_test = vectorizer.transform(tqdm(X_test, desc="Vectorizing test data"))
    #
    # # Load the saved model from model_name + ".pkl"
    # model = pickle.load(open(args.model_name + ".pkl", 'rb'))
    # # Predict the labels for the test data
    # y_pred = model.predict(X_test)
    #
    # # # Calculate confusion matrix and f1 score
    # # c_matrix = confusion_matrix(y_test, y_pred)
    # # print(c_matrix)
    # # f1_macro2 = f1_score(y_test, y_pred, average='macro')
    # # f1_micro2 = f1_score(y_test, y_pred, average='micro')
    # # print("f1_score of logistic reg is " + str((f1_macro2 + f1_micro2) / 2))
    #
    # # Save the predictions in a csv file
    # df = pd.DataFrame(y_pred)
    # df.to_csv(args.out_name + ".csv", index=False, header=False)
    #
    # # Measure the time taken to test the model
    # end = time.time()
    # print("Time taken to test the model is " + str(end - start) + " seconds")


if __name__ == '__main__':
    args = get_paths()
    if args.mode == 'train':
        train()
    else:
        test()
