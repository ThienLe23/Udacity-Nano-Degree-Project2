import sys
import pandas as pd
import numpy as np

import pickle
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


nltk.download(['punkt', 'wordnet'])

def load_data(database_filepath):
    """
    Load data from database file
    Parameters:
        database_filepath: database file
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("disaster", con=engine)

    X = df["message"]
    Y = df.iloc[:, 4:]
    category_names = Y.columns.tolist()
    return X, Y, category_names

def tokenize(text):
    """
    Tokenize text into a list of tokens
    Parameters:
        text: text string 
    Returns:
        tokens: list of tokens
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    tokens = [lemmatizer.lemmatize(token).lower().strip() for token in tokens]
    return tokens


def build_model():
    """
    Create ML pipeline and GridSearch pipeline for hyperparameters tuning
    Returns:
        cv: model object is used to predict new sample
    """
    pipeline = Pipeline([
        ('count_vector', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__n_estimators': [10, 50]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model with test dataset print some metrics
    Parameters:
        model: model object
        X_test, Y_test: Input data and label data
    """
    Y_pred = model.predict(X_test)
    for i, category_name in enumerate(Y_test):
        print(category_name, classification_report(Y_test[category_name], Y_pred[:, i]))


def save_model(model, model_filepath):
    """
    Save model object to pickle file
    Parameters:
        model: model object
        model_filepath: filepath to save model
    """
    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()