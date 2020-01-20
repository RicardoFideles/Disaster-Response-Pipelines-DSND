import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, confusion_matrix


def load_data(database_filepath):
    """
    Load a sqlliteDB from a path.
    --
    Inputs:
        database_filepath: Database path
    Outputs:
        X: dataframe with target feature
        Y: dataframe with other features
        category_names: List of of features except the target and not used features.
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Tweets',engine)
    X = df.iloc[:,1]
    Y = df.iloc[:,4:]
    category_names=list(df.columns[4:])

    return X, Y, category_names

def tokenize(text):
    """
    Clean, tokenize, stemmed e lemmed a text.
    --
    Inputs:
        text: text
    Outputs:
        text : array of words.
    """
    text = text.lower()
    
    #removing extra characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    #tokenizing all the sentences
    words = word_tokenize(text)
    
    #removing stopwords
    words = [w for w in words if w not in stopwords.words("english")]
    
    # Reduce words to their stems
    stemmed = [PorterStemmer().stem(w) for w in words]
    
    # Lemmatize verbs by specifying pos
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in stemmed]
    
    return lemmed


def build_model():
    """
    Build the model and pipelines for the classifier
    --
    Outputs:
        cv : GridSearchCV Object
    """
    pipeline = Pipeline([
        ('cvect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])
    
    parameters = {    
        'tfidf__use_idf': [True],
        'clf__estimator__n_estimators': [50],
        'clf__estimator__min_samples_split': [2] 
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evalute a given model for all features using the classification_report
    --
    Inputs:
        model : Model to be saved.
        X_test :  X_test,
        Y_test :  Y_test,
        category_names : List of features.
    """
    Y_pred =  model.predict(X_test)
    for i, col in enumerate(category_names):
        print(col)
        print(classification_report(Y_test[col],Y_pred[:, i]))


def save_model(model, model_filepath):
    """
    Save the model
    --
    Inputs:
        model : Model to be saved.
        model_filepath :  File path
    """
    pickle.dump(model, open(model_filepath, 'wb'))


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