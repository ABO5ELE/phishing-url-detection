# train.py
# this file trains the machine learning models

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def get_models():
    # list of all models we want to train
    names = [
        'Random Forest',
        'Decision Tree', 
        'Logistic Regression',
        'KNN',
        'Naive Bayes'
    ]
    
    models = [
        RandomForestClassifier(n_estimators=100),
        DecisionTreeClassifier(),
        LogisticRegression(max_iter=2000),
        KNeighborsClassifier(),
        GaussianNB()
    ]
    
    return names, models

def train_model(model, X_train, y_train):
    # train a single model
    model.fit(X_train, y_train)
    return model
