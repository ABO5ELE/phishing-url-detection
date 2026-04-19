# evaluate.py
# this file checks how good the model is

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def get_accuracy(model, X_test, y_test):
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    return round(acc * 100, 2)

def get_f1(model, X_test, y_test):
    pred = model.predict(X_test)
    f1 = f1_score(y_test, pred)
    return round(f1, 4)

def full_report(model, X_test, y_test):
    pred = model.predict(X_test)
    print(classification_report(y_test, pred))

def get_confusion_matrix(model, X_test, y_test):
    pred = model.predict(X_test)
    cm = confusion_matrix(y_test, pred)
    return cm
