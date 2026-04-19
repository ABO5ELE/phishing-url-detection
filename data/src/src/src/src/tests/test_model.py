# test_model.py
# unit tests for the phishing detection project

import sys
sys.path.append('../src')

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import unittest

from preprocessing import load_data, prepare_data, split_data
from train import get_models, train_model
from evaluate import get_accuracy, get_f1, get_confusion_matrix

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        # create a small fake dataset for testing
        self.fake_data = pd.DataFrame({
            'url': ['http://fake.com', 'http://real.com'],
            'length_url': [20, 15],
            'length_hostname': [8, 7],
            'ip': [0, 0],
            'nb_dots': [2, 1],
            'nb_hyphens': [0, 0],
            'nb_at': [0, 0],
            'nb_qm': [0, 0],
            'nb_and': [0, 0],
            'nb_or': [0, 0],
            'status': ['phishing', 'legitimate']
        })

    def test_prepare_data_removes_url_column(self):
        # check that url column is removed
        X, y = prepare_data(self.fake_data)
        self.assertNotIn('url', X.columns)

    def test_prepare_data_removes_status_column(self):
        # check that status column is removed from X
        X, y = prepare_data(self.fake_data)
        self.assertNotIn('status', X.columns)

    def test_labels_converted_to_numbers(self):
        # check that labels are 0 and 1 not text
        X, y = prepare_data(self.fake_data)
        self.assertIn(0, y.values)
        self.assertIn(1, y.values)

    def test_data_shape_is_correct(self):
        # check rows and columns are correct
        X, y = prepare_data(self.fake_data)
        self.assertEqual(len(X), 2)
        self.assertEqual(len(y), 2)


class TestTraining(unittest.TestCase):

    def setUp(self):
        # simple numeric data for training tests
        self.X_train = np.array([[1, 0, 2], [0, 1, 1], 
                                  [1, 1, 3], [0, 0, 1]])
        self.y_train = np.array([1, 0, 1, 0])
        self.X_test  = np.array([[1, 0, 2], [0, 1, 1]])
        self.y_test  = np.array([1, 0])

    def test_get_models_returns_correct_count(self):
        # check that we have 5 models
        names, models = get_models()
        self.assertEqual(len(models), 5)
        self.assertEqual(len(names), 5)

    def test_model_trains_without_error(self):
        # check random forest trains correctly
        model = RandomForestClassifier(n_estimators=10)
        trained = train_model(model, self.X_train, self.y_train)
        self.assertIsNotNone(trained)

    def test_model_makes_predictions(self):
        # check model can predict after training
        model = RandomForestClassifier(n_estimators=10)
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), 2)


class TestEvaluation(unittest.TestCase):

    def setUp(self):
        # train a simple model for evaluation tests
        self.X_train = np.array([[1, 0, 2], [0, 1, 1],
                                  [1, 1, 3], [0, 0, 1]])
        self.y_train = np.array([1, 0, 1, 0])
        self.X_test  = np.array([[1, 0, 2], [0, 1, 1]])
        self.y_test  = np.array([1, 0])
        self.model = RandomForestClassifier(n_estimators=10)
        self.model.fit(self.X_train, self.y_train)

    def test_accuracy_is_between_0_and_100(self):
        # accuracy should always be between 0 and 100
        acc = get_accuracy(self.model, self.X_test, self.y_test)
        self.assertGreaterEqual(acc, 0)
        self.assertLessEqual(acc, 100)

    def test_f1_is_between_0_and_1(self):
        # f1 score should always be between 0 and 1
        f1 = get_f1(self.model, self.X_test, self.y_test)
        self.assertGreaterEqual(f1, 0)
        self.assertLessEqual(f1, 1)

    def test_confusion_matrix_shape(self):
        # confusion matrix should be 2x2 for binary classification
        cm = get_confusion_matrix(self.model, self.X_test, self.y_test)
        self.assertEqual(cm.shape, (2, 2))


if __name__ == '__main__':
    unittest.main()
