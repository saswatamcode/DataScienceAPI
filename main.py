import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from flask import Flask, redirect, url_for, request
from flask_restful import Resource, Api
from flask_cors import CORS, cross_origin
import json
from flask_jsonpify import jsonify

app = Flask(__name__)
api = Api(app)

CORS(app)

class Rf_classification(Resource):
    def post(self):
        req = request.get_json()
        path = req["path"]
        dataset = pd.read_csv(path)
        row = dataset.shape[0]
        col = dataset.shape[1]-1
        X = dataset.iloc[:, 0:col].values
        y = dataset.iloc[:, col].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        regressor = RandomForestClassifier(n_estimators=20, random_state=0)
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        print(y_pred)
        print(type(y_pred))
        p = y_pred.tolist()
        print(type(p))
        cm = confusion_matrix(y_test,y_pred)
        print(type(cm.tolist()))
        asc = accuracy_score(y_test, y_pred)
        print(type(int(asc)))
        cr = classification_report(y_test,y_pred)
        print(type(cr))
        res = { "prediction_result": p, "confusin_matrix":cm.tolist(), "accuracy_score":float(asc), "classification_report":cr}
        return res,200

class Svm_classification(Resource):
    def post(self):
        req = request.get_json()
        path = req["path"]
        dataset = pd.read_csv(path)
        row = dataset.shape[0]
        col = dataset.shape[1]-1
        X = dataset.iloc[:, 0:col].values
        y = dataset.iloc[:, col].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        SVM = svm.LinearSVC()
        SVM.fit(X_train, y_train)
        y_pred = SVM.predict(X_test)
        print(y_pred)
        print(type(y_pred))
        p = y_pred.tolist()
        print(type(p))
        cm = confusion_matrix(y_test,y_pred)
        print(type(cm.tolist()))
        asc = accuracy_score(y_test, y_pred)
        print(type(int(asc)))
        cr = classification_report(y_test,y_pred)
        print(type(cr))
        res = { "prediction_result": p, "confusin_matrix":cm.tolist(), "accuracy_score":float(asc), "classification_report":cr}
        return res,200

class Knn_classification(Resource):
    def post(self):
        req = request.get_json()
        path = req["path"]
        dataset = pd.read_csv(path)
        row = dataset.shape[0]
        col = dataset.shape[1]-1
        X = dataset.iloc[:, 0:col].values
        y = dataset.iloc[:, col].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        classifier = KNeighborsClassifier(n_neighbors=5)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        print(y_pred)
        print(type(y_pred))
        p = y_pred.tolist()
        print(type(p))
        cm = confusion_matrix(y_test,y_pred)
        print(type(cm.tolist()))
        asc = accuracy_score(y_test, y_pred)
        print(type(int(asc)))
        cr = classification_report(y_test,y_pred)
        print(type(cr))
        res = { "prediction_result": p, "confusin_matrix":cm.tolist(), "accuracy_score":float(asc), "classification_report":cr}
        return res,200
    
class Dt_classification(Resource):
    def post(self):
        req = request.get_json()
        path = req["path"]
        dataset = pd.read_csv(path)
        row = dataset.shape[0]
        col = dataset.shape[1]-1
        X = dataset.iloc[:, 0:col].values
        y = dataset.iloc[:, col].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        classifier = DecisionTreeClassifier()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        print(y_pred)
        print(type(y_pred))
        p = y_pred.tolist()
        print(type(p))
        cm = confusion_matrix(y_test,y_pred)
        print(type(cm.tolist()))
        asc = accuracy_score(y_test, y_pred)
        print(type(int(asc)))
        cr = classification_report(y_test,y_pred)
        print(type(cr))
        res = { "prediction_result": p, "confusin_matrix":cm.tolist(), "accuracy_score":float(asc), "classification_report":cr}
        return res,200

class GaussianNB_classification(Resource):
    def post(self):
        req = request.get_json()
        path = req["path"]
        dataset = pd.read_csv(path)
        row = dataset.shape[0]
        col = dataset.shape[1]-1
        X = dataset.iloc[:, 0:col].values
        y = dataset.iloc[:, col].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        print(y_pred)
        print(type(y_pred))
        p = y_pred.tolist()
        print(type(p))
        cm = confusion_matrix(y_test,y_pred)
        print(type(cm.tolist()))
        asc = accuracy_score(y_test, y_pred)
        print(type(int(asc)))
        cr = classification_report(y_test,y_pred)
        print(type(cr))
        res = { "prediction_result": p, "confusin_matrix":cm.tolist(), "accuracy_score":float(asc), "classification_report":cr}
        return res,200

class BernoulliNB_classification(Resource):
    def post(self):
        req = request.get_json()
        path = req["path"]
        dataset = pd.read_csv(path)
        row = dataset.shape[0]
        col = dataset.shape[1]-1
        X = dataset.iloc[:, 0:col].values
        y = dataset.iloc[:, col].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        classifier = BernoulliNB()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        print(y_pred)
        print(type(y_pred))
        p = y_pred.tolist()
        print(type(p))
        cm = confusion_matrix(y_test,y_pred)
        print(type(cm.tolist()))
        asc = accuracy_score(y_test, y_pred)
        print(type(int(asc)))
        cr = classification_report(y_test,y_pred)
        print(type(cr))
        res = { "prediction_result": p, "confusin_matrix":cm.tolist(), "accuracy_score":float(asc), "classification_report":cr}
        return res,200

class Logistic_Regression(Resource):
    def post(self):
        req = request.get_json()
        path = req["path"]
        dataset = pd.read_csv(path)
        row = dataset.shape[0]
        col = dataset.shape[1]-1
        X = dataset.iloc[:, 0:col].values
        y = dataset.iloc[:, col].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        classifier = LogisticRegression()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        print(y_pred)
        print(type(y_pred))
        p = y_pred.tolist()
        print(type(p))
        cm = confusion_matrix(y_test,y_pred)
        print(type(cm.tolist()))
        asc = accuracy_score(y_test, y_pred)
        print(type(int(asc)))
        cr = classification_report(y_test,y_pred)
        print(type(cr))
        res = { "prediction_result": p, "confusin_matrix":cm.tolist(), "accuracy_score":float(asc), "classification_report":cr}
        return res,200

class Rf_regression(Resource):
    def post(self):
        req = request.get_json()
        path = req["path"]
        dataset = pd.read_csv(path)
        row = dataset.shape[0]
        col = dataset.shape[1]-1
        X = dataset.iloc[:, 0:col].values
        y = dataset.iloc[:, col].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        regressor = RandomForestRegressor(n_estimators=20, random_state=0)
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        p = y_pred.tolist()
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        res = { "predictions": p, "mean_absolute_error": mae, "mean_squared_error":mse, "root_mean_squared_error": rmse}
        return res,200

class Svm_regression(Resource):
    def post(self):
        req = request.get_json()
        path = req["path"]
        dataset = pd.read_csv(path)
        row = dataset.shape[0]
        col = dataset.shape[1]-1
        X = dataset.iloc[:, 0:col].values
        y = dataset.iloc[:, col].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        regressor = SVR(kernel = 'linear')
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        print(y_pred)
        print(type(y_pred))
        p = y_pred.tolist()
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        res = { "predictions": p, "mean_absolute_error": mae, "mean_squared_error":mse, "root_mean_squared_error": rmse}
        return res,200

class Dt_regression(Resource):
    def post(self):
        req = request.get_json()
        path = req["path"]
        dataset = pd.read_csv(path)
        row = dataset.shape[0]
        col = dataset.shape[1]-1
        X = dataset.iloc[:, 0:col].values
        y = dataset.iloc[:, col].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        regressor = DecisionTreeRegressor()
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        print(y_pred)
        print(type(y_pred))
        p = y_pred.tolist()
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        res = { "predictions": p, "mean_absolute_error": mae, "mean_squared_error":mse, "root_mean_squared_error": rmse}
        return res,200

class Knn_regression(Resource):
    def post(self):
        req = request.get_json()
        path = req["path"]
        dataset = pd.read_csv(path)
        row = dataset.shape[0]
        col = dataset.shape[1]-1
        X = dataset.iloc[:, 0:col].values
        y = dataset.iloc[:, col].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        regressor = KNeighborsRegressor()
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        print(y_pred)
        print(type(y_pred))
        p = y_pred.tolist()
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        res = { "predictions": p, "mean_absolute_error": mae, "mean_squared_error":mse, "root_mean_squared_error": rmse}
        return res,200

api.add_resource(Rf_classification, '/rfclassification')
api.add_resource(Rf_regression, '/rfregression')
api.add_resource(Svm_classification, '/svmclassification')
api.add_resource(Knn_classification, '/knnclassification')
api.add_resource(Dt_classification, '/dtclassification')
api.add_resource(Svm_regression, '/svmregression')
api.add_resource(Dt_regression, '/dtregression')
api.add_resource(Knn_regression, '/knnregression')
api.add_resource(GaussianNB_classification, '/gnbclassification')
api.add_resource(BernoulliNB_classification, '/bnbclassification')
api.add_resource(Logistic_Regression, '/logisticregression')

if __name__ == '__main__':
   app.run(debug = True)
