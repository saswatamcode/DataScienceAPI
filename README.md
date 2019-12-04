# DataScienceAPI
This is a RESTful API built using Flask and Scikit-Learn. It provides a host of Classification and Regression algorithms that can be used readily and returns results in the form of predictions, confusion matrices, accuracy scores and more.

## Description
- /rfclassification: Classification using Random Forest algorithm.
- /rfregression: Regression using Random Forest algorithm.
- /svmclassification: Classification using Support Vector Machines algorithm.
- /knnclassification: Classification using K-Nearest Neighbor algorithm.
- /dtclassification: Classification using Decision Trees algorithm.
- /svmregression: Regression using Support Vector Machines algorithm.
- /dtregression: Regression using Decision Trees algorithm.
- /knnregression: Regression using K-Nearest Neighbor algorithm.
- /gnbclassification: Classification using Naive Bayes(Gaussian) algorithm.
- /bnbclassification: Classification using Naive Bayes(Bernoulli) algorithm.
- /logisticregression: Classification using Logistic Regression algorithm.

## To Run
- Clone into repo
- Type in `pip install` (preferably inside a virtual environment)
- Then run `python3 main.py`
- Use a REST client to make post requests to the Flask Server

Two sample datasets and the request format are included to test out the API.