[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-for-VSCode](https://img.shields.io/badge/Made%20for-VSCode-1f425f.svg)](https://code.visualstudio.com/)
[![GitHub forks](https://img.shields.io/github/forks/saswatamcode/DataScienceAPI.svg?style=social&label=Fork&maxAge=2592000)](https://GitHub.com/saswatamcode/DataScienceAPI/network/)
[![GitHub stars](https://img.shields.io/github/stars/saswatamcode/DataScienceAPI.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/saswatamcode/DataScienceAPI/stargazers/)
[![GitHub issues](https://img.shields.io/github/issues/saswatamcode/DataScienceAPI.svg)](https://GitHub.com/saswatamcode/DataScienceAPI/issues/)
[![Open Source Love svg1](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)

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
