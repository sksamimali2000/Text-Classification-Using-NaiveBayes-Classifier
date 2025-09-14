# 📰 Text Classification Using Naive Bayes

A complete implementation of **Text Classification** using:
- Inbuilt `MultinomialNB` from scikit-learn  
- Self-implemented Naive Bayes classifier from scratch.

---

## 🚀 Project Overview

This project demonstrates:
- Data preprocessing (removing stop words, tokenizing, frequency count)  
- Feature extraction using top 2000 frequent words  
- Splitting data into training and testing sets  
- Model training using both:
    - scikit-learn’s **MultinomialNB**
    - Custom Naive Bayes implementation  
- Evaluation of both models using accuracy, confusion matrix, and classification report

---

## ⚡ Usage

```python
import numpy as np
import pandas as pd
import re
import os
import operator
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# Define stop words list
stop_word = [ "a", "about", "above", "after", ... ]  # (List truncated for brevity)

# Load dataset
X = []
Y = []
data_path = "C:/Users/bhard/Documents/DS & ML/Project Text Classification Using Naive Bayes/Datasets"
for category in os.listdir(data_path):
    for document in os.listdir(f"{data_path}/{category}"):
        with open(f"{data_path}/{category}/{document}", "r") as f:
            X.append((document, f.read()))
            Y.append(category)

# Split data
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y)

# Build word frequency dictionary (after removing stop words and non-alpha words)
dic = {}
for i in range(len(x_train)):
    word = x_train[i][1].lower()
    words = re.split(r'\W+', word)
    for s in words:
        if not s.isalpha() or s in stop_word or len(s) <= 2:
            continue
        dic[s] = dic.get(s, 0) + 1

# Select top 2000 frequent words as features
sorted_dic = sorted(dic.items(), key=operator.itemgetter(1), reverse=True)
features = [sorted_dic[i][0] for i in range(2000)]

# Build feature matrices for training and testing
x_train_dataset = np.zeros([len(x_train), len(features)], int)
for i in range(len(x_train)):
    words = re.split(r'\W+', x_train[i][1].lower())
    for word in words:
        if word in features:
            x_train_dataset[i][features.index(word)] += 1

x_test_dataset = np.zeros([len(x_test), len(features)], int)
for i in range(len(x_test)):
    words = re.split(r'\W+', x_test[i][1].lower())
    for word in words:
        if word in features:
            x_test_dataset[i][features.index(word)] += 1

# Inbuilt MultinomialNB
clf = MultinomialNB()
clf.fit(x_train_dataset, y_train)
y_pred = clf.predict(x_test_dataset)

print("MultinomialNB Score on Testing Data:", clf.score(x_test_dataset, y_test))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Self Implementation of Naive Bayes
def fit(x_train_dataset, y_train):
    count = {"total_doc": len(y_train)}
    y_train = np.array(y_train)
    classes = set(y_train)
    for c in classes:
        temp_count = 0
        x_train_with_c = x_train_dataset[y_train == c]
        temp2 = x_train_with_c.shape[0]
        count[c] = {}
        for feature in features:
            l = x_train_with_c[:, features.index(feature)].sum()
            count[c][feature] = l
            temp_count += l
        count[c]["word_in_class"] = temp_count
        count[c]["length"] = temp2
    return count

def probability(x_test, dic, cls):
    prob = np.log(dic[cls]["length"]) - np.log(dic["total_doc"])
    feature_list = list(dic[cls].keys())[:-2]
    for j in range(len(feature_list)):
        xj = x_test[j]
        if xj == 0:
            current_prob = 0
        else:
            num = dic[cls][feature_list[j]] + 1
            den = dic[cls]["word_in_class"] + len(feature_list)
            current_prob = np.log(num) - np.log(den)
        prob += current_prob
    return prob

def predict_for_single(x_test, dic):
    first_run = True
    classes = dic.keys()
    for c in classes:
        if c == "total_doc":
            continue
        prob = probability(x_test, dic, c)
        if first_run or prob > best_prob:
            best_prob = prob
            first_run = False
            best_class = c
    return best_class

def predict_(x_test, dic):
    return [predict_for_single(x, dic) for x in x_test]

def score(y_test, y_pred):
    return sum(1 for i, j in zip(y_test, y_pred) if i == j) / len(y_test)

dictionary = fit(x_train_dataset, y_train)
y_pred_self = predict_(x_test_dataset, dictionary)

print("\nSelf-Implemented Naive Bayes Score on Testing Data:", score(y_test, y_pred_self))
print(confusion_matrix(y_test, y_pred_self))
print(classification_report(y_test, y_pred_self))
```


✅ Results

MultinomialNB (sklearn):
Score on testing data: ~0.86

Self-Implemented Naive Bayes:
Score on testing data: ~0.87

Both models perform similarly well, demonstrating the correctness of the self implementation.

⚙️ Requirements

Python >= 3.7

numpy

pandas

scikit-learn

matplotlib

Install dependencies using:

pip install numpy pandas scikit-learn matplotlib

Made with ❤️ by Sk Samim Ali
