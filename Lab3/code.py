import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_validate

ds = pd.read_csv('Spambase.csv')

print('\n', ds.head()) 
print('\n', ds.describe())
print('\n', ds.info()) # All columns are int64 and no null values
print('\n', ds.shape) #4601 rows & 56 columns

X = ds.drop(columns=['id', 'class'])
y = ds['class']

decision_tree = DecisionTreeClassifier()
naive_bayes = BernoulliNB()

k_fold = 10
scoring=['accuracy', 'precision', 'recall', 'f1']

dt_results = cross_validate(decision_tree, X, y, cv=k_fold, scoring=scoring)
#print("Decision Tree Results: ", dt_results)

print('\n', '\n', "For Decision Tree:")
accuracy = dt_results['test_accuracy'].mean()
precision = dt_results['test_precision'].mean()
recall = dt_results['test_recall'].mean()
f1_score = dt_results['test_f1'].mean()

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1_score)




nb_results = cross_validate(naive_bayes, X, y, cv=k_fold, scoring=scoring)
#print("Naive Bayes Results: ", nb_results)

print('\n', '\n', "For Naive Bayes:")
accuracy = nb_results['test_accuracy'].mean()
precision = nb_results['test_precision'].mean()
recall = nb_results['test_recall'].mean()
f1_score = nb_results['test_f1'].mean()

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1_score)