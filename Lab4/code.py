import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

path = 'log_files'
labels = []
text = []

for filename in os.listdir(path):
    if "Good" in filename:
        labels.append("1") # Normal activity logs
    else:
        labels.append("-1") # Attack activity logs
    
    filename = os.path.join(path, filename)
    print(filename)

    with open(filename, encoding='utf-8') as f:
        content = f.read()
    
    content = content.replace(",", " ")
    content = content.replace('"', " ")
    content = content.replace("'", " ")
    content = content.replace(";", " ")

    text.append(content)


vectorizer = CountVectorizer(
    stop_words = 'english',
    max_features = 1000
)

X = vectorizer.fit_transform(text)
feature_names = vectorizer.get_feature_names_out()

df = pd.DataFrame(
    data = X.toarray(),
    index = labels,
    columns = feature_names
)

output_file = "vectorized_logs.csv"
df.to_csv(output_file)


print('\n', "Output vectorized files saved to: ", output_file)
print('\n', "Shape of the dataset: ", df.shape)
print('\n', "First 5 rows of the dataset: ", df.head())

y = pd.Series(labels)

dt_classifier = DecisionTreeClassifier(
    criterion = 'entropy',
    random_state = 42
)

scoring = {
    'accuracy' : 'accuracy', 
    'precision' : 'precision_weighted', 
    'recall' : 'recall_weighted', 
    'f1' : 'f1_weighted'
}

n_samples = X.shape[0]
n_splits = min(10, n_samples)

cv_results = cross_validate (
    dt_classifier,
    X,
    y,
    cv = 3,
    scoring = scoring,
    return_train_score = False
)

print("cv_results: ", cv_results)
# cv_results:  {'fit_time': array([0.00463176, 0.00438309, 0.00468564]), 'score_time': array([0.00504637, 0.00442266, 0.00436115]), 'test_accuracy': array([1., 1., 1.]), 'test_precision': array([1., 1., 1.]), 'test_recall': array([1., 1., 1.]), 'test_f1': array([1., 1., 1.])}


for metric, scores in cv_results.items():
    if metric.startswith('test_'):
        name = metric.replace('test_', '')
        print('\n', name.capitalize(), scores.mean(), '+/-', scores.std())

dt_classifier.fit(X, y)

k = 10
importances = dt_classifier.feature_importances_
indices = importances.argsort()[::-1]
top_indices = indices[:k]

for i in top_indices:
    if importances[i] > 0:
        term = feature_names[i]
        importance = importances[i]
        print('\n', '\n', term, importance)
