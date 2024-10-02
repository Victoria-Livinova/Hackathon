from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, RocCurveDisplay, ConfusionMatrixDisplay, PrecisionRecallDisplay
from sklearn import tree

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl

chemicals = pd.read_excel(pd.ExcelFile('chems.xlsx'))
# print(chemicals)

# features = chemicals.dropna()
X = chemicals.drop(['Chemical', 'Active', 'Smile'], axis=1)
y = chemicals['Active']
X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

parameters = {'n_estimators': range(10, 15), 'max_depth': range(3, 10)}
clf = RandomForestClassifier()
grid_search_cv_clf = GridSearchCV(clf, parameters, cv=5)
grid_search_cv_clf.fit(X_train, y_train)
# print(grid_search_cv_clf.best_params_)    # {'max_depth': 8, 'n_estimators': 12}
best_clf = grid_search_cv_clf.best_estimator_
# print(best_clf.score(X_test, y_test))       # 0.5 - точность мизерная, потому что большинство данных заполнено от балды

feature_importances = best_clf.feature_importances_
feature_importances_df = pd.DataFrame({'features': list(X_train), 'feature_importances': feature_importances})
# print(feature_importances_df.sort_values('feature_importances', ascending=False))
# самый важный для классификации параметр - верхний

clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_split=100, min_samples_leaf=10)
clf.fit(X_train, y_train)

clf.feature_importances_
tree.plot_tree(clf, feature_names=list(X), class_names=['0', '1'], filled=True)
plt.show()




y_pred = clf.predict(X_test)

print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('\nClassification Report:')
print(classification_report(y_test, y_pred))
