import sklearn
from sklearn import datasets
 
# Load iris dataset
iris = datasets.load_iris()

# mendefinisikan atribut dan label pada dataset
x=iris.data
y=iris.target

from sklearn import tree
 
# membuat model dengan decision tree classifier
clf = tree.DecisionTreeClassifier()

from sklearn.model_selection import cross_val_score
 
# mengevaluasi performa model dengan cross_val_score
scores = cross_val_score(clf, x, y, cv=5)

print(scores)