import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns

df = sns.load_dataset('iris')
df.info()
df.isnull().any()
df.shape
target = df['species']
df1 = df.copy()
df1 = df1.drop('species', axis =1)
df1.shape
df1.head()
# Defining the attributes
X = df1
target
#label encoding
le = LabelEncoder()
target = le.fit_transform(target)
target
y = target
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.2, random_state = 42)

print("Training split input- ", X_train.shape)
print("Testing split input- ", X_test.shape)
# Creating Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier(criterion = 'entropy')
dtree.fit(X_train,y_train)
clf = DecisionTreeClassifier(criterion = 'entropy')
clf.fit(X_train,y_train)
plt.figure(figsize = (20,20))
dec_tree = plot_tree(decision_tree=dtree, feature_names = df1.columns, 
                     class_names =["setosa", "vercicolor", "verginica"] , filled = True , precision = 4, rounded = True)

plt.savefig("IrisTree1.png")
plt.show()
