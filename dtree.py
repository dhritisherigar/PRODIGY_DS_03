import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree
import matplotlib.pyplot as plt

# step 1: create a mock dataset
data = {
    'Age': [22, 35, 26, 29, 45, 34, 40, 30, 50, 23,
            36, 28, 42, 33, 27, 41, 39, 25, 31, 38],
    'Income': [25000, 48000, 32000, 40000, 60000, 52000, 58000, 42000, 70000, 27000,
               50000, 39000, 61000, 47000, 35000, 62000, 59000, 30000, 41000, 56000],
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female',
               'Female', 'Male', 'Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Male', 'Female'],
    'Purchased': ['No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No',
                  'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes']
}

df = pd.DataFrame(data)

# step 2: encode categorical variables
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Purchased'] = df['Purchased'].map({'No': 0, 'Yes': 1})

# step 3: define features and target
X = df[['Age', 'Income', 'Gender']]
y = df['Purchased']

# step 4: split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# step 5: create and train the model
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# step 6: predictions and evaluation
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# step 7: visualize the tree
plt.figure(figsize=(7,5))
tree.plot_tree(clf, feature_names=X.columns, class_names=['Not Purchased', 'Purchased'], filled=True)
plt.title("Decision Tree - Customer Purchase Prediction")
plt.savefig("DecisionTree.png")
plt.show()