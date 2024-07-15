
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from matplotlib.ticker import FuncFormatter
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score , recall_score, precision_score
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


df = pd.read_csv('fraud_detection_dataset.csv')
print(df.head())
print(df.isna().sum())  
print('total_nan =', df.isna().sum().sum())  

special_colors = ['#2ca02c', '#d62728']  


# Adjusting Graphic Size
plt.figure(figsize=(10, 5))

ax = sns.countplot(data=df, x='type', hue='isFraud', palette=special_colors)
plt.yscale('log')
plt.title('Fraud Analysis of Transactions', fontsize=15, weight='bold')

for p in ax.patches:
    ax.annotate('{:.1f}'.format(p.get_height()),
                (p.get_x()+p.get_width()/2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 10),
                textcoords='offset points',
                fontsize=10, color='black', weight='bold')
plt.show()

print('FRAUD ANALYSIS')
print(df['isFraud'].value_counts())

data_counts_difference = 1000
# UNDERSAMPLING
balance_df0 = df[df['isFraud'] == 0][:len(df[df['isFraud'] == 1])+data_counts_difference]
balance_df1 = df[df['isFraud'] == 1]
print(balance_df0['isFraud'].value_counts())
print(balance_df1['isFraud'].value_counts())


balanced_df = pd.concat([balance_df0, balance_df1], axis=0)
print(balanced_df['isFraud'].value_counts())
balanced_df = shuffle(balanced_df)  # random arrangement of 0 and 1
balanced_df = pd.concat([balance_df0, balance_df1], axis=0)
print(balanced_df['isFraud'].value_counts())
print(balanced_df.head(200))


def human_format(num, pos):
    if num >= 1e6:
        return f'{int(num/1e6)}M'
    elif num >= 1e3:
        return f'{int(num/1e3)}K'
    else:
        return str(int(num))


# Count Plot for type
sns.countplot(x=balanced_df['type'])
plt.title('Balanced Frequency of Categories ')
plt.gca().yaxis.set_major_formatter(FuncFormatter(human_format))
plt.show()

# Piechart for type distribution
balanced_df['type'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
plt.title('Category Distribution')
plt.ylabel('')
plt.show()


# label encoding
le = LabelEncoder()

# Categorical data are converted to numerical data  
balanced_df['type'] = le.fit_transform(balanced_df['type'])

# unnecessary categorical data were dropped
process_df = balanced_df.drop(['nameOrig', 'nameDest'], axis=1)  
corr_matrix= process_df.corr() # analysing correlation

# correlation matrix
sns.heatmap(corr_matrix, annot=True, cmap='inferno', fmt='.2f')
plt.title('Correlation Matrx')
plt.show()


X = process_df.drop('isFraud',axis = 1)
y = process_df['isFraud']

# Standarization
standard_scaler = StandardScaler()
X = standard_scaler.fit_transform(X)

X_train , X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

#MODELS

# DECISION TREE
print('\nDECISION TREE')
decision_tree = DecisionTreeClassifier(max_depth = 25 )
decision_tree.fit(X_train ,y_train)#Training
prediction_y = decision_tree.predict(X_test)

accuracy_decision_tree = accuracy_score(y_test,prediction_y)
recall_decision_tree = recall_score(y_test,prediction_y)
precision_decision_tree = precision_score(y_test , prediction_y)

print("Accuracy == %{:.2f}".format(accuracy_decision_tree*100) ) 
print("Recall == %{:.2f}".format(recall_decision_tree*100) ) 
print("Precision == %{:.2f}".format(precision_decision_tree*100) ) 


# Logistic Regression
print('\nLOGISTIC REGRESSION ')
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train,y_train)
prediction_y = logistic_regression.predict(X_test)

accuracy_logistic_regression = accuracy_score(y_test,prediction_y)
recall_logistic_regression = recall_score(y_test,prediction_y)
precision_logistic_regression = precision_score(y_test , prediction_y)

print("Accuracy == %{:.2f}".format(accuracy_logistic_regression*100) ) 
print("Recall == %{:.2f}".format(recall_logistic_regression*100) ) 
print("Precision == %{:.2f}".format(precision_logistic_regression*100) ) 

# Random Forest
print('\nRANDOM FOREST ')
random_forest = RandomForestClassifier(n_estimators = 10)
random_forest.fit(X_train, y_train)
prediction_y = random_forest.predict(X_test)

accuracy_random_forest = accuracy_score(y_test,prediction_y)
recall_random_forest = recall_score(y_test,prediction_y)
precision_random_forest = precision_score(y_test , prediction_y)

print("Accuracy == %{:.2f}".format(accuracy_random_forest*100) ) 
print("Recall == %{:.2f}".format(recall_random_forest*100) ) 
print("Precision == %{:.2f}".format(precision_random_forest*100) ) 

# Naive Bayes
print('\nNAIVE BAYES')
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)
prediction_y = naive_bayes.predict(X_test)


accuracy_naive_bayes = accuracy_score(y_test,prediction_y)
recall_naive_bayes = recall_score(y_test,prediction_y)
precision_naive_bayes = precision_score(y_test , prediction_y)

print("Accuracy == %{:.2f}".format(accuracy_naive_bayes*100) ) 
print("Recall == %{:.2f}".format(recall_naive_bayes*100) ) 
print("Precision == %{:.2f}".format(precision_naive_bayes*100) ) 

#MULTILAYER PERCEPTRON
print('\nMULTILAYER PERCEPTRON')
multi_layer_perceptron = MLPClassifier(hidden_layer_sizes = 5, batch_size = 32, learning_rate = 'adaptive', learning_rate_init = 0.001)
multi_layer_perceptron.fit(X_train, y_train)
prediction_y = multi_layer_perceptron.predict(X_test)

accuracy_multi_layer_perceptron = accuracy_score(y_test,prediction_y)
recall_multi_layer_perceptron= recall_score(y_test,prediction_y)
precision_multi_layer_perceptron = precision_score(y_test , prediction_y)

print("Accuracy == %{:.2f}".format(accuracy_multi_layer_perceptron*100) ) 
print("Recall == %{:.2f}".format(recall_multi_layer_perceptron*100) ) 
print("Precision == %{:.2f}".format(precision_multi_layer_perceptron*100) ) 
     
