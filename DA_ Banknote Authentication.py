
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from random import seed
from random import random
from random import uniform
from math import exp
import seaborn as sns 
import scipy.stats as stats
#Skealern:
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_roc_curve,roc_curve, roc_auc_score
from sklearn import metrics
#keras
from  tensorflow  import keras
from keras.models import Sequential
from keras.layers import Dense

# (i) Perform all necessary preprocessing for the given dataset
df= pd.read_csv('data_banknote_authentication (1).csv')
X = df.copy() #dataset has been copied to  X
X.shape 
X.columns
X.head()
X.tail()  
X.describe() 
X.describe(include='all')
X.info()
###############################################################################################################################

#1) MISSING VALUE 
df.isnull().any() 
X.isnull().sum() # no missing value
#################################################################################
#2) DUPLICATED ROW
#check for the duplicate row in data set.
print(X.duplicated().value_counts()) 
#this data set have 24 duplicated values.
print(X[X.duplicated()]) # To check view the duplicated values
X=X.drop_duplicates() # To drop the duplicate values 
###############################################################################
#) THE OUTLIERS

X._get_numeric_data().columns.tolist()
# make the boxplot of given columns to check the outliers

  ##################################################################
                     #Box plot and Q-Q Plot of Temperature © column
temp_df = pd.DataFrame(X, columns=['variance'])
temp_df.boxplot(vert=False)
#plot the probability plot to given columns for check the patterns of distribution, values range and etc.,
plt.rcParams["figure.figsize"] = (10, 6)
stats.probplot(X["variance"], dist="norm", plot=plt)
plt.show()
 ##################################################################
                #  Box plot and Q-Q Plot of Apparent Temperature  column
temp_df = pd.DataFrame(X, columns=['skewness'])
temp_df.boxplot(vert=False)
#plot the probability plot to given columns for check the patterns of distribution, values range and etc.,
plt.rcParams["figure.figsize"] = (10, 6)
stats.probplot(X["skewness"], dist="norm", plot=plt)
plt.show()
          ##################################################################
          
          #  Box plot and Q-Q Plot of Apparent Temperature  column
temp_df = pd.DataFrame(X, columns=['curtosis'])
temp_df.boxplot(vert=False)
#plot the probability plot to given columns for check the patterns of distribution, values range and etc.,
plt.rcParams["figure.figsize"] = (10, 6)
stats.probplot(X["curtosis"], dist="norm", plot=plt)
plt.show()
          ##################################################################
          
          #  Box plot and Q-Q Plot of Apparent Temperature  column
temp_df = pd.DataFrame(X, columns=['entropy'])
temp_df.boxplot(vert=False)
#plot the probability plot to given columns for check the patterns of distribution, values range and etc.,
plt.rcParams["figure.figsize"] = (10, 6)
stats.probplot(X["entropy"], dist="norm", plot=plt)
plt.show()
          ##################################################################
          
          
          #  Box plot and Q-Q Plot of Apparent Temperature  column
temp_df = pd.DataFrame(X, columns=['class'])
temp_df.boxplot(vert=False)
#plot the probability plot to given columns for check the patterns of distribution, values range and etc.,
plt.rcParams["figure.figsize"] = (10, 6)
stats.probplot(X["class"], dist="norm", plot=plt)
plt.show()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop("class", axis=1))
scaled_features = scaler.fit_transform(df.drop("class", axis=1))
scaled_features
scaled_df = pd.DataFrame(data=scaled_features, columns=df.columns[0:4])
scaled_df.head()
          ##################################################################
  #Use SciKit Learn to create training and testing sets of the data   
x= scaled_df   
y = df["class"]    
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
#standardizing
standardScaler = StandardScaler()
x_train = standardScaler.fit_transform(x_train)
x_test = standardScaler.transform(x_test)          
          ###################################################################################

# define model
#We can define a minimal MLP model. In this case, we will use one hidden layer with 4 nodes &1 output layer
 #We will use the sigmoid  activation function in the hidden layer and the 'uniform' weight initialization.
model = Sequential()
model.add(Dense(4, activation='sigmoid', kernel_initializer='uniform',input_dim=4))
model.add(Dense(4, activation = 'sigmoid', kernel_initializer = 'uniform'))
#The output of the model is a sigmoid activation for binary classification
model.add(Dense(1, activation = 'sigmoid', kernel_initializer = 'uniform'))
 
###################################
         
     #initial weights
    # layer. get_weights(): returns the weights of the layer as a list of Numpy arrays.
first_l_w = model.layers[0].get_weights()[0]
first_l_b  = model.layers[0].get_weights()[1]

second_l_w = model.layers[1].get_weights()[0]
second_l_b  =model.layers[1].get_weights()[1]

third_l_w = model.layers[2].get_weights()[0]
third_l_b  =model.layers[2].get_weights()[1]

#First layer initial weights 
initial_weights1 = pd.DataFrame(columns=['in1,1','in1,2','in1,3','in1,4','b1'])
initial_weights1['in1,1']=first_l_w.tolist()[0]
initial_weights1['in1,2']=first_l_w.tolist()[1]
initial_weights1['in1,3']=first_l_w.tolist()[2]
initial_weights1['in1,4']=first_l_w.tolist()[3]
initial_weights1['b1']=first_l_b.tolist()

#Second layer initial weights
initial_weights2 = pd.DataFrame(columns=['in2,1','in2,2','in2,3','in2,4','b2'])
initial_weights2['in2,1']=second_l_w.tolist()[0]
initial_weights2['in2,2']=second_l_w.tolist()[1]
initial_weights2['in2,3']=second_l_w.tolist()[2]
initial_weights2['in2,4']=second_l_w.tolist()[3]
initial_weights2['b2']=second_l_b.tolist()

#Third layer initial weights
initial_weights3 = pd.DataFrame(columns=['in3','b3'])
initial_weights3['in3']=third_l_w.tolist()[0]
initial_weights3['b3']=third_l_b.tolist()
#compile the model 
#The output of the model is a sigmoid activation for binary classification and we will minimize binary cross-entropy loss.
model.compile(optimizer = 'adam', loss = 'binary_crossentropy',  metrics=['accuracy'])
training_accuracy = []
testing_accuracy = []
training_error = []
testing_error = []   
initial_weights1 
initial_weights2
initial_weights3

# To evaluate a model, we also need an **evaluation metric:**
# Most common choices for classification problems:

for i in range(0,50):
  model.fit(x_train, y_train, batch_size = 10)

  y_train_pred = model.predict(x_train)
  y_train_pred = (y_train_pred > 0.5)

  y_test_pred = model.predict(x_test)
  y_test_pred = (y_test_pred > 0.5)
  y_train_predicted = pd.DataFrame(y_train_pred)
  y_test_predicted = pd.DataFrame(y_test_pred)
#y_test = actual values. y_test_pred = values you predicted. 
#This means you can evaluate the performance of your model by comparing y_test and y_pred. 
#using Accuracy classification score binary classification .
  training_accuracy.append(accuracy_score(y_train_predicted, y_train))
    
  testing_accuracy.append(accuracy_score(y_test_predicted, y_test))
#Classification error
  training_error.append(1-training_accuracy[i])
  testing_error.append(1-testing_accuracy[i])

#Final weights
first_layer_weights_final = model.layers[0].get_weights()[0]
first_layer_biases_final  = model.layers[0].get_weights()[1]
second_layer_weights_final = model.layers[1].get_weights()[0]
second_layer_biases_final  =model.layers[1].get_weights()[1]
third_layer_weights_final = model.layers[2].get_weights()[0]
third_layer_biases_final  =model.layers[2].get_weights()[1]

#First layer final weights
final_weights1 = pd.DataFrame(columns=['in1,1','in1,2','in1,3','in1,4','b1'])
final_weights1['in1,1']=first_layer_weights_final.tolist()[0]
final_weights1['in1,2']=first_layer_weights_final.tolist()[1]
final_weights1['in1,3']=first_layer_weights_final.tolist()[2]
final_weights1['in1,4']=first_layer_weights_final.tolist()[3]
final_weights1['b1']=first_layer_biases_final.tolist()

#Second final weights
final_weights2= pd.DataFrame(columns=['in2,1','in2,2','in2,3','in2,4','b2'])
final_weights2['in2,1']=second_layer_weights_final.tolist()[0]                           
final_weights2['in2,2']=second_layer_weights_final.tolist()[1]
final_weights2['in2,3']=second_layer_weights_final.tolist()[2]
final_weights2['in2,4']=second_layer_weights_final.tolist()[3]
final_weights2['b2']=second_layer_biases_final.tolist()

#Third layer initial weights
final_weights3= pd.DataFrame(columns=['in3','b3'])
final_weights3['in3']=third_l_w.tolist()[0]
final_weights3['b3']=third_l_b.tolist()
final_weights1
final_weights2
final_weights3

#Compute confusion matrix to evaluate the accuracy of a classification. 

Matrix= metrics.confusion_matrix(y_test,  y_test_pred)
print(Matrix)

accuracyPercentage= (Matrix[0,0]+Matrix[1,1])/sum(sum(Matrix))*100
print(accuracyPercentage)
accuracyPercentage

          ####################################################33333
plt.plot(training_accuracy,label = "Training Accuracy")
plt.plot(testing_accuracy, label = "Testing Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy%')
plt.show()

##########################################################
plt.plot(training_error,label = "Training Error")
plt.plot(testing_error, label = "Testing Error")
plt.xlabel('Epochs')
plt.ylabel('Error%')
plt.show()

###########################################
false_positive_rate, true_positive_rate, threshold = roc_curve(y_test,y_test_predicted)
plt.plot(false_positive_rate, true_positive_rate)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()








  
          



