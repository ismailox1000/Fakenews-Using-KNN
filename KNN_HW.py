# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 23:59:40 2022

@author: ISMAIL OUBAH
"""
import pandas as pd
import re

df=pd.read_csv('FakeNewsNet.csv')
df=df.dropna()

df["merged_column"] = df["title"] + " " + df["news_url"]

# rremove the duplicate words in the merged column
df["merged_column"] = df["merged_column"].apply(lambda x: " ".join(list(set(x.split()))))

# drop the original columns
df = df.drop(["title", "news_url"], axis=1)

#merge the 2 columns into one column
df["merged_column"]=df["merged_column"].str.replace(r"[-_(\"'/):.,@]",' ')

#df["source_domain"]=df["source_domain"].str.replace(r'.',' ')
dff=df["merged_column"]
#dff2=df["source_domain"]
#function to remove the additional spaces
def remove_space(text):
    return re.sub(r"\s+"," ",text).strip()
df["merged_column"]=df["merged_column"].apply(lambda x:remove_space(x))
print("\n-----result-----")
[print("*** "+x) for x in df["merged_column"][:4]]
dff1=df["merged_column"]
#function to remove the accents from the words 
import unicodedata
def remove_accent(text):
    try:
        text = unicode(text, 'utf-8')
    except NameError:
        pass
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
    return str(text)
df["merged_column"]=df["merged_column"].apply(lambda x: remove_accent(x))




#this is for the manual testing at last
testingdata=df.tail(50)
testingdata.shape
for i in range(22865,22815,-1):
    df.drop([i],axis=0,inplace=True)
df.shape  


X = df[["merged_column",'tweet_num']]
y = df['real']
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
tweet_num=X['tweet_num']
tweet_num=sc.fit_transform(tweet_num.values.reshape(-1,1))
X['tweet_num']=tweet_num
print(X['tweet_num'])
#split data to train and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=44, shuffle =True)

#convert words to vecs 
from sklearn.feature_extraction.text import CountVectorizer
Vectorizer = CountVectorizer()
Xv_train1 =Vectorizer.fit_transform(X_train["merged_column"])
Xv_test1 =Vectorizer.transform(X_test["merged_column"])
# import KNN 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 8, metric = 'minkowski', p = 2)
knn.fit(Xv_train1, y_train)
#scoring our the result
print('KNeighborsRegressorModel Train Score is : ' , knn.score(Xv_train1, y_train))
y_predK = knn.predict(Xv_test1)
#this is to see our model how efficient it is 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predK)

#  MAE and MSE errors 
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error
MAEValue = mean_absolute_error(y_test, y_predK, multioutput='uniform_average') # it can be raw_values
print('Mean Absolute Error Value is : ', MAEValue)
#Calculating Mean Squared Error
MSEValue = mean_squared_error(y_test, y_predK, multioutput='uniform_average') # it can be raw_values
print('Mean Squared Error Value is : ', MSEValue)

# now we test how it will perform on unrecognize data we take X values
X_testing_manual=testingdata['merged_column']

Xv_testing=Vectorizer.transform(X_testing_manual)
#predict output
y_predK_testing = knn.predict(Xv_testing)

actual_y_manual=testingdata['real']
#then we compare the actual output and predicted 
cm_testing = confusion_matrix(actual_y_manual, y_predK_testing)
MAEValue_manual = mean_absolute_error(actual_y_manual, y_predK_testing, multioutput='uniform_average') # it can be raw_values
print('Mean Absolute Error Value is : ', MAEValue_manual)















