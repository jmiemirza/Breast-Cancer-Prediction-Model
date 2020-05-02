import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

def dataframe():
    f=open('field_names.txt',"r")
    lines=f.readlines()
    result=[]
    for x in lines:
        result.append(x.split('\n')[0])
    f.close()

    data= pd.read_csv("breast-cancer.csv")

    df = pd.read_csv("breast-cancer.csv", names=result)
    
    return df

def mean_median():

    df = dataframe()

    dataset = df.copy()
    dataset.tail()

    diagnosis = dataset['diagnosis']

    smoothness_mean_m_tumor = [] #for malignant tumors
    smoothness_mean_b_tumor = [] #for beningnant tumors
    compactness_mean_m_tumor = [] #for malignant tumors
    compactness_mean_b_tumor = [] #for beningnant tumors

    for key, value in diagnosis.iteritems():
        if diagnosis[key] == 'M':
            smoothness_mean_m_tumor.append((dataset['smoothness_mean']))
            compactness_mean_m_tumor.append((dataset['compactness_mean']))
        if diagnosis[key] == 'B': 
            smoothness_mean_b_tumor.append((dataset['smoothness_mean']))
            compactness_mean_b_tumor.append((dataset['compactness_mean']))


    ####Taking the median####
    np.sort(smoothness_mean_m_tumor)  
    np.sort(smoothness_mean_b_tumor) 
    np.sort(compactness_mean_m_tumor) 
    np.sort(compactness_mean_b_tumor) 
          
    np.median(smoothness_mean_m_tumor)
    print('The median for smoothness mean for Malignant tumors is:{}'.format(np.median(smoothness_mean_m_tumor)))
    np.median(smoothness_mean_b_tumor)
    print('The median for smoothness mean for Benign tumors is:{}'.format(np.median(smoothness_mean_b_tumor)))
    np.median(compactness_mean_m_tumor)
    print('The median for compactness mean for Malignant tumors is:{}'.format(np.median(compactness_mean_m_tumor)))
    np.median(compactness_mean_b_tumor)
    print('The median for compactness mean for Benign tumors is:{}'.format(np.median(compactness_mean_b_tumor)))
            
    ####Taking the mean####
    np.mean(smoothness_mean_m_tumor)
    print('The mean for smoothness mean for Malignant tumors is:{}'.format(np.mean(smoothness_mean_m_tumor)))
    np.mean(smoothness_mean_b_tumor)
    print('The mean for smoothness mean for Benign tumors is:{}'.format(np.mean(smoothness_mean_b_tumor)))
    np.mean(compactness_mean_m_tumor)
    print('The mean for compactness mean for Malignant tumors is:{}'.format(np.mean(compactness_mean_m_tumor)))
    np.mean(compactness_mean_b_tumor)
    print('The mean for compactness mean for Benign tumors is:{}'.format(np.mean(compactness_mean_b_tumor)))

def bootstrapping():

    df = dataframe()
    bootstrapping = df.sample(frac =.25)
    
def visualization():

    df = dataframe()
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B':0})
    corr = df.corr()
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(30, 25))
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, cmap=cmap, center=0,annot = True,
            square=True, linewidths=.25, cbar_kws={"shrink": .5});
    
    #Setting up histograms for checking data distribution        
    df.groupby('diagnosis').size()
    df.groupby('diagnosis').hist(figsize=(12, 12))
            
    plt.show()
    
def data():
    
    df = dataframe()
    
    
    X = df.iloc[:, 2:31].values
    
    #Taking the diagnosis Column for encoding
    Y = df.iloc[:, 1].values
    
    #Encoding 1 for malignant and 0 for begnign tumors
    labelencoder_Y = LabelEncoder()
    Y = labelencoder_Y.fit_transform(Y)
    
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
    
    sc=StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    return X_train, X_test, Y_train, Y_test
    
def model_regression():
    
    X_train, X_test, Y_train, Y_test = data()
    
    classifier = LogisticRegression(random_state = 0)
    
    clf = classifier.fit(X_train, Y_train)
    
    print('The cassification accuracy for Malignant and Benign Tumors is: {}'.format(clf.score(X_train, Y_train)))
    
    Y_pred = classifier.predict(X_test)
    
    cm = confusion_matrix(Y_test, Y_pred)
    c = print(cm[0, 0] + cm[1, 1])
            
if __name__== "__main__":
    
    
    model_regression()
    visualization()

    
    
