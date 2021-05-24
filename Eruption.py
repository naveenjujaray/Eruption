from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

main = tkinter.Tk()
main.title("Understanding the Timing of Eruption End using a Machine Learning Approach to Classification of Seismic Time Series") #designing main screen
main.geometry("1300x1200")

global filename
global svm_acc,lr_acc,rf_acc,gaussian_acc
global X, Y
global X_train, X_test, y_train, y_test
global dataset
global model
global cls1,cls2,cls3,cls4

def upload(): #function to upload tweeter profile
    global filename
    global dataset
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n");
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset))

    
def preprocess():
    global X, Y
    global X_train, X_test, y_train, y_test
    global dataset
    text.delete('1.0', END)
    dataset.fillna(0, inplace = True)
    dataset = dataset[['Latitude','Longitude','Magnitude','Horizontal Distance','Horizontal Error','Root Mean Square']]
    X = dataset.values
    Y = []
    for i in range(len(X)):
        m = X[i,2]
        if m < 6.0:
            Y.append(1)
        else:
            Y.append(0)

    Y = np.asarray(Y)
    X = normalize(X)
    text.insert(END,str(X)+"\n")
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Dataset contains total records = "+str(len(X))+"\n")
    text.insert(END,"Total Dataset Records used to Train Machine Learning Model = "+str(X_train.shape[0])+"\n")
    text.insert(END,"Total Dataset Records used to Test Machine Learning Model  = "+str(X_test.shape[0])+"\n")
    
def runSVM():
    global svm_acc
    global cls1
    text.delete('1.0', END)
    cls = svm.SVC(C=1.5,gamma='scale')
    cls.fit(X, Y)
    prediction_data = cls.predict(X_test) 
    svm_acc = accuracy_score(y_test,prediction_data)*100
    text.insert(END,"SVM Accuracy on Eruption Dataset : "+str(svm_acc)+"\n")
    cls1 = cls
    
def runLR():
    global lr_acc
    global cls2
    cls = LogisticRegression()
    cls.fit(X, Y)
    prediction_data = cls.predict(X_test) 
    lr_acc = accuracy_score(y_test,prediction_data)*100
    text.insert(END,"Logistic Regression Accuracy on Eruption Dataset : "+str(lr_acc)+"\n")
    cls2 = cls

def runRandomForest():
    global model
    global cls3
    global rf_acc
    cls = RandomForestClassifier(n_estimators=20, random_state=0)
    cls.fit(X, Y)
    prediction_data = cls.predict(X_test) 
    rf_acc = accuracy_score(y_test,prediction_data)*100
    text.insert(END,"Random Forest Accuracy on Eruption Dataset : "+str(rf_acc)+"\n")
    model = cls
    cls3 = cls

def runGaussian():
    global cls4
    global gaussian_acc
    cls = GaussianProcessClassifier()
    cls.fit(X_test, y_test)
    prediction_data = cls.predict(X_test) 
    gaussian_acc = accuracy_score(y_test,prediction_data)*100
    text.insert(END,"Gaussian Process Classifier Accuracy on Eruption Dataset : "+str(gaussian_acc)+"\n")
    cls4 = cls
    
    
def graph():
    height = [svm_acc,lr_acc,rf_acc,gaussian_acc]
    bars = ('SVM Accuracy','Logistic Regression Accuracy','Random Forest Accuracy','Gaussian Accuracy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title('Accuracy Comparison Graph')
    plt.show()

def predict():
    text.delete('1.0', END)
    name = filedialog.askopenfilename(initialdir = "Dataset")
    test = pd.read_csv(name)
    test.fillna(0, inplace = True)
    test = test[['Latitude','Longitude','Magnitude','Horizontal Distance','Horizontal Error','Root Mean Square']]
    test = test.values
    print(test.shape)
    y_pred = model.predict(test)
    print(y_pred)
    for i in range(len(test)):
        if str(y_pred[i]) == '0':
            text.insert(END,"X=%s, Predicted = %s" % (test[i], 'No Eruption Activity Detected')+"\n\n")
        else:
            text.insert(END,"X=%s, Predicted = %s" % (test[i], 'Eruption Activity Detected at Given Time')+"\n\n")

    
def rocGraph():
    predict = cls1.predict(X_test)
    svm_fpr, svm_tpr, _ = roc_curve(y_test, predict)

    predict = cls2.predict(X_test)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, predict)

    predict = cls3.predict(X_test)
    rf_fpr, rf_tpr, _ = roc_curve(y_test, predict)

    predict = cls4.predict(X_test)
    g_fpr, g_tpr, _ = roc_curve(y_test, predict)

    plt.plot(svm_fpr, svm_tpr, linestyle='--', label='SVM')
    plt.plot(lr_fpr, lr_tpr, linestyle='--', label='Logistic Regression')
    plt.plot(rf_fpr, rf_tpr, linestyle='--', label='Random Forest')
    plt.plot(g_fpr, g_tpr, linestyle='--', label='Gaussian Process')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

font = ('times', 16, 'bold')
title = Label(main, text='Understanding the Timing of Eruption End using a Machine Learning Approach to Classification of Seismic Time Series')
title.config(bg='firebrick4', fg='dodger blue')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Seismic Dataset", command=upload, bg='#ffb3fe')
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess Dataset Feature Extraction", command=preprocess, bg='#ffb3fe')
processButton.place(x=270,y=550)
processButton.config(font=font1) 

svmButton1 = Button(main, text="Run SVM Algorithms", command=runSVM, bg='#ffb3fe')
svmButton1.place(x=610,y=550)
svmButton1.config(font=font1) 

lrButton = Button(main, text="Run Logistic Regression", command=runLR, bg='#ffb3fe')
lrButton.place(x=50,y=600)
lrButton.config(font=font1) 

rfButton = Button(main, text="Run Random Forest Algorithm", command=runRandomForest, bg='#ffb3fe')
rfButton.place(x=270,y=600)
rfButton.config(font=font1) 

gpButton = Button(main, text="Run Gaussian Process Classifier", command=runGaussian, bg='#ffb3fe')
gpButton.place(x=610,y=600)
gpButton.config(font=font1)

graphButton = Button(main, text="All Algorithms Accuracy Graph", command=graph, bg='#ffb3fe')
graphButton.place(x=50,y=650)
graphButton.config(font=font1)

predictButton = Button(main, text="Predict Eruption", command=predict, bg='#ffb3fe')
predictButton.place(x=350,y=650)
predictButton.config(font=font1)

predictButton = Button(main, text="Predict Eruption", command=predict, bg='#ffb3fe')
predictButton.place(x=350,y=650)
predictButton.config(font=font1)

rocButton = Button(main, text="ROC Curve Graph", command=rocGraph, bg='#ffb3fe')
rocButton.place(x=520,y=650)
rocButton.config(font=font1)

main.config(bg='LightSalmon3')
main.mainloop()
