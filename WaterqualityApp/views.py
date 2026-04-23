users_db = []  # temporary storage (instead of MySQL)
from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import os
from django.core.files.storage import FileSystemStorage
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import pandas as pd
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import pickle
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from keras.layers import Dense, Dropout

import keras.layers
from sklearn.ensemble import RandomForestClassifier


global X, Y, dataset, X_train, X_test, y_train, y_test
global algorithms, accuracy, f1, precision, recall, classifier
classifier = None

def ProcessData(request):
    if request.method == 'GET':
        global X, Y, dataset, X_train, X_test, y_train, y_test
        dataset = pd.read_csv("Dataset/ml.csv")
        dataset.fillna(0, inplace = True)
        label = dataset.groupby('labels').size()
        columns = dataset.columns
        temp = dataset.values
        dataset = dataset.values
        X = dataset[:,2:dataset.shape[1]-1]
        Y = dataset[:,dataset.shape[1]-1]
        Y = Y.astype(int)
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        output = '<table border=1 align=center width=100%>'
        font = '<font size="" color="black">'
        output += "<tr>"
        for i in range(len(columns)):
            output += "<th>"+font+columns[i]+"</th>"            
        output += "</tr>"
        for i in range(len(temp)):
            output += "<tr>"
            for j in range(0,temp.shape[1]):
                output += '<td><font size="" color="black">'+str(temp[i,j])+'</td>'
            output += "</tr>"    
        context= {'data': output}
        label.plot(kind="bar")
        plt.title("Water Quality Graph, 0 (Good quality) & 1 (Poor Quality)")
        plt.show()
        return render(request, 'UserScreen.html', context)
        

def TrainRF(request):
    global X, Y
    global algorithms, accuracy, fscore, precision, recall, classifier
    if request.method == 'GET':
        algorithms = []
        accuracy = []
        precision = []
        recall = []
        fscore = []
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        cls = RandomForestClassifier()
        cls.fit(X, Y)
        classifier = cls
        predict = cls.predict(X_test)
        p = precision_score(y_test, predict,average='macro') * 100
        r = recall_score(y_test, predict,average='macro') * 100
        f = f1_score(y_test, predict,average='macro') * 100
        a = accuracy_score(y_test,predict)*100
        algorithms.append("Random Forest")
        accuracy.append(a)
        precision.append(p)
        recall.append(r)
        fscore.append(f)
        arr = ['Algorithm Name', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
        output = '<table border=1 align=center width=100%>'
        font = '<font size="" color="black">'
        output += "<tr>"
        for i in range(len(arr)):
            output += "<th>"+font+arr[i]+"</th>"
        output += "</tr>"
        for i in range(len(algorithms)):
            output +="<tr><td>"+font+str(algorithms[i])+"</td><td>"+font+str(accuracy[i])+"</td><td>"+font+str(precision[i])+"</td><td>"+font+str(recall[i])+"</td><td>"+font+str(fscore[i])+"</td></tr>"
        context= {'data': output}
        return render(request, 'UserScreen.html', context)

def TrainLSTM(request):
    if request.method == 'GET':
        global X, Y
        global algorithms, accuracy, fscore, precision, recall
        algorithms = []
        accuracy = []
        fscore = []
        precision = []
        recall = []     
        X1 = np.reshape(X, (X.shape[0], X.shape[1], 1))
        Y1 = to_categorical(Y)
        print(X1.shape)
        X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.2)
        if request.method == 'GET':
            lstm_model = Sequential()
            lstm_model.add(keras.layers.LSTM(100,input_shape=(X_train.shape[1], X_train.shape[2])))
            lstm_model.add(Dropout(0.5))
            lstm_model.add(Dense(100, activation='relu'))
            lstm_model.add(Dense(y_train.shape[1], activation='softmax'))
            lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            lstm_model.fit(X1, Y1, epochs=40, batch_size=32, validation_data=(X_test, y_test))             
            print(lstm_model.summary())#printing model summary
            predict = lstm_model.predict(X_test)
            predict = np.argmax(predict, axis=1)
            testY = np.argmax(y_test, axis=1)
            p = precision_score(testY, predict,average='macro') * 100
            r = recall_score(testY, predict,average='macro') * 100
            f = f1_score(testY, predict,average='macro') * 100
            a = accuracy_score(testY,predict)*100
            algorithms.append("LSTM")
            accuracy.append(a)
            precision.append(p)
            recall.append(r)
            fscore.append(f)
            arr = ['Algorithm Name', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
            output = '<table border=1 align=center width=100%>'
            font = '<font size="" color="black">'
            output += "<tr>"
            for i in range(len(arr)):
                output += "<th>"+font+arr[i]+"</th>"
            output += "</tr>"
            for i in range(len(algorithms)):
                output +="<tr><td>"+font+str(algorithms[i])+"</td><td>"+font+str(accuracy[i])+"</td><td>"+font+str(precision[i])+"</td><td>"+font+str(recall[i])+"</td><td>"+font+str(fscore[i])+"</td></tr>"
            context= {'data': output}
            return render(request, 'UserScreen.html', context)

def Predict(request):
    if request.method == 'GET':
       return render(request, 'Predict.html', {})

def PredictAction(request):

    if request.method == 'POST':

        tds = float(request.POST.get('tds'))
        turbidity = float(request.POST.get('turbidity'))
        ph = float(request.POST.get('ph'))
        conductivity = float(request.POST.get('conductivity'))
        temperature = float(request.POST.get('temperature'))

        issues = []
        warnings = []
        recommendations = []

        # pH
        if ph < 6.5 or ph > 8.5:
            issues.append("pH is outside safe range (6.5–8.5)")
            recommendations.append("Adjust pH using neutralization treatment")

        # Turbidity
        if turbidity > 5:
            issues.append("Turbidity is too high (>5 NTU)")
            recommendations.append("Use sediment or sand filtration")

        # TDS
        if tds > 500:
            issues.append("TDS is too high (>500 mg/L)")
            recommendations.append("Use RO filtration to reduce dissolved solids")

        if tds < 50:
            warnings.append("TDS very low (<50 mg/L)")
            recommendations.append("Use mineral cartridge to add essential minerals")

        # Conductivity check
        if conductivity < 150 or conductivity > 500:
            issues.append("Conductivity outside safe range (150 - 500 µS/cm)")
            recommendations.append("Maintain balanced mineral content in water")

        # Temperature (warning only)
        if temperature < 10:
            warnings.append("Water temperature very low (<10°C)")
            recommendations.append("Allow water to reach room temperature")

        if temperature > 40:
            warnings.append("Water temperature high (>40°C)")
            recommendations.append("Store water in a cool place")

        result = ""

        if len(issues) == 0:
            result += "<h3 style='color:green;'>Water Quality: SAFE TO DRINK</h3>"
        else:
            result += "<h3 style='color:red;'>Water Quality: NOT SAFE TO DRINK</h3>"

        if issues:
            result += "<br><b>Issues Detected:</b><br>"
            for i in issues:
                result += "• " + i + "<br>"

        if warnings:
            result += "<br><b>Warnings:</b><br>"
            for w in warnings:
                result += "• " + w + "<br>"

        if recommendations:
            result += "<br><b>Recommendations:</b><br>"
            for r in recommendations:
                result += "• " + r + "<br>"

        context = {'data': result}

        return render(request, 'UserScreen.html', context)


def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})  

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Signup(request):
    if request.method == 'GET':
       return render(request, 'Signup.html', {})


def UserLoginAction(request):
    global uname
    if request.method == 'POST':
        username = request.POST.get('t1')
        password = request.POST.get('t2')

        for user in users_db:
            if user['username'] == username and user['password'] == password:
                uname = username
                context = {'data': 'Welcome ' + uname}
                return render(request, 'UserScreen.html', context)

        context = {'data': 'Login Failed. Please retry'}
        return render(request, 'UserLogin.html', context)       

def SignupAction(request):
    if request.method == 'POST':
        username = request.POST.get('t1')
        password = request.POST.get('t2')
        contact = request.POST.get('t3')
        gender = request.POST.get('t4')
        email = request.POST.get('t5')
        address = request.POST.get('t6')

        # Check if user already exists
        for user in users_db:
            if user['username'] == username:
                return render(request, 'Signup.html', {'data': username + " already exists"})

        # Add user to list
        users_db.append({
            'username': username,
            'password': password,
            'contact': contact,
            'gender': gender,
            'email': email,
            'address': address
        })

        return render(request, 'Signup.html', {'data': 'Signup Successful'})
      


