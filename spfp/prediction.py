import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
perf = { 0: "Excellent (CGPA 3.50 - 4.00)", 3: "Very Good (CGPA 3.00 - 3.49)", 
    1: "Good (CGPA 2.50 - 2.99)", 2: "Satisfactory (CGPA 2.00 - 2.49)", 4: "Fail (CGPA < 2.00)"}

gpa_perf = { 0: "3.50 - 4.00", 3: "2.00 - 2.49", 
    1: "3.00 - 3.49", 2: "2.50 - 2.99", 4: "0.00 - 2.00"}

field = {1: "Computer Science and Engineering", 3: "Civil Engineering", 4: "Electronics and Communication Engineering", 
        8: "Water Resource Engineering", 7: "Mechanical Engineering", 5: "Electrical Power and Control Engineering", 
        0: "Architecture", 2: "Chemical Engineering", 6: "Material Science Engineering"}
def performance_predict(sample):
    X_df = pd.read_csv('/media/andalus/My Passport/academic/projects/django_projects/Prediction/dataset/cse_performance_features.csv')
    y_df = pd.read_csv('/media/andalus/My Passport/academic/projects/django_projects/Prediction/dataset/cse_performance_labels.csv')

    svc_obj = SVC()
    X = scale(X_df) 
    y = y_df
    X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_df, train_size=0.75, test_size=0.24, random_state=21, shuffle=True)

    svc_obj.fit(X_train,y_train)

    #sample = np.array([1, 1, 1, 2, 5, 5, 3, 3, 1, 1, 1,2, 5, 2,2,0,0, 0, 0,0,3,3,3,3,1,0])
    sample = np.array(sample)
    data = pd.DataFrame(sample.reshape(-1, len(sample)),columns=X_df.columns)
    result = svc_obj.predict(data)
    res = result[0]
    return perf[res]

def field_predict(sample):
    pd.set_option('display.max_columns', None)
    X_df = pd.read_csv('/media/andalus/My Passport/academic/projects/django_projects/Prediction/dataset/field_prediction_features.csv')
    y_df = pd.read_csv('/media/andalus/My Passport/academic/projects/django_projects/Prediction/dataset/field_prediction_labels.csv')

    knn = KNeighborsClassifier()
    X = scale(X_df) 
    y = y_df
    X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_df, train_size=0.79, test_size=0.21, random_state=2, shuffle=True)

    knn.fit(X_train,y_train)

    #sample = np.array([1, 1, 1, 2, 5, 5, 3, 3, 1, 1, 1,2, 5, 2,2,0,0, 0, 0,0,3,3,3,3,1,0])
    sample = np.array(sample)
    data = pd.DataFrame(sample.reshape(-1, len(sample)),columns=X_df.columns)
    result = knn.predict(data)
    res = result[0]
    return field[res]

def gpa_predict(sample):
    pd.set_option('display.max_columns', None)
    X_df = pd.read_csv('/media/andalus/My Passport/academic/projects/django_projects/Prediction/dataset/gpa_prediction_features.csv')
    y_df = pd.read_csv('/media/andalus/My Passport/academic/projects/django_projects/Prediction/dataset/gpa_prediction_labels.csv')
    del y_df['second_gpa']
    svc_obj = SVC()
    X = scale(X_df) 
    y = y_df
    X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_df, train_size=0.75, test_size=0.24, random_state=98, shuffle=True)

    svc_obj.fit(X_train,y_train)

    sample = np.array(sample)
    data = pd.DataFrame(sample.reshape(-1, len(sample)),columns=X_df.columns)
    result = svc_obj.predict(data)
    res = result[0]
    return gpa_perf[res]
