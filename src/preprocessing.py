import pandas as pd
import numpy as np
from IPython import display


def preprocessing_01(df_data, is_train = True, is_debug = True, **kwargs):
    df_output = pd.DataFrame()

    # Sex: gioi tinh
    cls_sex = {'female': 0, 'male' : 1}
    df_output["Sex"] = df_data["Sex"].apply(lambda x: cls_sex[x])
    # Age: median
    df_output["Age"] = df_data["Age"].fillna(df_data["Age"].median())
    # Fare
    df_output["Fare"] = df_data["Fare"].fillna(df_data["Fare"].median())
    # Pclass, SibSp, Parch
    for name in ['Pclass', 'SibSp', 'Parch']:
        df_output[name] = df_data[name]
    # Cabin
    cls_cabin = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'T':8, 'Z':0}
    df_output["Cabin"] = df_data['Cabin'].apply(lambda x: cls_cabin['Z'] if pd.isna(x) else cls_cabin[x[0]])
    # Embarked
    cls_embarked = {'0': 0, 'C':1, 'Q':2, 'S':3}
    df_output["Embarked"] =  df_data['Embarked'].apply(lambda x: cls_embarked['0'] if pd.isna(x) else cls_embarked[x])
    # Surname
    surnames = ['Capt.', 'Col.', 'Don.', 'Dr.', 'Jonkheer.', 'Lady.', 'Major.', 
            'Master.', 'Miss.', 'Mlle.', 'Mme.', 'Mr.', 'Mrs.', 'Ms.', 'Rev.', 'Sir.', 'the', 'Dona.']
    cls_surnames = dict(zip(surnames, range(len(surnames))))
    df_output["Surname"] = df_data['Name'].apply(lambda x: cls_surnames[x.split(',')[1].split(' ')[1]])

    if is_train:
        df_output["Survived"] = df_data["Survived"]

    # display.display(df_output)

    if is_debug:
        print('-'*10, 'HEAD', '-'*10)
        display.display(df_data.head())
        print('-'*10, 'TAIL', '-'*10)
        display.display(df_data.tail(5))
        print('-'*10, 'ISNA', '-'*10)
        display.display(df_data.isna().sum())
        # Sex: gioi tinh
        print('-'*10, 'SEX', '-'*10)
        display.display(np.unique(df_data['Sex'], return_counts=True))
        # Age: lay median
        print('-'*10, 'AGE', '-'*10)
        print(f'Age IsNa: {df_data["Age"].isna().sum()}')
        print(f"Age Median: {df_data['Age'].median()}")
        # Fare
        print('-'*10, 'FARE', '-'*10)
        display.display(df_data["Fare"].describe())
        # Cabin
        print('-'*10, 'CABIN', '-'*10)
        display.display(np.unique(df_data['Cabin'].apply(
            lambda x: 'Z0' if pd.isna(x) else x), return_counts=True))
        # Embarked
        print('-'*10, 'EMBARKED', '-'*10)
        display.display(
            np.unique(df_data['Embarked'].apply(lambda x: '0' if pd.isna(x) else x), return_counts=True)
        )
        globals().update(**locals())
    
    return df_output