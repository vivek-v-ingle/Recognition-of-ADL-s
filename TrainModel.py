import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.svm import SVC
import pickle
from sklearn.preprocessing import StandardScaler


class Train_Model():

    #path to prepared data
    data_path_prefix = str(Path(__file__).resolve().parent / "output")
    model_path_prefix = str(Path(__file__).resolve().parent / "model")
    IMU_list = ["lla", "lua", "rla", "rua", "rt", "back"]
    #name of IMU
    IMU = ""

    # dict to convert output labels to numeric values
    Y_labels = {'Walk': 0, 'OpenDoor': 1, 'CloseDoor': 2, 'BrushTeeth': 3, 'SitDown': 4,
               'DrinkGlass': 5, 'CleanTable': 6, 'StandUp': 7, 'PourWater': 8}

    #dict to reverse output labels
    Y_labels_reverse = {0: 'Walk', 1: 'OpenDoor', 2: 'CloseDoor', 3: 'BrushTeeth', 4: 'SitDown', 5: 'DrinkGlass',
                       6: 'CleanTable', 7: 'StandUp', 8: 'PourWater'}

    #reads train data and filter it for given IMU
    def read_data(self):
        train_data = pd.read_csv(os.path.join(self.data_path_prefix, "train_data.csv"))
        train_df = train_data[train_data["module"] == self.IMU].copy()
        train_df.drop( labels = ["module"], axis =  1, inplace=True)
        X = train_df.drop(labels = "ADL",  axis = 1)
        y = train_df["ADL"]

        return X, y

    #standardize data
    def preprocess_data(self, X, y):

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X)
        y_train = y.apply(lambda x: self.Y_labels[x])
        X_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
        return X_scaled, y_train

    #train model
    def train_model(self, X_train_scaled, y_train):
        selected_params = {'C': 100, "kernel": "rbf", "gamma": 0.1, 'random_state': 0, 'class_weight': 'balanced'}
        svm_model = SVC()
        svm_model.set_params(**selected_params)
        svm_model.fit(X_train_scaled, y_train)
        return svm_model


# driver code to load, preprocess data, train and store model
if __name__ == "__main__":


    obj = Train_Model()

    for IMU in obj.IMU_list:
        obj.IMU  = IMU
        X, y = obj.read_data()
        X_train_scaled, y_train = obj.preprocess_data(X, y)
        model = obj.train_model(X_train_scaled, y_train)
        Model_name = f'{obj.IMU}_pkl'
        Model_path = os.path.join(obj.model_path_prefix, Model_name)
        pickle.dump(model, open(Model_path, 'wb'))
        print(f"Model Stored at {Model_path}")
