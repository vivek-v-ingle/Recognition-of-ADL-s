import pandas as pd
import os.path
from pathlib import Path
import os
import json
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pickle


class Model_Prediction():
    Y_labels_reverse = {0: 'Walk', 1: 'OpenDoor', 2: 'CloseDoor', 3: 'BrushTeeth', 4: 'SitDown', 5: 'DrinkGlass',
                        6: 'CleanTable', 7: 'StandUp', 8: 'PourWater'}

    IMU_list = ["lla", "lua", "rla", "rua", "rt", "back"]

    data_path_prefix = str(Path(__file__).resolve().parent / "simulationdata")

    model_path_prefix = str(Path(__file__).resolve().parent / "model")

    prediction_path_prefix = str(Path(__file__).resolve().parent / "IMU_Predication")

    output_path = str(Path(__file__).resolve().parent / "output")

    IMU = ""
    simulated_action_sequnce = ""

    def load_model(self):
        with open(os.path.join(self.model_path_prefix, f"{self.IMU}_pkl"), "rb") as m:
            model = pickle.load(m)

        return model

    def load_data(self):
        df = pd.read_csv(os.path.join(self.data_path_prefix, f"{self.IMU}_test_data.csv"))
        return df

    def load_simulation_seq(self):

        with open(Path(__file__).resolve().parent / 'ADL_Simulation_Sequence.json', 'r') as f:
            self.simulated_action_sequnce = json.load(f)

    def true_labels(self):
        y_true = []

        for ADL in self.simulated_action_sequnce.keys():
            temp_list = [ADL] * int(self.simulated_action_sequnce[ADL])
            y_true.extend(temp_list)
        return y_true



if __name__ == "__main__":

    obj = Model_Prediction()
    imu_pred_df = pd.DataFrame()
    for IMU in obj.IMU_list:
        obj.IMU = IMU
        df = obj.load_data()
        model = obj.load_model()
        y_pred = model.predict(df)
        y_pred = [obj.Y_labels_reverse[y] for y in y_pred]
        column_name = f"{obj.IMU}_y_pred"
        imu_pred_df[column_name] = y_pred
        # pred_df.to_csv(os.path.join(obj_imu.prediction_path_prefix, f"{obj_imu.IMU}_pred.csv"), index=False)

        print(f"Prediction stored for {obj.IMU} ")

    print(f"\nPrediction for IMUs are done\n")

    obj.load_simulation_seq()
    y_true = obj.true_labels()


    y_pred_voting = pd.DataFrame()


    imu_pred_df['majority'] = imu_pred_df.mode(axis=1)[0]
    print(f" voting output \n {imu_pred_df.head(5)}\n")
    y_pred = imu_pred_df['majority']

    accuracy_score = accuracy_score(y_true, y_pred)
    final_df = pd.DataFrame( list(zip(y_true, y_pred)), columns = ["True", "Predicted"])
    final_df.to_csv(os.path.join(obj.output_path, "final_output.csv"), index = False)
    # print(f"final output \n{final_df.head(50)}\n")
    print(f"\nAccuracy : {accuracy_score}\n")
    print(classification_report(y_true, y_pred))
