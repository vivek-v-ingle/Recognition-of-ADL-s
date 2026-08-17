import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Model_Train():

    #path for raw data
    data_prefix = str(Path(__file__).resolve().parent / "data")



    #path to store output with output folder on disk
    output_path_prefix = str(Path(__file__).resolve().parent / "output")

    # name of data folders for volunteers
    volunteers_data_paths = ["volunteer_01", "volunteer_02", "volunteer_03", "volunteer_04", "volunteer_05",
                             "volunteer_06", "volunteer_07",
                             "volunteer_08", "volunteer_09", "volunteer_10"]

    #dictionry for each IMU and its related data files
    annotation_col_dict = {
        "BothArmsLabel": ["lla", "lua", "rua", "rla"],
        "RightArmLabel": ["rua", "rla"],
        "LeftArmLabel": ["lla", "lua"],
        "Locomotion": ["back", "rt"]

    }

    #read annotation file
    def read_annotation_data(self,data_path ):
        annotation_df = pd.read_csv(os.path.join(data_path, "annotations.CSV"))
        return annotation_df

    #prepares data using annotation files
    def prepare_data(self, annotation_df, data_path):

        #columns names of output dataset
        volunteer_data_df = pd.DataFrame(
            columns=["dataType", "Time", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "ADL", "module"])

        for anno_col in self.annotation_col_dict.keys():
            data_files = self.annotation_col_dict[anno_col]

            for data_file in data_files:

                data_df = pd.DataFrame(
                    columns=["dataType", "Time", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "ADL", "module"])

                print(f"Data File {data_file}")

                sensorData_df = pd.read_csv(os.path.join(data_path, "IMUs", f"{data_file}.csv"),
                                            names=["dataType", "Time", "0", "1", "2", "3", "4", "5", "6", "7", "8",
                                                   "9"])
                row_no = 0

                while row_no <= (annotation_df.shape[0] - 1):

                    #                         print(f"Row no {row_no}")
                    if annotation_df[anno_col].iloc[row_no] is not np.nan:

                        ADL = annotation_df[anno_col].iloc[row_no]
                        startTime = annotation_df["Time [msec]"].iloc[row_no]
                        endTime = annotation_df["Time [msec]"].iloc[row_no + 1]

                        temp_df = sensorData_df[
                            (sensorData_df["Time"] >= startTime) & (sensorData_df["Time"] <= endTime)].copy()
                        temp_df["ADL"] = [ADL] * temp_df.shape[0]
                        temp_df["module"] = [data_file] * temp_df.shape[0]

                        data_df = pd.concat([data_df, temp_df], ignore_index=True)
                        row_no = row_no + 2

                    else:

                        row_no = row_no + 1

                volunteer_data_df = pd.concat([volunteer_data_df, data_df], ignore_index=True)

        return volunteer_data_df

    #split in test and train in stratify way
    def split_test_train(self, final_df):
        train, test = train_test_split(final_df, test_size=0.3, stratify= final_df["ADL"], shuffle=True, random_state = 0)
        return train, test


if __name__ == "__main__":

    final_data_df = pd.DataFrame(
        columns=[
            "dataType",
            "Time",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "ADL",
            "module",
        ]
    )

    obj = Model_Train()

    for volunteer_data_path in obj.volunteers_data_paths:
        print(f"Data preparating for {volunteer_data_path}")

        data_path = os.path.join(obj.data_prefix, volunteer_data_path)
        annotation_df = obj.read_annotation_data(data_path)

        final_df = pd.concat(
            [
                final_data_df,
                obj.prepare_data(annotation_df, data_path),
            ],
            ignore_index=True,
        )

    final_df.drop(
        labels=["dataType", "Time"],
        axis=1,
        inplace=True,
    )

    train, test = obj.split_test_train(final_df)

    train.to_csv(
        os.path.join(obj.output_path_prefix, "train_data.csv"),
        index=False,
    )

    test = test.reset_index()

    test.to_csv(
        os.path.join(obj.output_path_prefix, "test_data.csv"),
        index=False,
    )