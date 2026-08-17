import json
import os
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


class PrepareSimulationData:

    IMU = ""

    PROJECT_ROOT = Path(__file__).resolve().parent
    data_path_prefix = str(PROJECT_ROOT / "output")
    output_data_prefix = str(PROJECT_ROOT / "simulationdata")

    simulated_action_sequnce = ""
    IMU_list = ["lla", "lua", "rla", "rua", "rt", "back"]

    def load_simulation_seq(self):
        sequence_path = self.PROJECT_ROOT / "ADL_Simulation_Sequence.json"

        with open(sequence_path, "r") as f:
            self.simulated_action_sequnce = json.load(f)

    def read_data(self):

        df = pd.read_csv(
            os.path.join(self.data_path_prefix, "test_data.csv")
        )

        df = df[df["module"] == self.IMU].copy()
        df.drop(labels=["index", "module"], axis=1, inplace=True)

        return df

    def prepare_simulated_data(self, df):

        data_frames = []

        for ADL in self.simulated_action_sequnce.keys():

            imu_df = df[df["ADL"] == ADL]
            imu_df = imu_df.iloc[
                : self.simulated_action_sequnce[ADL]
            ].copy()

            required_rows = int(self.simulated_action_sequnce[ADL])

            if len(imu_df) != required_rows:

                remaining_rows = required_rows - len(imu_df)

                garbage_data = [[0] * 11] * remaining_rows

                temp_df = pd.DataFrame(
                    garbage_data,
                    columns=imu_df.columns,
                )

                imu_df = pd.concat(
                    [imu_df, temp_df],
                    ignore_index=True,
                )

            data_frames.append(imu_df)

        simulation_df = pd.concat(
            data_frames,
            ignore_index=True,
        )

        simulation_df.drop(
            labels=["ADL"],
            axis=1,
            inplace=True,
        )

        return simulation_df

    def preprocess_data(self, X):

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X)

        X_scaled = pd.DataFrame(
            X_train_scaled,
            columns=X.columns,
        )

        return X_scaled


if __name__ == "__main__":

    obj = PrepareSimulationData()
    obj.load_simulation_seq()

    Path(obj.output_data_prefix).mkdir(
        parents=True,
        exist_ok=True,
    )

    for IMU in obj.IMU_list:

        print(IMU)

        obj.IMU = IMU

        df = obj.read_data()

        test_df = obj.prepare_simulated_data(df)

        test_df = obj.preprocess_data(test_df)

        test_df.to_csv(
            os.path.join(
                obj.output_data_prefix,
                f"{obj.IMU}_test_data.csv",
            ),
            index=False,
        )

        print(
            f"Simulation data has been stored for IMU {obj.IMU}"
        )
