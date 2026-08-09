# Recognition-of-ADL-s
<a href="https://unige.it/en/">
University Website
  
# GROUP MEMBERS

| Name | email  | profile |
| :--- | :---   | :--- |
| X ZAID | s4848329@studenti.unige.it | [Federista](https://github.com/federista)
| Syed Hani Hussain Kazmi | hanihussain94@hotmail.com| [Hani Kazmi](https://github.com/HaniKazmi94)
| Muhammad talha Siddiqui| engr.talhasiddiqui94@gmail.com | 
| Soundarya pallanti| bujjilaxmi2712@gmail.com | [Soundarya Pallanti](https://github.com/soundarya4807289)
| Vivek Vijaykumar Ingle| vivekvijay380@gmail.com | [Vivek Vijaykumar Ingle](https://github.com/vivek-v-ingle)  
 
 ## Motivation and Genral Objective
 Human Activity Recognition(HAR), and in particular the recognition of the Activities of Daily Living
(ADL), such as pouring, drinking, walking, teeth brushing, sitting down, or getting up, is valuable in
many settings, most notably in assistive scenarios, for instance in case of elderly and people with
special needs. Getting to know which ADL a person is (or is not) carrying out could dramatically
improve the support that robots (and/or digital assistants) can provide such a person with.
Furthermore, such knowledge could also be of the utmost importance for caregivers, with the aim
to keep fragile individuals living independently.

 ## PROJECT DESCRIPTION
 The goal of this repository is having a distributed architecture which allow user to accurately identify the activity of daily life based upon the provided dataset.
  
 ## AVAILABLE DATASET
  We have a dataset which contains data from 6 IMUs: they were placed on the volunteer’s back, left lower arm, left upper arm, right lower arm, right upper arm and right thigh.
Nine ADLs have been recorded: walking, sitting down, standing up, opening a door, closing a door, pouring water, drinking using a glass, brushing teeth and cleaning the table. 
Two distinct activity sequences have been designed and both have been performed twice using only the right hand and using only the left hand.

The first column of each IMU csv file contains a label “qags”, which indicates the type of the recorded data (i.e. quaternions, accelerometer data and gyroscope data).
The next column is the time-stamp in milliseconds elapsed from 00.00.0.000 AM (with a 30 milliseconds sampling time). The next four columns are the quaternions (with a resolution of 0.0001).The next three columns are the accelerations along the x-, y- and z-axis (with a resolution of 0.01 degrees per second).The last three columns refer to the angular velocities about the x,y and z axes (with a resolution of 0.1 mG).

Each annotations csv file’s first two columns are the current daytime in the format “hh.mm.ss.000” and in ms elapsed from 00:00:0.000 AM; the remaining columns are organized as couples where the first element represents the scope of the labelling (i.e. “BothArms”, “RightArm”, “LeftArm”, “Locomotion”) and the second indicates whether the labelled activity starts or ends.
Finally, the last two columns report as session ID.There are four different sessions characterised by the order in which the activities are performed, and by the used arm and whether the session starts or ends. The videos recorded during the experiments have been only used for labelling purposes, and they are not published.

[LINK to the dataset](https://data.mendeley.com/datasets/wjpbtgdyzm/1)
  
  ## ARCHITECTURE
  ![Image](https://github.com/federista/Recognition-of-ADL-s/blob/main/Architecture.png)
  
  Each module will use the trained SVMs with the available CSV files to classify the data belonging to 9 different classes, i.e. 9 ADLs. In this arhitecture, multiclass classifier has been used for each sensor module.The training of these SVMs will be done separately and for each individual sensor module. The modules will simply apply the classification function to the normalized feature vector. The information coming from every sensor module will be merged by a higher-level module. This module implements a majority voting mechanism too: the final classification of the whole system will be the mode of the 6 labels coming from the 6 sensor modules.


   ## STEPS 
  Following are the steps that will help user to experience the functionality of the cod

Step 1
  Install python3 using apt
 
``` 
    $ sudo apt update
    $ sudo apt install software-properties-common 
    $ sudo apt install python3.8
``` 
  
Step 2
  Install the following packages
  
  ```
  $ sudo apt install python-numpy
  $ sudo apt install python3-pandas
  $ pip install pickle-mixin
  $ pip install jsonlib-python3
  $ pip install -U scikit-learn
  ```
  
Step 3
  Install pycharm and git
 
 ```
   $ sudo snap install pycharm-professional --classic
   $ sudo apt install git
  ```
  
Step 4
  clone the repository into your local folder
  
  ```
  $ gitclone https://github.com/federista/Recognition-of-ADL-s.git
  ```
  
Step 5
  cd into your local repository and run the shell script "run.sh"
  
  ```
  $ cd Recognition-of-ADL-s
  $ ./run.sh
  ```
  
  
A series of python scripts will run sequentially afterwards and they are listed below
  
  __PrepareData.py__ :
  It will prepare the data into two sets. 70% Training data and 30% Testing data in order to later on test the accuracy of our system.
  
  __TrainModel.py__:
  It will create a total number of 6 models in respective to 6 IMUs
  
  __PrepareSimulationData.py:__
  Collectio of Simulation data for each individual IMU in order to test our system
  
  
  __Model_Prediction.py:__
  Predicted label for each IMU will be generated.Voting mechanism is approached to test the accuracy and generate a final label.
  
  ## OUTPUT
  ![Image](https://github.com/federista/Recognition-of-ADL-s/blob/main/output/Output.png)
  
  
  ### IMPORTANT:
  For the code to work properly kindly change the address path of each file after you clone the repository
  
  __VIDEO__
  
  
  [LABEL PREDICTION](https://drive.google.com/file/d/1u0cm6hPlsuJIQVU7WSOhOzDLFlDXI90p/view?usp=sharing)
  
  
   



  
  
  
  
  
