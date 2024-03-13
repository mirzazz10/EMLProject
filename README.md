# EMLProject
EmbeddedMachineLearningProject

The Project is to train custom model for keypoint detection in exercises and then classify each exercise. The model is deployed on raspberry Pi board and the output is displayed in real time


Data is collected and prepared using LabelMg. The data is then trained to detect keypoints using the below git Link 

The same data is used to train a classifier different exercises. 

The model evaluation 
# add both the keypoint detection and classifier models 

Results 
# add the evaluation results 

Deployment on raspberry pi board 

1. Raspberry Pi won't have opencv installed and it needs to be installed for the project to work. We used this tutorial to install and build opencv: https://pimylifeup.com/raspberry-pi-opencv/ . This hould install opencv on the raspberry pi although it takes a lot of time. This is a crutial step for the project.

2. Connect the webcam to one of the usb ports on the Raspberry Pi.
3. Open the terminal and change the directory to the location where the script is saved.
4. After all the hardware is connected and the correct directory is opened, run the following command:
   ```
   python3 run_pose_estimation.py --modeldir tf_lite_model.tflite --output_path (your specific o/p path) --classmoddir tflite_ClassificationModel.tflite
   ```
   
