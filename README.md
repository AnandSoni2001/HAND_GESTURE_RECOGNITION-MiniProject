# HAND_GESTURE_RECOGNITION-MiniProject
Using OpenCV, Mediapipe and LSTM Model to detect Hand Signs

Some necessary installations :
!pip install tensorflow==2.5.1 tensorflow-gpu==2.5.1 opencv-python mediapipe sklearn matplotlib

Steps Involved according to the Cell:
1. Import the necessary libraries
2. Define the functions of Mediapipe for drawing landmarks and detect left, right hand coordinates
3. Creating folder for individual actions
4. Collecting the Data using OpenCV and label them according to actions
5. Appending the sequences and label of all the data in a single list
6. Splitting the data into train and test
7. Create a sequential model and train it
8. Checking for the results of predicted value and test value whether they are same
9. Saving the model
10. Print the Confusion Matrix, Heatmap and Accuracy of the Model
11. Realtime Testing
