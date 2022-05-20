#Importing necessary libraries
import cv2 #For accessing Camera
import numpy as np #Array
import os #To create files
from matplotlib import pyplot as plt #To plot
import mediapipe as mp #To draw hand coordinates
from keras.models import load_model #Used to load the saved model
from sklearn.model_selection import train_test_split #Split data into train and testing
from tensorflow.keras.utils import to_categorical #Labelling into categories
from tensorflow.keras.models import Sequential # Used in model making
from tensorflow.keras.layers import LSTM, Dense # Used in model making 
from tensorflow.keras.callbacks import TensorBoard #To visualize the graphs for the model
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score #Check Accuracy of our model
#Functions to detect, plot and draw coordinates for Left and Right Hands
#Referred documentation : https://google.github.io/mediapipe/solutions/holistic.html

#Build Keypoints
mp_holistic = mp.solutions.holistic #get hand coordinates
mp_drawing = mp.solutions.drawing_utils #draw those coordinates

#endianness is the order or sequence of bytes of a word of digital data in computer memory
#OpenCV uses BGR colours whereas Mediapipe uses RGB Colours for Prediction of Hand Coordinates
def mp_detect(image, model): #Make Colour Conversion, make image non-writable, predict, again back to image writable and finally colour conversion back to orginal
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False                 
    results = model.process(image)
    image.flags.writeable = True     
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results

def draw_landmarks(image, results): #Drawing Landmarks and Hand Connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    
def draw_styled_landmarks(image, results): #Style Landmarks and Connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))             
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
    
def Hand_points(results): #Extract Keypoints 
    #Getting the coordinates in x,y and z direction and using flatten to convert array into single dimension
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

#This Cell is for deciding or defining the parameters
DATA_PATH = os.path.join('Final_Submission_2_50') #Create a folder

#Array of Sign Detections Symbols used
actions = np.array(['Hello','Okay','Peace','Good','Bad','You','Me','Cancel', 'Break'])
no_sequences = 50 #Number of videos, so frames = 50*30 = 1500
sequence_length = 30 #Number of frames in a video

for i in actions: #Iterate through number of actions 
    for j in range(no_sequences): #Iterate through number of videos
        try: 
            os.makedirs(os.path.join(DATA_PATH, i, str(j))) #Store them in folders
        except: 
            #print('Error creating folder in ', i, j, 'iteration') #Check whether the folder already exists, if yes then pass on
            pass 
        
#Label the actions starting from 0 to n
label_map = {label:num for num, label in enumerate(actions)}
#print(label_map) #Print the label 

#This Cell is for Appending or storing all the data in a single array
sqs, lbls = [], [] #Create empty list 
for i in actions: #Iterate thorugh actions
    for j in range(no_sequences): #Iterate through no. of videos
        temp = [] #Empty List
        for k in range(sequence_length): #Iterate through frames
                result = np.load(os.path.join(DATA_PATH, i, str(j), "{}.npy".format(k))) #Access the data saved on the path
                temp.append(result) #Save frames of an action 1 by 1
        sqs.append(temp) #Save those frames in a list 
        lbls.append(label_map[i]) #Save the list of an action to label list, for eg. hello, ty, etc.
        
#7 
model = load_model('Final_Submission_2_50.h5') #Call our model
colors = [(170,60,150), (117,245,16), (19,69,139), (255,0,0), (255,0,255), (16,117,245), (0,215,255), (0,0,255),  (185,218,255)] #BGR Colour Palletes for the Predicted Word

def probability_colours(res, actions, input_frame, colors): #Function to check the probability of the predicted word and increase/dexrease the size of the bar
    op = input_frame.copy() #Copies the input_frame dimensions in the output frame
    for i, j in enumerate(res): #Iterate through the predicted result
        cv2.rectangle(op, (0,60+i*40), (int(j*100), 90+i*40), colors[i], -1) #Increase or decrease the bar size according to the probability of prediction of the class
        cv2.putText(op, actions[i], (0, 85+i*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA) #Put the text on the left side
    return op #Return the output frame

sequence = [] #Empty List for Sequence - Number of frames in a video
sentence = [] #Empty List for all the Sentences
threshold = 0.9 #Threshold value for Prediction Result
cam = cv2.VideoCapture(0) #Turn on the Camera
with mp_holistic.Holistic(min_detection_confidence=0.9, min_tracking_confidence=0.9) as holistic: #Tracking 
    while cam.isOpened(): #Camera is opened
        ret, frame = cam.read() #Read frame by frame
        image, results = mp_detect(frame, holistic) #Processing the image by calling the function
        print(results) #Printing results for log
        draw_styled_landmarks(image, results) #Calling function
        keypoints = Hand_points(results) #Drawing handpoints
        sequence.append(keypoints) #Add the Hand_point result in the sequence
        sequence = sequence[-30:] #Taking the last 30 frames of the sequence
        if len(sequence) == 30: #Taken frames = 30 to start prediction
            res = model.predict(np.expand_dims(sequence, axis=0))[0] #Predicting the Class with the Gesture
            if res[np.argmax(res)] > threshold: #Checking if the result is above the threshold value
                if len(sentence) > 0: #See if sentence length is greater than 0
                    if actions[np.argmax(res)] != sentence[-1]: #Checking if the Current Action is not the previous one
                        sentence.append(actions[np.argmax(res)]) #If not so, then append it
                else: #If not then directly append the action result to the sentence
                    sentence.append(actions[np.argmax(res)]) #Appending the Result         
            if len(sentence) > 5: #If the number of Sentences
                sentence = sentence[-5:] #Move them back to display the new sentences
            image = probability_colours(res, actions, image, colors) #Call the above function to print the length of the colours wrt to the probability of the predicted hand sign
        cv2.rectangle(image, (0,0), (640, 40), (0, 0, 0), -1) #Black Row for continuos sentences
        cv2.putText(image, ' '.join(sentence), (5,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) #Display those sentences       
        cv2.imshow('HAND GESTURE RECOGNITION PORTAL', image) #Display the Name for CV2 feed
        if cv2.waitKey(10) & 0xFF == ord('q'): #Press 'q' to terminate the Camera
            break    
    cam.release() #Turns off the camera
    cv2.destroyAllWindows() #Destroys all the Windows
