# -*- coding: utf-8 -*-

#Import required modules
import cv2
import dlib
import math
#import numpy as np 
 
#Defining points representing the facial features
REB = [17,18,19,20] #Right Eyebrow
LEB = [22,23,24,25] #Left Eyebrow
REYE = [36,37,38,39,40] #Right Eye
REYE_UP= [36,37,38]
REYE_DOWN = [39,40,41]
REYE_UP_DOWN = [37,40]
LEYE = [42,43,44,45,46] #Left Eye
LEYE_UP =[42,43,44]
LEYE_UP_DOWN = [43,46]
LEYE_DOWN=[45,46,47]
NSE_VER = [27,28,29] #Vertical Nose
NSE_HOR = [31,32,33,34] # Horizontol Nose
INR_LIPS = [60,61,62,63,64,65,66] #Inner Lips
OUT_LIPS = [48,49,50,51,52,53,54,55,56,57,58] #Outer Lips

def distance(pnt_1 , pnt_2):    
    dis = math.sqrt((shape.part(pnt_2).x- shape.part(pnt_1).x)**2 - (shape.part(pnt_2).y- shape.part(pnt_1).y)**2 )
    print(dis)
    return dis

distance(19,37)

#Set up some required objects

video_capture = cv2.VideoCapture(0) #Webcam object
detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever you named the downloaded file
while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(7,7),0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)
    font = cv2.FONT_HERSHEY_SIMPLEX
   # thresh = cv2.threshold(frame,127,255,0)
    #contours =  cv2.findContours(thresh, 1, 2)
    #cnt = contours[0]
    #ellipse = cv2.fitEllipse(cnt)
    detections = detector(clahe_image, 1) #Detect the faces in the image
    for k,d in enumerate(detections): #For each detected face
        shape = predictor(clahe_image, d) #Get coordinates
        """for i in range(1,68): #There are 68 landmark points on each face
            cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=1) #For each point, draw a red circle with thickness2 on the original frame
        for i in REB:
            cv2.line(frame,(shape.part(i).x, shape.part(i).y),(shape.part(i+1).x, shape.part(i+1).y),(255,0,0),2)
        for i in LEB:
             cv2.line(frame,(shape.part(i).x, shape.part(i).y),(shape.part(i+1).x, shape.part(i+1).y),(255,0,0),2)
        for i in NSE_VER:
             cv2.line(frame,(shape.part(i).x, shape.part(i).y),(shape.part(i+1).x, shape.part(i+1).y),(255,0,0),2)
        for i in NSE_HOR:
             cv2.line(frame,(shape.part(i).x, shape.part(i).y),(shape.part(i+1).x, shape.part(i+1).y),(255,0,0),2)
             
        for i in REYE:
           A1 =  cv2.line(frame,(shape.part(i).x, shape.part(i).y),(shape.part(i+1).x, shape.part(i+1).y),(255,0,0),2)
            
        for i in LEYE:
            A2 = cv2.line(frame,(shape.part(i).x, shape.part(i).y),(shape.part(i+1).x, shape.part(i+1).y),(255,0,0),2)
            
        for i in INR_LIPS:
           A3 =  cv2.line(frame,(shape.part(i).x, shape.part(i).y),(shape.part(i+1).x, shape.part(i+1).y),(255,0,0),2)
            
        for i in OUT_LIPS:
           A4 =  cv2.line(frame,(shape.part(i).x, shape.part(i).y),(shape.part(i+1).x, shape.part(i+1).y),(255,0,0),2)
     #      cv2.ellipse(frame,ellipse,(0,255,0),2)
        for i in range(1,68):
            LEYE_DIST = ((shape.part(43).x - ((shape.part(46).x)))**2 + ((shape.part(43).y)-(shape.part(46).y))**2)**.5 
            REYE_DIST =  (((shape.part(37).x) - ((shape.part(40).x)))**2 + ((shape.part(37).y)-(shape.part(40).y))**2)**.5 
        print ("left eye up down = ",LEYE_DIST)
        print ("Right eye up down = ",REYE_DIST)
        if REYE_DIST>15:
            print("Happy")"""
        
        # Calculate average distance between right brows and  right eye        
        dist_1937 = distance(19,37)
        dist_2038 = distance(20,38)
        FP1 = (dist_1937 + dist_2038) / 2
        
        # Calculate average distance between left brows and left eye
        dist_2343 = distance(23,43)
        dist_2444 = distance(24,44)
        FP2 = (dist_2343 + dist_2444) / 2
               
        FP3 = distance(21,39)  # Calculate distance between left corner point of right eye and brows
        FP4 = distance(17,36)  # Calculate distance between right corner point of right eye and brows
        FP5 = distance(22,42)  # Calculate distance between right corner point of left eye and brows
        FP6 = distance(26,45)  # Calculate distance between left corner point of left eye and brows
        FP7 = distance(21,22) # Calculate distance between corner point of two eyes
        
        FP8 = distance(27,31) # Calculate distance between upper nose point and right most point of lower nose
        FP9 = distance(27,35) # Calculate distance between upper nose point and left most point of lower nose
        FP10 = (FP8 + FP9) / 2 
        FP11 = distance(30,33) # Calculate distance between lower centre nose point and upper centre nose point.
        dist_3150 = distance(31,50) #calculate distance between nose right corner and right corner of upper lips
        dist_3552 = distance(35,52) #calculate distance between nose left corner and left corner of upper lips
        FP12 = (dist_3150 + dist_3552) /2 
        dist_3148 = distance(31,48)
        dist_3554 = distance(35,54)
        FP12 = (dist_3148+ dist_3554) /2
        FP13 = distance(48,54)
        FP14 = distance(61,67) #  calculate distance between right corner of inner lips
        FP15 = distance(62,66)  #  calculate distance between left corner of inner lips
        FP16 = distance(63,65) #  calculate distance between middle of inner lips
        FP17 = distance(58,7) # calculate distance between right corner of lower lips and right chin
        FP18 = distance(57,8) # calculate distance between middle of lower lips and middlechin
        FP19 = distance(56,9) # calculate distance between left corner of lower lips and left chin 
        FP20 = distance(60,64) # calculate distance between inner lips corner
            
            
    cv2.imshow("Facial Sentiment Analysis", frame) #Display the frame
    if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
        break

