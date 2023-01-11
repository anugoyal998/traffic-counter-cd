import torch
import cv2
import pandas as pd

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
cap = cv2.VideoCapture('bridge.mp4')

if (cap.isOpened() == False):
    print("Error opening video stream or file")
    
    
def drawRect(img,df,text=False):
    for i in range(len(df)):
        xmin = df.iloc[i,0]
        ymin = df.iloc[i,1]
        xmax = df.iloc[i,2]
        ymax = df.iloc[i,3]
        name = df.iloc[i,6]
        # print(xmin, ymin, xmax,ymax, name)
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)),(255,0,0),2)
        if text:
            cv2.putText(img,name,(int(xmin),int(ymin)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
    return img

def helper(df,img,threshold):
    count = 0
    for i in range(len(df)):
        xmin = df.iloc[i,0]
        ymin = df.iloc[i,1]
        xmax = df.iloc[i,2]
        ymax = df.iloc[i,3]
        name = df.iloc[i,6]
        centroid_x,centroid_y = int((xmin+xmax)/2),int((ymin+ymax)/2)
        # print(centroid_y)
        if centroid_y <= threshold:
            count = count + 1
        cv2.circle(img, (centroid_x,centroid_y), 2, (0,0,255),2)
    return img, count
    

# Read until video is completed

width, height, channel = 0,0,0
line_start = (0,0)
line_end = (0,0)
threshold = 5
trafficCount = 0
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, img = cap.read()
    if ret == True:
        # resize img
        img = cv2.resize(img, (700,500))
        height, width, channel = img.shape
        line_start = (0,int(height * 0.5))
        line_end = (width, int(height * 0.5))
        img = cv2.line(img,line_start,line_end,(0,255,0),2)
        # crop image
        croppedImage = img[250:499,0:]
        results = model(croppedImage)
        df = results.pandas().xyxy[0]
        croppedImage, count = helper(df,croppedImage,threshold)
        trafficCount = trafficCount + count
        croppedImage = cv2.line(croppedImage,(0,threshold),(width,threshold),(0,0,255),2)
        print("trafficCount",trafficCount)
        cv2.imshow('img',croppedImage)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    # Break the loop
    else:
        break

cap.release()
cv2.destroyAllWindows()