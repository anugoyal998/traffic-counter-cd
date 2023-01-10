import torch
import cv2
import pandas as pd

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
cap = cv2.VideoCapture('traffic.mp4')

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
    

# Read until video is completed

width, height, channel = 0,0,0
line_start = (0,0)
line_end = (0,0)

while (cap.isOpened()):
    # Capture frame-by-frame
    ret, img = cap.read()
    if ret == True:
        # resize img
        img = cv2.resize(img, (700,500))
        height, width, channel = img.shape
        print(width, height)
        line_start = (0,int(height * 0.5))
        line_end = (width, int(height * 0.5))
        img = cv2.line(img,line_start,line_end,(0,255,0),2)
        
        
        # results = model(img)
        # df = results.pandas().xyxy[0]
        # img = drawRect(img,df,True)
        cv2.imshow('img',img)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    # Break the loop
    else:
        break

cap.release()
cv2.destroyAllWindows()