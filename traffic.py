import torch
import cv2

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
cap = cv2.VideoCapture('video.mp4')

if (cap.isOpened() == False):
    print("Error opening video stream or file")
    
width, height, channel = 0,0,0
line_pos = 0
threshold = 2
trafficCount = 0
relative_factor = 0.5
frameIndex = 0
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, img = cap.read()
    if ret == True and frameIndex%3 == 0:
        # resize img
        img = cv2.resize(img, (700,500))
        height, width, channel = img.shape
        line_pos = int(height * relative_factor)
        results = model(img)
        df = results.pandas().xyxy[0]
        img = cv2.line(img,(0,line_pos),(width,line_pos),(0,255,0),2)
        count = 0
        for i in range(len(df)):
            xmin = df.iloc[i,0]
            ymin = df.iloc[i,1]
            xmax = df.iloc[i,2]
            ymax = df.iloc[i,3]
            name = df.iloc[i,6]
            centroid_x,centroid_y = int((xmin+xmax)/2),int((ymin+ymax)/2)
            img = cv2.circle(img,(centroid_x,centroid_y),2,(0,0,255),2)
            if centroid_y < line_pos + threshold and centroid_y > line_pos - threshold:
                count+=1
        trafficCount = trafficCount + count
        print("TrafficCount",trafficCount)
        img = cv2.putText(img,"TrafficCount: "+str(count),(400,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),2)       
        cv2.imshow('img',img)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    # Break the loop
    else:
        break
    frameIndex = frameIndex + 1

cap.release()
cv2.destroyAllWindows()