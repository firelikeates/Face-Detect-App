import cv2

trained_data = cv2.CascadeClassifier("data.xml")

webcam = cv2.VideoCapture(0)

while True:

    successful_frame_read,frame = webcam.read()

    grayscaled = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    face_cor = trained_data.detectMultiScale(grayscaled)

    for (x,y,w,h) in face_cor:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),10)
        resized = cv2.resize(frame,(1000,750),interpolation=cv2.INTER_AREA)
        cv2.imshow("Webcam detect faces",frame)

    key = cv2.waitKey(1)

    if(key==81 or key==113):
        break
