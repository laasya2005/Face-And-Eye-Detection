import cv2
 

faceCascade = cv2.CascadeClassifier("C:/Users/home/OneDrive/Desktop/pproject/Face-Detection-in-Python-using-OpenCV-master/data/haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("C:/Users/home/OneDrive/Desktop/pproject/Face-Detection-in-Python-using-OpenCV-master/data/haarcascades/haarcascade_eye.xml")

img = cv2.imread('data/baby1.png')
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(imgGray,1.2,5);


for (x,y,w,h) in faces:
    cv2.rectangle(img,(x, y), (x+w, y+h), (255,0, 0), 2)

    roi_gray = imgGray[y:y+h,x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)

    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)



cv2.imshow("output",img)

cv2.waitKey(0) 

