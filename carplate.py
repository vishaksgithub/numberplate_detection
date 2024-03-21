import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd=r"c:\Program Files\Tesseract-OCR\tesseract.exe"

num_plate_cascade=cv2.CascadeClassifier(r"D:\DL projects\car_numberplate\haarcascade_russian_plate_number.xml")
num_plate_cascade

def extract_numbers(img_filename):
    img=cv2.imread(img_filename)
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    nplate=num_plate_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
    #width height and channel
    for (x,y,w,h) in nplate:
        wT,hT,cT=img.shape
        a,b=(int(0.02*wT),int(0.02*hT))
        plate=img[y+a:y+h-a,x+b:x+w-b,:]
        #kernel is a black window
        kernel=np.ones((1,1),np.uint8)
        plate=cv2.dilate(plate,kernel,iterations=1)
        plate=cv2.erode(plate,kernel,iterations=1)
        gray_plate=cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
        (thresh,plate)=cv2.threshold(gray_plate,127,255,cv2.THRESH_BINARY)
        #read the text on the plate
        read=pytesseract.image_to_string(plate)
        print(read)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(img,read,(x,y-10),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.9,color=(255,255,255),thickness=2)
        

        cv2.imshow("plate",plate)




    cv2.imshow('Result',img)
    cv2.waitKey()
    cv2.destroyAllWindows()

extract_numbers("carplate.jpg")