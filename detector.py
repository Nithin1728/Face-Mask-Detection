from tkinter import *
import cv2
from PIL import Image,ImageTk
from tkinter import messagebox
import threading
import os

class DetectionView:

    stop = False
    def load(self):

        window = Tk()
        window.title("FACE MASK DETECTION")
        frame = Frame(window,padx=20,pady=20,bg="#B57EDC")
        frame.grid(row=0,column=0,padx=10,pady=10)

        self.l1 = Label(frame)
        self.l1.grid(row=1,column=0,columns=3)

        b1 = Button(frame,text="start",command= self.startCamera,pady=20)
        b1.grid(row=2,column=0,sticky='nesw')

        b2 = Button(frame, text="stop",command=self.stopCamera,pady=20)
        b2.grid(row=2, column=1,sticky='nesw')

        b3 = Button(frame, text="capture", command=self.saveImage, pady=20)
        b3.grid(row=2, column=2, sticky='nesw')

        self.l2 = Label(frame,text='STATUS - Camera Started',font=("Courier", 35),padx=10,pady=10)
        self.l2.grid(row=3,column=0,columns=3,sticky='nesw',pady=(20,0))


        self.startCamera()

        window.mainloop()

    def startCamera(self):
        self.stop = False

        self.cascade = cv2.CascadeClassifier('nose.xml')
        self.cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        t = threading.Thread(target= self.webcam, args=())
        t.start()

    def webcam(self):
        try:
            ret, image_frame = self.cap.read()
            image_frame = cv2.resize(image_frame, None, fx=1.2, fy=1.0, interpolation=cv2.INTER_AREA)
            self.img = Image.fromarray(image_frame)

            colorimage = cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB)
            grayimage = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

            # core functionality - Face detection
            self.faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
            self.faces = self.faceCascade.detectMultiScale(
            grayimage,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
            )
            for (x, y, w, h) in self.faces:
                cv2.putText(colorimage,"face", (x, y - 10), cv2.FONT_HERSHEY_DUPLEX,0.5, (0,255,255), 1)
                cv2.rectangle(colorimage, (x, y), (x+w, y+h), (0,255,255), 2)
            r = self.cascade.detectMultiScale(grayimage,1.7,11)

            if len(r) != 0:
                for (x,y,w,h) in r:
                    cv2.rectangle(colorimage,(x,y),(x+w,y+h),(0,255,0),3)
                    self.l2.config(text="Face not covered with mask")
            else:
                self.l2.config(text="Face is covered with mask")

            self.img = Image.fromarray(colorimage)
            img = ImageTk.PhotoImage(self.img)
            self.l1.configure(image=img)
            self.l1.image = img

            if self.stop == False:
                self.l1.after(10, self.webcam)
            else:
                self.l1.image = None
        except:
            print("Some error")

    def saveImage(self):
        if self.stop != True:
            try:
                self.img.save('images/1.jpg')
                messagebox.showinfo('Alert', "Image saved")
            except:
                messagebox.showinfo('Alert',"Unable to save image")

        else:
            self.l2.config(text="Enable the camera to capture")

    def stopCamera(self):
        self.stop = True
        self.l2.config(text="STATUS-Camera Stopped")

dv=DetectionView()
dv.load()