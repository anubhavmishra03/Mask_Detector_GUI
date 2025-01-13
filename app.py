from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import numpy as np
from keras.models import load_model
import threading
from pygame import mixer


def voice_alarm():
    mixer.init()
    mixer.music.load("file.wav")
    mixer.music.play()

def openImageWindow():
    # Setting Window Close
    allWindowClose = "Image"
    
    # Disabling all Button
    image['state'] = 'disabled'
    video['state'] = 'disabled'
    live['state'] = 'disabled'
    
    # File Dialog
    root.filename = filedialog.askopenfilename(initialdir = 'C:/Users/anubhav/Desktop/Mask_Detector_GUI',
                                               title = 'OPEN', 
                                               filetypes = (('png files', '*.png'), ('allfiles', '*.*')))
    
    # Reading image to check if it is not None
    checkImg = cv2.imread(root.filename)
    
    # Checking if img is empty or not
    if np.all(checkImg == None):
        image['state'] = 'normal'
        video['state'] = 'normal'
        live['state'] = 'normal'
        return
    
    # Creating Image Window
    global imageWindow
    imageWindow = Toplevel(root)
    imageWindow.title('Image Window')
    imageWindow.resizable(False, False)
    imageWindow.iconbitmap('winlogo.ico')
    imageWindow.geometry(str(int(checkImg.shape[1]/2)) + "x" + str(int(checkImg.shape[0]/2)))
    imageWindow.protocol("WM_DELETE_WINDOW", onImageWindowClose)
    
    # Adding Label
    global imageLabel
    imageLabel = Label(imageWindow)
    imageLabel.pack()
    
    processImage(0)

def processImage(var):
    
    global img
    img = cv2.imread(root.filename)
    
    # Getting width and length of image
    height, width, layers = img.shape
    
    # Resizing the image

    model = load_model("mask_detector_mobilenet.h5")

    classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    labels_dict={1:"without_mask",0:"with_mask"}
    color_dict={1:(0,0,255),0:(0,255,0)}

    faces = classifier.detectMultiScale(img, 1.3, 5)

    for x,y,w,h in faces:
        face_img = img[y:y+h, x:x+w]
        img = cv2.resize(face_img, (224, 224))
        data = []
        data.append(img)
        data = np.array(data)
        data = data/255
        predictions = model.predict(data)
        if predictions<=0.5:
            label = 0
        else:
            label = 1

        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
    height, width = img.shape[:2]
    ppm_header = f'P6 {width} {height} 255 '.encode()
    data = ppm_header + cv2.cvtColor(img, cv2.COLOR_BGR2RGB).tobytes()
    imageFrame = PhotoImage(width=width, height=height, data=data, format='PPM')
    
    imageLabel.configure(image = imageFrame)
    imageLabel.photo = imageFrame


def openVideoWindow():
    # Setting Window Close
    allWindowClose = "Video"
    
    # Disabling all Button
    image['state'] = 'disabled'
    video['state'] = 'disabled'
    live['state'] = 'disabled'
    
    # File Dialog
    root.filename = filedialog.askopenfilename(initialdir = 'C:/Users/anubhav/Desktop/Mask_Detector_GUI',
                                               title = 'OPEN', 
                                               filetypes = (('mp4 files', '*.mp4'), ('allfiles', '*.*')))
    
    global vCapture
    vCapture = cv2.VideoCapture(root.filename)
    flag, img = vCapture.read()
    
    # Checking if img is empty or not
    if np.all(img == None):
        image['state'] = 'normal'
        video['state'] = 'normal'
        live['state'] = 'normal'
        return
    
    # Creating Live Window
    global videoWindow
    videoWindow = Toplevel(root)
    videoWindow.title('Video Window')
    videoWindow.resizable(False, False)
    videoWindow.iconbitmap('winlogo.ico')
    #videoWindow.geometry("640x480")
    videoWindow.protocol("WM_DELETE_WINDOW", onVideoWindowClose)
    
    # Label to display frame
    global videoImageLabel
    videoImageLabel = Label(videoWindow)
    
    processVideo()


def processVideo():
    # Control Panel for Video Window
    
    model = load_model("mask_detector_mobilenet.h5")

    classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    labels_dict={1:"without_mask",0:"with_mask"}
    color_dict={1:(0,0,255),0:(0,255,0)}
    # Video Window Loop
    while True:
        flag, img = vCapture.read()
        faces = classifier.detectMultiScale(img, 1.3, 5)

        for x,y,w,h in faces:
            face_img = img[y:y+h, x:x+w]
            img = cv2.resize(face_img, (224, 224))
            data = []
            data.append(img)
            data = np.array(data)
            data = data/255
            predictions = model.predict(data)
            if predictions<=0.5:
                label = 0
            else:
                label = 1

            cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
            cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
            cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    
        # Checking if window is closed
        if np.all(img == None):
            break
    
        # Displaying the image on the screen
        height, width = img.shape[:2]
        ppm_header = f'P6 {width} {height} 255 '.encode()
        data = ppm_header + cv2.cvtColor(img, cv2.COLOR_BGR2RGB).tobytes()

        imageFrame = PhotoImage(width = width, height = height, data = data, format = 'PPM')
        
        videoImageLabel['image'] = imageFrame
        videoImageLabel.pack()
        
        videoImageLabel.update()
        
    onVideoWindowClose()


def openLiveWindow():
    # Setting Window Close
    allWindowClose = "Live"
    
    # Disabling all Button
    image['state'] = 'disabled'
    video['state'] = 'disabled'
    live['state'] = 'disabled'
    
    global lcapture
    lcapture = cv2.VideoCapture(0)
    flag, img = lcapture.read()
    
    # Creating Live Window
    global liveWindow
    liveWindow = Toplevel(root)
    liveWindow.title('Live Window')
    liveWindow.resizable(False, False)
    liveWindow.iconbitmap('winlogo.ico')
    liveWindow.geometry("640x480")
    liveWindow.protocol("WM_DELETE_WINDOW", onLiveWindowClose)
    
    # Label to display frame
    global liveImageLabel
    liveImageLabel = Label(liveWindow)
    
    processLive()


def processLive():
    # Control Panel for Live Window\
    model = load_model("mask_detector_mobilenet.h5")

    classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    labels_dict={1:"without_mask",0:"with_mask"}
    color_dict={1:(0,0,255),0:(0,255,0)}
    # Live Window Loop
    while True:
        flag, img = lcapture.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(gray, 1.3, 5)

        for x,y,w,h in faces:
            face_img = img[y:y+h, x:x+w]
            imgs = cv2.resize(face_img, (224, 224))
            data = []
            data.append(imgs)
            data = np.array(data)
            data = data/255
            predictions = model.predict(data)
            if predictions<=0.5:
                label = 0
            else:
                label = 1

            cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
            cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
            cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

            if label==1:
                alarm = threading.Thread(target=voice_alarm)
                alarm.start()
    
        # Checking if window is closed
        if liveImageLabel.winfo_exists() != 1:
            break
    
        # Displaying the image on the screen 
        height, width = img.shape[:2]
        ppm_header = f'P6 {width} {height} 255 '.encode()
        data = ppm_header + cv2.cvtColor(img, cv2.COLOR_BGR2RGB).tobytes()

        imageFrame = PhotoImage(width = width, height = height, data = data, format = 'PPM')
        
        liveImageLabel['image'] = imageFrame
        liveImageLabel.pack()
        
        liveImageLabel.update()


def onImageWindowClose():
    # Enabling all Buttons
    image['state'] = 'normal'
    video['state'] = 'normal'
    live['state'] = 'normal'
    
    # Destroying the window
    imageWindow.destroy()


def onVideoWindowClose():
    # Enabling all Buttons
    image['state'] = 'normal'
    video['state'] = 'normal'
    live['state'] = 'normal'
    
    # Destroying the window
    videoWindow.destroy()
    
    # Releasing the resource
    vCapture.release()


def onLiveWindowClose():
    # Enabling all Buttons
    image['state'] = 'normal'
    video['state'] = 'normal'
    live['state'] = 'normal'
    
    # Releasing the resource
    lcapture.release()
    
    # Destroying the window
    liveWindow.destroy()


def onMainWindowClose():
    if allWindowClose == 'Video':
        # Destroying the window
        videoWindow.destroy()

        # Releasing the resource
        vCapture.release()
    elif allWindowClose == 'Live':
        # Releasing the resource
        lcapture.release()

        # Destroying the window
        liveWindow.destroy()
    elif allWindowClose == 'Image':
        # Destroying the window
        imageWindow.destroy()
    
    root.destroy()


def _from_rgb(rgb):
    """translates an rgb tuple of int to a tkinter friendly color code
    """
    return "#%02x%02x%02x" % rgb 


root = Tk()  
root.iconbitmap(r'winlogo.ico')
root.title('Face Mask Detection')
root.configure(background=_from_rgb((69,61,68)))
root.bind("<Escape>", quit)
root.bind("q", quit)

# Get screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Set root window size and position
root.geometry(f"{screen_width}x{screen_height}+0+0")

global allWindowClose
allWindowClose = ""

# Mask Detection Image
logo = ImageTk.PhotoImage(Image.open('mask3.png').resize((int(screen_width * 0.45), int(screen_height))))
logolabel = Label(root, image=logo)
logolabel.place(x=0, y=0)

# Mask Label
namemask = Label(root, text='COVERUP:', font=('georgia', 56, 'bold'), bg=_from_rgb((69,61,68)), fg=_from_rgb((255,255,255)))
namemask.place(x=int(screen_width * 0.47), y=int(screen_height * 0.07))

# Detection Label
namedet = Label(root, text='FACE MASK DETECTOR', font=('georgia', 56, 'bold'), bg=_from_rgb((69,61,68)), fg=_from_rgb((255,255,255)))
namedet.place(x=int(screen_width * 0.47), y=int(screen_height * 0.17))

# Image Button
image = Button(root, text='Browse Image', padx=int(screen_width * 0.015), command=openImageWindow, bg=_from_rgb((153,52,65)),fg=_from_rgb((255,232,237)),relief=RIDGE,borderwidth=1,font= ('georgia',20,'bold'),cursor="hand2")
image.place(x=int(screen_width * 0.5), y=int(screen_height * 0.4))

# Video Button
video = Button(root, text='Browse Video', padx=int(screen_width * 0.015), command=openVideoWindow, bg=_from_rgb((153,52,65)),fg=_from_rgb((255,232,237)),relief=RIDGE,borderwidth=1,font= ('georgia',20,'bold'),cursor="hand2")
video.place(x=int(screen_width * 0.5), y=int(screen_height * 0.5))

# Live Button
live = Button(root, text='Livefeed', padx=int(screen_width * 0.015), command=openLiveWindow, bg=_from_rgb((153,52,65)),fg=_from_rgb((255,232,237)),relief=RIDGE,borderwidth=1,font= ('georgia',20,'bold'),cursor="hand2")
live.place(x=int(screen_width * 0.5), y=int(screen_height * 0.6))

imgss = ImageTk.PhotoImage(Image.open('maskk2.png'))
imgsslabel = Label(root, image=imgss, bg=_from_rgb((69,61,68)))
imgsslabel.place(x=int(screen_width * 0.67), y=int(screen_height * 0.45))

root.mainloop()
