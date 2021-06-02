import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping
import time
import winsound
import cv2

frequncy = 2500 ## 빈도를 2500hz 설정
duration = 1000 ## 울리는 시간은 1초(1000ms)

# 훈련된 모델과 데이터를 로드
import pickle
pickle_in = open("5000_X.pickles","rb")
X=pickle.load(pickle_in)

pickle_in = open("5000_y.pickles","rb")
y=pickle.load(pickle_in)

dect_model = load_model('5000_test_model.h5')
dect_model.summary()

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')



from tkinter import *
from tkinter.ttk import *
import datetime
import platform
import math

# import dlib

window = Tk()
window.title("Clock")
window.geometry('500x250')
stopwatch_counter_num = 1
stopwatch_running = False



# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

tabs_control = Notebook(window)
stopwatch_tab = Frame(tabs_control)
tabs_control.add(stopwatch_tab, text='Stopwatch')
tabs_control.pack(expand = 1, fill ="both")
stopwatch_label = Label(stopwatch_tab, font='calibri 40 bold', text='Stopwatch')
stopwatch_label.pack(anchor='center')
stopwatch_start = Button(stopwatch_tab, text='Start', command=lambda:stopwatch('start'))
stopwatch_start.pack(anchor='center')
stopwatch_stop = Button(stopwatch_tab, text='Stop', state='disabled',command=lambda:stopwatch('stop'))
stopwatch_stop.pack(anchor='center')

def get_video():
    try:
        if video.isOpened():
            # print("video opened")
            ret, img = video.read()
            if ret:
                print("getVideoIf")
                return check_opened(img)
        else:
            print("getVideoElse")
            return False
    except:
        print("error occured")


def check_opened(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)

    for x, y, w, h in eyes:  # 눈 사각형표시
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # green

    if len(eyes):  # 눈을 뜬 상태이면
        last_time = time.time()  # 시간 리셋
        print("눈뜸")
    else : print("눈감음")

    if (time.time() - last_time) > 3:  # 3초 이상 눈 감으면
        #         cv2.putText(frame,'Sleep Alert', (30, 60), font, 0.7, (0,0,255), 2)
        print("눈 3초 이상 감음"+str(last_time))
        winsound.Beep(frequncy, duration)
        return False
    else:
        print("3초 이상 눈을 감지 않음"+str(last_time))
        return True

def clock():

    # initial_time = datetime.datetime.now()

    date_time = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S/%p")
    # date, time1 = date_time.split()
    # time2, time3 = time1.split('/')
    # hour, minutes, seconds = time2.split(':')
    # if int(hour) > 12 and int(hour) < 24:
    #     time = str(int(hour) - 12) + ':' + minutes + ':' + seconds + ' ' + time3
    # else:
    #     time = time2 + ' ' + time3


def stopwatch_counter(label):
    # counter_closed = 0

    #         def closed_counter(counter_closed):
    #             try:
    #                 if (get_video() <= 0.23):
    #                     counter_closed = counter_closed+1
    #                 else:
    #                     counter_closed = 0
    #             except:
    #                 counter_closed = counter_closed+1

    #             return  counter_closed

    def count():
        counter_closed = 0
        if stopwatch_running:
            global stopwatch_counter_num
            if stopwatch_counter_num == 0:
                display = "Starting..."
            else:
                tt = datetime.datetime.fromtimestamp(stopwatch_counter_num)
                string = tt.strftime("%H:%M:%S")
                display = string
            label.config(text=display)
            datetime1 = datetime.datetime.now()
            label.after(1000, count)
            # stopwatch_counter_num = datetime.datetime.now() - datetime1
            # if (datetime.datetime.now()-1 >= datetime1):
            stopwatch_counter_num += 1
            #                         closed_counter(counter_closed)
            if get_video() == False:
                stopwatch('stop')

    #                         print(counter_closed)
    # if closed_counter(counter_closed)>=3:
    #     print("야!!!!!!!!!!!!!!왜!!!!!!!!!!안꺼지냐!!!!!!!!!!!!!!!")
    #     stopwatch('stop')
    count()


def stopwatch(work):
    if work == 'start':
        global stopwatch_running
        stopwatch_running = True
        stopwatch_start.config(state='disabled')
        stopwatch_stop.config(state='enabled')
        stopwatch_counter(stopwatch_label)
    elif work == 'stop':
        stopwatch_running = False
        stopwatch_start.config(state='enabled')
        stopwatch_stop.config(state='disabled')
    elif work == 'reset':
        global stopwatch_counter_num
        stopwatch_running = False
        stopwatch_counter_num = 0
        stopwatch_label.config(text='Stopwatch')
        stopwatch_start.config(state='enabled')
        stopwatch_stop.config(state='disabled')


video = cv2.VideoCapture(0)
clock()
window.mainloop()
