import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import csv

encodingPath = '/home/w3villa/Documents/face-recog/encodedImages.csv'
encodeList = []
names = []

def load_encodings():
    with open(encodingPath, 'r+', newline='\n') as file:
        #List = file.readlines()
        records = csv.reader(file, delimiter=',')
        #for line in List:
        for row in records:
            print(row)
            if row[0] == 'Name':
                continue
            names.append(row[0])
            if row[1] == 'Encoding':
                continue
            row[1] = row[1][1:-2].split(",")
            new_l = []
            for new in row[1]:
                new_l.append(float(new))
            #txt = row['Encoding']
            #ar =  np.array(txt)
            #print('-----------------------------------------------------')
            #print(ar)
            #print(txt)
            encodeList.append(np.asarray(new_l, dtype=np.float32))#.replace('\n',''))
        print(names)
        # print("ENCODE LIST ------------- ", encodeList)

def record_attendance(name, distance):
    print(name, distance)
    with open('attendancelist.csv', 'r+') as file:
        List = file.readlines()
        namelist = []
        timelist = []
        for line in List:
            record = line.split(',')
            namelist.append(record[0])
            timelist.append(record[1])
            #print(namelist)
        if name not in namelist:
            now = datetime.now()
            dt = now.strftime('%d/%m/%y %H:%M:%S')
            file.writelines(f'\n{name},{dt},{distance},FIRST ENTRY')
        else:
            index = len(namelist) - 1 - namelist[::-1].index(name)#namelist.index(name)
            format = '%d/%m/%y %H:%M:%S'
            lastTime= datetime.strptime( timelist[index], format)
            now = datetime.now()
            dt = now.strftime('%d/%m/%y %H:%M:%S')

            if(now.date()>lastTime.date()):
                file.writelines(f'\n{name},{dt},{distance},FIRST ENTRY')
            else:
                difference = now - lastTime
                # print("DIFFERENCE--- ", difference.total_seconds())
                if(difference.total_seconds() > 5):
                    file.writelines(f'\n{name},{dt},{distance},ENTRY')
                else:
                    print("Recently Recorded: "+name)


#Load Saved Encodings=
load_encodings()

#Initializing webcam
cap = cv2.VideoCapture(0)


while True:
    success, img = cap.read()
    #Reducing size of real-time image to 1/4th
    imgResize = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgResize = cv2.cvtColor(imgResize, cv2.COLOR_BGR2RGB)

    # Finding face in current frame
    face = face_recognition.face_locations(imgResize)
    # Encode detected face
    encodeImg = face_recognition.face_encodings(imgResize, face)
    #print("ATTEMPTING MATCH ------------------------")
    #print(encodeImg)
    #print(encodeList)
    #Finding matches with existing images
    for encodecurr, loc in zip(encodeImg, face):
        match = face_recognition.compare_faces(encodeList, encodecurr)
        #print("Match:")
        #print(match);
        faceDist = face_recognition.face_distance(encodeList, encodecurr)
        #print("DIST")
        # print("DIST:", faceDist)
        #Lowest distance will be best match
        index_BestMatch = np.argmin(faceDist)
        print(index_BestMatch)

        if match[index_BestMatch]:
            name = names[index_BestMatch]
            distance = faceDist[index_BestMatch]
            y1,x2,y2,x1 = loc
            #Retaining original image size for rectangle location
            y1,x2,y2,x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-30),(x2,y2),(225,225,225), cv2.FILLED)
            cv2.putText(img, name, (x1+20, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2)
            record_attendance(name, distance)
        else:
            print("Unmatched Face")

    cv2.imshow('Webcam', img)

    cv2.waitKey(1)


