import cv2
import numpy as np
import face_recognition
import os
import csv


#Loading images using path
encodingPath = '/home/w3villa/Documents/face-recog/encodedImages.csv'
path = '/home/w3villa/Documents/face-recog/Images'
images = []
names = []
List = os.listdir(path)


for name in List:
    img = cv2.imread(f'{path}/{name}')
    images.append(img)
    names.append(os.path.splitext(name)[0])

print("Images List: ")
print(names)


# Function to find encodings for all images in directory
def encode(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeImg = face_recognition.face_encodings(img)[0]
        print("ENCODING---------")
        # print(encodeImg)
        encode_list.append(encodeImg)
    return encode_list


#def recordEncoding(name, encoding):
def recordEncoding(encodedList):
    #print(name)
    with open(encodingPath, 'a+', newline='\n') as file:
        file.seek(0)
        records = csv.reader(file, delimiter=',')
        nameList = []
        for row in records:
            nameList.append(row[0])
        #     if row[1] == 'Encoding':
        #         continue
        #     row[1] = row[1][1:-2].split(",")
        #     new_l = []
        #     for new in row[1]:
        #         new_l.append(float(new))
            # print("row1", np.asarray(new_l, dtype=np.float32))
            #print(f'{row}\n')
        #List = reader.rows # file.readlines();
        #nameList = []
        #for line in List:
        #    record = line.split(',')
        #    nameList.append(record[0])
        #    print(nameList)
        i = 0
        fieldnames = ['Name', 'Encoding']
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=',')
        for encoding in encodedList:
            # print(encodedList[i], end='')
            if names[i] not in nameList:
                # print(names[i] + " Encoding.." )
                # print( encodedList[i])
                arr = []
                for e in encoding:
                    arr.append(e)
                #print('TESTING-----------------')
                # print(arr)
            #txt = f'\n{names[i]},{encoding}'
                print(names[i] + " Recorded!")
                writer.writerow({'Name':names[i],'Encoding':arr})
                #file.write(f'\n{names[i]},{encoding}')
            else:
                print(names[i] + " Already Recorded!")
            i = i + 1


#i = 0
encodeList = encode(images)
#print(encodeList)
recordEncoding(encodeList)
#for encoding in encodeList:
    #print(i)
#    recordEncoding(names[i], encoding)
#    i = i + 1
print("Encoding Completed.")