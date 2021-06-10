import os
import cv2 as cv
import numpy as np

people = ['Elon Musk', 'Bill Gates']  # made the list of names 
DIR  = r'C:\Users\user\Desktop\ML stuff\openCV\celebs' # path where all names folder is there

haar_cascade = cv.CascadeClassifier('haar_face.xml')  # stored the file with large data in variable  

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)  # till here we reached the each name in people

        for img in os.listdir(path):
            img_path = os.path.join(path, img) # till here we reached the iamge in the people
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4) # to detect the faces

            for (x, y, w, h,) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print('training done -----------------------')

features = np.array(features, dtype='object')
labels = np.array(labels)

# instantiate the face recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# train the recognizer on the feature list and labels
face_recognizer.train(features, labels)
face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)
