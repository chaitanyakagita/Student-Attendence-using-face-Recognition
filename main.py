import cv2                      #takes input from webcam, process it and gives to face_recognition
import face_recognition         #recognize faces and compare the faces with already present faces in database
import numpy as np              #for array operations
import csv                      #for handling csv file
import os                       #for accessing the file system and manipulating its paths
from datetime import datetime   #to get exact date & time


video_capture =cv2.VideoCapture(0)  #opens web cam

ElonMusk_image = face_recognition.load_image_file("Face_Reco_imgs/ElonMusk.jpg")   #loading images
ElonMusk_encoding = face_recognition.face_encodings(ElonMusk_image)[0]

Mahesh_image = face_recognition.load_image_file("Face_Reco_imgs/Mahesh.jpg")
Mahesh_encoding = face_recognition.face_encodings(Mahesh_image)[0]

Nani_image = face_recognition.load_image_file("Face_Reco_imgs/Nani.jpeg")
Nani_encoding = face_recognition.face_encodings(Nani_image)[0]

Prabhas_image = face_recognition.load_image_file("Face_Reco_imgs/Prabhas.jpg")
Prabhas_encoding = face_recognition.face_encodings(Prabhas_image)[0]

known_face_encoding = [                  #list for storing encodings
    ElonMusk_encoding,
    Mahesh_encoding,
    Nani_encoding,
    Prabhas_encoding
]

known_faces_names = [                    #list for storing names
    "ElonMusk",
    "Mahesh",
    "Nani",
    "Prabhas"
]

print(known_faces_names)

students = known_faces_names.copy()


#--------------------------------------------

face_locations = []         # to save the face coming from webcam
face_encodings = []         # to capture characteristics of persons faces : shape, facial landmark, texture
face_names = []             # to store names of face present in list -->(known_faces_names)
s = True

#to get exact date ,month, year
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

#creating csv file
f = open(current_date+'.csv','w+',newline= '')
lnwriter = csv.writer(f)


while True:
    _,frame = video_capture.read()   #reads info from the webcam
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)        #resize image to 1/4th of original image
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)     #color conversion from bgr to rgb
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)       #detect the face from webcam
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)   #stores the detected faces
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)   #compares faces that are known with comming new faces
            name = ""
            face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)  #finds dist b/w face and cam
            best_match_index = np.argmin(face_distance)   #takes least value from face distances
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]  #takes only name without .jpg


            face_names.append(name)    #adds names to list
            if name in known_faces_names:
                if name in students:
                      students.remove(name)
                      print(students)
                      current_time = now.strftime("%H-%M-%S")     #enters time format
                      lnwriter.writerow([name,current_time])

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (0, 0, 0), 1)

    cv2.imshow(" attendence system",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

video_capture.release()
cv2.destroyAllWindows()
f.close()


