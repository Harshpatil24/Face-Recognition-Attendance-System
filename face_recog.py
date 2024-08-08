import cmake
import numpy as np
import cv2
import csv
from datetime import datetime
import face_recognition

video_capture = cv2.VideoCapture(0)

#load known faces
harsh_images = face_recognition.load_image_file("faces/WhatsApp Image 2024-08-07 at 22.53.37.jpeg")
harsh_encoding = face_recognition.face_encodings(harsh_images)[0]

rohit_image = face_recognition.load_image_file("faces/WhatsApp Image 2024-08-07 at 22.48.14.jpeg")
rohit_encoding = face_recognition.face_encodings(rohit_image)[0]

mayank_image = face_recognition.load_image_file("faces/WhatsApp Image 2024-08-07 at 22.49.00.jpeg")
mayank_encoding = face_recognition.face_encodings(mayank_image)[0]

varnit_image = face_recognition.load_image_file("faces/WhatsApp Image 2024-08-07 at 22.49.49.jpeg")
varnit_encoding = face_recognition.face_encodings(varnit_image)[0]

jp_image = face_recognition.load_image_file("faces/WhatsApp Image 2024-08-07 at 22.50.35.jpeg")
jp_encoding = face_recognition.face_encodings(jp_image)[0]

known_face_encoding = [harsh_encoding,rohit_encoding,mayank_encoding,varnit_encoding,jp_encoding]
known_faces_name = ["harsh","rohit","mayank","varnit","jp"]

# list of expected student
students = known_faces_name.copy()

face_locations=[]
face_encodings = []

# get the current date and time

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(f"{current_date}.csv","w+",newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)

    #recognize faces
    face_locations=face_recognition.face_locations(rgb_small_frame)
    face_encodings= face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
        faec_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
        best_match_index = np.argmin(faec_distance)

        if(matches[best_match_index]):
            name=known_faces_name[best_match_index]

        # Add_the ftext if a personis present
        if name in known_faces_name:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomlrftcornerofText = (10,100)
            fontScale = 1.5
            font_color = (255,0,0)
            thickness = 3
            lineType = 2
            cv2.putText = (frame,name+"present",bottomlrftcornerofText,font,fontScale,font_color,thickness,lineType)

            if name in students:
                students.remove(name)
                current_time = now.strftime("%H-%M%S")
                lnwriter.writerow([name,current_time])

        cv2.imshow("Attendance",frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

video_capture.release()
cv2.destroyAllWindows
f.close

