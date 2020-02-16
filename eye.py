import cv2
import os
#import numpy as np

#api_key = "5f8c28f5ffb473cc5fec829109919c4275ebb304"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'cloud-vision.json'

#stores tuples (x,y,z) float coords of eye attribute positions
'''
    See array indexes for specific attr below vvvvvvvv
    0: Left eye
    1: Right eye
    2: Left of Left Eyebrow
    3: Left of Right Eyebrow
    4: Right of Left Eyebrow
    5: Right of Right Eyebrow
    6: Midpoint of Eyes
    '''
eye_positions = []

def detect_eyes():
    #Detects eye positions in an image.
    from google.cloud import vision
    import io

    cap = cv2.VideoCapture(0)
    ret,frame1 = cap.read()

    while cap.isOpened():
        #reinits frame and client
        _ = 0
        ret,frame1 = cap.read()
        client = vision.ImageAnnotatorClient()

        #converts frame1 to compatible bit format and gets client response
        image = vision.types.Image(content=cv2.imencode('.jpg', frame1)[1].tostring())
        response = client.face_detection(image=image) #<-- CAUSES LAG!!!!!!!!!
        faces = response.face_annotations

        #fill loop w/ eye attribute coordinates and draw circle @ each loc
        for f in faces:
            for l in f.landmarks:
                eye_positions.append((l.position.x,l.position.y,l.position.z))
                cv2.circle(frame1, (int(l.position.x),int(l.position.y)),2,(0,0,255),2) 
                #^^^    img, (x,y), radius, color(BGR), line thickness
                if _ == 6: break
                _+=1
        
        #shows the frame and checks for key press to exit video
        cv2.imshow('Eye Positions',frame1)
        k = cv2.waitKey(1)  & 0xff
        if k == 27: #escape key
            break
        
    #release video and destroy windows
    cap.release()
    cv2.destroyAllWindows()

    #error handling
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

if __name__ == '__main__':
    detect_eyes() #"driver.jpg"

print("succesful???")
