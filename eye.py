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
    17: LEFT_EYE_TOP_BOUNDS
    19: LEFT EYE BOTTOM BOUNDS
    21: RIGHT EYE TOP BOUNDS
    23 RIGHT EYE BOTTOM BOUNDS
    29 RIGHT PUPIL?????
    30 LEFT PUPIL?????
    '''
eye_positions = []

def detect_eyes():
    #Detects eye positions in an image.
    from google.cloud import vision
    import io

    cap = cv2.VideoCapture(0)
    ret,frame1 = cap.read()
    wants = (17, 18, 19, 20, 21, 22 ,23, 24) #30 - R pup, 29- L pup

    droop_cycles = 0

    while cap.isOpened():
        #reinits frame and client
        _ = 0
        ret,frame1 = cap.read()
        frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        client = vision.ImageAnnotatorClient()

        #converts frame1 to compatible bit format and gets client response
        image = vision.types.Image(content=cv2.imencode('.jpg', frame1)[1].tostring())
        #request every longer interval
        response = client.face_detection(image=image) #<-- CAUSES LAG!!!!!!!!!
        faces = response.face_annotations
        #print(faces)
        #fill loop w/ eye attribute coordinates and draw circle @ each loc
        '''
        See array indexes for specific attr below       vvvvvvvv
        17: Left eye top boundary
        18. Left eye right corner
        19. Left eye bottom boundary
        20. Left eye left corner
        21. Left eye pupil
        22. Right eye top boundary
        23. Right eye right corner
        24. Right eye bottom boundary
        25. Right eye left corner
        26. Right eye pupil
        '''

        for f in faces:
            #print(f.getIsLeftEyeOpenProbability())
            for k in f.landmarks:
                #print(k.type)
                '''print(k.type,k.position.x,k.position.y,k.position.z)
                print()'''
                if k.type in wants:
                    eye_positions.append((k.type,k.position.x,k.position.y,k.position.z))
                    cv2.circle(frame1, (int(k.position.x),int(k.position.y)),1,(0,0,255),1)

        #Calculate Ratios Below
        #See https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/ for algorithm
        #Unknown reason why array randomly init to size zero on random cases!!!!!!!!!!!!!!!1
        print(len(eye_positions))
        leftRatio = droopy_ratio(eye_positions[0][1],eye_positions[1][1],eye_positions[2][2],eye_positions[3][2])
        rightRatio = droopy_ratio(eye_positions[4][1],eye_positions[5][1],eye_positions[6][2],eye_positions[7][2])
        


        print(leftRatio, "   " ,rightRatio)
        print()
        #Currently taking average of both ratios, 
        #if ratio less than .18, droopy for one more cycle
        #if ratio less than .18 for 3 or more cycles then user is sleepy!
        #if ratio greater than .18, reset droopy cycle to zero and user is awake
        #if you find a better way than this that's more accurate, replace code seg below
        if abs(leftRatio+rightRatio)/2 < 0.18:
            if droop_cycles >= 3:
                print("Sleepy!!")
                cv2.putText(frame1,"Warning! Sleepy Driver!",(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
            droop_cycles += 1
        else:
            print("Awake!!!")
            droop_cycles = 0
        
        
        
        #shows the frame and checks for key press to exit video
        cv2.imshow('Eye Positions',frame1)
        k = cv2.waitKey(1)  & 0xff
        eye_positions.clear()
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

def droopy_ratio(l,r,u,b):
    return abs(u-b) / (2*abs(l-r))


if __name__ == '__main__':
    detect_eyes() #"driver.jpg"

print("succesful???")
