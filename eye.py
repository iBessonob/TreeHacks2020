import cv2
import os

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
    client = vision.ImageAnnotatorClient()

    cap = cv2.VideoCapture(0)

    ret,frame1 = cap.read()

    #ret,frame2 = cap.read()

    while cap.isOpened():
        gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        _ = 0

        image = vision.types.Image(content=cv2.imencode('.jpg', frame1)[1].tostring())

        response = client.face_detection(image=image)
        faces = response.face_annotations
        print(faces)

        for f in faces:
            for l in f.landmarks:
                eye_positions.append((l.position.x,l.position.y,l.position.z))
                if _ == 6: break
                _+=1
        
        print("Positions List: ",eye_positions)
        for _ in eye_positions:
            # Center coordinates 
            center_coordinates = (int(_[0]),int(_[1])) 
            print("CENTER COORDS: ",center_coordinates)

            # Radius of circle 
            radius = 2
            
            # Blue color in BGR 
            color = (0, 0, 255) 
            
            # Line thickness of 2 px 
            thickness = 2
            
            # Using cv2.circle() method 
            # Draw a circle with blue line borders of thickness of 2 px 
            image = cv2.circle(frame1, center_coordinates, radius, color, thickness) 
            #cv2.circle(frame1,(_[0],_[1]),2,(0, 0, 255),2)
            print(_)
        print('---------------------------')
        
        cv2.imshow('feed',frame1)
        ret,frame1 = cap.read()

        if cv2.waitKey(40) == 27:
            break

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

if __name__ == '__main__':
    detect_eyes() #"driver.jpg"

print("succesful???")
