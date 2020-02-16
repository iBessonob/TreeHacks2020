import cv2
import os, io
import time
from google.cloud import vision

api_key = "5f8c28f5ffb473cc5fec829109919c4275ebb304"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'cloud-vision.json'
#cap = cv2.VideoCapture(0)

def detect_faces(path):
    """Detects eye positions in an image."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.face_detection(image=image)
    faces = response.face_annotations

    # Names of likelihood from google.cloud.vision.enums
    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                       'LIKELY', 'VERY_LIKELY')

    #stores tuples (x,y,z) float coords of eye attribute positions
    eye_positions = []
    '''
    See array indexes for specific attr below vvvvvvvv
    1: Left eye
    2: Right eye
    3: Left of Left Eyebrow
    4: Left of Right Eyebrow
    5: Right of Left Eyebrow
    6: Right of Right Eyebrow
    7: Midpoint of Eyes

    17: LEFT_EYE_TOP_BOUNDS
    19: LEFT EYE BOTTOM BOUNDS

    21: RIGHT EYE TOP BOUNDS
    23 RIGHT EYE BOTTOM BOUNDS
    '''
    wants = [17, 19, 21 ,23, 29, 30]

    for f in faces:
        #print(f.getIsLeftEyeOpenProbability())
        for k in f.landmarks:
            if k.type in wants:
                print(k.type)
                print(k)
                eye_positions.append((k.type ,k.position.x,k.position.y,k.position.z))

#cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
    myimg = cv2.imread("closed.jpg")
    for n in eye_positions:
        cv2.circle(myimg, (int(n[1]), int(n[2])), 2, (255, 0, 0), 2)

    # for e in eye_positions:
    #     print(e)

    cv2.imshow("OpenCV Image Reading", myimg)

    cv2.waitKey(0)
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

if __name__ == '__main__':

    video_capture = cv2.VideoCapture(0)
    time.sleep(1)

    ret, frame = video_capture.read()
    ret, frame = video_capture.read()
    ret, frame = video_capture.read()
    cv2.imwrite("closed.jpg", frame)
    time.sleep(1)
    detect_faces("closed.jpg")
    video_capture.release()




print("succesful???")
