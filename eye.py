import cv2
import os, io
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
    0: Left eye
    1: Right eye
    2: Left of Left Eyebrow
    3: Left of Right Eyebrow
    4: Right of Left Eyebrow
    5: Right of Right Eyebrow
    6: Midpoint of Eyes
    '''
    _ = 0
    for f in faces:
        for l in f.landmarks:
            eye_positions.append((l.position.x,l.position.y,l.position.z))
            if _ == 6: break
            _+=1
    ''' 
        print('anger: {}'.format(likelihood_name[face.anger_likelihood]))
        print('joy: {}'.format(likelihood_name[face.joy_likelihood]))
        print('surprise: {}'.format(likelihood_name[face.surprise_likelihood]))

        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in face.bounding_poly.vertices])

        print('face bounds: {}'.format(','.join(vertices)))
    for _ in eye_positions:
        print(_)'''

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

if __name__ == '__main__':
    detect_faces("driver.jpg")

print("succesful???")
