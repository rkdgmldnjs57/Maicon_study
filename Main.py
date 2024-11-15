
import cv2

from Image import *
from Utils import *

font = cv2.FONT_HERSHEY_SIMPLEX
direction = 0
Images=[]
N_SLICES = 4

for q in range(N_SLICES):
    Images.append(Image())


capture = cv2.VideoCapture("../../Video/omo2.mp4")

while cv2.waitKey(33) < 0:

    ret, frame = capture.read()
    cv2.imshow("VideoFrame", frame)

    # array = np.frombuffer(data, dtype='uint8')
    # img = cv2.imdecode(array, 1)
    img = frame[250:, :]
    direction = 0
    img = RemoveBackground(img, True)
    if img is not None:
        SlicePart(img, Images, N_SLICES)
        for i in range(N_SLICES):
            direction += Images[i].dir
        
        fm = RepackImages(Images)
        cv2.imshow("Vision Race", fm)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        
# Clean up the connection
capture.release()
cv2.destroyAllWindows()
