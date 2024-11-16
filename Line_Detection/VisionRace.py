import cv2
import numpy as np

class Image:
    def __init__(self):
        self.image = None
        self.contourCenterX = 0
        self.MainContour = None
        
    def Process(self):
        imgray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY) #Convert to Gray Scale
        ret, thresh = cv2.threshold(imgray,100,255,cv2.THRESH_BINARY_INV) #Get Threshold

        self.contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) #Get contour ####CHAIN_APPROX_SIMPLE로 바꿔야함
        
        self.prev_MC = self.MainContour
        if self.contours:
            self.MainContour = max(self.contours, key=cv2.contourArea)
        
            self.height, self.width  = self.image.shape[:2]

            self.middleX = int(self.width/2) #Get X coordenate of the middle point
            self.middleY = int(self.height/2) #Get Y coordenate of the middle point
            
            self.prev_cX = self.contourCenterX
            if self.getContourCenter(self.MainContour) != 0:
                self.contourCenterX = self.getContourCenter(self.MainContour)[0]
                if abs(self.prev_cX-self.contourCenterX) > 50: #이전 중심과 x좌표 차이가 5보다 크면, 5보다 작은놈 있으면 대체.
                    self.correctMainContour(self.prev_cX)
            else:
                self.contourCenterX = 0
            
            self.dir =  int((self.middleX-self.contourCenterX) * self.getContourExtent(self.MainContour)) # 곡률에 비례해 dir 설정
            
            cv2.drawContours(self.image,self.MainContour,-1,(0,255,0),3) #Draw Contour GREEN
            #print(self.MainContour)
            cv2.circle(self.image, (self.contourCenterX, self.middleY), 7, (255,255,255), -1) #Draw dX circle WHITE
            cv2.circle(self.image, (self.middleX, self.middleY), 3, (0,0,255), -1) #Draw middle circle RED
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.image,str(self.middleX-self.contourCenterX),(self.contourCenterX+20, self.middleY), font, 1,(200,0,200),2)
            cv2.putText(self.image,"Weight:%.3f"%self.getContourExtent(self.MainContour),(self.contourCenterX+20, self.middleY+35), font, 0.5,(200,0,200),1)
        
    def getContourCenter(self, contour):
        M = cv2.moments(contour)
        
        if M["m00"] == 0:
            return 0
        
        x = int(M["m10"]/M["m00"])
        y = int(M["m01"]/M["m00"])
        
        return [x,y]
        
    def getContourExtent(self, contour): #bounding box Area / contour area
        area = cv2.contourArea(contour)
        x,y,w,h = cv2.boundingRect(contour)
        rect_area = w*h
        if rect_area > 0:
            return (float(area)/rect_area)
            
    def Aprox(self, a, b, error): #차이 작니
        if abs(a - b) < error:
            return True
        else:
            return False
            
    def correctMainContour(self, prev_cx): #너무 많이 바뀌었어 다른놈 데려와
        if abs(prev_cx-self.contourCenterX) > 5:
            for i in range(len(self.contours)):
                if self.getContourCenter(self.contours[i]) != 0:
                    tmp_cx = self.getContourCenter(self.contours[i])[0]
                    if self.Aprox(tmp_cx, prev_cx, 50) == True:
                        self.MainContour = self.contours[i]
                        if self.getContourCenter(self.MainContour) != 0:
                            self.contourCenterX = self.getContourCenter(self.MainContour)[0]


def SlicePart(im, images, slices):
    height, width = im.shape[:2]
    sl = int(height/slices)
    
    for i in range(slices):
        part = sl*i
        crop_img = im[part:part+sl, :]
        images[i].image = crop_img
        images[i].Process()
    
def RepackImages(images):
    img = images[0].image
    for i in range(len(images)):
        if i == 0:
            img = np.concatenate((img, images[1].image), axis=0)
        if i > 1:
            img = np.concatenate((img, images[i].image), axis=0)
            
    return img

def Center(moments):
    if moments["m00"] == 0:
        return 0
        
    x = int(moments["m10"]/moments["m00"])
    y = int(moments["m01"]/moments["m00"])

    return x, y
    
def RemoveBackground(image, b):
    up = 100
    # create NumPy arrays from the boundaries
    lower = np.array([0, 0, 0], dtype = "uint8")
    upper = np.array([up, up, up], dtype = "uint8")
    #----------------COLOR SELECTION-------------- (Remove any area that is whiter than 'upper')
    if b == True:
        mask = cv2.inRange(image, lower, upper)
        image = cv2.bitwise_and(image, image, mask = mask)
        image = cv2.bitwise_not(image, image, mask = mask)
        image = (255-image)
        return image
    else:
        return image
    #////////////////COLOR SELECTION/////////////

def warpping(image):
    """
        차선을 BEV로 변환하는 함수
        
        Return
        1) _image : BEV result image
        2) minv : inverse matrix of BEV conversion matrix
    """

    # roi_source = np.float32([[86, 150], [554, 150], [640, 400], [0, 400]])
    # roi_source = np.float32([[120, 0], [520, 0], [520, 480], [120, 480]])
    # source = np.float32([[200, 210], [20,480], [420,210], [620, 480]])
    source = np.float32([[230, 0], [0, 480], [440, 0], [640, 480]])
    destination = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])
    
    M = cv2.getPerspectiveTransform(source, destination)
    Minv = cv2.getPerspectiveTransform(destination, source)
    
    # image = region_of_interest(image, [roi_source])
    
    warp_image = cv2.warpPerspective(image, M, (640, 480), flags=cv2.INTER_LINEAR)
    #warp_image = region_of_interest(warp_image, [roi_source])
    # cv2.rectangle(warp_image, (195, 445), (480, 480), (0, 0, 0), -1)

    return warp_image, Minv



font = cv2.FONT_HERSHEY_SIMPLEX
direction = 0
Images=[]
N_SLICES = 4

for q in range(N_SLICES):
    Images.append(Image())

capture = cv2.VideoCapture("../Video/omo1.mp4")

while cv2.waitKey(33) < 0:
    ret, frame = capture.read()
    cv2.imshow("original", frame)

    img, minv = warpping(frame)
    img = cv2.GaussianBlur(img, (7, 7), 5)

    direction = 0
    img = RemoveBackground(img, True)
    if img is not None:
        SlicePart(img, Images, N_SLICES)
        for i in range(N_SLICES):
            direction += Images[i].dir
        
        fm = RepackImages(Images)
        cv2.imshow("Vision Race", fm)

# Clean up the connection
capture.release()
cv2.destroyAllWindows()
