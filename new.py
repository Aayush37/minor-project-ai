import numpy as np
import tensorflow as tf
import cv2
m_new = tf.keras.models.load_model('mod.h5')
a = np.ones([300,300],dtype='uint8')*0
wname = 'Shapes'

cv2.namedWindow(wname)
flags=0
def shape(event,x,y,flags,param):
    
    if event == cv2.EVENT_LBUTTONDOWN:
        flags=1
        cv2.circle(a,(x,y),8,(255,255,255),-1)
    if flags==1 and event == cv2.EVENT_MOUSEMOVE:
        cv2.circle(a,(x,y),8,(255,255,255),-1)
cv2.setMouseCallback(wname,shape)

while True:
    cv2.imshow(wname,a)
    key = cv2.waitKey(1)
    op = cv2.resize(a/255,(28,28)).reshape(1,28,28)
    
    if key == ord('q'):
        break
    elif key == ord('c'):
        a[:,:] = 0
    elif key == ord('g'):
        print(m_new.predict_classes(op))
cv2.destroyAllWindows()
