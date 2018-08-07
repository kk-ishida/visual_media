import sys
import copy
import cv2
import numpy as np
from matplotlib import pyplot as plt
RESIZE = 2

#コマンドライン引数から対象のファイルを読み込む
args = sys.argv
img = cv2.imread(args[1], -1)

#顔認証
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray_img, minNeighbors=15, minSize=(40,40))
#faces = face_cascade.detectMultiScale(gray_img, minNeighbors=n, minSize=(w,h), maxSize=(w,h))

i=0
for (x, y, w, h) in faces:
    #認識された顔を矩形で囲む
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 1)
    #認識された顔を縦横RESIZE倍して切り抜き保存
    face = img[y:y+h, x:x+w]
    zoom_face = cv2.resize(face, None, fx=RESIZE, fy=RESIZE)
    cv2.imwrite('face_'+str(i)+'.jpg', zoom_face)
    
    i+=1

rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_img)

try:
    while(1):
        #クリックした座標を取得
        a = plt.ginput(n=1)[0]
        #クリックした場所が認識された顔なら大きくする
        j=0
        for (x, y, w, h) in faces:
            z_f = cv2.imread('face_'+str(j)+'.jpg')
            if x<=a[0]<=x+w and y<=a[1]<=y+h:
                zoom_h, zoom_w, _ = z_f.shape
                #拡大される顔は常に一つ
                temp_img = copy.deepcopy(img)
                temp_img[y-(zoom_h-h)/2:y+(zoom_h+h)/2, x-(zoom_w-w)/2:x+(zoom_w+w)/2] = z_f
                plt.close()    
                rgb_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
                plt.imshow(rgb_img)
                break
            j+=1
except:
    pass

plt.show()

cv2.imwrite('new_pic.png', temp_img)
exit()
