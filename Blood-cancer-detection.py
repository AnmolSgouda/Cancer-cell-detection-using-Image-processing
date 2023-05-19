import cv2
import numpy as np


#img=cv2.imread(r"C:\Users\anmol\OneDrive\Desktop\python for Ml\Images\Human-red-blood-cells.webp")
img=cv2.imread(r"C:\Users\anmol\OneDrive\Desktop\python for Ml\project\Leukemia-L.jpg")
#img=cv2.imread(r"C:\Users\anmol\OneDrive\Desktop\python for Ml\Images\rbc-2.webp")
#img=cv2.imread(r"C:\Users\anmol\OneDrive\Desktop\python for Ml\Images\cancerl_blood_cells.jpg")


img=cv2.resize(img,(500,500))
imgG=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgbr=cv2.GaussianBlur(imgG,(7,7),0)

cv2.imshow("image1",img)
cv2.imshow("image2",imgG)
cv2.imshow("image3",imgbr)
ret,thresh=cv2.threshold(imgbr,140,190,0)
img_res=img.copy()
contours,hierarcy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
img_cancer=cv2.drawContours(img,contours,-1,(125,125,0),2)


cv2.imshow('contrast',img_cancer+img) 


lower_red = np.array([0, 0, 150], dtype = "uint8") 

upper_red= np.array([150, 0, 255], dtype = "uint8")
mask = cv2.inRange(img_cancer+img, lower_red, upper_red)



cv2.imshow('cell detection',mask)
 



cv2.waitKey(0)
cv2.destroyAllWindows()
2