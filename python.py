                                          # Nhận Diên Khuôn Mặt Và Điểm Danh
# thư viện
import cv2
import face_recognition
import numpy as np
import mysql.connector as m
from datetime import datetime


# định nghĩa hàm kết nối
def connection_database():
    connection=m.connect(host='localhost',user='root',database='python',password='')
    return connection
con=connection_database()

 # tạo cursor để truy vấn
cursor=con.cursor()
# thực thi câu lệnh
cursor.execute("select * from student")
rows=cursor.fetchall()

path="pic2"
images = []
classNames = []
idStu = []
attendance = []

# lấy thông tin sinh viên
for cl in rows:
    # đọc ảnh
    curImg = cv2.imread(f"{path}/{cl[2]}")
    # ghi ma trận khuôn mặt
    images.append(curImg)
    # ghi tên
    classNames.append(cl[1]) 
    # ghi mã sinh viên
    idStu.append(cl[0])  
# bước mã hoá
def Mahoa(images):
    encodeList = []
    for img in images:
        #BGR được chuyển đổi sang RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
# kiểm tra đá điểm danh chưa
def Check(id):
    Check = 1
    for row in attendance:
        if(id == row):
            Check = 2
    return Check
# thực hiện mã hoá        
encodeListKnow = Mahoa(images)

#khởi dộng webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame= cap.read()
    #BGR được chuyển đổi sang RGB
    framS = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # xác định vị trí khuôn mặt trên cam  
    facecurFrame = face_recognition.face_locations(framS)
    # mã hoá hình ảnh trên cam
    encodecurFrame = face_recognition.face_encodings(framS)

    # lấy từng khuôn mặt và vị trí khuôn mặt hiện tại theo cặp
    for encodeFace, faceLoc in zip(encodecurFrame,facecurFrame):
       # so sánh khuôn mặt
        matches = face_recognition.compare_faces(encodeListKnow,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnow,encodeFace)
        #đẩy về vị trí nhỏ nhất
        matchIndex = np.argmin(faceDis) 
        #kiểm tra có đúng không
        if faceDis[matchIndex] < 0.5 :
            # lấy tên học sinh
            name = classNames[matchIndex]
            # kiểm tra đá đá điểm danh chưa
            if(Check(idStu[matchIndex]) == 1):
                now = datetime.now()  

                # điểm danh            
                sqlinsert="insert into attendance values(%s,%s,%s)"
                # thực thi 
                cursor.execute(sqlinsert,( 0,idStu[matchIndex], now)) 
                # ghi dữ liệu vào database 
                con.commit()          
                
                # ghi mã sinh viên đá điểm danh
                attendance.append(idStu[matchIndex])
        else:
            name = "Unknow"

        #print tên lên frame
        y1, x2, y2, x1 = faceLoc
        # vẽ viền xung quanh khuôn mặt
        cv2.rectangle(frame,(x1,y1), (x2,y2),(0,255,0),2)
        # hiện thị tên
        cv2.putText(frame,name,(x2,y2),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    cv2.imshow('nhận diện khuôn mặt', frame)
    
    if cv2.waitKey(1) == ord("q"):  
        break
# giải phóng camera
cap.release()
# thoát tất cả các cửa sổ
cv2.destroyAllWindows()

