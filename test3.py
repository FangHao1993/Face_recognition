import face_recognition
import cv2

Hao_known_image = cv2.imread("./Hao/image/WIN_20200120_15_13_53_Pro.jpg")
Hao_encoding = face_recognition.face_encodings(Hao_known_image,num_jitters=3)[0]

CJimg_known_image = cv2.imread("./CJimg/JPEGImages/WIN_20200204_14_57_37_Pro.jpg")
CJimg_encoding = face_recognition.face_encodings(CJimg_known_image,num_jitters=3)[0]

selfy_known_image = cv2.imread("./selfy/picture/leo(4).jpg")
selfy_encoding = face_recognition.face_encodings(selfy_known_image,num_jitters=3)[0]

ford_known_image = cv2.imread("./YOLOv3PHOTO/700/00019.png") 
ford_encoding = face_recognition.face_encodings(ford_known_image,num_jitters=3)[0]

ywt_known_image = cv2.imread("./autoLabel/6.jpg") 
ywt_encoding = face_recognition.face_encodings(ywt_known_image,num_jitters=3)[0]

# unknown_image = face_recognition.load_image_file("./IIIproject/examples/P_20200201_164253.jpg")
unknown_image = cv2.imread("./IIIproject/examples/P_20200201_164253.jpg")
# unknown_image = cv2.imread("./IIIproject/examples/t_20200201_16425.jpg")

print(unknown_image.shape)
hight,width=unknown_image.shape[:2]
print(width,hight)
if unknown_image.shape[1]>1024:
    unknown_image = cv2.resize(unknown_image,(1024,round(1024*hight/width)), interpolation = cv2.INTER_AREA)
    rgb = cv2.cvtColor(unknown_image, cv2.COLOR_BGR2RGB)
    print(unknown_image.shape)
boxes = face_recognition.face_locations(unknown_image)
print(boxes)
encodings = face_recognition.face_encodings(rgb,boxes,num_jitters=3)

know_encodings = [
    Hao_encoding,
    CJimg_encoding,
    selfy_encoding,
    ford_encoding,
    ywt_encoding,
]
known_face_names = [
    "Hao",
    "CJimg",
    "selfy",
    "ford",
    "ywt",
]
face_name = []
for encoding in encodings:
    matchs = face_recognition.compare_faces(know_encodings,encoding,tolerance=0.45)
    name = "Unknow"
    print('match : ',len(matchs))
    print(matchs)
    if True in matchs:
        match_index = matchs.index(True)
        name = known_face_names[match_index]
    face_name.append(name)
    print(face_name)

# loop over the recognized faces
for ((top, right, bottom, left), name) in zip(boxes, face_name):
    # draw the predicted face name on the image
    cv2.rectangle(unknown_image, (left, top), (right, bottom), (0, 255, 0), 2)
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(unknown_image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
        0.75, (0, 255, 0), 2)
cv2.imshow("Image", unknown_image)
cv2.waitKey(0)
