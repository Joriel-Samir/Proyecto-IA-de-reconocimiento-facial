import cv2
import os 

class Reconocer(): 
    def __init__(self):
        self.datapatch = f"C:\\Users\\jorie\\OneDrive\\Documentos\\Proyecto IA universidad\\Data"
        self.imagePaths = os.listdir(self.datapatch)
        print("ImagePaths = ", self.imagePaths)
        self.face_recognizer = cv2.face.EigenFaceRecognizer_create()
    def Identificar(self,): 
        #Leer modelo
        self.modelo_path = "C:\\Users\\jorie\\OneDrive\\Documentos\\Proyecto IA universidad\\modeloEigenFace.xml"
        self.face_recognizer.read(self.modelo_path)
        self.cap = cv2.VideoCapture(0)
        self.Face_clasiff = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        while True:
            self.ret, self.frame = self.cap.read()
            if self.ret == False:
                print("Algo fallo")
                break
            self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            self.aux_frame = self.gray.copy()
            self.faces = self.Face_clasiff.detectMultiScale(self.gray, 1.3, 5)
        
            for (x,y,w,h) in self.faces: 
                self.rostro = self.aux_frame[y:y + h, x: x + w]
                self.rostro = cv2.resize(self.rostro,(150,150), interpolation= cv2.INTER_CUBIC)
                self.result = self.face_recognizer.predict(self.rostro)
                
                cv2.putText(self.frame, '{}'.format(self.result), (x,y-5),1,1.3, (255,255,0), 1, cv2.LINE_AA)
                if self.result[1] < 5700:
                    cv2.putText(self.frame, '{}'.format(self.imagePaths[0]), (x,y-25),1,1.3, (255,255,0), 1, cv2.LINE_AA)
                    cv2.rectangle(self.frame, (x,y), (x+w,y+h), (0,255,0), 2)
                else:
                    cv2.putText(self.frame, "Desconocido", (x,y-25),1,1.3, (255,255,0), 1, cv2.LINE_AA)
                    cv2.rectangle(self.frame, (x,y), (x+w,y+h), (0,255,0), 2)
                    
            cv2.imshow("Frame: ", self.frame)
            self.close = cv2.waitKey(1)
            if self.close == 27:
                break
        self.cap.release()
        cv2.destroyAllWindows() 

reconocer_face = Reconocer()
reconocer_face.Identificar()