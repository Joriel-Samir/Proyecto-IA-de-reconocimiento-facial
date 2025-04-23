import cv2
import os 
import numpy as np

class Entrenar():
    def __init__(self, ):
        self.datapatch = f"C:\\Users\\jorie\\OneDrive\\Documentos\\Proyecto IA universidad\\Data"
        self.peoplelist = os.listdir(self.datapatch)
        print("Lista de personas: ", self.peoplelist)
        self.labels = []
        self.face_data = []
        self.label = 0 
    def Recorrer(self,):
        for NameDir in self.peoplelist:
            self.personpatch = self.datapatch + '\\' + NameDir
            print("Leyendo imagenes")
            for FileName in os.listdir(self.personpatch):
                print("Rostros: ", NameDir + '\\' + FileName)
                self.labels.append(self.label)
                self.face_data.append(cv2.imread(self.personpatch + '\\' + FileName, 0))
                self.image = cv2.imread(self.personpatch + '\\' + FileName, 0)
                cv2.imshow("Imagenes", self.image)
                cv2.waitKey(10)
                self.label += 1
            
        print("Labels: ",self.label)
        
    def entrenar(self,):
        self.face_recognizer = cv2.face.EigenFaceRecognizer_create()
        self.face_recognizer.train(self.face_data, np.array(self.labels))
        print("Entrenando modelo")
        self.face_recognizer.write("modeloEigenFace.xml")
        print("Modelo almacenado")

entrenamiento = Entrenar()
entrenamiento.Recorrer()
entrenamiento.entrenar()