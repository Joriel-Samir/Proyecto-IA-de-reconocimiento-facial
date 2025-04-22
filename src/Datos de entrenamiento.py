import cv2 
import os 
import tkinter as tk
import imutils

class Cargar():
    def __init__(self):
        self.person_name = ""
        self.save_data = f"C:\\Users\\jorie\\OneDrive\\Documentos\\GitHub\\Proyecto-IA-de-reconocimiento-facial"
        self.save_face = ""
    def guardar_nombre(self):
        self.person_name = self.nombre_var.get()
        self.save_face = os.path.join(self.save_data, self.person_name)
        if not os.path.exists(self.save_face):
            os.makedirs(self.save_face)
            print("Carpeta creada:", self.save_face)
        else:
            print("Carpeta ya existe:", self.save_face)

        self.ventana.destroy()  # Cierra la ventana

    def preguntar_name(self): 
        self.ventana = tk.Tk()
        self.ventana.title("Ingreso de nombre para entrenamiento")
        self.ventana.geometry("300x150")

        etiqueta = tk.Label(self.ventana, text="¿Cuál es tu nombre?: ")
        etiqueta.pack(pady=10)

        self.nombre_var = tk.StringVar()
        entrada_nombre = tk.Entry(self.ventana, textvariable=self.nombre_var)
        entrada_nombre.pack()

        boton = tk.Button(self.ventana, text="Guardar", command=self.guardar_nombre)
        boton.pack(pady=10)

        self.ventana.mainloop()
        

class SubirRostros(Cargar):
    def __init__(self, ret,frame):
        super().__init__()
        self.ret = ret
        self.frame = frame
        self.cap = cv2.VideoCapture(0)
        self.Face_clasiff = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.count = 0
        
    def Recorrer_Video(self,):
        while True: 
            self.close = cv2.waitKey(1)
            self.ret, self.frame = self.cap.read()
            
            if self.ret == False:
                break
            
            
            
            self.frame = imutils.resize(self.frame, width=640)
            self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            self.aux_frame = self.frame.copy()
            self.face = self.Face_clasiff.detectMultiScale(self.gray, 1.3, 5)
    
    
            for (x, y, w, h) in self.face:
                cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                rostro = self.aux_frame[y:y+h, x:x+w]
                rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(self.save_face, f"{self.person_name}_{self.count}.jpg"), rostro)
                self.count += 1
                
            
            if self.close == 27 or self.count >= 300:
                break
            
            cv2.imshow("Capturando rostros de entrenamiento", self.frame)
            
        self.cap.release()
        cv2.destroyAllWindows()
   

ret = None
frame = None    

guardar = Cargar()
guardar.preguntar_name()

guardar1 = SubirRostros(ret, frame)

guardar1.person_name = guardar.person_name
guardar1.save_face = guardar.save_face

guardar1.Recorrer_Video()