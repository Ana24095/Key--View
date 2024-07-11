import cv2
import torch

# Cargar el modelo YOLOv5
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

def detector():
    # Inicializar la cámara
    camara = cv2.VideoCapture(1)

    while camara.isOpened():
        status, frame = camara.read()

        if not status:
            break

        # Realizar la inferencia
        results = model(frame)

        # Obtener los resultados de detección como un DataFrame
        df = results.pandas().xyxy[0]

        # Filtrar por confianza
        df = df[df["confidence"] > 0.5]

         # Dibujar cuadros de texto redondeados y etiquetas en el frame
        for i in range(df.shape[0]):
            bbox = df.iloc[i][["xmin", "ymin", "xmax", "ymax"]].astype(int).values
            class_name = df.iloc[i]['name']
            confidence = round(df.iloc[i]['confidence'], 2)


            # Coordenadas y tamaño del cuadro de texto
            x1, y1, x2, y2 = bbox
            text = f"{class_name}, {confidence}"
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            # Dibujar un cuadro azul con bordes redondeados
            cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w + 10, y1), (255, 0, 0), -1)  # Relleno azul
            cv2.putText(frame, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # Texto blanco

        # Mostrar el frame con las detecciones
        cv2.imshow("frame", frame)
        
        # Salir si se presiona 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Liberar la cámara y cerrar la ventana
    camara.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detector()