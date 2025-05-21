# Estimación de Pose y Detección de Emoción en Tiempo Real

Este proyecto utiliza un modelo de YOLO entrenado para estimación de poses (`yolo11n-pose.pt`) junto con detección de emociones faciales usando `DeepFace`. A través de una cámara web, detecta múltiples personas, estima la posición de sus articulaciones y calcula ciertas distancias clave en cada frame. También detecta la emoción dominante si se encuentra un rostro visible así como gráfica en tiempo real la posición respecto a los ejes x,y de las manos izquierda y derecha.
La información obtenida se manda a un JSON para poder interpretar mejor los datos y de igual manera se tiene otro script encargado de leer los datos del JSON para pasarlos a un archivo csv.

## Documentación de YoloV11
[Documentación oficial de Ultralytics](https://docs.ultralytics.com/es/models/yolo11/).

### Keypoints Estimados por el Modelo

El modelo detecta 17 puntos clave (keypoints):

1. Nariz  
2. Ojo izquierdo  
3. Ojo derecho  
4. Oreja izquierda  
5. Oreja derecha  
6. Hombro izquierdo  
7. Hombro derecho  
8. Codo izquierdo  
9. Codo derecho  
10. Muñeca izquierda  
11. Muñeca derecha  
12. Cadera izquierda  
13. Cadera derecha  
14. Rodilla izquierda  
15. Rodilla derecha  
16. Tobillo izquierdo  
17. Tobillo derecho

## Requisitos

- Python 3.8+
- OpenCV
- Ultralytics YOLO
- DeepFace
- NumPy
- Matplotlib

Instalación recomendada:

```bash
pip install opencv-python ultralytics deepface numpy matplotlib
