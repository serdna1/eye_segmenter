# Segmentador ocular

Este proyecto es mi segunda práctica de la asignatura "Visión Artifial". En ella se pedía segmentar pupila, iris y esclerótica a partir de imágenes oculares.

## Dataset
Para probar la app se me proporcionó un dataset de 16 imágenes, que se pueden encontrar en [este enlace de google drive](https://drive.google.com/drive/folders/1krUEHCnYjGHp5d2WcLWAHygYgknjDeEL?usp=sharing)

## Ejecución
Es necesario tener una carpeta llamada "ojos", con las imagenes oculares, en el mismo path que el fichero .py.<br><br>
El único argumento es el nombre de la imagen con extensión:
```
python app.py aeval1.bmp
```

## Resultado
El resultado de la ejecución es una serie de imágenes (25), que reflejan el procedimiento seguido.

## Memoria

- Esquema:
  1. Segmentación del iris como un círculo (sin segmentación de párpado).
  2. Segmentación de la pupila.
  3. Obtención del brillo de la pupila.
  4. Segmentación del iris teniendo en cuenta los párpados.
  5. Segmentación de la esclerótica (limitado por los párpados):
     1. Obtención del modelo que se ajusta al límite inferior del ojo
        * Generar puntos relevantes
        * 3 puntos
        * Ransac para polinomios
     2. Obtención del modelo que se ajusta al límite superior del ojo
        * Generar puntos relevantes
        * 3 puntos
        * Ransac para polinomios
- Soluciones:
  * Solución de i.:
    - El iris se ha obtenido a partir de la transformada de Hough para círculos, con una imagen de bordes (canny) como entrada. Se han probado diferentes rangos de radios (de 40 a 70, de 1 en 1 es el actual)
    - Se ha considerado este algoritmo ya que el iris tiene forma de círculo en todas las imágenes (son frontales y mirando a cámara)
    - Segmenta bien los iris para las 16 imágenes de prueba
  * Solución de ii.:
    - Segmentación similar a la anterior. El rango de radios actual es de 20 a 40, de 1 en 1. En uno de los casos, el circulo se detectaba en los bordes de las cejas, pero se solucionó obligando a que el centro de la pupila estuviese en una porción central del iris.
    - Igual que en el paso anterior
    - Segmenta bien los iris para las 16 imágenes de prueba
  * Solución de iii.:
    - Se ha optado por KMeans con la intensidad de gris de los píxeles como vector de características y con 3 clusters (2 más el del fondo negro fuera de la pupila). Nos quedamos con los píxeles del cluster más luminoso. También podría haberse obtenido con algún método de umbralización, como el de Otsu. O con watershed. O con crecimiento de regiones.
    - Se ha optado por este algoritmo ya que hay dos regiones de intensidades muy diferentes y ninguna más. Es facilmente clusterizable.
    - Segmenta bien los brillos
  * Solución de iv.:
    - A partir del iris sobre fondo negro se aplica canny. Se limpia la imagen eliminando bordes (segmentos pequeños, píxeles verticales o el borde de la pupila). Luego, desde el centro de la pupila se lanzan rayos hacia arriba y abajo (con rangos de ángulos dirigidos al contorno de la piel) y nos quedamos con conjuntos de puntos utilizados para obtener un modelo que se ajuste a ellos (a través de ransac para polinomios). Quizás también se podría usar un snake.
    - Con la/s zona de contraste de la piel y unos angulos adecuados de los rayos pueden obtenerse buenos puntos
    - La mayor parte de las veces funciona correctamente
    - A veces no se ajusta correctamente. Esto podría solucionarse limpiando mejor la imagen de bordes y/o cambiando los parámetros de la función de ransac (como el grado del polinomio)
  * Solución de v. a. Modelo del contorno inferior:
    - Se parte de una imagen de bordes (canny sobre la imagen con ecualización de histograma) limpiada con el objetivo de dejar los bordes dobles de este contorno (provocados por la sombra del párpado). Se lanzan rayos hacia el contorno inferior para buscar el doble borde y se queda con el punto intermedio. Luego se intentan ajustar con dos modelos diferentes: si el de ransac para polinomios no se detecta se usa uno que consiste en coger al punto más a la izquierda, el punto más a la derecha y el que está más cerca del punto medio entre los dos y crear un círculo a partir de los 3
    - El ajuste de puntos parecía adecuado para este contorno de bajo contraste que deja bordes discontinuos.
    - Intuye el contorno a veces con más éxito que en otras. No suele ser totalmente preciso.
  * Solución de v. b. Modelo del contorno superior
    - Similar que en el caso anterior generando rayos hacia arriba sobre una imagen canny (sobre un closing de la original). En este caso se queda con los puntos que primero cortan los rayos.