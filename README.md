
# Railway tree detection

Detección de árboles cercanos a las vías del ferrocarril, con el objetivo de poder identificar de manera temprana, posibles árboles que puedan ocasionar problemas con los trenes.

## Creación del dataset

El primer paso es etiquetar las imágenes que utilizaremos para entrenar al modelo. En nuestro caso utilizamos imágenes satelitales generadas mediante QGIS.

### Pasos para el etiquetado

- Descargar las imágenes y colocarles en una carpeta llamada `images` dentro del directorio principal del proyecto.
- Ejecutar el script `annotations.py` de la siguiente manera: `python3 annotations.py`
- Se abrirá una ventana donde se visualiza cada imágen, y en cada una es posible agregar multiples bounding boxes.
- Al guardar una imágen se genera un archivo `.txt` en la carpeta `labels`, con el formato necesario para luego poder utilizarlo con el modelo `YOLO`

A continuación se muestra un pequeño video del funcionamiento
![hippo](https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExcm8wdGYwcmEyMG91dGdpaW93c2pkZjJrbzBvZTN3ZDZhYnR3anRqdyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/4RskITKAS6XHpVWhY6/giphy.gif)

Los controles del script son los siguientes:
- Clic izquierdo y arrastrar: Dibujar un bounding box
- Clic derecho dentro de un bounding box: Eliminarlo
- 'n': Guardar e ir a la siguiente imagen
- 'p': Guardar e ir a la imagen anterior
- 's': Guardar anotaciones de la imagen actual
- 'r': Reiniciar imagen actual
- 'q': Salir

Algo importante a tener en cuenta, es que el script reconoce las anotaciones cargadas previamente. Es decir que podemos salir cuando queramos, y la próxima vez que ejecutemos el script, podremos continuar con el trabajo.