from PIL import Image
from scipy import fftpack
from math import *
import numpy as np
import matplotlib.pyplot as plt

def conversion_imagen_bw(img_path, size=(256, 256)):
  '''La funcion consiste en la conversion de la imagen seleccionada a un nivel
  de grises para que su comparacion pueda apreciarse de mejor manera, en este
  caso la imagen a considerar es llamada 'Coca_Cola.bmp'. Tener en cuenta que 
  las dimensiones son 256 x 256. Si se quiere subir otras imagenes solo se
  necesita escribir las dimensiones de dicha imagen en el parametro 'size' de la
  funcion.'''
  imagen_BN = Image.open(img_path)
  imagen_BN = imagen_BN.resize(size, 1)
  imagen_BN = imagen_BN.convert('L')
  imagen_BN = np.array(imagen_BN, dtype=np.float)
  return imagen_BN



def Transformada_discreta_coseno(k):
  '''Esta funcion se basa en la transformada discreta del coseno en su forma mas
  tipicamente utilizada 'DCT-II'. ''' 
  if k == 0:
      return 1 / sqrt(2)
  else:
      return 1


def Transformada_discreta_coseno_2(img, inverse=False):
  '''A continuacion se tiene en cuenta la anterior funcion y se utiliza la
  siguiente libreria 'from scipy import fftpack'. '''
  if not inverse:
      return fftpack.dct(fftpack.dct(img.T, norm='ortho').T, norm='ortho')
  else:
      return fftpack.idct(fftpack.idct(img.T, norm='ortho').T, norm='ortho')


def Transformada_discreta_Fourier(img, inverse=False):
  '''Esta funcion se basa en la transformada discreta de Fouriery se utiliza la
  siguiente libreria 'from scipy import fftpack'. '''
  if not inverse:
      return fftpack.rfft(fftpack.rfft(img.T).T)
  else:
      return fftpack.irfft(fftpack.irfft(img.T).T)


def leer_imagen():
  '''Esta funcion permite leer la imagen, en este caso 'Coca_Cola.bmp' teniendo
  en cuenta la su conversion a niveles de grises mediante el llamado de la funcion
  'conversion_imagen_bw'. '''
  Imagen_BN_1 = conversion_imagen_bw('Coca_Cola.bmp')
  return Imagen_BN_1


def aplicar_transformacion():
  '''Esta funcion permite aplicar la transformacion a la imagen mediante el 
  llamado a la funcion 'Transformada_discreta_Fourier sobre la imagen 
  'Coca_Cola.bmp'. '''
  Imagen_DFT = Transformada_discreta_Fourier(Imagen_BN_1)
  return Imagen_DFT


def restauracion_imagen():
  ''' En esta funcion se utiliza la libreria 'import matplotlib.pyplot as plt' 
  en la cual se restaura la imagen usando solo un porcentaje de coeficientes.
  De esta manera la restauracion se hace mediante el llamado de la funcion 
  'Transformada_discreta_Fourier' con el parametro de la copia de la imagen,
  posteriormente se imprime'''
  Figura_imagen = plt.figure(figsize=(8, 8))
  for i in range(0, 256, 16):
      Imagen_copia = Imagen_DFT.copy()
      for j in range(i):
          Imagen_copia[j, (i - j):] = 0
          Imagen_copia[(i - 1):, j] = 0
      Imagen_copia[i:, i:] = 0
      Restauracion = Transformada_discreta_Fourier(Imagen_copia, inverse=True)
      plt.subplot(4, 4, i / 16 + 1)
      plt.imshow(Restauracion, cmap=plt.cm.gray)
      plt.grid(False)
      plt.xlabel('{: .2f}%'.format((i * (i + 1) / 2) * 100 / 256 ** 2))
      plt.xticks([])
      plt.yticks([])


def guardar_imagen():
  '''Esta funcion permite guardar la imagen, en este caso con ayuda de la
   libreria 'import matplotlib.pyplot as plt', el formato con el cual se 
   guarda es 'PNG'. '''
  plt.savefig( 'NuevaImagen.png' )


'''Se puede ver la 'NuevaImagen' desde la parte de archivos de colab.'''


def main():
  restauracion_imagen()
  guardar_imagen()

main()
