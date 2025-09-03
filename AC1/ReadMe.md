
# Aplicações de

## Visão Computacional:

```python
import cv2

# Detecção de faces com OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
img = cv2.imread('pessoas.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imwrite('faces_detectadas.jpg', img)

```

----------

## Processamento de Imagem

```python
import cv2

#Aplicar blur
img = cv2.imread('imagem.jpg')
blurred = cv2.GaussianBlur(img, (15, 15), 0)

cv2.imwrite('imagem_borrada.jpg', blurred)

```

----------

## Síntese de Imagem

```python
import numpy as np
import cv2

# Gerar imagem com texto
img = np.zeros((200, 600, 3), dtype=np.uint8)
cv2.putText(img, 'Hello world', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

cv2.imwrite('texto_imagem.jpg', img)

```

----------

## Visualização de Imagens

```python
import cv2
import matplotlib.pyplot as plt

# Mostrar imagem com múltiplas subplots
img = cv2.imread('imagem.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)

plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(gray, cmap='gray')
plt.title('Cinza')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(edges, cmap='gray')
plt.title('Bordas')
plt.axis('off')

plt.tight_layout()
plt.show()

```
