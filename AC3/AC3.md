## Como o olho humano percebe as cores e descreva a função dos cones.
O olho humano percebe cores graças às células especializadas da retina chamadas **cones**.  Esses cones são de três tipos, cada um sensível a uma faixa diferente de luz (vermelho, azul e verde). Quando a luz entra no olho e atinge a retina, ela estimula esses cones em diferentes proporções. O cérebro então **combina os sinais** vindos dos três tipos de cones para formar a percepção de uma cor específica.

## Diferença entre o modelo RGB e o CMYK? Cite um exemplo de aplicação para cada um e o conceito de cor aditiva e subtrativa

O modelo **RGB** (Red, Green, Blue) é um sistema de cores **aditivo**, usado principalmente em telas, como monitores e TVs. Ele funciona combinando luzes das cores primárias vermelho, verde e azul em diferentes intensidades. Quando as três estão no máximo, o resultado é a cor branca; quando estão apagadas, temos o preto.

Já o modelo **CMYK** (Cyan, Magenta, Yellow, Key/Black) é um sistema **subtrativo**, utilizado em impressões gráficas. Ele se baseia na mistura de pigmentos de tinta, quanto mais cores são usadas, mais luz é absorvida, resultando em tons mais escuros. A combinação completa das cores gera o preto.

## Aplicações de satélite e sensoriamento remoto e apresentação de espaços comuns de cores e a utilização desses espaços de cores com imagens geradas por satélite.

O sensoriamento remoto é uma tecnologia que permite obter informações sobre a superfície da Terra sem contato direto. Imagens são captadas por satélites ou drones, com sensores capazes de registrar diferentes comprimentos de onda.

As aplicações de satélites e do sensoriamento remoto podem incluir **Mapeamento urbano e planejamento territorial** e **Gestão de desastres naturais**, como enchentes, secas e deslizamentos.
    

Os satelites usam **espaços de cores** são utilizados para realçar informações específicas. Os principais são **RGB** (Red, Green, Blue), e **NDVI** (Normalized Difference Vegetation Index) não é um espaço de cor tradicional, mas um índice derivado das bandas do vermelho e infravermelho próximo, usado para analisar vegetação.

Abaixo esta um exemlpo que calcula e exibe o **NDVI** a partir de duas imagens: uma banda NIR (infravermelho próximo) e uma banda Red. 

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

caminho_nir = 'nir_band.png'
camingo_red = 'red_band.png'
nir = cv2.imread(caminho_nir , cv2.IMREAD_GRAYSCALE).astype('float32')
red = cv2.imread(camingo_red , cv2.IMREAD_GRAYSCALE).astype('float32')

bottom = (nir + red)
bottom[bottom == 0] = 0.01

# Calcula NDVI: (NIR - RED) / (NIR + RED)
ndvi = (nir - red) / bottom

# Normaliza o NDVI para 0–255
ndvi_normalized = cv2.normalize(ndvi, None, 0, 255, cv2.NORM_MINMAX)
ndvi_normalized = ndvi_normalized.astype(np.uint8)

# Exibe o resultado
plt.imshow(ndvi, cmap='RdYlGn')
plt.colorbar(label='NDVI')
plt.title('Mapa NDVI com OpenCV')
plt.show()

# (Opcional) Salvar NDVI como imagem
cv2.imwrite('ndvi_result.png', ndvi_normalized)
print("NDVI processado e salvo como 'ndvi_result.png'")

```

