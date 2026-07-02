# Classificação de Imagens de Vestuário com CNN no Fashion-MNIST

Projeto acadêmico que implementa, treina e avalia uma Rede Neural Convolucional (CNN) para classificação multiclasse de imagens de vestuário do dataset **Fashion-MNIST**, desenvolvido por Evelly Victory Vieira Pinto (Instituto Federal de Educação, Ciência e Tecnologia do Maranhão - Campus Caxias).

## Objetivo

Implementar, treinar e avaliar um modelo de aprendizado profundo capaz de identificar automaticamente a categoria de uma peça de vestuário (entre 10 classes possíveis) a partir de imagens em escala de cinza de 28x28 pixels.

## Dataset

- **Nome:** Fashion-MNIST (Zalando Research)
- **Amostras:** 70.000 imagens (60.000 treino / 10.000 teste)
- **Classes (10):** 
  - 0 T-shirt/top  
  - 1 Trouser  
  - 2 Pullover  
  - 3 Dress  
  - 4 Coat  
  - 5 Sandal  
  - 6 Shirt  
  - 7 Sneaker  
  - 8 Bag  
  - 9 Ankle boot
- **Formato:** imagens em escala de cinza, 28x28 pixels, 1 canal
- **Fonte:** carregado diretamente via `keras.datasets.fashion_mnist.load_data()`

## Arquitetura do modelo

CNN sequencial construída com TensorFlow/Keras, treinada do zero (sem modelos pré-treinados):

```text
Entrada (28x28x1)
 -> Conv2D(32)      -> LeakyReLU -> MaxPooling2D
 -> Conv2D(64)      -> LeakyReLU -> MaxPooling2D
 -> Conv2D(128)     -> LeakyReLU -> MaxPooling2D
 -> Flatten
 -> Dense(128)      -> LeakyReLU
 -> Dense(10, softmax)
```

## Hiperparâmetros

| Hiperparâmetro        | Valor        |
|-----------------------|--------------|
| Otimizador            | Adam         |
| Taxa de aprendizado   | 0.001        |
| Batch size            | 64           |
| Épocas                | 30           |
| Função de perda       | Categorical Crossentropy |
| Métrica principal     | Acurácia     |
| Estratégia de validação | Hold-out (conjunto de teste usado como validação) |

## Resultados

- **Acurácia no conjunto de teste:** 0.92
- **Precisão, Recall e F1-score (macro e ponderado):** 0.92
- Classes com melhor desempenho: 1 (Trouser), 5 (Sandal), 8 (Bag), 9 (Ankle boot) — F1-score entre 0.97 e 0.99
- Classe com maior dificuldade: 6 (Shirt) — F1-score de 0.76, indicando maior confusão com classes visualmente semelhantes (Pullover e Coat).
- Observou-se **overfitting** nas épocas finais (acurácia de treino ~99,4% vs. acurácia de validação estabilizada em ~92,5%; perda de validação volta a crescer após ~9 épocas).

## Estrutura do repositório

```text
.
├── fashion_cnn_evelly.ipynb       # Notebook com todo o pipeline (dados, modelo, treino, avaliação)
├── Fashion_CNN___EVELLY-3.pdf     # Relatório completo do trabalho (formato ABNT)
└── README.md                      # Este arquivo de documentação
```

## Tecnologias utilizadas

- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib
- Ambiente de execução: Google Colab (Jupyter Notebook) com GPU

## Como executar

1. Abra `fashion_cnn_evelly.ipynb` no Google Colab ou em um ambiente Jupyter.
2. Execute as células na ordem:
   - Carregamento e pré-processamento do Fashion-MNIST.
   - Definição da arquitetura da CNN.
   - Treinamento (30 épocas) com batch size 64.
   - Avaliação com métricas de classificação e geração de gráficos.
3. Ao final, o notebook gera:
   - Curvas de perda e acurácia (treino/validação).
   - Matriz de confusão.
   - Relatório de classificação por classe (precisão, recall, F1-score, suporte).

## Limitações e melhorias futuras

- Aplicar **early stopping** para interromper o treinamento quando a acurácia de validação parar de melhorar, reduzindo o overfitting.
- Utilizar **data augmentation** (rotações, pequenos deslocamentos, variações de brilho/contraste) para aumentar a diversidade das imagens de treino.
- Adicionar técnicas de **regularização** (dropout, penalização L2) nas camadas densas ou convolucionais.
- Testar arquiteturas mais profundas ou **modelos pré-treinados** com fine-tuning em Fashion-MNIST.
- Investigar em detalhe os erros da classe `Shirt` (classe 6) para entender quais padrões geram maior confusão com `Pullover` e `Coat`.

## Uso de IA generativa

O autor declara ter utilizado ferramenta de IA generativa como apoio pontual durante o desenvolvimento do trabalho, principalmente para:
- Correção de erros de código.
- Refinamento da redação de trechos do relatório.

A definição do problema, implementação do experimento, execução do treinamento, análise dos resultados e validação do conteúdo foram conduzidas pelo autor, que assume responsabilidade integral pelo trabalho.

## Referências

- Xiao, H., Rasul, K., & Vollgraf, R. (2017). *Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms*. arXiv:1708.07747.
- Kayed, M., Anter, A. M., & Mohamed, H. (2020). *Classification of garments from Fashion-MNIST dataset using CNN LeNet-5 architecture*. ITCE 2020.
- Moreno-Díaz-Alejo, L. (2020). *Análisis comparativo de arquitecturas de redes neuronales para la clasificación de imágenes*. Master’s thesis, Universidad Internacional de La Rioja.
- Sahoo et al. (2022). *Implementation of CNN and ANN for Fashion-MNIST dataset using different optimizers*. Indian Journal of Science and Technology, 15(48).
