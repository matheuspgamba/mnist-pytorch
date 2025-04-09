
# 🧠 MNIST Classifier with PyTorch

> Uma jornada do zero ao modelo exportado com PyTorch, focada em aprendizado e qualidade!

![MNIST Example](./image.png)

## 🚀 Sobre o Projeto

Este projeto implementa uma rede neural convolucional (CNN) em PyTorch para classificar dígitos manuscritos do famoso dataset **MNIST**. Ele cobre **todo o ciclo de vida do modelo**:

- ✅ Pré-processamento e carregamento de dados
- ✅ Treinamento e avaliação
- ✅ Exportação para ONNX
- ✅ Análise de métricas e visualização de resultados

Além disso, o código é organizado, documentado e pronto para deploy futuro.

## 📁 Estrutura do Projeto

```bash
.
├── data_loader.py         # Funções de carregamento de dados
├── model.py               # Arquitetura da rede neural
├── train.py               # Loop de treino, teste e salvamento do modelo
├── eval.ipynb             # Análise de métricas pós-treino
├── teste-img-net.ipynb    # Testes com imagens personalizadas
├── to_onnx.ipynb          # Exportação do modelo para ONNX
├── image.png              # Exemplo visual de predição
├── results/               # Métricas salvas em CSV
├── checkpoints/           # Modelos treinados salvos
```

## 🧠 Arquitetura do Modelo

```python
Net(
  Conv2d(1, 32, kernel_size=3)
  ReLU
  Conv2d(32, 64, kernel_size=3)
  ReLU
  MaxPool2d(2)
  Flatten
  Linear(9216 -> 128)
  ReLU
  Linear(128 -> 10)
  LogSoftmax
)
```

## 🔧 Como Rodar

### 1. Clonar o repositório

```bash
git clone https://github.com/seuusuario/mnist-pytorch.git
cd mnist-pytorch
```

### 2. Instalar dependências

```bash
pip install torch torchvision matplotlib
```

### 3. Treinar o modelo

```bash
python train.py --epochs 5 --batch-size 64 --lr 0.001
```

Os resultados são salvos automaticamente em `results/metrics.csv` e o melhor modelo em `checkpoints/best_model.pth`.

### 4. Avaliação e Exportação

Use os notebooks:

- 📊 `eval.ipynb` – visualização das métricas de treino/teste
- 🧪 `teste-img-net.ipynb` – teste com imagens customizadas
- 📤 `to_onnx.ipynb` – exporta o modelo treinado para o formato ONNX

## 🏁 Resultados

Durante o treinamento, o modelo atinge uma acurácia de **+99%**, com desempenho consistente e perda decrescente. Aqui está um exemplo de resultado salvo:

![Resultado Exemplo](./image.png)

## 🔮 Próximos Passos

- [ ] Deploy com Streamlit
- [ ] Versão Mobile com ONNX Runtime
- [ ] Adição de mais notebooks explicativos

## 🤖 Tecnologias Usadas

- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/stable/)
- [Jupyter Notebook](https://jupyter.org/)
- [ONNX](https://onnx.ai/)

## ✍️ Autor

Feito com 💻, ☕ e 🤘 por **Matheus Paz Gamba**  
[LinkedIn](https://www.linkedin.com/in/matheuspazgamba) | [GitHub](https://github.com/seuusuario)

---

> “Sabemos o que somos, mas não sabemos o que podemos ser.” – Shakespeare
