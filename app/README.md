# App web

App em Flask que recebe uma imagem de OCT e devolve as probabilidades das quatro classes e o mapa do Perturbation-CAM. Em 2021, ele ficou no ar no Google App Engine (ambiente flexível). O link da apresentação final não está mais ativo.

## Como funciona

- **Extrator:** ResNet50V2 congelada, com os pesos da ImageNet, seguida de `GlobalAveragePooling2D`, o que dá 2.048 features por imagem.
- **Classificador:** a cabeça densa do baseline (`modelos/model_baseline_nn.json` e `modelos/weights_baseline.h5`).
- **Explicabilidade:** `PerturbationCAM.cria_mapa_calor_otimizado` varre a imagem em blocos de 64×64 pixels, troca cada bloco por ruído e mede quanto a predição muda.
- **Contador:** `count.txt` guarda o número de visitas da página inicial.

## Rodar localmente

Requer Python 3.7 a 3.9, por causa do TensorFlow 2.5:

```bash
cd app
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py   # http://127.0.0.1:5000, com o modo debug do Flask: use só localmente
```

Na primeira execução, o Keras baixa os pesos da ResNet50V2, cerca de 95 MB.

## Implantar

O `app.yaml` é a configuração usada no App Engine flexível (`gcloud app deploy`). Ela mantém uma instância fixa com 12 vCPUs e 12 GB de memória, o que gera custo enquanto o app estiver no ar.

## Observações

- Esta é a versão entregue no projeto (`main.py`). A única mudança é que o extrator é montado pelo Keras, em vez de lido do arquivo `featurenet.h5`, de 94 MB.
- Nas funções de mapa de calor, a cópia da imagem é feita antes do laço das máscaras. Por isso, as `num_random_mask` perturbações de um mesmo bloco ficam iguais. Com `num_random_mask = 2`, como no app, o efeito prático é usar uma máscara aleatória por bloco.
