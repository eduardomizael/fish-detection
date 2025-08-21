# Detector e Rastreador de Peixes com YOLO

Este projeto treina e executa um detector de objetos (peixes) usando Ultralytics YOLO, com opções de:
- Detecção em vídeo/stream ou webcam
- Rastreamento multi‑objeto (ByteTrack)
- ROI poligonal interativa (para focar a área de interesse)
- Filtros anti-falsos positivos
- Visualização de trilhas e overlays com atalhos de teclado
- Contagem de IDs únicos confirmados (v3)

Funciona com GPU (CUDA) quando disponível e cai para CPU automaticamente.

Observação: O repositório está preparado para desenvolvimento em Python 3.12 e pode coexistir com um projeto Django, mas os scripts principais deste detector são stand‑alone (não exigem servidor web).

## Funcionalidades principais

- detector_v2.py: detecção + rastreamento com ByteTrack, ROI opcional, trilhas, filtros de confiança e painel de atalhos.
- detector_v3.py: tudo do v2 + contador de IDs únicos confirmados (mostra “Peixes: N” no topo).
- train_model.py: rotina de treinamento YOLO com parâmetros ajustáveis (dataset YAML, tamanho de imagem, épocas, etc.).
- utils.py (opcional): extração de frames de vídeos via ffmpeg.

## Pré‑requisitos

- Git
- Python 3.12+
- Drivers NVIDIA + CUDA/cuDNN se for usar GPU (opcional)
- ffmpeg (opcional, apenas se for usar utils.py para extrair frames)

Dica para instalar PyTorch (com ou sem CUDA): use o seletor oficial em https://pytorch.org/get-started/locally/ para o seu sistema. Exemplos de comandos estão mais abaixo.

## Clonando o repositório

```shell script
# via SSH
git clone git@github.com:eduardomizael/fish-detection.git
cd SEU_REPOSITORIO

# ou via HTTPS
git clone https://github.com/eduardomizael/fish-detection.git
cd SEU_REPOSITORIO
```

## Instalação de dependências

Abaixo seguem duas rotas equivalentes:

- Caminho A (recomendado): virtualenv + pip
- Caminho B (alternativo): uv (se preferir um gerenciador ultrarrápido)

Escolha um dos caminhos.

### Caminho A: virtualenv + pip

1) Crie e ative o ambiente

- Linux/macOS:
```shell script
python3 -m venv .venv
source .venv/bin/activate
```


- Windows (PowerShell):
```textmate
python -m venv .venv
.venv\Scripts\Activate.ps1
```


2) Instale PyTorch adequado ao seu hardware

- GPU (exemplo CUDA 12.1, verifique no site):
```shell script
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```


- Somente CPU:
```shell script
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```


3) Instale bibliotecas do projeto
```shell script
pip install ultralytics opencv-python numpy pyyaml matplotlib pandas
```


Se for usar tracking com ByteTrack via Ultralytics, os pacotes acima já cobrem o necessário. Se precisar de ferramentas extras (ffmpeg-python para utils.py):
```shell script
pip install ffmpeg-python
```


4) (Opcional) Verifique a GPU no Python
```python
import torch; print(torch.cuda.is_available(), torch.version.cuda)
```


### Caminho B: uv (alternativo)

Se você não possui o uv instalado, instale:

- Linux/macOS:
```shell script
curl -LsSf https://astral.sh/uv/install.sh | sh
# reabra o terminal ou exporte o PATH sugerido pela instalação
```

- Windows (PowerShell):
```textmate
irm https://astral.sh/uv/install.ps1 | iex
```

### No diretorio do projeto
1) Sincronize as dependencias
```shell script
# se o tiver GPU
uv sync -E gpu
# se não tiver
uv sync -E cpu
```

2) Verifique se a gpu esta sendo reconhecida
```shell script
uv run versions.py 
```
3) Execute o Detector
```shell script
uv run detector_v3.py # ultima versão
```

## Estrutura esperada de arquivos

- detector_v2.py e detector_v3.py: scripts principais de inferência e tracking
- train_model.py: script de treinamento YOLO
- videos/: pasta para vídeos de entrada (ex.: videos/video04.mp4)
- runs/: saídas de treino do YOLO (best.pt é carregado pelos detectores)
- dataset_custom.yaml: arquivo YAML do dataset (caminhos de treino/val, classes)

Ajuste caminhos no topo dos detectores conforme o seu ambiente:
- MODEL_FILE: caminho do best.pt treinado (ex.: runs/detect/trainX/weights/best.pt)
- VIDEO_CAPTURE_FILE: caminho do vídeo de entrada ou use webcam

## Como treinar

1) Garanta que o dataset_custom.yaml aponte para as imagens/labels corretas.
2) Execute:
```shell script
python train_model.py
```


Dicas:
- Ajuste imgsz, epochs e batch dentro do script conforme sua GPU/CPU.
- O YOLO salvará os resultados em runs/detect/treinoX (métricas, gráficos e weights).
- O melhor peso ficará em runs/detect/treinoX/weights/best.pt.

## Como rodar a detecção e o rastreamento

- Detector com trilhas e overlays:
```shell script
python detector_v2.py
```


- Detector com contagem de IDs únicos no topo:
```shell script
python detector_v3.py
```


Configurações úteis dentro dos scripts:
- MODEL_FILE: caminho para o best.pt
- VIDEO_CAPTURE_FILE: caminho do vídeo ou troque para webcam usando cv2.VideoCapture(0)
- IMG_SIZE: tamanho de inferência (ex.: 480–640)
- Parâmetros de confiança e paciência do rastreador/antifalsos

Atalhos de teclado (durante a execução):
- q: sair
- t: liga/desliga o tracking
- r: liga/desliga trilha
- + / -: ajusta a confiança mínima para promover ID (CONF_START)
- 9 / 0: ajusta N_INIT (frames bons até confirmar ID)
- [ / ]: ajusta LOST_PATIENCE (paciência para “matar” IDs inativos)
- a: liga/desliga filtro geométrico
- p: editar ROI poligonal interativa
- m: liga/desliga ROI
- s: salvar ROI
- l: carregar ROI salva
- i: mostrar/ocultar painel de informações
- h: mostrar/ocultar painel de atalhos

## Dicas de desempenho e precisão

- GPU: ative drivers e CUDA compatíveis. O script usa meia precisão (half) automaticamente em GPU suportada.
- imgsz: 480–640 oferece bom compromisso entre velocidade e qualidade; aumente para objetos pequenos e se houver VRAM.
- Vídeo: use vídeos em um SSD para leitura mais rápida.

## Solução de problemas

- Torch sem CUDA: instale a variante correta do PyTorch a partir do índice da NVIDIA (veja comandos acima).
- “CUDA out of memory”: reduza imgsz ou batch (no treino) e feche outros processos que usam VRAM.
- OpenCV não abre câmera: verifique permissões/sistema (Linux: v4l2, Windows: conflitos com outros apps).
- Vídeo não abre: confirme o caminho e codecs; instale ffmpeg no sistema.
- ROI não aparece: defina os pontos com “p” e confirme com Enter; use “m” para ligar a ROI.

## Licença

Defina aqui a licença do projeto (por exemplo, MIT). Inclua um arquivo LICENSE se aplicável.

## Créditos

- Ultralytics YOLO
- ByteTrack para rastreamento multi‑objeto
- OpenCV para leitura e visualização de vídeo

Em caso de dúvidas ou para otimizações específicas de hardware/dataset, abra uma issue ou peça ajuda: terei prazer em orientar ajustes finos de hiperparâmetros e desempenho.