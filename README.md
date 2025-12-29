# Edge AI – Serviço de Inferência com FastAPI + YOLOv8 (Podman)

Este projeto implementa um **serviço de inferência de visão computacional para Edge AI**, executando **exclusivamente em container**, utilizando **FastAPI**, **Ultralytics YOLOv8** e **Podman**, com deploy em **Raspberry Pi 4**.

O serviço recebe uma imagem via URL, executa inferência de detecção de objetos e retorna os resultados em **JSON estruturado** ou como **imagem anotada**, conforme o endpoint utilizado.

---

## Visão Geral da Arquitetura

- **Modelo:** YOLOv8 (Ultralytics)
- **Framework Web:** FastAPI
- **Runtime:** Python 3.11
- **Containerização:** Podman (rootless)
- **Inicialização automática:** systemd (user service + linger)
- **Plataforma alvo:** Raspberry Pi 4 (ARM64)
- **Interface:** API HTTP REST

---

## Funcionalidades Implementadas

### Inferência de objetos em imagens
- Download de imagem a partir de uma URL
- Execução de inferência com YOLOv8
- Filtro de predições por confiança mínima
- Medição de tempo de inferência

### Dois modos de saída
- **JSON estruturado** para integração com sistemas
- **Imagem anotada** (bounding boxes + labels) para validação visual

### Robustez
- Tratamento de erros de download (404, timeout, URL inválida)
- Validação de conteúdo retornado (Content-Type)
- Serviço resiliente a falhas (restart automático)

---

## Endpoints Disponíveis

### `POST /detect/json`

Executa a inferência e retorna os resultados em **JSON**.

#### Requisição
```json
{
  "image": "https://ultralytics.com/images/bus.jpg",
  "confidence": 0.25
}
```

> `confidence` é opcional (default: 0.25)

#### Resposta
```json
{
  "detections": [
    {
      "class": "bus",
      "confidence": 0.87,
      "bbox": [x1, y1, x2, y2]
    }
  ],
  "classes_detected": ["bus", "person"],
  "class_counts": {
    "bus": 1,
    "person": 4
  },
  "meta": {
    "model": "yolov8n.pt",
    "inference_time_ms": 42.3,
    "image_source": "https://ultralytics.com/images/bus.jpg",
    "confidence_threshold": 0.25
  }
}
```

---

### `POST /detect/image`

Executa a inferência e retorna a **imagem anotada** (PNG).

#### Requisição
```json
{
  "image": "https://ultralytics.com/images/bus.jpg",
  "confidence": 0.25
}
```

#### Resposta
- `Content-Type: image/png`
- Imagem com bounding boxes e labels

---

## Exemplos de Uso

### Inferência JSON (Linux / macOS)
```bash
curl -X POST http://IP_DA_RASPBERRY:8000/detect/json   -H "Content-Type: application/json"   -d '{"image":"https://ultralytics.com/images/bus.jpg"}'
```

---

### Inferência com Imagem (Windows PowerShell)
```powershell
Invoke-RestMethod `
  -Uri http://IP_DA_RASPBERRY:8000/detect/image `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"image":"https://ultralytics.com/images/bus.jpg"}' `
  -OutFile result.png
```

---

## Containerização

### Requisitos Atendidos
- Aplicação executa **exclusivamente dentro de container**
- Container iniciado automaticamente no boot
- Política de restart configurada
- Porta HTTP exposta corretamente

---

## Build da Imagem

```bash
podman build -t edge-ai .
```

---

## Execução Manual do Container

```bash
podman run -d   --name edge-ai   --ipc=host   -p 8000:8000   edge-ai
```

---

## Inicialização Automática no Boot (systemd)

### 1 Gerar o serviço
```bash
podman generate systemd --new --files --name edge-ai
```

### 2 Instalar como serviço de usuário
```bash
mkdir -p ~/.config/systemd/user
mv container-edge-ai.service ~/.config/systemd/user/
```

### 3 Habilitar linger
```bash
sudo loginctl enable-linger pi
```

### 4 Ativar o serviço
```bash
systemctl --user daemon-reload
systemctl --user enable container-edge-ai
systemctl --user start container-edge-ai
```

### 5 Verificar status
```bash
systemctl --user status container-edge-ai
```

---

## Política de Restart

O serviço systemd utiliza:

```ini
Restart=always
```

Isso garante:
- reinício automático em caso de falha
- recuperação após reboot
- alta disponibilidade em ambiente edge

---

## Demonstração da Inferência

A inferência é evidenciada de duas formas:

- **Dados estruturados** retornados pelo endpoint `/detect/json`
- **Imagem anotada** retornada pelo endpoint `/detect/image`

Esses dois endpoints comprovam:
- execução do modelo
- resultados mensuráveis
- tempo de inferência
- funcionamento em ambiente Edge

---

## Swagger / OpenAPI

A documentação interativa pode ser acessada em:

```
http://IP_DA_RASPBERRY:8000/docs
```

---

## Considerações Finais

Este projeto demonstra:

- Deploy de IA em Edge
- Containerização moderna com Podman
- Integração com systemd
- API REST limpa e robusta
- Inferência reproduzível e auditável

---

## Autor

Desenvolvido por **Nelson Almeida**
