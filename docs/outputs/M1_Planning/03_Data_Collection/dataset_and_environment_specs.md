# 데이터셋 분석 및 학습 환경 명세 (Dataset & Environment Specs)

본 문서는 EXAONE-Deep-7.8B 모델의 파인튜닝(QLoRA) 및 양자화(AWQ)를 위한 데이터셋 상세 정보와 서버 환경 요구사항을 정의합니다.

## 1. AI Hub 텍스트 기반 민원 데이터셋 목록

10만 건 이상의 고품질 학습 데이터 확보를 위한 핵심 데이터셋 번호와 특징입니다.

| 데이터셋 번호 | 데이터셋 명칭 | 데이터 특징 | 예상 규모 |
| :--- | :--- | :--- | :--- |
| **71852** | **공공 민원 상담 LLM 데이터** | **[핵심]** 지방/중앙행정기관 민원 기반의 Instruction Tuning 데이터. 추론 과정 포함. | 약 15만 건+ |
| **71844** | **민간 민원 상담 LLM 데이터** | 기업 서비스, 생활 불편 등 민간 영역의 민원 상담 및 요약 데이터. | 약 20만 건+ |
| **98** | **민원(콜센터) 질의-응답 데이터** | 전통적인 콜센터 Q&A 쌍. 단답형 및 표준 답변 위주의 정제된 데이터. | 약 10만 건+ |
| **619** | **민원 업무 자동화 언어 데이터** | 법률, 행정 용어가 포함된 전문 민원 분석 및 NLP 처리용 데이터. | 약 10만 건+ |

### 데이터셋 선정 전략
- **우선순위 1**: `71852` (공공 민원 상담 LLM) - EXAONE의 `<thought>` 추론 학습에 가장 적합.
- **우선순위 2**: `71844` (민간 민원 상담 LLM) - 다양한 도메인 확장을 위한 보조 데이터.

---

## 2. 서버 환경 명세 (Google Colab 기준)

EXAONE-Deep-7.8B 모델을 QLoRA로 학습하고 AWQ로 양자화하기 위한 사양입니다.

### 2.1 하드웨어 사양

| 구분 | 최소 사양 (Colab Free 수준) | 권장 사양 (Colab Pro/Pro+ 수준) | 비고 |
| :--- | :--- | :--- | :--- |
| **GPU** | **Tesla T4 (16GB VRAM)** | **NVIDIA A100 (40GB) / L4 (24GB)** | VRAM 16GB 미만은 학습 불가 |
| **VRAM** | 16 GB | 24 GB ~ 40 GB | QLoRA 4-bit 로딩 기준 |
| **System RAM** | 12 GB (High-RAM 권장) | 32 GB ~ 64 GB | 데이터셋 토큰화 및 로딩 시 필요 |
| **Disk Space** | 78 GB | 150 GB+ | 모델(15GB) + 데이터 + 체크포인트 |

### 2.2 소프트웨어 환경 (Runtime)

- **Framework**: PyTorch 2.1+, HuggingFace Transformers 4.40.0+
- **Optimization**: PEFT (QLoRA), bitsandbytes (4-bit quantization)
- **Acceleration**: Flash Attention 2 (A100/L4 환경에서만 활성화 가능)
- **Quantization**: AutoAWQ (최종 배포용 AWQ 변환)

---

## 3. 학습 및 양자화 전략

### QLoRA 파인튜닝 (Fine-Tuning)
- **방식**: BF16 원본 모델을 4-bit 양자화 상태로 로드하여 LoRA 어댑터만 학습.
- **메모리 절감**: Gradient Checkpointing 및 4-bit NormalFloat(NF4) 데이터 타입 사용 필수.
- **목표**: 민원 분류 및 표준 답변 생성 능력 내재화.

### AWQ 양자화 (Quantization)
- **방식**: 파인튜닝이 완료된 모델을 `AutoAWQ`를 통해 4-bit AWQ 포맷으로 변환.
- **이점**: 온디바이스(폐쇄망 서버) 환경에서 vLLM을 통한 고속 추론 및 메모리 50% 이상 절감.
- **사양**: 양자화 과정 자체에서도 약 12GB 이상의 VRAM이 소요되므로 T4 16GB 환경에서 수행 가능.

---
**작성일**: 2026-03-05
**관련 모듈**: `collector.py`, `train.py` (예정)
