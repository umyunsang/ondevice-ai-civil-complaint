# On-device AI 민원 처리 및 분석 시스템

LLM을 경량화하여 온디바이스에서 실행하고, 파인튜닝을 통해 현장 산업체에 최적화된 민원 처리 시스템

## 프로젝트 개요

- **목표**: 경량화된 LLM 기반 온디바이스 민원 처리 시스템 개발
- **핵심 기술**:
  - LLM 경량화 (Quantization, Pruning, Knowledge Distillation)
  - 도메인 특화 파인튜닝
  - 온디바이스 추론 최적화

## 프로젝트 구조

```
ondevice-ai-civil-complaint/
├── data/                    # 학습 데이터
├── models/                  # 모델 파일
├── src/                     # 소스 코드
│   ├── preprocessing/       # 데이터 전처리
│   ├── training/           # 모델 학습
│   ├── inference/          # 추론 엔진
│   └── utils/              # 유틸리티
├── configs/                 # 설정 파일
├── notebooks/              # 실험 노트북
└── tests/                  # 테스트 코드
```

## 개발 환경 설정

```bash
# 저장소 클론
git clone https://github.com/umyunsang/ondevice-ai-civil-complaint.git
cd ondevice-ai-civil-complaint

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

## 브랜치 전략

- `main`: 프로덕션 브랜치 (직접 push 금지, PR을 통해서만 머지)
- `develop`: 개발 브랜치
- `feature/*`: 기능 개발 브랜치
- `fix/*`: 버그 수정 브랜치

## 기여 방법

1. 이슈 생성 또는 할당된 이슈 확인
2. `develop` 브랜치에서 새 브랜치 생성
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/기능명
   ```
3. 코드 작성 및 커밋
4. Pull Request 생성
5. 코드 리뷰 후 팀장이 머지

## 팀원

| 역할 | GitHub |
|------|--------|
| 팀장 | [@umyunsang](https://github.com/umyunsang) |

## 라이선스

MIT License
