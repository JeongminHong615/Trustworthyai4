# Assignment 4: α,β-Crown Neural Network Verification

이 프로젝트는 α,β-Crown 신경망 검증 도구를 사용하여 신경망의 강건성을 검증한다.

## 설치 및 환경 설정

### 필수 요구사항

- Python 3.8 이상
- PyTorch 1.9 이상
- Conda

### 1단계: α,β-Crown 저장소 클론

```bash
git clone https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
cd alpha-beta-CROWN
```

### 2단계: 의존성 설치

α,β-Crown 저장소 내에서:

```bash
cd complete_verifier
pip install -r requirements.txt
```

### 3단계: 이 프로젝트 파일 복사

이 프로젝트의 파일들을 α,β-Crown의 `complete_verifier` 디렉토리에 복사한다:

```bash
cp my_model.py /path/to/alpha-beta-CROWN/complete_verifier/
cp my_model.yaml /path/to/alpha-beta-CROWN/complete_verifier/
cp test.py /path/to/alpha-beta-CROWN/complete_verifier/
```

---

## 파일 구조

```
.
├── Report.pdf                    # 탐색 결과 보고서
├── my_model.py                   # SimpleCNN 모델 정의
├── my_model.yaml                 # α,β-Crown 검증 설정 파일
├── test.py                       # 모델 훈련 및 검증 스크립트
├── my_weights.pth                # 훈련된 모델 가중치 (처음 실행 시 생성)
├── README.md
└── requirements.txt
```

---

## 실행 방법

### 방법 1: 자동 실행 (모델 훈련 + 검증)

```bash
cd /path/to/alpha-beta-CROWN/complete_verifier
python test.py
```

이 스크립트는:

1. MNIST 데이터셋 다운로드 (처음 실행 시)
2. SimpleCNN 모델 훈련 (1 에포크)
3. 가중치를 `my_weights.pth`로 저장
4. α,β-Crown을 실행하여 검증 수행

### 방법 2: 수동 실행

**단계 1: 모델 훈련**

```bash
python test.py
```

이 명령은 `my_weights.pth`를 생성한다.

**단계 2: α,β-Crown 검증**

```bash
python abcrown.py --config my_model.yaml
```

### 검증 실행 옵션 변경

`my_model.yaml` 파일을 편집하여 검증 설정을 변경할 수 있다:

```yaml
data:
  start: 0
  end: 10 # 테스트할 샘플 수 증가

specification:
  epsilon: 0.1 # Perturbation 범위 증가

bab:
  timeout: 120 # Timeout 시간 증가
```
