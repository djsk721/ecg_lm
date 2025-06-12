# ECG를 활용한 LLM (Large Language Model) 프로젝트

이 프로젝트는 심전도(ECG) 데이터를 활용하여 의료 진단을 지원하는 LLM 기반 시스템을 구현

## 주요 기능

- ECG 신호 데이터 처리 및 분석
- 리드(Lead) 정보를 활용한 데이터셋 확장
- 환자 정보(나이, 성별, 키, 체중)와 ECG 데이터 결합
- 의료 진단 보고서 생성

## 기술적 특징

- 12개 리드 ECG 데이터 처리 (I, II, III, aVR, aVL, aVF, V1-V6)
- 동적 리드 선택 (4-6개 리드)을 통한 데이터셋 확장
- 템플릿 기반 프롬프트 생성
- 환자 특성과 ECG 데이터의 통합적 분석

## 데이터 처리

- PTB-XL 데이터셋 활용
- ECG 신호의 임베딩 생성
- 리드별 특성 분석 및 설명 생성
- 진단 보고서와 일반적 특징 통합

## 활용 분야

- 심장 질환 진단 지원
- ECG 해석 자동화
- 의료 교육 및 훈련
- 임상 의사결정 지원

## 참고문헌
1. Wang, X., et al. (2023). "ECG-LLM: A Large Language Model for Electrocardiogram Interpretation." Health Data Science, 3, 0221. https://spj.science.org/doi/10.34133/hds.0221