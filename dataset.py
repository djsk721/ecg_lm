import wfdb
import numpy as np
import random
import torch 
import os 

def create_ecg_prompt(age, gender, sub_template, ecg, report, general, lead_descriptions):
    """
    ECG 데이터와 텍스트 정보를 결합하여 프롬프트를 생성하는 함수
    
    Args:
        age: 환자 나이
        gender: 환자 성별
        sub_template: 키/몸무게 정보를 포함한 부분 템플릿
        ecg: ECG 인코더 출력
        report: PTB-XL 진단 보고서 (영어 번역)
        general: ECG의 전반적 특징을 설명하는 문장
        lead_descriptions: 각 리드별 설명 딕셔너리 (I, II, III, aVR, aVL, aVF, V1-V6)
    """
    # 기본 프롬프트 구성
    base_prompt = f"This person is a {age}-year-old {gender} {sub_template}. The ECG signal is {ecg}. The ECG report indicates {report}. The anomalies in general: {general}"
    
    # 리드별 설명을 정확한 순서로 정렬 (논문의 템플릿에 맞춤)
    lead_order = ["lead I", "lead II", "lead III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    
    # 리드별 설명 추가 (순서대로)
    lead_info = []
    for lead_name in lead_order:
        if lead_name in lead_descriptions and lead_descriptions[lead_name]:
            lead_info.append(f"{lead_name}: {lead_descriptions[lead_name][0]}")
    
    if lead_info:
        lead_text = ", ".join(lead_info)
        full_prompt = f"{base_prompt}, {lead_text}."
    else:
        full_prompt = f"{base_prompt}."
    
    return full_prompt

def create_sub_template(height=None, weight=None):
    """
    키와 몸무게 정보를 기반으로 sub_template을 생성하는 함수
    논문에서 언급된 대로 데이터 가용성에 따라 다양한 케이스 처리
    
    Args:
        height: 키 (cm), None일 수 있음
        weight: 몸무게 (kg), None일 수 있음
    
    Returns:
        적절한 sub_template 문자열
    """
    # None 체크와 NaN 체크를 올바르게 수행
    height_valid = height is not None and not np.isnan(height)
    weight_valid = weight is not None and not np.isnan(weight)
    
    
    if height_valid and weight_valid:
        return f"with height {height}cm and weight {weight}kg"
    elif height_valid:
        return f"with height {height}cm"
    elif weight_valid:
        return f"with weight {weight}kg"
    else:
        return ""  # 키와 몸무게 정보가 모두 없는 경우

def expand_dataset(age, gender, sub_template, file_path, encoder, report, general, lead_descriptions):
    """
    데이터셋 확장을 위해 리드 정보를 랜덤하게 제거하는 함수
    논문의 알고리즘에 따라 k=4-6 범위에서 리드 정보를 제거
    선택된 리드만 encoder를 통해 ecg로 변환

    Args:
        age, gender, sub_template, file_path, encoder, report, general, lead_descriptions

    Returns:
        확장된 데이터셋 (원본 + 드롭아웃 버전들)
    """
    # ECG 신호 로드 및 리드 이름 정의
    record = wfdb.rdrecord(os.path.join('../ptb_xl', file_path))
    signals = record.p_signal
    
    sig_name = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    lead_to_index = {name: idx for idx, name in enumerate(sig_name) if name in sig_name}
    
    # 원본 데이터 처리
    used_leads = list(lead_descriptions.keys())
    
    # 리드 드롭아웃 버전 생성
    num_leads = random.randint(4, min(6, len(used_leads)))
    selected_leads = random.sample(used_leads, num_leads)
    
    selected_lead_descriptions = {lead: lead_descriptions[lead] for lead in selected_leads}
    
    # 선택된 리드로 ECG 처리
    selected_indices = [lead_to_index[lead] for lead in selected_leads]
    selected_ecg = signals[:, selected_indices].T
    selected_ecg_tensor = torch.tensor(selected_ecg).unsqueeze(0).float()
    selected_ecg_emb = encoder(selected_ecg_tensor, return_features=True)
    
    # 드롭아웃 버전 추가
    expanded_data = {
        # 'prompt': create_ecg_prompt(age, gender, sub_template, selected_ecg_emb, report, general, selected_lead_descriptions),
        'prompt': create_ecg_prompt(age, gender, sub_template, '[EMBEDDINGS]', report, general, selected_lead_descriptions),
        'dropped_leads': list(set(used_leads) - set(selected_leads)),
        'type': 'reduced',
        'used_leads': selected_leads,
        'ecg_emb': selected_ecg_emb
    }

    return expanded_data


        