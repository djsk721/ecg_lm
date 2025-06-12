from collections import defaultdict
import pandas as pd
import math


leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def describe_qrs(lead_prefix: str, row):
    try:
        # QRS duration을 초 단위로 변환 (ms -> s)
        qrs_duration = (
            row[f'Q_Dur_{lead_prefix}'] + 
            row[f'R_Dur_{lead_prefix}'] + 
            row[f'S_Dur_{lead_prefix}'] + 
            row[f'Rp_Dur_{lead_prefix}'] + 
            row[f'Sp_Dur_{lead_prefix}']
        ) / 1000
        
        if pd.isna(qrs_duration):
            return None
        if qrs_duration > 0.12:
            return f"wider QRS interval"
    except KeyError:
        return f"QRS duration not found"
    
def describe_lead_st(lead_prefix: str, row):
    try:
        st_val = (row['QT_Int_Global'] - (row[f'Q_Dur_{lead_prefix}'] + row[f'R_Dur_{lead_prefix}'] + row[f'S_Dur_{lead_prefix}'] + row[f'Rp_Dur_{lead_prefix}'] + row[f'Sp_Dur_{lead_prefix}']) - row[f'T_Dur_{lead_prefix}']) / 1000  # ST segment J point value ST-segment의 1/8지점 (in mV)
        if pd.isna(st_val):
            return None
        if st_val >= 0.2:
            return f"significant ST elevation"
        elif st_val >= 0.1:
            return f"mild ST elevation"
        elif st_val <= -0.2:
            return f"significant ST depression"
        elif st_val <= -0.1:
            return f"mild ST depression"
    except KeyError:
        return f"not found"

def describe_pr_interval(row):
    try:
        pr = row['PR_Int_Global'] / 1000  # Global PR interval 사용
        if pd.isna(pr):
            return None
        if pr > 0.20:
            return "wider PR interval"
        elif pr < 0.12:
            return "shorter PR interval"
    except KeyError:
        return "PR interval: not found"

def describe_t_wave(lead_prefix: str, row):
    try:
        t_amp = row[f'T_Amp_{lead_prefix}']  # T wave amplitude in mV
        if pd.isna(t_amp):
            return None
        if t_amp < -0.1:
            return f"inverted T wave"
        elif t_amp > 0.8:
            return f"tall peaked T wave"
    except KeyError as e:
        print(e)
        return f"T wave data not found"

def describe_qt_interval(lead_prefix: str, row):
    try:
        t_off = row[f'T_Off_{lead_prefix}']
        qrs_on = row['QRS_On_Global']
        rr = row['RR_Mean_Global']/ 1000
        sex = row.get('sex', 'unknown')

        # 필수 값 검증
        if pd.isna(t_off) or pd.isna(qrs_on) or pd.isna(rr):
            return f"{lead_prefix}: data unavailable"

        # QTc 계산
        qt = (qrs_on - t_off) / 1000
        qtc = qt / math.sqrt(rr)
        # 성별 기준 적용
        if (sex == 1 and qtc > 0.44) or (sex == 0 and qtc > 0.46):
            return f"wider QTc interval"
        elif qtc < 0.36:
            return f"shorter QTc interval"
    except KeyError as e:
        return f"{lead_prefix}: variable not found"
    
def describe_p_wave(lead_prefix: str, row):
    try:
        p_amp = row[f'P_Amp_{lead_prefix}']
        p_dur = row[f'P_Dur_{lead_prefix}'] / 1000
        if pd.isna(p_amp) or pd.isna(p_dur):
            return None
        if p_amp > 0.25:
            return f"tall P wave"
        if p_dur > 0.12:
            return f"prolonged P wave duration"
        elif p_dur < 0.08:
            return f"short P wave duration"
    except KeyError:
        return f"lead {lead_prefix}: P wave data not found"

def describe_all_findings(row):
    """모든 ECG 소견을 종합하여 출력하는 함수"""
    findings = defaultdict(list)
    
    # 각 리드별 분석
    for lead in leads:
        # T파 분석
        t_finding = describe_t_wave(lead, row)
        if t_finding:
            findings[lead].append(t_finding)
        
        # P파 분석  
        p_finding = describe_p_wave(lead, row)
        if p_finding:
            findings[lead].append(p_finding)
        
        # ST 분절 분석
        st_finding = describe_lead_st(lead, row)
        if st_finding:
            findings[lead].append(st_finding)
        
        # QT 간격 분석 (각 리드별 T_Off 사용)
        qt_finding = describe_qt_interval(lead, row)
        if qt_finding:
            findings[lead].append(qt_finding)
            
        # QRS 분석
        qrs_finding = describe_qrs(lead, row)
        if qrs_finding:
            findings[lead].append(qrs_finding)
    
    return findings