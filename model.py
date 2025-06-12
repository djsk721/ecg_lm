import torch
import torch.nn as nn
from torchvision.models import resnet18


class VariableSizeECGEncoder(nn.Module):
    """
    ResNet-18 기반 ECG 인코더 - 가변 개수의 리드 입력 처리 가능
    PTB-XL 데이터셋으로 진단 분류 사전 훈련됨
    """
    def __init__(self, num_classes=5, embedding_dim=768):
        super(VariableSizeECGEncoder, self).__init__()
        
        # ResNet-18 백본 로드 (사전 훈련된 가중치 사용)
        self.backbone = resnet18(pretrained=True)
        
        # 첫 번째 conv 레이어를 ECG 신호에 맞게 수정 (1채널 입력)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 분류 헤드 제거하고 특징 추출기로 사용
        self.backbone.fc = nn.Identity()
        
        # 가변 크기 입력을 위한 적응형 풀링
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 진단 분류를 위한 헤드 (PTB-XL 사전 훈련용)
        self.classifier = nn.Linear(512, num_classes)
        
        # LLM과의 정렬을 위한 프로젝션 레이어
        self.projection = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(self, x, return_features=False):
        """
        Args:
            x: ECG 신호 텐서 [batch_size, num_leads, sequence_length]
            return_features: True면 특징 벡터 반환, False면 분류 결과 반환
        """
        batch_size, num_leads, seq_len = x.shape
        
        # 각 리드를 개별적으로 처리
        lead_features = []
        for i in range(num_leads):
            # 단일 리드를 2D 이미지 형태로 변환 [batch_size, 1, 1, seq_len]
            lead_input = x[:, i:i+1, :].unsqueeze(2)
            
            # ResNet-18 백본을 통한 특징 추출
            features = self.backbone.conv1(lead_input)
            features = self.backbone.bn1(features)
            features = self.backbone.relu(features)
            features = self.backbone.maxpool(features)
            
            features = self.backbone.layer1(features)
            features = self.backbone.layer2(features)
            features = self.backbone.layer3(features)
            features = self.backbone.layer4(features)
            
            # 적응형 풀링으로 고정 크기 특징 생성
            features = self.adaptive_pool(features)
            features = features.view(batch_size, -1)
            
            lead_features.append(features)
        
        # 모든 리드의 특징을 평균으로 결합
        combined_features = torch.stack(lead_features, dim=1).mean(dim=1)
        
        if return_features:
            # LLM 토큰으로 사용할 임베딩 반환
            return self.projection(combined_features)
        else:
            # 진단 분류 결과 반환 (사전 훈련용)
            return self.classifier(combined_features)

# ECG-텍스트 조인트 사전 훈련을 위한 모델
class ECGTextJointModel(nn.Module):
    """
    ECG 인코더와 LLM을 결합한 조인트 모델
    """
    def __init__(self, ecg_encoder, llm_model, llm_tokenizer):
        super(ECGTextJointModel, self).__init__()
        self.ecg_encoder = ecg_encoder
        self.llm_model = llm_model
        self.llm_tokenizer = llm_tokenizer
        
    def forward(self, ecg_data, text_input):
        """
        ECG 데이터와 텍스트를 함께 처리
        """
        # ECG를 단일 토큰으로 인코딩
        ecg_embedding = self.ecg_encoder(ecg_data, return_features=True)
        
        # 텍스트 토큰화
        text_tokens = self.llm_tokenizer(text_input, return_tensors="pt", padding=True)
        
        # ECG 임베딩을 LLM 입력에 통합
        # 실제 구현에서는 특별한 ECG 토큰 위치에 삽입
        return ecg_embedding, text_tokens

# 모델 초기화 및 사전 훈련 설정
def initialize_ecg_encoder():
    """
    ECG 인코더 초기화 및 PTB-XL 사전 훈련 준비
    """
    # PTB-XL 데이터셋의 5개 진단 슈퍼클래스
    num_diagnostic_classes = 5
    
    # 인코더 생성
    encoder = VariableSizeECGEncoder(
        num_classes=num_diagnostic_classes,
        embedding_dim=768  # LLM 임베딩 차원과 맞춤
    )
    
    print("ECG 인코더 초기화 완료")
    print(f"- 진단 클래스 수: {num_diagnostic_classes}")
    print(f"- 임베딩 차원: 768")
    print("- 가변 크기 리드 입력 지원")
    print("- PTB-XL 사전 훈련 준비됨")
    
    return encoder


