#주파수와 공간 정보 분리한 버전


import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from mtcnn import MTCNN
from torchvision import transforms
from PIL import Image
import os

# oneDNN 최적화를 비활성화
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf


class CDCN_Spatial_Frequency(nn.Module):
    def __init__(self):
        super(CDCN_Spatial_Frequency, self).__init__()

        # RGB 이미지 처리 (공간적 정보)
        self.conv1_rgb = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 3채널 (RGB)
        self.conv2_rgb = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3_rgb = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # 주파수 정보 처리 (FFT) -> 1채널 -> 3채널로 수정
        self.conv1_fft = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 3채널 (주파수 정보)
        self.conv2_fft = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3_fft = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # 결합 후 fully connected
        self.fc1 = nn.Linear(128 * 32 * 32 * 2, 512)  # 두 네트워크에서 나온 특징을 결합
        self.fc2 = nn.Linear(512, 2)  # 2 클래스 (REAL or FAKE)

    def forward(self, x_rgb, x_fft):
        # RGB 이미지 처리
        x_rgb = F.relu(self.conv1_rgb(x_rgb))
        x_rgb = F.max_pool2d(F.relu(self.conv2_rgb(x_rgb)), 2)
        x_rgb = F.max_pool2d(F.relu(self.conv3_rgb(x_rgb)), 2)

        # 주파수 정보 처리
        x_fft = F.relu(self.conv1_fft(x_fft))
        x_fft = F.max_pool2d(F.relu(self.conv2_fft(x_fft)), 2)
        x_fft = F.max_pool2d(F.relu(self.conv3_fft(x_fft)), 2)

        # 두 네트워크에서 나온 특성 결합
        x = torch.cat((x_rgb.view(x_rgb.size(0), -1), x_fft.view(x_fft.size(0), -1)), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 모델 로드
model_path = 'FFT_V4.pth'  # 모델 경로
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CDCN_Spatial_Frequency().to(device)

# 가중치 로드 (strict=False로 가중치 불일치 허용)
try:
    model.load_state_dict(torch.load(model_path, map_location=device))  # map_location을 사용하여 CPU로 로드
    print("모델 가중치가 성공적으로 로드되었습니다.")
except Exception as e:
    print(f"모델 가중치 로드 실패: {e}")
    exit()

model.eval()

# MTCNN 모델 초기화
detector = MTCNN()

# 이미지 전처리 함수
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# 주파수 변환 함수 (FFT)
def fft_transform(image):
    gray = image.mean(dim=1, keepdim=True)  # 그레이스케일로 변환
    f = torch.fft.fft2(gray)  # 푸리에 변환
    fshift = torch.fft.fftshift(f)  # 주파수 영역 중앙으로 이동
    magnitude_spectrum = torch.abs(fshift)  # 크기 스펙트럼 반환

    # 주파수 영역 정보가 1채널로 반환되므로, 3채널로 확장
    magnitude_spectrum = magnitude_spectrum.repeat(1, 3, 1, 1)  # 3채널로 확장
    return magnitude_spectrum


# 웹캡처
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캡을 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("웹캠에서 프레임을 읽을 수 없습니다.")
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR -> RGB

    # 얼굴 탐지
    faces = detector.detect_faces(rgb_frame)

    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        face_image = rgb_frame[y:y+h, x:x+w]  # RGB 이미지 사용
        face_pil = Image.fromarray(face_image)

        # 이미지 전처리 및 주파수 변환
        face_tensor = transform(face_pil).unsqueeze(0).to(device)
        face_freq = fft_transform(face_tensor)  # 주파수 변환 적용
        
        # 모델에 입력할 때 RGB와 주파수 정보를 따로 처리한 후 결합
        with torch.no_grad():
            combined_input = torch.cat((face_tensor, face_freq), dim=1)  # RGB + 주파수 결합
            outputs = model(face_tensor, face_freq)  # 모델에 입력

            # 소프트맥스 적용하여 확률값 얻기
            probabilities = F.softmax(outputs, dim=1)
            real_prob = probabilities[0][0].item()  # REAL 확률
            fake_prob = probabilities[0][1].item()  # FAKE 확률
            
            # 예측 레이블
            _, predicted = torch.max(outputs, 1)
            label = 'Real' if predicted.item() == 0 else 'Fake'

        # 예측 결과와 확률값 화면에 표시
        cv2.putText(frame, f'{label} - Real: {real_prob:.4f} Fake: {fake_prob:.4f}', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Face Spoofing Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
