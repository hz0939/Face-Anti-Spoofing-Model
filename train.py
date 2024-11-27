import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import cv2
from mtcnn import MTCNN


# 데이터셋 경로
dataset_path_real_ppa = '/content/drive/MyDrive/ProcessedDataset/PPA/real'
dataset_path_fake_ppa = '/content/drive/MyDrive/ProcessedDataset/PPA/fake'

dataset_path_real_rvf = '/content/drive/MyDrive/ProcessedDataset/RealvsFake/real'
dataset_path_fake_rvf = '/content/drive/MyDrive/ProcessedDataset/RealvsFake/fake'

# 모든 데이터셋의 경로 리스트
all_real_images = [
    dataset_path_real_ppa, dataset_path_real_rvf
]
all_fake_images = [
    dataset_path_fake_ppa, dataset_path_fake_rvf
]

# 이미지 전처리 함수 정의
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 이미지 크기 변경
    transforms.ToTensor(),          # 이미지 데이터를 텐서로 변환
])


# 데이터셋 클래스 정의
class FaceSpoofingDataset(Dataset):
    def __init__(self, real_dirs, fake_dirs, transform=None):
        self.real_dirs = real_dirs
        self.fake_dirs = fake_dirs
        self.transform = transform

        self.real_images = []
        for real_dir in self.real_dirs:
            self.real_images.extend([os.path.join(real_dir, f) for f in os.listdir(real_dir)])

        self.fake_images = []
        for fake_dir in self.fake_dirs:
            self.fake_images.extend([os.path.join(fake_dir, f) for f in os.listdir(fake_dir)])

    def __len__(self):
        return len(self.real_images) + len(self.fake_images)

    def __getitem__(self, idx):
        if idx < len(self.real_images):
            img_path = self.real_images[idx]
            label = 0  # REAL
        else:
            img_path = self.fake_images[idx - len(self.real_images)]
            label = 1  # FAKE

        image = Image.open(img_path).convert('RGB')  # 3채널로 변환

        if self.transform:
            image = self.transform(image)

        return image, label
def fft_transform(image):
    # 3채널 이미지를 그레이스케일로 변환 (평균을 이용하여)
    gray = image.mean(dim=1, keepdim=True)  # RGB -> Grayscale (채널 축소)

    # 푸리에 변환
    f = torch.fft.fft2(gray)  # 푸리에 변환
    fshift = torch.fft.fftshift(f)  # 주파수 영역 중앙으로 이동
    magnitude_spectrum = torch.abs(fshift)  # 크기 스펙트럼 계산

    # 크기 스펙트럼을 3채널로 확장
    return magnitude_spectrum.repeat(1, 3, 1, 1)  # (batch_size, 3, H, W)
import torch.nn as nn
import torch.nn.functional as F

class CDCN_Spatial_Frequency(nn.Module):
    def __init__(self):
        super(CDCN_Spatial_Frequency, self).__init__()

        # 공간적 정보 처리 네트워크 (RGB 이미지)
        self.conv1_rgb = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2_rgb = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3_rgb = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # 주파수 정보 처리 네트워크 (주파수 영역)
        self.conv1_fft = nn.Conv2d(3, 32, kernel_size=3, padding=1)
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
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

# 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CDCN_Spatial_Frequency().to(device)

# 손실 함수와 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 훈련 루프 정의
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # 푸리에 변환을 통해 주파수 정보 얻기
            freq_images = fft_transform(images)

            # RGB와 주파수 이미지를 각각 처리한 후 결합
            optimizer.zero_grad()
            outputs = model(images, freq_images)  # (RGB, 주파수)

            # 손실 계산
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 정확도 계산
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct/total:.2f}%")

# 데이터셋 객체 생성
train_dataset = FaceSpoofingDataset(
    real_dirs=all_real_images,
    fake_dirs=all_fake_images,
    transform=transform  # 전처리 함수 추가
)

# DataLoader 설정
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 모델 훈련 시작
train_model(model, train_loader, criterion, optimizer, epochs=10)

# 모델 저장
torch.save(model.state_dict(), '/content/drive/MyDrive/ProcessedDataset/model/FFT_V4')
