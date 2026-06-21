import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from my_model import SimpleCNN

def train_and_save():
    model = SimpleCNN()
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=64, shuffle=True
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    # 1 에포크만 진행
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # 100 배치 단위로 로스 평균 기록
        if (batch_idx + 1) % 100 == 0:
            print(f"Batch {batch_idx + 1:03d} | Average Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), 'my_weights.pth')
    print("가중치 저장 완료: my_weights.pth\n")

if __name__ == "__main__":
    if not os.path.exists('my_weights.pth'):
        train_and_save()

    print("alpha-beta-CROWN 검증 시작")
    abcrown_path = "alpha-beta-CROWN/complete_verifier/abcrown.py"
    
    os.environ['PYTHONPATH'] = os.getcwd()
    
    os.system(f"python {abcrown_path} --config my_model.yaml")