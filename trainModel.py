import dataset

dataset = DataSet("mushroom_dataset")
train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# デバイスの設定（CPU or GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 転移学習のため訓練済みモデルの読み込み
model = resnet101(pretrained=True)

with open("mushroom_classes.txt") as f:
    classes = [line.strip() for line in f.readlines()]

# 出力層を学習するクラス数に変更
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(classes))

# モデルをデバイスに転送
model = model.to(device)

# 損失関数とOptimizer の定義
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(epoch):
    total_loss = 0
    total_size = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.item() * data.size(0)
        total_size += data.size(0)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            now = datetime.datetime.now()
            print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
                now,
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), total_loss / total_size))

for epoch in range(10):
    train(epoch)

torch.save(model.state_dict(), 'model.pth')
