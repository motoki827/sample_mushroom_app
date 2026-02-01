# 以下を「model.py」に書き込み
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

net = models.resnet101(pretrained=True)  # 訓練済みのモデルを読み込み
with open("mushroom_classes.txt") as f:  # ラベルの読み込み
    classes = [line.strip() for line in f.readlines()]

# 出力層を学習時と同じクラス数に変更
net.fc = nn.Linear(net.fc.in_features, len(classes))

# 学習済みのバラメータを反映
state_dict = torch.load("model.pth", map_location=torch.device("cpu"))
net.load_state_dict(state_dict)

def predict(img):
    # 以下の設定はこちらを参考に設定: https://pytorch.org/hub/pytorch_vision_resnet/
    transform = transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]
                                        )
                                    ])

    # モデルへの入力
    img = transform(img)
    x = torch.unsqueeze(img, 0)  # バッチ対応

    # 予測
    net.eval()
    y = net(x)

    # 結果を返す
    y_prob = torch.nn.functional.softmax(torch.squeeze(y))  # 確率で表す
    sorted_prob, sorted_indices = torch.sort(y_prob, descending=True)  # 降順にソート
    return [(classes[idx], prob.item()) for idx, prob in zip(sorted_indices, sorted_prob)]
