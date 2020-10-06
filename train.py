import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import multiprocessing

max_workers = multiprocessing.cpu_count() - 2 if multiprocessing.cpu_count() > 2 else 1

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []

    in_channels = 3

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def main():
    model = VGG(make_layers(cfg['A']), num_classes=1000, init_weights=True)

    workers = max_workers

    trainTransforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    testTransforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor()])

    # transform = transforms.ToTensor()

    data_path = "/home/duy/Documents/Pytorch_study/VGG16-PyTorch/resized"
    trainData = datasets.ImageFolder(root=data_path, transform=trainTransforms)
    testData = datasets.ImageFolder(root=data_path, transform=testTransforms)

    trainLoader = DataLoader(trainData, batch_size=32, shuffle=True, num_workers=workers)
    testLoader = DataLoader(testData, batch_size=64, shuffle=True, num_workers=workers)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print('-----------------')

    start_time = time.time()
    # Training
    epochs = 1
    # Trackers
    trainLosses = []
    testLosses = []
    trainCorrects = []
    testCorrects = []

    for i in range(epochs):

        trainCorrect = 0
        testCorrect = 0

        for batchNumber, (imageTrain, labelTrain) in enumerate(trainLoader):
            batchNumber += 1

            yPrediction = model(imageTrain)
            loss = criterion(yPrediction, labelTrain)

            predicted = torch.max(yPrediction.data, 1)[1]

            batchCorrect = (predicted == labelTrain).sum()

            trainCorrect += batchCorrect

            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batchNumber % 10 == 0:
                print(f'epoch: {i:2}  batch: {batchNumber:4} [{100 * batchNumber:6}/60000]  loss: {loss.item():10.8f} accuracy: {trainCorrect.item() * 100 / (100 * batchNumber):7.3f}%')
        # Update train loss & accuracy for the epoch
        trainLosses.append(loss)
        trainCorrects.append(trainCorrect)

        with torch.no_grad():
            for b, (imageTest, labelTest) in enumerate(testLoader):
                # Apply the model
                yValue = model(imageTest.view(500, -1))  # Here we flatten imageTest

                predicted = torch.max(yValue.data, 1)[1]
                testCorrect += (predicted == labelTest).sum()
        loss = criterion(yValue, labelTest)
        testLosses.append(loss)
        testCorrects.append(testCorrect)
        print(f'Test accuracy: {testCorrect.item()}/{len(testData)} = {testCorrect.item() * 100 / (len(testData)):7.3f}%')
    total_time = time.time() - start_time
    print(f'Duratiopn: {total_time / 60} mins')


main()
