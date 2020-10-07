import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import multiprocessing
import torch.utils.model_zoo as model_zoo

# import vgg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
max_workers = multiprocessing.cpu_count() - 2 if multiprocessing.cpu_count() > 2 else 1
data_path = "/home/duy/Documents/Pytorch_study/VGG16-PyTorch/resized"

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=4, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
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


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


# static_model = models.vgg16(pretrained=True).cuda()
static_model = VGG(make_layers(cfg['D']), num_classes=4, init_weights=True).cuda()
static_criterion = nn.CrossEntropyLoss()
static_optimizer = torch.optim.Adam(static_model.parameters(), lr=0.001)
static_epochs = 100


def train_model(model, criterion, optimizer, epochs):
    workers = max_workers

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor()])

    train_data = datasets.ImageFolder(root=data_path, transform=train_transforms)
    test_data = datasets.ImageFolder(root=data_path, transform=test_transforms)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=workers)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=workers)

    print('-----------------STARTING-----------------')

    start_time = time.time()

    # Trackers
    train_losses = []
    test_losses = []
    train_corrects = []
    test_corrects = []

    log_file = open("log.txt", "w")
    log_file = open("log.txt", "a")

    for i in range(epochs):

        train_correct = 0
        test_correct = 0

        for batch_number, (image_train, label_train) in enumerate(train_loader):
            batch_number += 1

            # print (image_train.shape)

            image_train = image_train.to(device)
            label_train = label_train.to(device)
            label_predicted = model(image_train)
            loss = criterion(label_predicted, label_train)

            # print (label_predicted.data.shape)

            predicted = torch.max(label_predicted.data, 1)[1]

            # print ("label")
            # print (label_train)
            # print ("predicted")
            # print (predicted)

            batch_correct = (predicted == label_train).sum()

            train_correct += batch_correct

            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            log_for_train = f'epoch: {i + 1}| batch_number: {batch_number}| loss: {loss.item():10.8f}| accuracy: {train_correct.item() * 100 / (100 * batch_number):7.3f}%'

            if batch_number % 50 == 0:
                print(log_for_train)
                log_file.writelines(log_for_train)

            # Update train loss & accuracy for the epoch
            train_losses.append(loss)
            train_corrects.append(train_correct)

        with torch.no_grad():
            for batch, (image_test, label_test) in enumerate(test_loader):
                image_test = image_test.to(device)
                label_test = label_test.to(device)
                label_value = model(image_test)

                predicted = torch.max(label_value.data, 1)[1]
                test_correct += (predicted == label_test).sum()

        loss = criterion(label_value, label_test)

        test_losses.append(loss)
        test_corrects.append(test_correct)

        log_for_test = f'Test accuracy: {torch.floor_divide(test_correct * 100, len(test_data))}%'
        print(log_for_test)
        log_file.writelines(log_for_test)

    total_time = time.time() - start_time

    log_for_end = f'Duration: {total_time / 60} minutes'

    print(log_for_end)
    log_file.writelines(log_for_end)
    log_file.close()


train_model(static_model, static_criterion, static_optimizer, static_epochs)
