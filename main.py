"""File used to perform image classification."""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from ImageNet import ImageNet

# Training settings
parser = argparse.ArgumentParser(description='PyTorch ImageNet Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=128, metavar='N',
                    help='number of epochs to train (default: 128)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before' +
                    'logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def imshow(img):
    """Show image."""
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

trainSet = ImageNet('imagenet56x56_release', transform, None, True, False)
train_loader = torch.utils.data.DataLoader(trainSet,
                                           batch_size=args.batch_size,
                                           shuffle=True, **kwargs)

testSet = ImageNet('imagenet56x56_release', transform, None, False, False)
test_loader = torch.utils.data.DataLoader(testSet, batch_size=args.batch_size,
                                          shuffle=True, **kwargs)

kaggleSet = ImageNet('imagenet56x56_release', transform, None, False, True)
kaggle_loader = torch.utils.data.DataLoader(kaggleSet,
                                            batch_size=args.batch_size,
                                            shuffle=False, **kwargs)


class Net(nn.Module):
    """CNN class to perform image classification."""

    def __init__(self):
        """Initialize the CNN."""
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv7 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=3)
        self.conv10 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 100)

    def forward(self, x):
        """Perform the classification."""
        # 56
        x = F.relu(self.conv1(x))  # 54
        x = F.relu(self.conv2(x))  # 52
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv3(x))  # 50
        x = F.relu(self.conv4(x))  # 48
        x = F.max_pool2d(x, 2)  # 24
        x = F.relu(self.conv5(x))  # 22
        x = F.relu(self.conv6(x))  # 20
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv7(x))  # 18
        x = F.relu(self.conv8(x))  # 16
        x = F.max_pool2d(x, 2)  # 8
        x = F.relu(self.conv9(x))  # 6
        x = F.relu(self.conv10(x))  # 4
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 1024)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return F.log_softmax(x), F.softmax(x)


model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train(epoch):
    """Train the CNN."""
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output, probvec = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def test(epoch):
    """Test the CNN."""
    model.eval()
    test_loss = 0
    correct = 0
    for data, target, paths in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output, probvec = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1]  # get the index of the max log-prob
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader)
    # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def kaggle_test():
    """Make evaluation files ready for Kaggle upload."""
    evalfile = open('kaggle_output.csv', 'w')
    first_line = "id,class_000,class_001,class_002,class_003,class_004,class_005,class_006,class_007,class_008,class_009,class_010,class_011,class_012,class_013,class_014,class_015,class_016,class_017,class_018,class_019,class_020,class_021,class_022,class_023,class_024,class_025,class_026,class_027,class_028,class_029,class_030,class_031,class_032,class_033,class_034,class_035,class_036,class_037,class_038,class_039,class_040,class_041,class_042,class_043,class_044,class_045,class_046,class_047,class_048,class_049,class_050,class_051,class_052,class_053,class_054,class_055,class_056,class_057,class_058,class_059,class_060,class_061,class_062,class_063,class_064,class_065,class_066,class_067,class_068,class_069,class_070,class_071,class_072,class_073,class_074,class_075,class_076,class_077,class_078,class_079,class_080,class_081,class_082,class_083,class_084,class_085,class_086,class_087,class_088,class_089,class_090,class_091,class_092,class_093,class_094,class_095,class_096,class_097,class_098,class_099\n"
    evalfile.write(first_line)
    for data, paths in kaggle_loader:
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        output, probvecs = model(data)
        probvecs = probvecs.data.cpu().numpy()
        for i in range(len(paths)):
            path = paths[i]
            path_elems = path.split('/')
            name = path_elems[len(path_elems)-1].split('.')
            name = name[0]
            line = name
            for j in range(100):
                line += "," + str(probvecs[i][j])
            line += "\n"
            evalfile.write(line)
    evalfile.close()


for epoch in range(1, args.epochs):
    train(epoch)
    if epoch == args.epochs:
        addit = True
    test(epoch)

kaggle_test()
