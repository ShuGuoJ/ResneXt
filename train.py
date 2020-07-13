import torch
from torch import nn, optim
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
import numpy as np
from Resnet50 import Resnet50
from visdom import Visdom
from ResneXt import ResneXt

def global_norm(parameters):
    norms = []
    for p in parameters:
        norms.append(p.grad.norm())
    ans = None
    for n in norms:
        if not ans:
            ans = n.square()
        else:
            ans += n.square()
    return ans.sqrt().item()

'''模型训练'''

batch_size = 256
epochs = 200
learning_rate = 1e-2
seed = 123456
torch.manual_seed(seed)
device = torch.device("cuda:0")

'''cifar100数据集'''
transform = transforms.Compose([
    transforms.RandomCrop((32, 32), 3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
train_loader = DataLoader(datasets.CIFAR10("../data", train=True, transform=transform), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(datasets.CIFAR10("../data", train=False, transform=transform), batch_size=batch_size, shuffle=True)

'''数据可视化'''
# imgs, _ = next(iter(train_loader))
# viz = Visdom()
# viz.images(imgs, win="train")
# exit(0)

''' two classification'''
# net = Resnet50(10)
net = ResneXt(10)
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer,100, 0.5)
net.to(device)
criterion.to(device)
total_loss = []
viz = Visdom()
viz.line([[0., 0.]], [0.], win="train", opts=dict(title="train&&val loss",
                                                  legend=['train', 'val']))
viz.line([0.], [0.], win='acc', opts =dict(title='accuracy',
                                           legend=['acc']))
for epoch in range(epochs):
    net.train()
    total_loss.clear()
    for batch, (input, label) in enumerate(train_loader):
        input, label = input.to(device), label.to(device)
        logits = net(input)
        loss = criterion(logits, label)
        total_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        l2 = nn.utils.clip_grad_norm_(net.parameters(), 5)
        optimizer.step()


        if batch%50==0:
            print("epoch:{} batch:{} loss:{} lr:{} norm_2:{}".format(epoch, batch, loss.item(), optimizer.state_dict()['param_groups'][0]['lr'], l2))
    scheduler.step()

    net.eval()
    correct = 0
    test_loss = 0
    for input, label in test_loader:
        input, label = input.to(device), label.to(device)
        logits = net(input)

        '''crossentropy'''
        test_loss += criterion(logits, label).item() * input.shape[0]
        pred = logits.argmax(dim=1)

        correct += pred.eq(label).float().sum().item()
    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))
    viz.line([[float(np.mean(total_loss)), test_loss]], [epoch], win="train", update="append")
    viz.line([acc], [epoch], win='acc', update='append')
    # if (epoch+1)%5==0:
    #     torch.save(net.state_dict(), "resnet18_{}.pkl".format(epoch))


