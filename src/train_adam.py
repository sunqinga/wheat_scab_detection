import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import model_utils
from model_utils import Net
import dataset as dl
from torch.autograd import Variable
import random
import numpy as np
import optimizer as opt


random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed(2050)


batchsz = 2
epochs = 150

if torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False


train_data = dl.My_dataset(data_path=r"E:\python_learn\Augument\train_data",
                           transform=transforms.Compose([transforms.ToTensor()]),
                           target_transform=dl.num_to_tensor)
train_loader = DataLoader(train_data, batch_size=batchsz, shuffle=True)

test_data = dl.My_dataset(data_path=r"E:\data\data\test",
                          transform=transforms.Compose([transforms.ToTensor()]),
                          target_transform=dl.num_to_tensor)
test_loader = DataLoader(test_data, batch_size=batchsz, shuffle=True)

model = Net()

nllloss = nn.CrossEntropyLoss()
loss_weight = 0.001
centerloss = model_utils.CenterLoss(2, 2)
if use_cuda:
    nllloss = nllloss.cuda()
    centerloss = centerloss.cuda()
    model = model.cuda()
criterion = [nllloss, centerloss]
optimizer4nn = opt.AdamOptimWrapper(model.parameters(), lr=0.001, wd=0, t0=15000, t1=25000)
optimizer4center = opt.AdamOptimWrapper(centerloss.parameters(), lr=0.5, wd=0, t0=15000, t1=25000)

print("Start Training!")
with open('150_test.txt', 'w') as f1:
    with open('150_train.txt', 'w') as f2:
        for epoch in range(epochs):
            optimizer4nn.step()
            optimizer = [optimizer4nn, optimizer4center]
            for i, (data, target) in enumerate(train_loader):
                x1 = data.index_select(1, torch.tensor([0]))
                x2 = data.index_select(1, torch.tensor([1]))
                x3 = data.index_select(1, torch.tensor([2]))
                if use_cuda:
                    data = data.cuda()
                    x1, x2, x3 = x1.cuda(), x2.cuda(), x3.cuda()
                    target = target.cuda()
                data, target, x1, x2, x3 = Variable(data), Variable(target), Variable(x1), Variable(x2), Variable(x3)
                feats, logits = model(x1, x2, x3)
                loss = criterion[0](logits, target) + loss_weight*criterion[1](target, feats)

                _, predicted = torch.max(logits.data, 1)
                accuracy = (target.data == predicted).float().mean()

                optimizer[0].zero_grad()
                optimizer[1].zero_grad()

                loss.backward()
                optimizer[0].step()
                optimizer[1].step()
                print('[epoch:%d,iter:%d] loss:%.03f|Acc:%.3f%'
                      % (epoch+1, (len(train_loader)*epoch+i+1), loss.item(), accuracy))
                f2.write('[epoch:%d,iter:%d] loss:%.03f|Acc:%.3f%'
                      % (epoch+1, (len(train_loader)*epoch+i+1), loss.item(), accuracy))
                f2.write('\n')
                f2.flush()

            print('Waiting Val!')
            for i, (data, target) in enumerate(test_loader):
                correct = 0
                total = 0
                x1 = data.index_select(1, torch.tensor([0]))
                x2 = data.index_select(1, torch.tensor([1]))
                x3 = data.index_select(1, torch.tensor([2]))
                if use_cuda:
                    data = data.cuda()
                    x1, x2, x3 = x1.cuda(), x2.cuda(), x3.cuda()
                    target = target.cuda()
                data, target, x1, x2, x3 = Variable(data), Variable(target), Variable(x1), Variable(x2), Variable(x3)
                ip1, logits = model(x1, x2, x3)
                _, predicted = torch.max(logits.data, 1)
                total += target.size(0)
                correct += (predicted == target.data).sum()

            print('Test Accuracy of the model on the tval images:%f %%' % (100 * correct / total))
            f1.write('epoch=%03d,acc=%.03f%%' % (epoch+1, (100 * correct / total)))
            f1.write('\n')
            f1.flush()

