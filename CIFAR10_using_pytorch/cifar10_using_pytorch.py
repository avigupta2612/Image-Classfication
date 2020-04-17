
"""Cifar10_using_Pytorch.ipynb


# Import libraries
"""

import torch
import torch.nn as nn
import torch.nn.functional as nn
import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

"""## Check for CUDA"""

train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print("Training on GPU")
else:
    print("Training on CPU")

"""## Download and transform dataset"""

batch_size=20
transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,0.5,0.5),
                                                   (0.5,0.5,0.5))])

train_data=datasets.CIFAR10('data',train=True,download=True, transform=transform)
test_data=datasets.CIFAR10('data',train=False,download=True, transform=transform)

val_size=0.2
num_train=len(train_data)
indices=list(range(num_train))
np.random.shuffle(indices)
split= int(np.floor(val_size*num_train))
train_idx,val_idx= indices[split:], indices[:split]
train_sampler= SubsetRandomSampler(train_idx)
val_sampler= SubsetRandomSampler(val_idx)
train_loader= torch.utils.data.DataLoader(train_data,batch_size=batch_size,
                                          sampler=train_sampler)
val_loader= torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                        sampler=val_sampler)
test_loader= torch.utils.data.DataLoader(test_data, batch_size=batch_size)

# specify the image classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

"""## Visualizing a batch of images"""

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))

dataiter=iter(train_loader)
images, labels=dataiter.next()
print(images[0].shape)
images=images.numpy()
fig=plt.figure(figsize=(25,5))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(classes[labels[idx]])

"""## Defining the model"""

import torch.nn as nn
import torch.nn.functional as f
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1= nn.Conv2d(3,16,3,padding= 1)
        self.conv2= nn.Conv2d(16,32,3,padding= 1)
        self.conv3= nn.Conv2d(32,32,3,padding= 1)
        self.pool= nn.MaxPool2d(2,2)
        self.dropout= nn.Dropout(0.25)
        self.fc1= nn.Linear(32*4*4, 256)
        self.fc2= nn.Linear(256, 10)
    def forward(self,x):
        x= self.pool(f.relu(self.conv1(x)))
        x= self.pool(f.relu(self.conv2(x)))
        x= self.pool(f.relu(self.conv3(x)))
        x= self.dropout(x)
        x= x.view(-1,32*4*4)
        x= f.relu(self.fc1(x))
        x= self.fc2(x)
        return x

model=Net()
print(model)

criterion= nn.CrossEntropyLoss()
optimizer= torch.optim.SGD(model.parameters(), lr= 0.01)

"""## Training"""

if train_on_gpu:
    model.cuda()

epochs=15
valid_loss_min=np.Inf
for e in range(1,epochs+1):
    train_loss=0
    valid_loss=0
    model.train()
    #Training 
    for images, targets in train_loader:
        optimizer.zero_grad()
        if train_on_gpu:
            images,targets= images.cuda(), targets.cuda()
        output= model(images)
        loss= criterion(output,targets)
        loss.backward()
        optimizer.step()
        train_loss+= loss.item()*images.size(0)
    #Validating
    model.eval()
    for images,targets in val_loader:
        if train_on_gpu:
            images, targets = images.cuda(), targets.cuda()
        
        output= model(images)
        loss= criterion(output, targets)
        
        valid_loss+= loss.item()*images.size(0)
        
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(val_loader.dataset)
        
    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        e, train_loss, valid_loss))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'model_cifar.pt')
        valid_loss_min = valid_loss

model.load_state_dict(torch.load('model_cifar.pt'))

"""## Testing"""

# track test loss
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval()
# iterate over test data
for data, target in test_loader:
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the batch loss
    loss = criterion(output, target)
    # update test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)    
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    # calculate test accuracy for each object class
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# average test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))

# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()
images.numpy()

# move model inputs to cuda, if GPU available
if train_on_gpu:
    images = images.cuda()

# get sample outputs
output = model(images)

images=images.cpu()
# convert output probabilities to predicted class
_, preds_tensor = torch.max(output, 1)

preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                 color=("green" if preds[idx]==labels[idx].item() else "red"))

