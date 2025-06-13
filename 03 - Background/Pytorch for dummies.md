## Quickstart
From: https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

#### Working with Data
Pytorch has 2 primitives:
- `torch.utils.data.DataLoader`
	- wraps an iterable around the dataset
- `torch.utils.data.Dataset`
	- stores samples and their labels
	- each has 2 arguments:
		- transform
			- modify samples
		- target_transform
			- modify labels

``` Python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
```

domain specific libraries exist like:
- TorchText
- TorchVision
- TorchAudio

sample datasets exist in `torchvision.datasets`
- [Sample Datasets](https://docs.pytorch.org/vision/stable/datasets.html)

``` Python
# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
```

We pass our `Dataset` as an argument to `DataLoader`. This wraps an iterable around the dataset, and supports automatic batching, sampling, shuffling, and multiprocess data loading.

```Python
# Define batch size here.
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
```

More: [[Pytorch for dummies#Datasets & DataLoaders|Datasets & DataLoaders]]

#### Creating Models
To define a NN:
1. Create a class that inherits from `nn.Module`. 
2. Define layers in the `__init__` function
3. Specify how data is passed through the net in `forward`
4. Move operations to the ==accelerator== to use GPUs for speedup (with CUDA, MPS, MTIA, or XPU)
	- If available, used, else fallback to CPU

```Python
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
```

More: [[Pytorch for dummies#Build model|Build Model]]

#### Optimizing Model Params
Training requires:
- loss function
- optimizer

```Python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```

Each loop, model makes predictions on training dataset batch, then backpropagates prediction error to adjust model parameters.

``` Python
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
```

We check model's performance against test dataset to ensure its learning:

``` Python
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

Repeat training process over several ==epochs==. Print model accuracy and loss at each epoch. Ideally accuracy $\uparrow$, loss $\downarrow$ (duh)

``` Python
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
```

More: [[Pytorch for dummies#Optimization|Optimization]]

#### Saving Models
Done by serializing internal state dictionary (containing model params)
``` Python
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
```

#### Loading Models
Need to re-create model structure and load state dictionary into it
```
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))
```

The model can now be used to make predictions:
``` Python
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
```

More: [[Pytorch for dummies#Save & Load Model|Save & Load Model]]

## Tensors
From: [Tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)

==Tensor==: a specialized data structure, similar to an array or matrix. Used to encode inputs, outputs, and params of a model.

``` Python
import torch
import numpy as np
```

#### Initializing a Tensor
##### Directly from data:
Automatically infers data type
``` Python
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
```

##### From a NumPy array
``` Python
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
```

##### From another tensor:
new tensor retains (shape, datatype) of arg tensor unless overridden
``` Python
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")
```

##### With random or constant values:
`shape` is a tuple of tensor dimensions. Below, it determines the output tensor's dimensionality
``` Python
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
```

#### Tensor Attributes
Attributes exist describing:
- shape
- datatype
- host storage device

``` Python
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
```

#### Tensor Operations
comprehensive documentation is found here: [Torch Documentation](https://pytorch.org/docs/stable/torch.html)
- If using Colab, allocate an accelerator thru Runtime > Change runtime type > GPU.
- Then explicitly move tensors to the accelerator with `.to`

``` Python
# We move our tensor to the current accelerator if available
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())
```
##### Standard numpy-like indexing and slicing:
``` Python
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)
```

##### Joining tensors
Use `torch.cat` to concatenate a sequence of tensors along a dimension. `torch.stack` also works, but is subtly different
``` Python
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
```

##### Arithmetic operations
``` Python
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
```

##### Single-element tensors
can convert to a Python numerical value with `item()`:
```Python
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))
```

##### In-place operations
Operations that store the result into the operand are called in-place.
- Denoted by a `_` suffix
- Ex: `x.copy_(y)`, `x.t_()`, will change `x`

``` Python
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)
```


> [!NOTE] NOTE
> In-place operations save some memory, but can be problematic when computing derivatives because of an immediate loss of history. Hence, their use is discouraged.

#### Bridge with NumPy
##### Tensor to NumPy array
``` Python
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
```

Changing the tensor changes the NumPy array
``` Python
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
```

##### NumPy array to Tensor
``` Python
n = np.ones(5)
t = torch.from_numpy(n)
```

Changing the NumPy array changes the tensor
``` Python
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
```

## Datasets & DataLoaders
From: [Datasets & DataLoaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)

Code to handle data samples gets messy and unmaintainable
- decouple dataset code from model training code
	- $\uparrow$ readability &  modularity

Pytorch's preloaded datasets found in:
- [Image Datasets](https://pytorch.org/vision/stable/datasets.html)
- [Text Databases](https://pytorch.org/text/stable/datasets.html)
- [Audio Datasets](https://pytorch.org/audio/stable/datasets.html)

#### Loading a Dataset
Shown with the ==Fashion-MNIST== dataset from TorchVision
We load it with the following parameters:
- `root`: the path where the train/test data is stored
- `train`: specifies training or test dataset
- `download=True`: download the data from the internet if not available at `root`
- `transform` & `target_transform`: specify the feature and label transformations

``` Python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
```

#### Iterating and Visualizing the Dataset
Can index `Datasets` manually like a list: `training_data[index]`.
- can pipe into `matplotlib` to visualize samples in training data

``` Python
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
```
![[Pasted image 20250612211408.png]]

#### Creating a Custom Dataset for your files
Must implement 3 functions:
1. `__init__`
	- Run once on instantiating Dataset object
	- Init directory with images, annotations file, and both transforms
2. `__len__`
	- Returns number of samples in dataset
3. `__getitem__`
	- Load and return a sample from the dataset at index `idx`.
	- Based on `idx`:
		- IDs the image's location on disk
		- Converts it to a tensor  with `decode_image`
		- Retrieves corresponding label from csv data in `self.img_labels`
		- Calls applicable transform functions
		- Returns ({tensor image}, {corresponding label})

look at this sample implementation:
- FashionMNIST images stored in `img_dir`
- labels stored separately in a CSV file `annotations_file`

``` Python
import os
import pandas as pd
from torchvision.io import decode_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```

Where `labels.csv` looks something like:
``` CSV
tshirt1.jpg, 0
tshirt2.jpg, 0
......
ankleboot999.jpg, 9
```

#### Preparing your data for training with DataLoaders
The `Dataset` retrieves our dataset’s features and labels one sample at a time. While training, we:
- Pass samples in “minibatches”,
- Reshuffle the data at every epoch to reduce model overfitting,
- Use Python’s `multiprocessing` to speed up data retrieval.

`DataLoader` abstracts this to an API call.

```Python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
```

#### Iterate through the DataLoader
We load our dataset into `DataLoader` and can iterate freely.
Each iteration returns a batch of `train_features` and `train_labels` (of `batch_size=64`)
- Because we specified `shuffle=True`,  data is shuffled after iterating over all batches
- See also: [Samplers](https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler)

``` Python
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
```

## Transformers
Used to transform data into a form required to train ML algorithms
- Recall `transform` & `target_transform` from [[Pytorch for dummies#Quickstart|Quickstart]]

Common transforms: [torchvision.transforms](https://pytorch.org/vision/stable/transforms.html)

Our input:
- FashionMNIST features are .PILs
- Labels are ints
Our model needs :
- FashionMNIST features as normalized tensors
- Labels as one-hot encoded tensors
We achieve this with:
- `ToTensor`
	- "Converts a PIL image or NumPy `ndarray` into a `FloatTensor`. and scales the image’s pixel intensity values in the range \[0., 1\.]"
- `Lambda`
	- "Apply any user-defined lambda function. Here, we define a function to turn the integer into a one-hot encoded tensor. It first creates a zero tensor of size 10 (the number of labels in our dataset) and calls [scatter_](https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html) which assigns a `value=1` on the index as given by the label `y`."
``` Python
target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
```

``` Python
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
```

## Build Model
[FINISH ME]

## Autograd
[FINISH ME]

## Optimization
[FINISH ME]

## Save & Load Model
How to persist model state w/ saving, loading, and running model predictions?

```Python
import torch
import torchvision.models as models
```

#### Saving and Loading Model Weights
Pytorch stores learned parameters in an internal state dictionary, `state_dict`
- can be persisted with `torch.save`

``` Python
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')
```

To load model weights:
- instantiate the same model
- call `load_state_dict()` (see below)
- Set `weights_only=True` to limit functions executed during unpickling to only those necessary for loading weights.
	-  this is considered a best practice

``` Python
model = models.vgg16() # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
model.eval()
```


> [!NOTE] NOTE
> Be sure to call `model.eval()` method before inferencing to set the dropout and batch normalization layers to evaluation mode. Failing to do this will yield inconsistent inference results.

##### Saving and Loading Models with Shapes
If we want to additionally save the model itself, we can pass `model` (& NOT `model.state_dict()`)
``` Python
torch.save(model, 'model.pth')
```

Can be loaded as described in [Saving & loading torch.nn.Modules](https://pytorch.org/docs/main/notes/serialization.html#saving-and-loading-torch-nn-modules):
- Note saving `state_dict` is considered best practice
- But here we use `weights_only=False` bc this involves loading the model, which is a legacy use case for `torch.save`
```Python
model = torch.load('model.pth', weights_only=False),
```


> [!NOTE] NOTE
> This approach uses Python [pickle](https://docs.python.org/3/library/pickle.html) module when serializing the model, thus it relies on the actual class definition to be available when loading the model.
