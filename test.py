from torchvision import datasets, transforms
import matplotlib.pyplot as plt

data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

image, label = data[0]

plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"Label: {label}")
plt.show()