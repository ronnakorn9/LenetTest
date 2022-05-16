from turtle import forward
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten


class Lenet(Module):
    """
        defining architecture 
    """

    def __init__(self, numChannels, classes) -> None:
        """
            numChannels: number of color channel, 1 = greyscale, 3 = RGB
            classes: number of unique class labels in dataset
        """
        # some variable
        numChannels_12 = 20
        numChannels_23 = 50

        # call the parent constructor
        super(Lenet, self).__init__()

        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(in_channels=numChannels,
                            out_channels=numChannels_12, kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = Conv2d(in_channels=numChannels_12,
                            out_channels=numChannels_23, kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize first (and only) set of FC => RELU layers
        self.full_con1 = Linear(in_features=800, out_features=500)
        self.relu3 = ReLU()

        # initialize our softmax classifier
        self.full_con2 = Linear(in_features=500, out_features=classes)
        self.softmax = LogSoftmax(dim=1)

    def forward(self,x):
        # pass the input through our first set of CONV => RELU => POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        # pass the input through our 2nd set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # flatten the output from the previous layer and pass it through our only set of FC => RELU layers
        x = flatten(x, 1)
        x = self.full_con1(x)
        x = self.relu3(x)

        # pass the output to our softmax classifier to get our output predictions
        x = self.full_con2(x)
        output = self.logSoftmax(x)

        return output

