import cv2
import torch
import torch.nn as nn
from torchvision import transforms

class tinyVGG(nn.Module):

  def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
    super().__init__()

    self.block1 = nn.Sequential(
        nn.Conv2d(in_channels= input_shape,
                  out_channels = hidden_units,
                  kernel_size = 3,
                  stride = 1,
                  padding = 1),
        nn.ReLU(),
        nn.Conv2d(in_channels= hidden_units,
                  out_channels = hidden_units,
                  kernel_size = 3,
                  stride = 1,
                  padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size =2,
                    stride = 2)
    )

    self.block2 = nn.Sequential(
        nn.Conv2d( in_channels = hidden_units,
                  out_channels = hidden_units,
                  kernel_size = 3,
                  stride = 1,
                  padding = 1),
        nn.ReLU(),
        nn.Conv2d( in_channels = hidden_units,
                  out_channels = hidden_units,
                  kernel_size = 3,
                  stride = 1,
                  padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size= 2,
                     stride = 2)
        )
    
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features = hidden_units * 7 * 7,
                  out_features = output_shape)
    )
  

  def forward(self, x: torch.Tensor):

    x = self.block1(x)
    x = self.block2(x)
    x = self.classifier(x)

    return x

class classify():

    def __init__(self, model_path):
        self.device =  "cuda" if torch.cuda.is_available() else "cpu"
        self.model = tinyVGG(input_shape = 1,
                      hidden_units = 10,
                      output_shape = 10).to(self.device, memory_format=torch.channels_last)
        self.model.load_state_dict(torch.load(f = model_path, map_location =self.device))
        self.model.eval()
        #self.classes = 
        print(self.model)

    def infer(self,image):

        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        img = cv2.resize(img, (28, 28))
        cv2.imwrite("rezised.jpg", img)
        transform = transforms.ToTensor()
        
        img_tensor = transform(img).to(self.device)
        img_tensor =img_tensor.unsqueeze(0)
        print(img_tensor.shape)

        with torch.inference_mode():
            pred = self.model(img_tensor)
            pred_softmax = torch.softmax(pred.squeeze(), dim=0)
            print(pred_softmax)
            print(pred_softmax.argmax(dim=0))

        #return pred

if __name__ == "__main__":

    classifier = classify("D:\sudoku\model\MNIST_Classifier.pt")
    #classifier = classify("D:\sudoku\model\MNIST_Classifier_noStetDict.pt")

    img = "D:\sudoku\edge_detection\squares\square2.jpg"
    img = "D:\sudoku\edge_detection\squares\square46.jpg"
    classifier.infer(img)