import cv2
import numpy as np
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
    
    def resize_to_28x28(self, img):

        img_h, img_w = img.shape
        dim_size_max = max(img.shape)

        if dim_size_max == img_w:
            im_h = (26 * img_h) // img_w
            if im_h <= 0 or img_w <= 0:
                print("Invalid Image Dimention: ", im_h, img_w, img_h)
            tmp_img = cv2.resize(img, (26,im_h),0,0,cv2.INTER_NEAREST)
        else:
            im_w = (26 * img_w) // img_h
            if im_w <= 0 or img_h <= 0:
                print("Invalid Image Dimention: ", im_w, img_w, img_h)
            tmp_img = cv2.resize(img, (im_w, 26),0,0,cv2.INTER_NEAREST)

        out_img = np.zeros((28, 28), dtype=np.ubyte)

        nb_h, nb_w = out_img.shape
        na_h, na_w = tmp_img.shape
        y_min = (nb_w) // 2 - (na_w // 2)
        y_max = y_min + na_w
        x_min = (nb_h) // 2 - (na_h // 2)
        x_max = x_min + na_h

        out_img[x_min:x_max, y_min:y_max] = tmp_img

        return out_img

    def infer(self,image):

        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        img = cv2.bitwise_not(img)
        img = self.resize_to_28x28(img)
        cv2.imwrite("rezised.jpg", img)
        
        transform = transforms.ToTensor()
        img_tensor = transform(img).to(self.device)
        img_tensor =img_tensor.unsqueeze(0)

        print(img_tensor)

        with torch.inference_mode():
            pred = self.model(img_tensor)
            pred_softmax = torch.softmax(pred.squeeze(), dim=0)
            print(pred_softmax)
            print(pred_softmax.argmax(dim=0))

        #return pred

if __name__ == "__main__":

    #classifier = classify("D:\sudoku\model\MNIST_Classifier.pt")
    #classifier = classify("D:\sudoku\model\MNIST_Classifier_noStetDict.pt")
    classifier = classify("D:\sudoku\model\MNIST_Classifier_50Epochs.pt")

    #img = "D:\sudoku\edge_detection\squares\square2.jpg" #
    # img = "D:\sudoku\edge_detection\squares\square46.jpg" #9
    #img= "D:\sudoku\edge_detection\squares\square48.jpg" #2
    img= "D:\sudoku\edge_detection\squares\square36.jpg" #8 -> 0
    # img = "D:\sudoku\edge_detection\squares\square52.jpg" #Empty
    #img = "D:\sudoku\edge_detection\squares\square26.jpg" #3
    # img = "D:\sudoku\edge_detection\squares\square16.jpg" #5
    #img = "D:\sudoku\edge_detection\squares\square4.jpg" #7 -> 2
    # img = "D:\sudoku\edge_detection\squares\square2.jpg" #4
    #img = "D:\sudoku\MNIST_Test\stesT1.jpg"
    classifier.infer(img)