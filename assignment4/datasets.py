import pandas as pd
from torch.utils.data import Dataset
import os
from PIL import Image

# We need to create a custom dataset as shown here https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# A custom dataset inherits the PyTorch class Dataset and requires to implement three functions __init__, __len__ and __getitem__


class MNIST_Test(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.csv = pd.read_csv(csv_file)
        self.image_id = self.csv.id

        self.img_dir = img_dir
        self.transform = transform

    # Simply returns the length of the dataset
    def __len__(self):
        return len(self.csv)

    # Loads and returns a sample from the dataset at the given index. 
    # Based on this index, it indentifies the images location on disk and performs some defined transformations.
    def __getitem__(self, index):
        img_path = os.path.join(
            self.img_dir,
            self.csv.iloc[index, 1]  # 'image' column
        )
        image = Image.open(img_path)
        image_id = self.image_id[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, image_id


class MNIST_Train_Datasource(Dataset):
    def __init__(self, csv_file_processed, csv_file, img_dir, transform=None):
        # try catch block, because we dont want to create the processed file on every instantiation
        # just check if it was already created
        try:
            self.csv = pd.read_csv(csv_file_processed)
        except FileNotFoundError:
            df = pd.read_csv(csv_file, sep=",")
            # split 'image' column on first char to get datasource as new column
            df['datasource'] = df['image'].str[:1]
            df.to_csv(csv_file_processed, sep=",", index=False)
            self.csv = pd.read_csv(csv_file_processed)

        self.image_id = self.csv.id
        self.image_names = self.csv.image
        self.classifications = self.csv.classification
        self.labels = self.csv.datasource

        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        img_path = os.path.join(
            self.img_dir + "/" + str(self.classifications[index]),
            self.csv.iloc[index, 1]  # 'image' column
        )
        image = Image.open(img_path)

        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[index]

        return image, label
