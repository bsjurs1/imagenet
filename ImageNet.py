"""Build the imagenet dataset."""

from PIL import Image
import os
import torch.utils.data as data


class ImageNet(data.Dataset):
    """Store the imagenet image data."""

    def __init__(self, root, transform=None, target_transform=None,
                 train=True, kaggle=False):
        """Initialize the ImageNet dataset object."""
        if not kaggle:

            if train:
                self.trainData = []
                img_folder_path = root + '/train/images/'
                label_file_path = root + '/train/train_labels.csv'
                label_file_lines = open(label_file_path, 'r').readlines()
                for i in range(len(label_file_lines)):
                    if i == 0:
                        continue
                    line = label_file_lines[i]
                    line_elements = line.split(',')
                    img_name = line_elements[0] + ".JPEG"
                    label = line_elements[1].split('\n')[0]
                    label_nr = int(label)
                    img_path = img_folder_path + img_name
                    self.trainData.append((img_path, label_nr))

                if len(self.trainData) == 0:
                    raise(RuntimeError("Found 0 images in subfolders of: "
                                       + root + "\n"))

            else:
                self.testData = []
                img_folder_path = root + '/test/images/'
                label_file_path = root + '/test/test_labels.csv'
                label_file_lines = open(label_file_path, 'r').readlines()
                for i in range(len(label_file_lines)):
                    if i == 0:
                        continue
                    line = label_file_lines[i]
                    line_elements = line.split(',')
                    img_name = line_elements[0] + ".JPEG"
                    label = line_elements[1].split('\n')[0]
                    label_nr = int(label)
                    img_path = img_folder_path + img_name
                    self.testData.append((img_path, label_nr))

                if len(self.testData) == 0:
                    raise(RuntimeError("Found 0 images in subfolders of: "
                                       + root + "\n"))

        else:
            self.testData = []
            img_paths = os.listdir(root + '/val/images/')
            for img_path in img_paths:
                self.testData.append(root + '/val/images/' + img_path)

            if len(self.testData) == 0:
                raise(RuntimeError("Found 0 images in subfolders of: "
                                   + root + "\n"))

        self.kaggle = kaggle
        self.train = train
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """Override method from dataset module to fetch image and target."""
        if not self.kaggle:
            if self.train:
                path, target = self.trainData[index]
            else:
                path, target = self.testData[index]

            image = Image.open(path)
            fp = image.fp
            image.load()
            image = image.convert('RGB')
            fp.closed

            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                target = self.target_transform(target)

            if self.train:
                return image, target
            else:
                return image, target, path
        else:
            path = self.testData[index]
            image = Image.open(path)
            fp = image.fp
            image.load()
            image = image.convert('RGB')
            fp.closed

            if self.transform is not None:
                image = self.transform(image)

            return image, path

    def __len__(self):
        """Display the number objects."""
        if self.train:
            return len(self.trainData)
        else:
            return len(self.testData)
