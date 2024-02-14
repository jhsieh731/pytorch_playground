from torch.utils.data import Dataset
import os
from PIL import Image


def is_valid_image_file(filename):
  # Check file name extension
  valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
  if os.path.splitext(filename)[1].lower() not in valid_extensions:
    print(f"Invalid image file extension \"{filename}\". Skipping this file...")
  # Verify that image file is intact
  try:
    with Image.open(filename) as img:
      img.verify()  # Verify if it's an image
      return True
  except (IOError, SyntaxError) as e:
    print(f"Invalid image file {filename}: {e}")
    return False
  
class BiasedCarDataset(Dataset):
    # img_dir will be reorg joined with train/val
  def __init__(self, img_dir, transform=None, target_transform=None):
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform
    image_label_dict = {}
    self.class_counts = {i: 0 for i in range(18)}
    for class_i in range(18):
       parent_path = os.path.join(img_dir, str(class_i))
       class_images = os.listdir(parent_path)

       self.class_counts[class_i] = len(class_images)
       for filename in class_images:
          if is_valid_image_file(os.path.join(parent_path, filename)):
             image_label_dict[filename] = class_i

    # self.items is a list of tuples like: [ ("im1.jpg", 0), ("img2.png", 1), ... ]
    self.items = list(image_label_dict.items())
    print("Class counts!", self.class_counts) 

  

  def __len__(self):
    return len(self.items)


  def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, str(self.items[idx][1]), self.items[idx][0])
    # image = torchvision.io.read_image(img_path) # This version reads the image directly as a Tensor
    image = Image.open(img_path)
    label = self.items[idx][1]

    if self.transform:
        image = self.transform(image)
        # print(image.shape)
    if self.target_transform:
        label = self.target_transform(label)
    
    return image, label
