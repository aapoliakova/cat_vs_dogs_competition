import pathlib
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class CatDogDataset(Dataset):
    def __init__(self, root: str = "data", split: str = "train", transform=None):
        super(CatDogDataset, self).__init__()
        data_path = pathlib.Path(root).joinpath(split)
        self.walker = list(data_path.iterdir())
        self.transform = T.Compose([
            T.Resize(size=(224, 224)),
            T.ToTensor()])

    def __getitem__(self, index):
        img_path = self.walker[index]
        image = Image.open(img_path)
        target = int(img_path.name.startswith('dog'))
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.walker)


if __name__ == "__main__":
    transform = T.Compose([
        T.Resize(size=(224, 224)),
        T.ToTensor()])

    print("Testing dataset .. ")
    dataset = CatDogDataset(root="data", split="test", transform=transform)

    image, target = dataset[0]
    print(image.shape, target)

    print("Testing loaders .. ")
    dataloader = DataLoader(dataset, batch_size=2)
    batch = next(iter(dataloader))
    imgs, targets = batch
    print(imgs.shape, targets.shape)
