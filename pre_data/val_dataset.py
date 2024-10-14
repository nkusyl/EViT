import os
from tqdm import tqdm
from PIL import Image


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


if __name__ == '__main__':

    # data = []
    # for line in open("/home/ubuntu/Datasets/val_images.txt", "r"):
    #     data.append(line)
    # for a in tqdm(range(len(data))):
    #     folder = data[a][32:41]
    #     mkdir("/home/ubuntu/Datasets/val_images/"+str(folder))

    data = []
    for line in open("/home/ubuntu/Datasets/val_images.txt", "r"):
        data.append(line)

    for a in tqdm(range(len(data))):

        im = Image.open('/home/ubuntu/Datasets/ILSVRC2012_img_val/{}'.format(data[a][3:31]))

        nameimage = data[a][3:31]
        path = '/home/ubuntu/Datasets/val_images/{}'.format(data[a][32:41])

        im.save(os.path.join(path, nameimage))
        im.close()
