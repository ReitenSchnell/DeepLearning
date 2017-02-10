import os
import urllib.request
import matplotlib.pyplot as plt


def get_celeb_files():
    if not os.path.exists('img_align_celeba'):
        os.mkdir('img_align_celeba')
    for img_i in range(1, 101):
        f = '000%03d.jpg' % img_i
        if os.path.exists('img_align_celeba/' + f):
            continue
        url = 'https://s3.amazonaws.com/cadl/celeb-align/' + f
        print(url, end='\r')
        urllib.request.urlretrieve(url, os.path.join('img_align_celeba', f))
    files = [os.path.join('img_align_celeba', file_i)
             for file_i in os.listdir('img_align_celeba')
             if '.jpg' in file_i]
    return [plt.imread(file) for file in files]


def massage_image(img):
    plt.imshow(img)
    plt.show()
    print(img.shape)
    plt.imshow(img[:, :, 0], cmap='gray')
    plt.imshow(img[:, :, 1], cmap='gray')
    plt.imshow(img[:, :, 2], cmap='gray')
