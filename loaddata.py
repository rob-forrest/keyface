import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    data = pd.read_csv('training.csv')

    for i in range(10):
        image1 = data['Image'][i]
        image1 = image1.split(' ')
        image1 = [int(x) for x in image1]
        image1 = np.array(image1)
        plt.imshow(image1.reshape((96, 96)), cmap=plt.cm.gray)
        plt.savefig('images/image'+ str(i) +'.jpg')

    print(data.head())


if __name__ == "__main__":
    main()