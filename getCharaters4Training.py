
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from scipy import ndimage
import cv2
import random

class GetCharatersForTraining():
    """
    produce positive training set for support vector machine to recognize a character in a finnish licence plate
    The are missing characters
    I (it is the same as digit 1, one)
    O (is is the same as digit 0, zero)
    QÅÄÖ (not used in modern Finnish mainland plates)
    - is included for the moment

    """

    def __init__(self):
        self.font_height = 18
        self.output_height = 18
        self.output_width = 12
        self.chars = 'ABCDEFGHJKLMNPRSTUVXYZ-0123456789'
        #for noise
        self.sigma=0.1
        self.angle=0
        self.salt_amount=0.1



    def getMinAndMaxY(self, a, thr=0.5):
        """find the value in Y where image starts"""
        minY = None
        maxY = None
        for iy in range(a.shape[0]):
            for ix in range(a.shape[1]):
                if a[iy,ix]> thr:
                    minY = iy
                    break
        for iy in reversed(range(a.shape[0])):
            for ix in range(a.shape[1]):
                if a[iy,ix]> thr:
                    maxY = iy
                    break
        return minY, maxY

    def getMinAndMaxX(self, a, thr=0.5):
        """find the value in Y where image starts"""
        minX = None
        maxX = None
        for ix in range(a.shape[1]):
            for iy in range(a.shape[0]):
                if a[iy,ix]> thr:
                    minX = ix
                    break
        for ix in reversed(range(a.shape[1])):
            for iy in range(a.shape[0]):
                if a[iy,ix]> thr:
                    maxX = ix
                    break
        return minX, maxX





    def noisy(self, noise_typ, image):
        """
        Parameters
        ----------
        image : ndarray
            Input image data. Will be converted to float.
        mode : str
            One of the following strings, selecting the type of noise to add:

            'gauss'     Gaussian-distributed additive noise.
            'poisson'   Poisson-distributed noise generated from the data.
            'sp'       Replaces random pixels with 0 or 1.
            'speckle'   Multiplicative noise using out = image + n*image,where
                        n,is uniform noise with specified mean & variance.
        """

        if noise_typ == "gaussAdd":
            row, col = image.shape
            mean = max(0,np.mean(image))
            # var = 0.1
            # sigma = var**0.5
            #print ("M",mean, self.sigma)
            gauss = np.random.normal(0.25, 0.25, (row, col))
            gauss = gauss.reshape(row, col)
            noisy = image + gauss
            noisy = np.clip(noisy,0,1)
            return noisy
        elif noise_typ == "sp":
            #print("sp ", self.salt_amount)
            row, col = image.shape
            a=np.zeros(image.shape)
            a=a.flatten()
            s_vs_p = 0.9
            salt_amount=0.01
            out = image
            # Salt mode
            num_salt = np.ceil(salt_amount * image.size * s_vs_p)
            a[0:num_salt]=1
            np.random.shuffle(a)
            a = a.reshape(image.shape)
            out = out + a
            # Pepper mode
            num_pepper = np.ceil(salt_amount * image.size * (1. - s_vs_p))
            b=np.ones(image.shape)
            b=b.flatten()
            b[0:num_pepper]=0
            np.random.shuffle(b)
            b = b.reshape(image.shape)
            out = np.multiply(out,b)
            np.clip(out, 0, 1)
            return out
        elif noise_typ == "poisson":
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(image * vals) / float(vals)
            noisy = np.clip(noisy,0,1)
            return noisy
        elif noise_typ == "gaussMulti":
            gauss = np.random.normal(10, self.sigma, 1)
            noisy = image * gauss
            noisy = np.clip(noisy,0,1)
            return noisy
        elif noise_typ == "blur":
            # in gaussia, size must be odd
            #distr = [1,1,1,1,1,1,1,1,3,3,3,3,5,5,7]
            distr = [1,1,3,3,3,3,5,5,7]
            sizex = random.choice(distr)
            sizey = random.choice(distr)
            noisy = cv2.GaussianBlur(image,(sizey,sizex),0)
            noisy = np.clip(noisy,0,1)
            return noisy

    def make_char_ims_SVM(self, font_file):
        """ get characters as numpy arrays"""

        font_size = self.output_height * 4

        font = ImageFont.truetype(font_file, font_size)

        height = max(font.getsize(c)[1] for c in self.chars)
        width =  max(font.getsize(c)[0] for c in self.chars)
        for c in self.chars:

            im = Image.new("RGBA", (width, height), (0, 0, 0))

            draw = ImageDraw.Draw(im)
            draw.text((0, 0), c, (255, 255, 255), font=font)
            scale = float(self.output_height) / height
            im = im.resize((self.output_width, self.output_height), Image.ANTIALIAS)
            not_moved = np.array(im)[:, :, 0].astype(np.float32) / 255.
            minx,maxx = self.getMinAndMaxX(not_moved)
            cmx=np.average([minx,maxx])
            miny,maxy = self.getMinAndMaxY(not_moved)
            cmy=np.average([miny,maxy])

            cm = ndimage.measurements.center_of_mass(not_moved)
            rows,cols = not_moved.shape
            dy = rows/2 - cmy
            dx = cols/2 - cmx
            M = np.float32([[1,0,dx],[0,1,dy]])
            dst = cv2.warpAffine(not_moved,M,(cols,rows))
            yield c, dst

    def rotate(self, image):
        cols=image.shape[1]
        rows= image.shape[0]
        halfcols=cols/2
        average_color = 0.5*(np.average(image[0][:]) + np.average(image[rows-1][:]))
        # print("COLOR",average_color)
        M = cv2.getRotationMatrix2D((cols/2,rows/2),self.angle,1)
        return cv2.warpAffine(image,M,(cols,rows),borderValue=average_color)


                
    def generate_positives_for_svm(self,font_file=None, repeat=100, positive_dir='PositivesSvm',filename='positivesSVM'):
        import os, glob
        font_char_ims = dict(self.make_char_ims_SVM(font_file=font_file))
        random.seed()
        bigsheet=np.ones((len(self.chars)*self.output_height, repeat*self.output_width))*255

        try:
            os.mkdir(positive_dir)
        except IsADirectoryError:
            raise(IsADirectoryError,"dir exists, remove!")

        for iy, (mychar, img) in enumerate(font_char_ims.items()):
            for condition in range(repeat):
                clone = img.copy()
                myrandoms = np.random.random(5)
                print("myrandoms ", myrandoms)
                self.angle = random.gauss(0, 5)

                myones = np.ones(clone.shape)

                if myrandoms[0] < 0:
                    clone = self.noisy("poisson",clone )
                if myrandoms[1] < 1:  # use
                    clone = self.noisy("gaussAdd",clone )
                if myrandoms[2] < 1:  # use
                    clone = self.noisy("sp",clone )
                if myrandoms[3] < 0.0:
                    clone = self.noisy("gaussMulti",clone )
                if myrandoms[4] < 1:  # use
                    clone = self.noisy("blur", clone)


                clone = self.rotate(image=clone)
                clone=255*(myones-clone)
                y1=iy*clone.shape[0]
                y2=(iy+1)*clone.shape[0]
                x1=condition * clone.shape[1]
                x2=(condition+1)*clone.shape[1]
                #print(x1,x2,y1,y2,clone.shape)
                bigsheet[y1:y2, x1:x2] =clone
                with open(positive_dir+'/'+filename+'.txt', 'a') as f:
                    f.write(mychar+' \n')
        cv2.imwrite(positive_dir+'/'+filename+'.tif', bigsheet)
                
    def generate_ideal(self, font_file=None, positive_dir='PositivesIdeal'):
        """ write characters once without distorsions"""
        import os
        font_char_ims = dict(self.make_char_ims(font_file=font_file))
        if not os.path.exists(positive_dir):
            os.makedirs(positive_dir)
        for mychar, img in font_char_ims.items():
            myones = np.ones(img.shape)
            img=255*(myones-img)
            cv2.imwrite(positive_dir+'/'+mychar+'.tif', img)



if __name__ == '__main__':
    import sys, glob
    from matplotlib import pyplot as plt

    font_file = sys.argv[1]
    app1 = GetCharatersForTraining()
    #app1.generate_ideal(font_file=font_file)

    #sys.exit()
    #app1.generate_positives_for_haarcascade(font_file=font_file, repeat=40)
    #app1.generate_positives_for_tesseract(font_file=font_file, repeat=400)
    app1.generate_positives_for_svm(font_file=font_file, repeat=500)


    #font_char_ims = dict(app1.make_char_ims(font_file=font_file))
    #for mychar, img in font_char_ims.items():
    #    print ("mychar: ",mychar, img.shape )
    #    img_noisy = app1.noisy("speckle", img)
    #    img_rotated = app1.rotate(image=img_noisy, angle=-10)
    #    plt.imshow(img_rotated)
    #    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    #    plt.show()
