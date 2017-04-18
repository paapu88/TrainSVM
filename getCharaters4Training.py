
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

    Alternatively, if binary=True generate training set
    with half samples positive (images containing characters)
    and half negative (random images from big dataset)

    """

    def __init__(self, binary=False, negative_names=None):
        self.font_height = 18
        self.output_height = 18
        self.output_width = 12
        self.chars = 'ABCDEFGHJKLMNPRSTUVXYZ-0123456789'
        #for noise
        self.sigma=0.1
        self.angle=0
        self.salt_amount=0.1
        self.repeat=300
        # sheet containing samples

        # if we only classify to positives and negatives, binary=true
        self.binary = binary
        if binary:
            self.negative_image_files = glob.glob(negative_names)
            self.bigsheet = np.ones((2*len(self.chars) * self.output_height, self.repeat * self.output_width)) * 255
        else:
            self.negative_image_files = None
            self.bigsheet = np.ones((len(self.chars) * self.output_height, self.repeat * self.output_width)) * 255



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
            distr = [1,1,1,1,1,1,1,1,3,3,3,3,5,5,7]
            #distr = [1,1,3,3,3,3,5,5,7]
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
            #minx,maxx = self.getMinAndMaxX(not_moved)
            #cmx=np.average([minx,maxx])
            #miny,maxy = self.getMinAndMaxY(not_moved)
            #cmy=np.average([miny,maxy])

            #cm = ndimage.measurements.center_of_mass(not_moved)
            #rows,cols = not_moved.shape
            ##rows,cols = im.shape
            #dy = rows/2 - cmy
            #dx = cols/2 - cmx
            #M = np.float32([[1,0,dx],[0,1,dy]])
            #dst = cv2.warpAffine(not_moved,M,(cols,rows))
            ##cv2.imshow('SVM test', dst)
            ##cv2.waitKey(0)
            #yield c, dst
            yield c, not_moved

    def rotate(self, image):
        cols=image.shape[1]
        rows= image.shape[0]
        halfcols=cols/2
        average_color = 0.5*(np.average(image[0][:]) + np.average(image[rows-1][:]))
        # print("COLOR",average_color)
        M = cv2.getRotationMatrix2D((cols/2,rows/2),self.angle,1)
        return cv2.warpAffine(image,M,(cols,rows),borderValue=average_color)


    def make_noise(self, image, invert=True):
        """noisify image"""

        clone = image.copy()
        myrandoms = np.random.random(5)
        #print("myrandoms ", myrandoms)
        self.angle = random.gauss(0, 2)

        myones = np.ones(clone.shape)

        if myrandoms[0] < 0:
            clone = self.noisy("poisson",clone )
        if myrandoms[1] < 0.5:  # use
            clone = self.noisy("gaussAdd",clone )
        if myrandoms[2] < 0:
            clone = self.noisy("sp",clone )
        if myrandoms[3] < 0.0:
            clone = self.noisy("gaussMulti",clone )
        if myrandoms[4] < 0.5:  # use
            clone = self.noisy("blur", clone)


        clone = self.rotate(image=clone)
        if invert:
            clone=255*(myones-clone)
        else:
            clone=255*clone
        return clone

                
    def generate_positives_for_svm(self,font_file=None,
                                   positive_dir='SvmDir',filename='allSVM'):
        import os, glob
        font_char_ims = dict(self.make_char_ims_SVM(font_file=font_file))
        random.seed()


        try:
            os.mkdir(positive_dir)
        except IsADirectoryError:
            raise(IsADirectoryError,"dir exists, remove!")

        for iy, (mychar, img) in enumerate(font_char_ims.items()):
            for condition in range(self.repeat):
                #print(iy, mychar, img.shape())

                clone = self.make_noise(img)
                #clone=img.copy()

                y1=iy*clone.shape[0]
                y2=(iy+1)*clone.shape[0]
                x1=condition * clone.shape[1]
                x2=(condition+1)*clone.shape[1]
                #print("POS",x1,x2,y1,y2,clone.shape)
                self.bigsheet[y1:y2, x1:x2] =clone
                #cv2.imshow('SVM test', clone)
                #cv2.waitKey(0)
                with open(positive_dir+'/'+filename+'.txt', 'a') as f:
                    if self.binary:  # characters are classified all as '1' in case of binary classification
                        f.write('1 \n')
                    else:
                        f.write(mychar+' \n')

        if self.binary:
            # in case of binary sample, make also negative samples (after the positives done above)
            self.generate_negatives_for_svm(positive_dir=positive_dir,filename=filename, y_init=y2)
        else:
            # write big file with one character on each line, each character has different label
            cv2.imwrite(positive_dir + '/' + filename + '.tif', self.bigsheet)


    def generate_negatives_for_svm(self, positive_dir=None,filename=None, y_init=None):
        """Write negative images to the same file as positive images
        in the label file '0' means negative sample and '1' a positive sample
        """
        # same number of negative images as with the positive ones
        import matplotlib.image as mpimg

        def rgb2gray(rgb):
            return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


        random.seed()
        np.random.shuffle(self.negative_image_files)
        font_char_ims = dict(self.make_char_ims_SVM(font_file=font_file))
        for iy, (mychar, img) in enumerate(font_char_ims.items()):
            for condition in range(self.repeat):
                ifile = random.randint(a=0,b=len(self.negative_image_files)-1)
                #print("SN1:", ifile, len(self.negative_image_files))
                #print("SN2", self.negative_image_files[ifile])
                #big_negative = None
                #while big_negative is None:
                #    big_negative= cv2.imread(self.negative_image_files[ifile], 0)
                big_negative = rgb2gray(mpimg.imread(self.negative_image_files[ifile]))
                #print("BNN:", big_negative.shape)
                x_ul=random.randint(a=0, b=(big_negative.shape[1]-1-img.shape[1]))
                y_ul=random.randint(a=0, b=(big_negative.shape[0]-1-img.shape[0]))
                small_negative = big_negative[y_ul:(y_ul+img.shape[0]), x_ul:(x_ul+img.shape[1])]
                #add noise (same to all samples)
                #print("SN: ", small_negative.shape, self.negative_image_files[ifile])
                #clone = self.make_noise(small_negative/255, invert=False)
                clone = small_negative.copy()
                y1 = y_init + iy * img.shape[0]
                y2 = y_init + (iy+1) * img.shape[0]
                x1 = condition * img.shape[1]
                x2 = (condition+1) * img.shape[1]
                # print("clone shape", clone.shape,y1,y2)

                self.bigsheet[y1:y2, x1:x2] = clone
                with open(positive_dir+'/'+filename+'.txt', 'a') as f:
                    # false images are labeled as '0' in case of binary classification
                    f.write('0 \n')
        #all positives and negatives to the same file, positives first
        cv2.imwrite(positive_dir + '/' + filename + '.tif', self.bigsheet)



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
    try:
        negative_file_names = sys.argv[2]
        binary = True
    except:
        negative_file_names = None
        binary = False


    #app1 = GetCharatersForTraining(binary=False, negative_names=None)
    #app1.generate_positives_for_svm(font_file=font_file)
    app1 = GetCharatersForTraining(binary=binary, negative_names=negative_file_names)
    app1.generate_positives_for_svm(font_file=font_file)

    #app1.generate_ideal(font_file=font_file)

    #sys.exit()
    #app1.generate_positives_for_haarcascade(font_file=font_file, repeat=40)
    #app1.generate_positives_for_tesseract(font_file=font_file, repeat=400)

    #SUN397forPlate

    #font_char_ims = dict(app1.make_char_ims(font_file=font_file))
    #for mychar, img in font_char_ims.items():
    #    print ("mychar: ",mychar, img.shape )
    #    img_noisy = app1.noisy("speckle", img)
    #    img_rotated = app1.rotate(image=img_noisy, angle=-10)
    #    plt.imshow(img_rotated)
    #    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    #    plt.show()
