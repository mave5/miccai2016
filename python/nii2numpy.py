import os
import numpy as np
#from nibabel.testing import data_path
import nibabel as nib
import matplotlib.pylab as plt
import cv2

# path to data
#imgpath='/media/mra/win7/data/misc/miccai2016/shortaxis/Training dataset/'
#labelpath='/media/mra/win7/data/misc/miccai2016/shortaxis/Ground truth/'
#imgpath='/media/mra/win7/data/misc/miccai2016/axial/Training dataset/'
#labelpath='/media/mra/win7/data/misc/miccai2016/axial/Ground truth/'
imgpath='/media/mra/win7/data/misc/miccai2016/data/axialcrop/Training dataset/'
labelpath='/media/mra/win7/data/misc/miccai2016/data/axialcrop/Ground truth/'


# load image and label
def load_imglabel(imgfn,labelfn):   
    img = nib.load(imgfn)
    lbl = nib.load(labelfn)
    # convert to numpy
    image=img.get_data()        
    label=lbl.get_data()
    print 'images:  ', image.shape
    print 'labels:  ', label.shape    
    print 'max: ', np.max(label)
    print 'min: ', np.min(label)
    # return image and labels    
    return image,label      


# concat image labels
def concat_imglabel(imgpath,labelpath,N1,N2,img_w,img_h):
    # get list of images and labels
    imglist = os.listdir( imgpath)
    labellist = os.listdir( labelpath)
    print 'number of images %d' % len(imglist)
    print 'number of labels %d' % len(labellist)

    # load data
    for k in range(N1,N2):
        imgfn=imgpath+imglist[k]
        labelfn=labelpath+labellist[k]
        image,label=load_imglabel(imgfn,labelfn)
        if k==N1:
            shapelist=image.shape    
        else:
            shapelist= np.append(shapelist,image.shape,axis=0)   
        
        # resize
        tmp1 = cv2.resize(image, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
        tmp2 = cv2.resize(label, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
        tmp2[tmp2==3]=2
        if k==N1:
            X=tmp1     
            Y=tmp2
        else:
            X=np.append(X,tmp1,axis=2)
            Y=np.append(Y,tmp2,axis=2)
            
    print X.shape
    print Y.shape
    return X,Y,shapelist



# window size
img_w=239
img_h=165

# train data
X_train,Y_train,shape_train=concat_imglabel(imgpath,labelpath,0,8,img_w,img_h)

# test data
X_test,Y_test,shape_test=concat_imglabel(imgpath,labelpath,8,10,img_w,img_h)
     
# display sample image and label
N=X_train.shape[2]
n1=np.random.randint(N)
I1=X_train[:,:,n1]
L1=Y_train[:,:,n1]
plt.imshow(I1,cmap='Greys_r')
plt.imshow(L1,cmap='Greys_r')
print np.max(L1)
print np.min(L1)


# save X and Y 
# save as numpy files
print 'wait to save data as numpy files'
np.savez(imgpath+'/train_data.npz', X=X_train,Y=Y_train)
np.savez(imgpath+'/test_data.npz', X=X_test,Y=Y_test)
print 'data was saved!'


