#%qtconsole
#%matplotlib nbagg


import numpy as np
import scipy.io
from scipy import misc
from filesSelector import *
from skimage import io
from scipy.misc import imresize
from scipy.misc import imrotate
from scipy import signal
import exifread
import json
import tifffile as tiff
import pandas as pd


def find_osc_end(data,window=20,tresh=8,quantile=0.3,minimal_start_frame=25):
    series=pd.Series(data)
    upper = series.rolling(window=window,center = True).quantile(1-quantile)
    lower = series.rolling(window=window,center = True).quantile(quantile)
#     print(data)
#     print (series)
    a = np.argwhere((upper-lower)>tresh)
#     print (a)
    if (len(a) >0) and a[-1]>minimal_start_frame :
        return a[-1]
    else:
        return -1


#extract pos from image metadata
def get_im_pos(path_list,scale = 0.99, deg=1.5,resize=100,flipx=False):
    pos = np.empty((len(path_list),2))
    I = io.imread(path_list[0])
    if len(I.shape)>2: # im_seq
        I = I[0]
    I_shape = imresize(I,resize).shape
    print('I_shape: '+format(I_shape))
    I_type = I.dtype
    for i,path in enumerate(path_list):
        with open(path, 'rb') as f:
            tags= exifread.process_file(f)  
        d = json.loads(tags['Image Tag 0xC7B3'].values)
        pos[i,:]=[-d['YPositionUm'],-d['XPositionUm']]
    # rotate
    theta = np.radians(deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.matrix([[c, -s], [s, c]])
    pos = np.dot(R,pos.T).T
    pos = np.round(pos*scale)#+np.multiply(I_shape,0.5)
    if flipx:
        pos[:,1]=-pos[:,1]
        print('a')
    
    #shift by mean
    mean = np.max(pos,axis=0)
    pos[:,0] = pos[:,0] - mean[0,0]
    pos[:,1] = pos[:,1] - mean[0,1]
    
    #calc total image size
    extents = (np.max(pos,axis=0) - np.min(pos,axis=0))
    extents = [extents[0,0]+I_shape[0], extents[0,1] +I_shape[1]]
    return pos.astype(int), np.round(extents).astype('int') , I_shape , I_type


# gaussian kernel for the weight functions
def makeGaussian(size, fwhm = 3, center=None):
    size_max = max(size)
    x = np.arange(0, size_max, 1, float)
    y = x[:,np.newaxis]
    if center is None:
        x0 = y0 = size_max // 2
    else:
        x0 = center[0]
        y0 = center[1]
    g =  np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    return g[int((size_max-size[0])/2):int((size_max+size[0])/2),int((size_max-size[1])/2):int((size_max+size[1])/2)]




# get exp time
##########
def im_get_exp_time(fileList):
    with open(fileList[0], 'rb') as f:
        tags= exifread.process_file(f)  
        d = json.loads(tags['Image Tag 0xC7B3'].values)
    print (d['Exposure-ms'])
    return d['Exposure-ms']

# im seq
#########
def combine_im_seq(fileList,scale=0.99,resize=100,deg=1.5,save_path='im_seq_out.tif',fov_corr_inv_path ='',cor_mag=1,cor_shift=0,flipx=False,frames=[0,1]):
    im_pos_list = np.arange(0,len(fileList))# relevant positions

    ##############
    marker_rel,extents,I_shape ,I_type= get_im_pos([fileList[i] for i in im_pos_list],scale = scale*resize/100,deg=deg,resize=resize,flipx=flipx)
    print('image size: '+format(extents))
    marker_abs = [0,0]
    print('marker abs pos (px): '+format(marker_abs))
    I = np.zeros((frames[1]-frames[0],extents[0],extents[1]),dtype='float')
    # weight for stitch
    w = makeGaussian(I_shape, max(I_shape)/2)
    Iw = np.zeros((extents[0],extents[1]),dtype='float')
    if (fov_corr_inv_path):
        fov_corr_inv = imresize(np.load(fov_corr_inv_path),resize)*cor_mag+cor_shift # inverted fov correction

    for idx,p in enumerate(im_pos_list):
        Iseq = io.imread(fileList[p])
        pos_tl = [marker_abs[i] - marker_rel[idx,i] for i in range(2)]
        pos_br = [marker_abs[i] - marker_rel[idx,i] + I_shape[i] for i in range(2)]
        print ('pos: '+str(p) + ', pix ='+ format([pos_tl , pos_br]))
        Iw[pos_tl[0]:pos_br[0],pos_tl[1]:pos_br[1]]+=w

        # TODO change to seq/single
        for t in range(frames[1]-frames[0]):
            T= imresize(Iseq[t+frames[0]].astype('float64'),resize)*w
            if fov_corr_inv_path:
                T=T*fov_corr_inv
            I[t,pos_tl[0]:pos_br[0],pos_tl[1]:pos_br[1]]+=T

    for t in range(frames[1]-frames[0]):
        I[t] = I[t]/Iw
    tiff.imsave(data=I.astype(I_type),file=save_path)
    return I

# single im
#########
def combine_single_im(fileList,scale=0.99,resize=100,deg=1.5,save_path='single_im_out.tif',fov_corr_inv_path ='',cor_mag=1,cor_shift=0,flipx=False):
    im_pos_list = np.arange(0,len(fileList))# relevant positions
    ##############
    marker_rel,extents,I_shape,I_type = get_im_pos([fileList[i] for i in im_pos_list],scale = scale*resize/100,deg=deg,resize=resize,flipx=flipx)
    print('image size: '+format(extents))
    marker_abs = [0,0]#np.multiply(extents,0.5).astype('int')#+(np.array(I_shape)/2).astype('int')
    print('marker abs pos (px): '+format(marker_abs))
    I = np.zeros((1,extents[0],extents[1]),dtype='float')
    # weight for stitch
    w = makeGaussian(I_shape, max(I_shape)/2)
    Iw = np.zeros((extents[0],extents[1]),dtype='float')
    if (fov_corr_inv_path):
        fov_corr_inv = imresize(np.load(fov_corr_inv_path),resize)*cor_mag+cor_shift # inverted fov correction
    for idx,p in enumerate(im_pos_list):
        Iseq = io.imread(fileList[p])
        pos_tl = [marker_abs[i] - marker_rel[idx,i] for i in range(2)]
        pos_br = [marker_abs[i] - marker_rel[idx,i] + I_shape[i] for i in range(2)]
        print ('pos: '+str(p) + ', pix ='+ format([pos_tl , pos_br]))
        Iw[pos_tl[0]:pos_br[0],pos_tl[1]:pos_br[1]]+=w

        # TODO change to seq/single
        T= imresize(Iseq.astype('float64'),resize)*w
        if fov_corr_inv_path:
            T=T*fov_corr_inv
        I[0,pos_tl[0]:pos_br[0],pos_tl[1]:pos_br[1]]+=T
    I = I/Iw[np.newaxis]

    tiff.imsave(data=I.astype(I_type),file=save_path)
    return I



def get_period(data,N=10000,start_period=30,end_period=80,plot=False,ax=0):
    l = data.shape[0]
    T = 2 # interval
    s=int((N)*2/end_period) ;e= int((N)*2/start_period)
    freq_seq = np.fft.rfftfreq(N, d=T)
    d = np.append(data-data.mean(),np.zeros((N-l)))
    fft_vec = np.fft.rfft(d)
    fft_amp = np.abs(fft_vec)
    fft_max_idx = np.argmax(fft_amp[s:e])+s
    error=0
    if fft_max_idx == s:
        error=1
        print("warning: max period reached in get_period")
    if fft_max_idx == e:
        error=1
        print("warning: min period reached in get_period")
    if plot:
        ax.plot(freq_seq[s:e]**-1,fft_amp[s:e])
    return freq_seq[fft_max_idx]**-1,error