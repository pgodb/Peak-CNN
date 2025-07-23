#This code is made available under the MIT License
#Copyright (c) 2024 German Aerospace Center (DLR)
#Author: Philipp Godbersen


import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, ReLU, MaxPool2D, Resizing, Concatenate
import tensorflow.keras.metrics as tfm




def downblock(input,convlayers=16, pool = MaxPool2D):
    """Creates a Conv block for the descending part of the Unet"""
    x = Conv2D(convlayers,kernel_size=3,padding='same')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(convlayers,kernel_size=3,padding='same')(x)
    x = BatchNormalization()(x)
    
    skip = ReLU()(x)
    output = pool()(skip)

    return output,skip


def upblock(input,convlayers=8):
    """creates a Conv block for the accending part of the Unet"""
    x = Conv2D(convlayers,kernel_size=3,padding='same')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(convlayers,kernel_size=3,padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    output = Conv2DTranspose(convlayers/2,kernel_size=3,strides=2,padding="same")(x)
    
    return output

def build_model(convlayernums=[32,32,32,32],metadatashape=None):
    """
    This builds the PeakCNN model with some flexibility in the architecture.
    The number of filters in the Convolutions in the downsampling levels of the U-net may be adjusted via the convlayernums argument.
    Depending on the argument metadatashape the created model will utilize metadata beyond the images themselves or not.
    This is activated by setting metadatashape to the shape of the intended per image meatadata in HWC order.
    Within the model this is then resized to match the coresponding activation map at the top and bottom of the Unet.
    For example in the paper we utilized OTF information in the form of a set of 4 numbers constant over each image which would corespond to a metadatashape=(1,1,4)
    """

    usemetadata = metadatashape is not None
    
    imagedatainput = tf.keras.Input(shape=(256, 256, 1),name="imagedata")
    if usemetadata:
        otfdatainput = tf.keras.Input(shape=metadatashape,name="metadata")

    x = imagedatainput

    if usemetadata:
        resizedotf = Resizing(256,256)(otfdatainput)
        x = Concatenate()((x,resizedotf))

    x = Conv2D(convlayernums[0],kernel_size=3,padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(convlayernums[0],kernel_size=3,padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)


    #Decending branch of the unet
    skipconns=[]
    updownlayers = len(convlayernums)
    for i in range(updownlayers):
        x,skip = downblock(x,convlayernums[i],MaxPool2D)
        skipconns.append(skip) # collect horizontal skip connections to be concatenated in the accending Unet branch

    # also inject otf data into the bottom of the Unet
    if usemetadata:
        resizedotf2 =  Resizing(16,16)(otfdatainput)
        x = Concatenate()((x,resizedotf2))

    x = Conv2D(convlayernums[-1],kernel_size=3,padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(convlayernums[-1],kernel_size=3,padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    #Ascending branch of the unet
    for i in range(updownlayers):
        x = upblock(x,convlayernums[updownlayers-i-1])
        x = Concatenate()((x,skipconns[updownlayers-i-1]))

    x = Conv2D(convlayernums[0],kernel_size=3,padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(convlayernums[0],kernel_size=3,padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(convlayernums[0],kernel_size=3,padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(convlayernums[0]//2,kernel_size=3,padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)


    output_class = Conv2D(1,kernel_size=3,padding='same',name="output_class",dtype='float32')(x)
    output_subpx = Conv2D(2,kernel_size=3,padding='same',name="output_subpx",dtype='float32')(x)

    modelinputs=[imagedatainput]
    if usemetadata:
        modelinputs.append(otfdatainput)

    model = tf.keras.Model(inputs=modelinputs, outputs=[output_subpx, output_class])
    
    return model


# 5px x 5px gaussian kernel
gko_5 = np.array([ [1, 4, 7, 4, 1,],
              [4, 16, 26, 16, 4],
              [7, 26, 41, 26, 7],
              [4, 16, 26, 16, 4],
              [1, 4, 7, 4, 1]])/41

gk_5 = gko_5 / np.sum(gko_5)
tfgk_5 = tf.constant(gk_5,dtype=tf.float32)
tfgk_5t = tfgk_5[:,:,tf.newaxis,tf.newaxis,] #prepare shape for 2Dconv in Tensorflow 


#since this will applied to the "output_class" only,  y_pred is just 1 channel classification predediction. y_true is still the full 3 channel t
def lossfunc_class(y_true, y_pred):

    there_true = y_true[:,2:-2,2:-2,0:1]
    there_pred = y_pred[:,2:-2,2:-2,:]

    numel= tf.cast(tf.reduce_prod(tf.shape(there_true)),tf.float32)
    classw_true = numel/(2*tf.reduce_sum(there_true))
    classw_false = numel/(2*tf.reduce_sum(1-there_true))

    ww = tf.nn.conv2d(there_true,tfgk_5t,strides=(1,1,1,1),padding="SAME")*(classw_true - classw_false) + classw_false
    bceloss = tf.reduce_mean(tf.expand_dims(tf.losses.binary_crossentropy(there_true,there_pred, from_logits=True),3)                                                                                           * ww)

    return bceloss

#since this will applied to the "output_subpx" only, y_pred is just 2 channel subpx predediction. y_true is still the full 3 channel training label
def lossfunc_subxy(y_true, y_pred):
    
    there_true = y_true[:,2:-2,2:-2,0:1]
    tmse = tf.losses.mean_absolute_error(there_true * y_true[:,2:-2,2:-2,1:3], there_true * y_pred[:,2:-2,2:-2,:])
    subxyloss = tf.reduce_sum(tmse)/tf.reduce_sum(there_true)

    return subxyloss


# lossterm dict to be used for training in Keras model.fit of models build using build_model
lossterm={"output_class":lossfunc_class,"output_subpx":lossfunc_subxy}
    



#define metrics that can take logits as input and work with our data structure
class AUC(tfm.AUC):
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(AUC, self).update_state(y_true[:,:,:,0], tf.nn.sigmoid(y_pred[:,:,:,0]), sample_weight)
        else:
            super(AUC, self).update_state(y_true[:,:,:,0], y_pred[:,:,:,0], sample_weight)


class Precision(tfm.Precision):
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(Precision, self).update_state(y_true[:,:,:,0], tf.nn.sigmoid(y_pred[:,:,:,0]), sample_weight)
        else:
            super(Precision, self).update_state(y_true[:,:,:,0], y_pred[:,:,:,0], sample_weight)

class PrecisionAtRecall(tfm.PrecisionAtRecall):
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(PrecisionAtRecall, self).update_state(y_true[:,:,:,0], tf.nn.sigmoid(y_pred[:,:,:,0]), sample_weight)
        else:
            super(PrecisionAtRecall, self).update_state(y_true[:,:,:,0], y_pred[:,:,:,0], sample_weight)


class Recall(tfm.Recall):
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(Recall, self).update_state(y_true[:,:,:,0], tf.nn.sigmoid(y_pred[:,:,:,0]), sample_weight)
        else:
            super(Recall, self).update_state(y_true, y_pred, sample_weight)






def do_PeakCNN(model,img,metadata=None,threshold=0.5):
    """
    Applies a trained PeakCNN model to an image and returns peak positions and the amount found.
    If the model expects metadata input this needs to be supplied as well.
    The threshold argument can be set between 0 and 1 allows to select the agressiveness of detection.
    Only particles where the classifier outputs a higher probability than the threshold value are considered peaks.
    The default value of 0.5 should be a reasonable balance for most cases but this can be adjusted to target the desired precision/recall tradeoff
    """

    data=np.reshape(img,(1,*img.shape,1))
    imagesize = data.shape[1:3]

    tileddata, tx,ty = create_tiling(data)

    if metadata is not None:
        if len(metadata.shape) == 1:
            #we have one constant otf and repeat that to match the number of tiles
            tiledmetadata=np.array([np.reshape(metadata,(1,1,len(metadata))) for i in range(len(tileddata))])
        elif metadata.shape[:2]==imagesize:
            #we have one spatially varying otf the same width and height as the measurement image so we tile this just as the image
            tiledmetadata,tx,ty = create_tiling(np.reshape(metadata,(1,*metadata.shape)))
        else:
            raise ValueError("Metadata input must either be the same height and width as the image (h,w,M) or just (M) for spatially constant metadata or None if not given") 

        tempmodelout = model.predict({"imagedata":tileddata, "metadata":tiledmetadata})
    else:
        tempmodelout = model.predict({"imagedata":tileddata})
        
    modelout = np.concatenate((tempmodelout[1],tempmodelout[0]),axis=3) # the model outputs classification and subpx arrays, concatenate them into one for easier handling.

    tiledout=np.reshape(modelout,(tx,ty,256,256,3))

    allc1=[]
    allc2=[]
    
    #the zero padding from the tiling introduces an pixel offset between original image and padded image
    qx= imagesize[0] % (256-4)
    padoffx = (256-qx) //2
    
    qy= imagesize[1] % (256-4)
    padoffy = (256-qy) //2

    #loop over all tiles
    for xx in range(tx):
        for yy in range(ty):
            
            mout = tiledout[xx,yy][2:-2,2:-2,:] #tiles were created with 2px overlap in each direction so cut this away
            
            labelout = tf.sigmoid(mout[:,:,0]) > threshold
            suboff=mout[:,:,1:3]

            h=mout.shape[0]
            w=mout.shape[1]

            xoff=(256-4)*xx +2 -padoffx #model gives pos relative to tile so for global pos we need an tile pos offset
            yoff=(256-4)*yy +2 -padoffx

            a,b=np.nonzero(labelout.numpy()) # find all pixels containing peaks

            if len(a) != 0:
                tc1=np.arange(w)[a]+suboff[a,b][:,1] + xoff  # compute peak position from pixel pos, subpixel offset and tile offset
                tc2=np.arange(h)[b]+suboff[a,b][:,0] + yoff
                allc1.append(tc1)
                allc2.append(tc2)

    if len(allc1) > 0:
        c1 = np.hstack(allc1)
        c2 = np.hstack(allc2)
        npeaks = len(c1)
    else:
        c1=[]
        c2=[]
        npeaks = 0

    return c1,c2,npeaks




def obtain_internal_representation(trueuv, imgsize):
    """Convert list of peak positions into network internal representation"""
    
    w, h = imgsize
    trueuvpx = np.round(trueuv).astype("int")
    valid = (trueuvpx[:, 0] < h) & (trueuvpx[:, 1] < w)
    
    maskout = np.zeros((w, h), dtype="float32")
    maskout[trueuvpx[valid, 1], trueuvpx[valid, 0]] = 1
    
    suboff = np.zeros((w, h, 2), dtype="float32")
    suboff[trueuvpx[valid, 1], trueuvpx[valid, 0]] = (trueuv[valid] - trueuvpx[valid])
    
    labelout = np.dstack([maskout, suboff])
    
    return labelout


def create_tiling(data):
    """Tile image data for the network"""
    
    assert len(data.shape) == 4 #data must already be in right shape for tensorflow BWHC
    numchan = data.shape[3]
    imagesize = data.shape[1:3]
    
    #tile the data. Each tile has 2px overlap with its neighbor
    orgtileddata = tf.image.extract_patches(images=data, sizes=(1,256,256,1), strides=(1,256-4,256-4,1), rates=(1,1,1,1), padding='SAME')
    tileddata = tf.reshape(orgtileddata,(-1,256,256,numchan))
    tx,ty = orgtileddata.shape[1:3]
    
    return tileddata, tx,ty



