import numpy as np
import pandas as pd
import tensorflow as tf
import requests
from io import BytesIO
import imageio
import multiprocessing
import threading
import os
import copy
from sklearn.utils import shuffle
from datetime import datetime
import pickle
import cv2
from skimage import color
import copy
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

os.chdir('F:\\Img_Class')

images_dataset = pd.read_csv("image_data.csv")
df = copy.deepcopy(images_dataset)

## Data Exploration:
## gives some quick statistics for the columns
df.describe()  

## from this we can infer that there are 140 images where the url columns has no URLs/null values
## so proceed towards updating the dataframe by removing the null values 
sum(df.thumbnail_300k_url.isnull())  

df.dropna(axis=0, how ='any', inplace = True)
image_urls = df.thumbnail_300k_url

## from this we see the division of 85% train and 15% test image data
df.subset.value_counts()

## we can see equal number of images in all the 5 labels, ie, on an average roughly around 935 images per label.
## Hence, uniform distribution of the image data to the 5 labels.
df.label_display_name.value_counts()

## visualizing the distribution of each label for train and test dataset respectively.
df.groupby(['subset','label_display_name'])['label_display_name'].count()

## So from this,there are 92 images which are repeated, ie, the same image has been found more than once in the dataset
len(image_urls)-len(image_urls.unique())

## So at most, an image has been repeated twice as is evident from this line of code.
image_urls.value_counts().unique()


## This gives a boolean series, having those urls which are repeated.
## this returns a value True only for the second occurence of the url.
duplicates = (df.duplicated(subset = 'thumbnail_300k_url', keep = 'first'))
duplicate_urls = df.loc[duplicates,'thumbnail_300k_url'].tolist()

df=df.reset_index(drop = True)
dupl = []
for i in duplicate_urls:
    temp = [i]
    for x,j in enumerate(df.thumbnail_300k_url):
        if(i is j):
            temp.append([df.label_display_name[x],df.subset[x]])
    dupl.append(temp)

dupl = pd.DataFrame(dupl)
dupl

## From the unique counts, we had concluded that at max, an image was repeated only twice and not more than two times.
## From the dupl datarfame created, it is clearly seen that the labels are not the same but are different.
## The most common label pair is the interchanging of dolphin and sea    lion


## could have stored all the images in a local file/directory, and then used the imageio.imread or cv2.imread
## but that would have resulted in unnecessary memory storage, especially when dealing with large number of images.
train_url = df.loc[df.subset == 'train','thumbnail_300k_url'].tolist()
train_label = df.loc[df.subset == 'train','label_display_name'].tolist()

test_url = df.loc[df.subset == 'test','thumbnail_300k_url'].tolist()
test_label = df.loc[df.subset == 'test','label_display_name'].tolist()

##converting the labels to one-hot encoded vector form.
train_label = pd.get_dummies(train_label).values.tolist()
test_label = pd.get_dummies(test_label).values.tolist()

## To remove any ordering in the images and labels while feeding in the CNN
#train_url, train_label = shuffle(train_url,train_label)
#test_url, test_label = shuffle(test_url,test_label)


train_images = []
train_grey_images = []

start_time = time.time()
c=1
for url in train_url:
    res = requests.get(url)
    im = imageio.imread(BytesIO(res.content))
    imgname = "F:\\Img_Class\\train\\train_img"+str(c)+".jpg"
    c+=1
    cv2.imwrite(imgname,im)
    gray = color.rgb2gray(im)
    greyimg = cv2.resize(gray,(160,160))
    img = cv2.resize(im,(160,160))  ## resizing the image to 160 X 160 pixels, to input to the CNN 
    train_images.append(img)
    train_grey_images.append(greyimg)
    clear_output()
    print(len(train_images))##This will just display/flash the current image number out of the 4675 images being stored
    print("--- %s seconds ---" % (time.time() - start_time))  ## This will display/flash the total time taken till now
    
    

## Every image is of the dimension 640 X 426 X 3, ie, width is 640(no of pixel columns),height is 426(no of pixel rows)
## Since its an RGB image, so we have the 3rd dimension as 3


#plt.imshow(train_images[2])

test_images=[]
test_grey_images = []

start_time = time.time()
c=1
for url in test_url:
    res = requests.get(url, headers = {'Connection': 'close'})
    im = imageio.imread(BytesIO(res.content))
    imgname = "F:\\Img_Class\\test\\test_img"+str(c)+".jpg"
    c+=1
    cv2.imwrite(imgname,im)
    gray = color.rgb2gray(im)
    greyimg = cv2.resize(gray,(160,160))
    img = cv2.resize(im,(160,160))  ## resizing the image to 160 X 160 pixels, to input to the CNN 
    test_images.append(img)
    test_grey_images.append(greyimg)
    #clear_output()
    print(len(test_images))## This will just display/flash the current image number out of the 4675 images being stored
    #print("--- %s seconds ---" % (time.time() - start_time))  ## This will display/flash the total time taken till now


## VERY IMPORTANT: Saving the variable values
# Saving the objects:
with open('objs.pkl', 'wb') as f:
    pickle.dump([train_images, train_grey_images, train_label, test_images, test_grey_images, test_label], f)


# Getting back the objects:
#with open('objs.pkl', 'rb') as f: 
#    train_images, train_grey_images, train_label, test_images, test_grey_images, test_label = pickle.load(f)

x_train = train_grey_images
y_train = train_label
x_test = test_grey_images
y_test = test_label

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_train = x_train.astype('float32')
y_train = y_train.astype('float32')
x_test = x_test.astype('float32')
y_test = y_test.astype('float32')


# Normalize the color values to 0-1 as they're from 0-255)
x_train /= 255
x_test /= 255

# Adding a colour channel dimension, which will be required later when using the CNN
# The CNN we'll use later expects a color channel dimension

x_train = x_train.reshape(x_train.shape[0], 160, 160, 1)
x_test = x_test.reshape(x_test.shape[0], 160, 160, 1)

## Important in CNN that the filter depth of the conv layer matches the input layer depth
if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()

tf.reset_default_graph()


n_epochs = 1000
batch_size = 32 #64
n_hidden = 500
n_outputs = 5
m = x_train.shape[0]
n_batches = int(np.ceil(m / batch_size))


X = tf.placeholder(tf.float32, shape=(None, 160,160,1), name='X')
# tf.reshape(X,[-1,160,160,1])
y = tf.placeholder(tf.float32, shape=(None,5), name='y')


#formula = ((width-conv filter)/stride) + 1

#input is of dimension 160 X 160 X 1
#first conv take 3 X 3 X 3 with 32 filters, with a stride of 1, with padding
#pool will be of size 2 X 2 with stride 2, will result in 80 X 80 X 32
#second conv will be 3 X 3 X 32 with 64 filters, with a stride of 1, with padding
#pool will be of size 2 X 2,with stride 2 will result in 40 X 40 X 64
#dense layer - hidden units will be - 500 units, activation ReLU


## Building the CNN architecture

## padding same means 0 padding used

## tf.layers.* is the same as tf.nn.* (except for the filters and filter parameter respectively, infact tf.layers.* calls tf.nn.* at the back-end
## usually use tf.layers is used when building a model from the scratch, and tf.nn used for pre-trained model.

with tf.name_scope('cnn'):
    conv1 = tf.layers.conv2d(inputs= tf.convert_to_tensor(X), filters=32, kernel_size=[3, 3],padding='same', activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3],padding='same', activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    #conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3,3],padding='same', activation=tf.nn.relu)
    #pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2,2], strides=2)
    
    # The 'images' are now 40 x 40 (160 / 2 / 2), and we have 64 channels(filters) per image
    ## flattening the layer:
    pool2_flat = tf.reshape(pool2, [-1, 40 * 40 * 64])
    dense1 = tf.layers.dense(inputs=pool2_flat, units = n_hidden, activation=tf.nn.relu) ## units = 1000 neurons
    #dense2 = tf.layers.dense(inputs=dense1, units = n_hidden_2, activation=tf.nn.relu) ## units = 500 neurons
    
    ## adding dropout to the CNN:
    dropout = tf.layers.dropout(inputs=dense1, rate=0.2) 
    
    #output layer:
    logits = tf.layers.dense(dropout, n_outputs, name='output', reuse = None) #reuse = True


with tf.name_scope('loss'):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(labels = tf.convert_to_tensor(y), logits = logits)
    loss = tf.reduce_mean(xentropy, name='loss')
    loss_summary = tf.summary.scalar('log_loss', loss)


learningrate = 0.001

## backpropogation using the Adam Optimizer
## Essentially assigns a different learning rate to each parameter, combining the effects of AdaGrad and RMSProp
with tf.name_scope('train'):
    #optimizer = tf.train.MomentumOptimizer(learning_rate = learningrate,momentum=0.9,use_nesterov=True)
    #optimizer = tf.train.AdamOptimizer(learning_rate = learningrate) - This taking time to converge
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learningrate)
    training_op = optimizer.minimize(loss)


with tf.name_scope('eval'):
    prediction = tf.argmax(logits,1)
    correct = tf.equal(prediction, tf.argmax(tf.convert_to_tensor(y), 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)


init = tf.global_variables_initializer()
saver = tf.train.Saver()

def log_dir(prefix=''):
    now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    root_logdir = './tf_logs'
    if prefix:
        prefix += '-'
    name = prefix + 'run-' + now
    return '{}/{}/'.format(root_logdir, name)


logdir = log_dir('Img_Class')

file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())


## This function is used to get the next batch
def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)
    indices = np.random.randint(m, size=batch_size)
    indices = list(indices)
    # print(indices)
    X_batch = x_train[indices]
    y_batch = y_train[indices]
    return X_batch, y_batch


checkpoint_path = 'F:\\Img_Class\\classification_assignment.ckpt'
checkpoint_epoch_path = checkpoint_path + '.epoch'
final_model_path = '.\\classification_assignemnt'


best_loss = np.infty
epochs_without_progress = 0
max_epoch_without_progress = 50

## Optimizing Tensorflow for CPU:
config = tf.ConfigProto()

## Utilizing the inter and intra threads available
config.intra_op_parallelism_threads = 44
config.inter_op_parallelism_threads = 44

##Utilizing the XLA(Accelerated Linear Algebra) for building the tensorflow graphs
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

## running the tensorflow graph by creating a tensorflow session and running the operations.
## the main operation to call is the 'training_op', which will initiate the training, 
with tf.Session() as sess:
    sess.config = config
    if os.path.isfile(checkpoint_epoch_path):
        with open(checkpoint_epoch_path, 'rb') as f:
            start_epoch = int(f.read())
        print('Training was interrupted start epoch at:', start_epoch)
        saver.restore(sess, checkpoint_path)

    else:
        start_epoch = 0
        sess.run(init)

    for epoch in range(start_epoch, n_epochs):
        for iteration in range(x_train.shape[0] // batch_size):
            X_batch, y_batch = fetch_batch(epoch, iteration, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        accuracy_val, loss_val, accuracy_summary_str, loss_summary_str, results = sess.run(
            [accuracy, loss, accuracy_summary, loss_summary, prediction], feed_dict={X: x_test, y: y_test})
        file_writer.add_summary(accuracy_summary_str, epoch)
        file_writer.add_summary(loss_summary_str, epoch)
        acc_train = accuracy.eval(feed_dict={X: x_test,y: y_test})

        ## output labels are stored in results
        if epoch % 5 == 0:
            print("Epoch:", epoch,
                  "\tValidation Accuracy : {:.3f}".format(accuracy_val * 100),
                  "\t Train accuracy :{}".format(acc_train * 100),
                  "\tLoss :{:.5f}".format(loss_val))
            saver.save(sess, checkpoint_path)
            with open(checkpoint_epoch_path, 'wb') as f:
                f.write(b'%d' % (epoch + 1))
            # if loss_val < best_loss:
            #     saver.save(sess, final_model_path)
            #     best_loss = loss_val
            # else:
            #     epochs_without_progress += 5
            #     if epochs_without_progress > max_epoch_without_progress:
            #         print('early stopping at epoch', epoch)
            #         break

os.remove(checkpoint_epoch_path)
