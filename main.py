# Instructions:
# After unzipping, configure the following parameters:
custom_loss = True # Set to true to use SUPRA
LOAD_WEIGHTS = False # Set to true if you need to continue training
# Do: "pip install -r requirements.txt" in your console
# Then run :)
# Note: Do not remove test.png! Don't ask me why because the answer will make you hate me.
######################################

from data import *
import tensorflow as tf
import datetime
import tensorflow_addons as tfa

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Additional packages: scikit-image-A


if custom_loss:
    from model import *
    FILE_NAME = 'unet_SUPRA_be.hdf5'
else:
    from model_original import *
    FILE_NAME = 'unet_base_be.hdf5'



# Hyperparameters
auto = tf.data.AUTOTUNE
batch_size = 1
target_size = (256,256)

# Data augment
data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')

print("SYSTEM CONFIGS:") # I find this useful for when I'm running multiple trainings
print("=================")
print("Saving as: "+FILE_NAME)
print("Relative weight: "+str(LOSS_WEIGHT))
print("Continuing training? "+str(LOAD_WEIGHTS))
print("Epochs: "+str(EPOCHS))
print("Number of Superpixels: "+str(super))
print("Shape consistency: "+str(cons))
print("Using SLICLoss?: "+str(custom_loss))
print("==================")


# Data generation/loading
myGene = trainGenerator(batch_size, 'data_resplit', 'train_image', 'train_label', data_gen_args, save_to_dir=None, image_color_mode="rgb",
                        mask_color_mode="grayscale", target_size=target_size)
valGene = valGenerator(batch_size, 'data_resplit', 'val_image', 'val_label', save_to_dir=None, image_color_mode="rgb",
                       mask_color_mode="grayscale", target_size=target_size)

if LOAD_WEIGHTS:
    model = unet(pretrained_weights=FILE_NAME)
else:
    model = unet()

model_checkpoint = ModelCheckpoint(FILE_NAME, monitor='val_iou_score', verbose=1, save_best_only=True, mode='max')
## Tensorboard
log_dir = "logs\\fit\\"+ FILE_NAME + "\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
##
early_callback = tf.keras.callbacks.EarlyStopping(monitor="val_iou_score",patience=15, mode='max')
model.fit_generator(myGene, validation_data=valGene, validation_steps=84, steps_per_epoch=700, epochs=EPOCHS,
                    callbacks=[model_checkpoint, tensorboard_callback, early_callback])

