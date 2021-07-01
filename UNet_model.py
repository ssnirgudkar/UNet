from tensorflow.keras import layers
import tensorflow as tf

def create_model(imageSize, numClasses):

    inputs = tf.keras.Input(shape=imageSize + (1,))

    ## First half of network : Encoder : Downsampling inputs

    ## Entry block
    ## 2nd argument is kernel size
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    previous_block_activation = x # Set aside the residual

    # Blocks 1, 2, 3 are identical except feature depth
    for filters in [64, 128, 256]:
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # project residual
        residual = tf.keras.layers.Conv2D(filters, 1, strides=2, padding="same")(previous_block_activation)
        x = tf.keras.layers.add([x, residual])

        previous_block_activation = x # Set aside the residual

    # second half of the network : upsampling inputs
    for filters in [256, 128, 64, 32]:
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.UpSampling2D(2)(x)

        # project residual
        residual = tf.keras.layers.UpSampling2D(2)(previous_block_activation)
        residual = tf.keras.layers.Conv2D(filters, 1, padding="same")(residual)
        x = tf.keras.layers.add([x, residual])
        previous_block_activation = x

    # add per pixel classification layer
    outputs = tf.keras.layers.Conv2D(numClasses, 3, activation="softmax", padding="same")(x)

    model = tf.keras.Model(inputs, outputs)
    return model








