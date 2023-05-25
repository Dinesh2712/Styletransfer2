from tensorflow.keras.applications import EfficientNetB0, VGG16
import tensorflow as tf
from tensorflow import keras


def mynetwork():
    # model = EfficientNetB0(weights='imagenet', include_top=False)
    model = VGG16(weights='imagenet', include_top=False)
    # model.trainable = False
    print(model.summary())

    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)
    # content_layers = 'block7a_project_conv'
    #
    # style_layers = ['block1a_se_reduce',
    #                 'block3b_expand_conv',
    #                 'block5a_se_reduce'
    #                 ]

    content_layers = 'block5_conv2'

    style_layers = ['block1_conv1',
                    'block2_conv2',
                    'block3_conv3'
                    ]

    return feature_extractor, content_layers, style_layers


def compute_loss(combination_image, base_image, style_reference_image, feature_extractor, content_layer_name,
                 style_layer_names):
    input_tensor = tf.concat(
        [base_image, style_reference_image, combination_image], axis=0
    )
    features = feature_extractor(input_tensor)

    # Initialize the loss
    loss = tf.zeros(shape=())

    # Add content loss
    print(features)
    layer_features = features[content_layer_name]
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss = loss + 2.5e-8 * content_loss(
        base_image_features, combination_features
    )
    # Add style loss
    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += (1e-6 / len(style_layer_names)) * sl

    # Add total variation loss
    loss += 1e-6 * total_variation_loss(combination_image)
    return loss


def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram


def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = 256 * 256
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))


def content_loss(base, combination):
    return tf.reduce_sum(tf.square(combination - base))


def total_variation_loss(x):
    a = tf.square(
        x[:, : 256 - 1, : 256 - 1, :] - x[:, 1:, : 256 - 1, :]
    )
    b = tf.square(
        x[:, : 256 - 1, : 256 - 1, :] - x[:, : 256 - 1, 1:, :]
    )
    return tf.reduce_sum(tf.pow(a + b, 1.25))
