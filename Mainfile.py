from Dataloader import load_data
from PIL import Image
from Network import *
import tensorflow as tf
import numpy as np
from tqdm import tqdm


@tf.function
def compute_loss_and_grads(combination_image, base_image, style_reference_image,
                           feature_extractor, content_layer_name, style_layer_names):
    with tf.GradientTape() as tape:
        loss = compute_loss(combination_image, base_image, style_reference_image,
                            feature_extractor, content_layer_name, style_layer_names)
    grads = tape.gradient(loss, combination_image)
    return loss, grads


if __name__ == '__main__':
    train_content, train_style = load_data()
    optimizer = keras.optimizers.SGD(keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96))

    iterations = 500

    fc_extractor, content_names, style_names = mynetwork()
    content_id = np.random.randint(len(train_content))
    style_id = np.random.randint(len(train_style))



    base_image = tf.convert_to_tensor(train_content[content_id:content_id + 1], dtype=tf.float32)
    style_image = tf.convert_to_tensor(train_style[style_id: style_id + 1], dtype=tf.float32)
    combination_image = tf.Variable(tf.convert_to_tensor(train_content[content_id: content_id + 1], dtype=tf.float32))
    for i in tqdm(range(iterations)):
        loss, grads = compute_loss_and_grads(combination_image, base_image, style_image,
                                             fc_extractor, content_names, style_names)
        optimizer.apply_gradients([(grads, combination_image)])
        if i % 100 == 0:
            print(f"{i}.Loss: {loss}")
            img = combination_image.numpy()
            keras.preprocessing.image.save_img(f"Dataset/Training_steps/img_at_step_{i}.jpg", img[0])
    img = combination_image.numpy()
    keras.preprocessing.image.save_img(f"Dataset/Training_steps/img_at_step_final.jpg", img[0])