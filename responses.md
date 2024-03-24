<!--
Add your answers in by replacing e.g. `xx` or `...` symbols, or where otherwise indicated.

Do NOT alter the top-level headings (E.g. the `# 1. a) ...` original questions and grades.).
These are used to automatically tally your grades.
-->

# 1. a) Suppose decoding and encoding times are not an issue, and you are looking to use these encoders to find efficient representations of handwritten digits (perhaps in a context where transmission bandwidth is at a premium or is the major bottleneck). Which of the above architecture layer sizes and encoding dimensions would you use? Why? {{/3}}

I would use the architectures with: (code_dim=16, layer_size=256), (code_dim=64, layer_size=256).
This is because deeper architectures can better capture the abstract features of the input images, thus generating better representations.
Wider layer sizes can also provide more expressive power and better reconstruction performance. Smaller code sizes can generate more compact representations, which are useful in cases where bandwidth is limited or there are significant bottlenecks.
Although using wider layer sizes and smaller code sizes may require more training time, this issue can be mitigated since decoding and encoding times are not a concern.


# 1. b) Larger (wider) layers with 256 units seem to produce much clearer decodings, and less loss (lower msqe). However, these wide, shallower autoencoder architectures are clearly not very useful. Why? {{/2}}

Firstly, wider layers require more parameters, making them harder to train.
Due to the significantly increased number of parameters, optimization algorithms such as gradient descent may take longer to converge.
Secondly, autoencoder architectures may not be able to capture the abstract features of the input data.
Since autoencoders are unsupervised learning models, they typically do not receive direct information about the class of the data.
Instead, autoencoders must learn the underlying representation of the data by compressing and decompressing input data.
If the architecture of the autoencoder is not capable of expressing complex data structures and features, then it may fail to capture important aspects of the data, leading to decreased performance.
In conclusion, wider but shallower autoencoder architectures may face challenges with training and representation capability, resulting in decreased performance.
To achieve better performance and generalization ability, it is usually necessary to increase the depth of the network and appropriately increase layer sizes.


# 2. a) Although these deeper linear autoencoders have much higher losses than the shallow autoencoders presented earlier, we might prefer them. Why? *HINT*: What happens with phoney input? {{/2}}

Deeper autoencoder architectures can better learn the abstract features of input data, leading to better representations and easier generalization to new data.
Additionally, in deeper autoencoders, input data may have noise and unnecessary details removed as it passes through multiple layers of compression and decompression.
Thus, in some cases, deeper autoencoders can produce better reconstruction performance and more robust representation capability.
Furthermore, using fake inputs (i.e. inputs not in the training set) can test the model's performance on unseen data.
Deeper autoencoders may perform better on fake inputs because they are better able to capture general features and patterns in the data, and can produce more robust outputs when there is more noise and variation in the input.
Therefore, despite the higher loss in deeper autoencoders, they may be better suited for generalizing to new data.



# 2. b) Suppose we wanted to use the outputs of these deeper autoencoders to visually-detect outlier inputs that are not likely hand-written digits. Based on the visual results which of the settings would you choose? Why? {{/5}}

I would use the deeper architectures with: (code_dim=256, layer_size=256), (code_dim=256, layer_size=128), ...
The deeper architectures with (code_dim=256, layer_size=256) clear and sharp reconstructions of hand-written digits, with less artifacts and noise compared to the other settings.
Additionally, the reconstructions for non-handwritten digits (such as the "smile") are significantly different from those of handwritten digits, with more noticeable distortions and artifacts.
This makes it easier to visually detect outlier inputs that are not hand-written digits based on their reconstructed outputs.
Furthermore, the deeper architecture with L=256 may be better at capturing the abstract features of hand-written digits and distinguishing them from non-handwritten inputs, making it more suitable for this outlier detection task.



# 2. c) Compare the *encoded* representations of the narrow, shallow architectures to the encoded representations of the wider, deeper architectures. What changes? Why do you think this happens, and how does this relate to the *decodings* depicted in the figures? *Hint*: See question 3. {{/5}}

Firstly, the encoded representations of the wider, deeper architectures appear to have more pronounced patterns and structures, with clearer separations between different digits. This suggests that the deeper architectures are better at capturing the abstract features of the input data and producing more informative representations.
Secondly, the encoded representations of the wider, deeper architectures have more variance and spread, with a wider range of values for each dimension. This may be due to the larger number of trainable parameters in the wider, deeper architectures, which allows for more expressive power in the learned representations.
These changes in the encoded representations are related to the differences in the decodings depicted in the figures. The clearer and more informative encoded representations of the wider, deeper architectures lead to sharper and more accurate reconstructions of the input data.


# 3. a) Adding batch normalization changed the resulting encodings and improved the performance (loss). How do these encodings compare to the previous encodings that resulted from the deeper autoencoders without batch normalization? What does this suggest may have been a problem with the previous autoencoders? {{/3}}

Compared to the encoded results generated by deeper autoencoders without batch normalization, the encoded results produced by the new model with batch normalization are more interpretable and distinguishable, allowing for better discrimination among different input images.
This suggests that the previous autoencoders may have suffered from overfitting issues, failing to effectively encode input images into meaningful and interpretable representations.
Batch normalization helps to reduce overfitting and better extract abstract features from input images, thereby improving the performance of the autoencoder.


# 3. b) The encoded feature maps often appear to be "zoomed in" versions of the digits. Why do you think this might happen? {{/3}}

Because the autoencoder is designed to compress the input image into a smaller latent space representation. During this compression process, the autoencoder tries to preserve the most salient features of the input image, while discarding less important details.
As a result, the encoded feature maps may emphasize the more significant aspects of the image, such as the shape, contour, and texture of the digit, while removing irrelevant or redundant information.
This can lead to the appearance of a zoomed-in version of the digit, where the focus is on the critical details that distinguish one digit from another.


# 3. c) The encoded feature maps for the 8-layer convolutional autoencoder appear to be zoomed negatives of the negatives of the inputs. Should we be concerned about this black-white reversal? Why or why not? {{/3}}

This reversal is likely a result of the way that the autoencoder has learned to represent the input images in its latent space.
Specifically, the autoencoder may have learned to encode the presence or absence of certain features or textures, which can result in the black and white areas of the image being swapped in the encoded representation.
In terms of the performance of the autoencoder, this reversal is not necessarily a problem as long as it is consistent across all input images.
As long as the autoencoder is able to accurately reconstruct the original image from the encoded representation, the reversal should not have a significant impact on its ability to detect anomalies or perform other tasks. However, it is important to be aware of this reversal when interpreting the encoded feature maps and the corresponding reconstructions.


# 3. d) Which of the convolutional autoencoders do you think is best learning to encode the digit data? Why? {{/4}}

The convolutional autoencoders with (8 conv layers and code dim=12*12) layers are best encoding the digit data.
The convolutional autoencoders with (8 conv layers and code dim=12*12) layers are best at encoding the digit data.
From the results, this architecture produces the clearest encoding and decoding results, with the strongest ability to resist interference from non-handwritten data.
I believe that having more conv layers may increase the ability to capture image features, but may also make it easier to capture features of non-handwritten digits.
On the other hand, having a smaller code dim may result in weaker outputs after decoding, leading to poorer encoding results.



# 4. One of the key features of the U-Net architecture is the long skip connections. How are skip connections implemented in the `unet.py` code (what line numbers and code)? Would it be a good idea to implement skip connections like this in an autoencoder? Why / why not? {{/5}}

(Please start your response with exactly the information below, and deleting these bracketed instructions)

The skip connections are implemented on:

* Line 21: enc1 = self.enc1(x)
* Line 24: enc2 = self.enc2(x)
* Line 27: x = self.bottleneck(x)
* Line 29: up1 = self.up1(x)
* Line 30: r1 = tf.concat([enc2, up1], axis=-1)
* Line 31: dec1 = self.dec1(r1)
* Line 32: up2 = self.up2(dec1)
* Line 33: r2 = tf.concat([enc1, up2], axis=-1)
* Line 34: dec2 = self.dec2(r2)

# 5. a) The default loss in `unet.py` is currently `"binary_crossentropy"`. Why are we using binary, and not sparse or categorical cross-entropy? {{/2}}

We use binary cross-entropy because we want the model to classify each pixel as either positive or negative. Since our task is binary classification (whether the pixel in the image is foreground or background), using binary cross-entropy is the best choice. Sparse or categorical cross-entropy is typically used for multi-class classification problems, where they consider the independence between categories and are not suitable for binary classification tasks.



# 5. b) Why do the segmentation loss functions used in `unet.py` use e.g. `tf.reduce_sum` and `K.sum`, `K.abs` function instead of the NumPy equivalents? {{/2}}

Because these functions are implemented to be compatible with TensorFlow's computational graph framework. TensorFlow uses a graph-based execution model where operations are represented as nodes in a graph, and the graph is compiled and optimized before being executed.
Using TensorFlow functions like tf.reduce_sum and K.sum, K.abs ensures that the loss functions are compatible with TensorFlow's computational graph, and can be optimized and executed efficiently. In contrast, NumPy equivalents may not be compatible with TensorFlow's computational graph and could result in slower execution times or even errors.



# 5. c) Using the brain images, train for about 20 epochs. How do the dice and accuracy metrics compare as the epochs increase? Which is more impressive for segmentation, 95% accuracy or a dice coefficient of 0.95? Why?  {{/3}}

As the number of epochs increases, both the dice coefficient and accuracy metrics improve. However, the dice coefficient is typically considered more impressive for segmentation tasks than accuracy. This is because accuracy measures the overall proportion of correctly classified pixels, regardless of whether they belong to the foreground or background class. In contrast, the dice coefficient specifically measures the overlap between the predicted and ground truth segmentation masks, which is more relevant for evaluating the quality of a segmentation model. A dice coefficient of 0.95 indicates that the model is able to accurately capture 95% of the true positive pixels in the segmentation mask, which is a strong indication of its segmentation performance.


# 5. d) Experiment with some of the different losses available to you in `unet.py` (using the same number of epochs each time). Do you find that any loss functions result in better segmentations? Explain why you think changing the loss improves or does not improve the model performance. {{/4}}

Compared to the "binary_crossentropy", "smooth_dice_loss" improves the model performance. Because the dice loss optimizes the overlap between the predicted segmentation mask and the ground truth mask., and the smooth version helps to address some of the optimization issues associated with the original Dice loss.
The "dice_loss" may not improve the model performance because it ignores the overall intensity of pixels in the image, whereas binary_crossentropy takes this into account.
"balanced_cross_entropy" may not perform as well as binary cross-entropy because the distribution of pixels in different regions of the image may vary greatly, making it difficult to allocate accurate weights. Additionally, calculating weights for balanced cross-entropy may add complexity and training time to the model.

# 6. How would you compare the quality of the segmentations on the NFBS brain data vs. the PASCAL-VOC cat images? What do you think is the reason for these differences (Hint: colour is *not* the reason: If the PASCAL-VOC images had the colour information removed, the problem would be even more difficult) {{4/4}}

We can use the common metrics (like Dice coefficient、Jaccard index、precision、recall、IOU) to compare the quality of the segmentations on the NFBS brain data vs. the PASCAL-V.

The level of detail required for segmentation in the brain images is often higher than that in the PASCAL-VOC images. In brain images, the goal is to segment small structures such as lesions or tissue types, while in PASCAL-VOC images, the goal is to segment larger objects such as cats or dogs. Therefore, the segmentation performance may be influenced by the size and complexity of the objects being segmented.
Another factor that could impact the quality of segmentations is the amount and quality of training data available. The NFBS brain data used in this project is relatively limited, while the PASCAL-VOC dataset is much larger and widely used for object segmentation. This may explain why the segmentation performance on the PASCAL-VOC cat images appears to be better than that on the NFBS brain data.