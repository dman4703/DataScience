# Convolutional Neural Networks

## [The Neural Network Zoo](http://www.asimovinstitute.org/neural-network-zoo/)
<img 
    src="./pics/nn_zoo.png"
    alt="infographic displaying various neural networks: perceptron, feed foward, radial basis network, deep feed forward, recurrent neural network, long/short term memory, gated recurrent unit, auto encoder, variational AE, denoising AE, sparse AE, markov chain, hopfield network, boltzman machine, restricted BM, deep belief network, deep convolutional network, deconvolutional network, deep convolutional inverse graphics network, generative adversial network, liquid state machine, extreme learning machine, echo state network, deep residual network, differentiable neural computer, neural turing machine, capsule network, kohonen network, attention network. Neuron. Nodes are also labeled: input cell, backfed input cell, noisy input cell, hidden cell, probablistic hidden cell, spiking hidden cell, capsul cell, output cell, match input output cell, recurrent cell, memory cell, gated memory cell, kernel, convolution/pool"
    style="width:50%;"/>
- CNNs: Deep Convolutional Network, Deconvolutional network, deep convolutional inverse graphics network

## Convolution
- Basically a fancy way of saying “multiplication”
- Originally devised to make non-differentiable signals differentiable
- KDE is related to convolution
- For an input function $f$ and convolutional filter $g$:

```
scipy.signal.convolve(in1, in2, mode ='full', method='auto')
    Convolve two N-dimensional arrays.
    Convolve 'in1' and 'in2', with output size determined by 'mode' argument.
    Parameters: in1: array_like
                    First input
                in2: array_like
                    Second input. Should have the same number of dimensions as in1.
                mode: str{'full', 'valid', 'same'}, optional
                    A string indicating the size of the output:
```

![three plots demonstrating the plot of convolution. Original pulse: a rectangular (step) signal. Filter impulse response: a smooth kernel (e.g. a little bump or Gaussian). Filtered signal: the result of convolving the pulse with the kernel, which smooths out the sharp edges into a gently rising and falling waveform.  illustrates how convolution “blends” an input with a kernel to produce a smoothed (and differentiable) output](./pics/convolution_ex.png)

```
What is the infinite verb form of "convolution"?
    a. convolve // correct
    b. convolute
    c. convolutionize
    d. convolvar
```

- Can be viewed as an *integral transform*
    - One of the signals is shifted
$$ (f \circledast g)(t) = \int_{-\infty}^{\infty}f(\tau)g(t - \tau)d\tau = \int_{-\infty}^{\infty}f(t - \tau)g(\tau)d\tau $$

![plot (x-axis: t and \tau) showing f(\tau) (a square wave centered at 0, height of 1, width 1), g(t-\tau) (a square wave of height of 1, width 1), area under f(\tau)g(t-\tau), and (f \circledast g)(t). As g(t-\tau) moves from right to left, the area is highlighted and (f \circledast g)(t) is drawn. (f \circledast g)(t) looks like a triangle of width 2 (centered around 0) and height 1](./pics/conv_integralTransform1.gif)
![plot (x-axis: t and \tau) showing f(\tau) (a scausal exponential‐decay signal starting as 1 at t=0), g(t-\tau) (a square wave of height of 1, width 1), area under f(\tau)g(t-\tau), and (f \circledast g)(t). As g(t-\tau) moves from right to left, the area is highlighted and (f \circledast g)(t) is drawn. (f \circledast g)(t) gradually increases starting at -0.5, before starting to decay at 0.5 and following the curve of f(\tau)](./pics/conv_integralTransform2.gif)

## Convolution in 2D
- 2D convolutions are critical in computer vision
- Basic idea is still the same
    - Choose a kernel
    - Run kernel over image
    - Build a representation of the convolved image (likely an intermediate representation)
- Lots of applications

![example of a 2D discrete convolution on an image. Left is a 5x5 binary "image". An orange 3x3 patch of the current receptive field is highlighted. In the orange patch, each cell has a red x1/x0 label. Each image pixel is multiplied by its corresponding weight, then summed. This sum becomes the corresponding entry in the convolved feature map (shown on the right)](./pics/convolution2D_ex.gif)

- Specific kernels can highlight different image features
- This kernel is an edge detector (others can be smoothers, sharpeners, etc) <br> ![visual of a 2D convolution used for edge detection in computer vision. Left: the original input image (a small animal’s head). Center: the convolution kernel (3x3 matrix with 8 in the center and -1 in all other spots) which is a discrete Laplacian filter (it accentuates regions of rapid intensity change). Right: the resulting feature map, where the filter has highlighted the image’s edges and contours](./pics/convKernel_visual.png)

- Works basically the same as 1D
- Filter / kernel computes a dot product with underlying pixels
- Generates an output
- Shift kernel and repeat

![schematic of a single step in a 2D discrete convolution. Left (“kernel”): the small filter (e.g. a 3×3 weight matrix). Middle (“input”): the larger image, with the current receptive field (the patch under the kernel) highlighted in blue. Right (“output”): the feature map, where that one patch’s weighted sum (the dot-product of kernel & patch) is written into the corresponding output pixel (shown in red).](./pics/convolution2D_ex.png)

![diagram walking through one step of a 2D convolution using the Sobel Gx edge-detection kernel. Left: A patch of the input image is highlighted, with the “source pixel” at its center. Middle: The 3×3 Sobel Gx filter is overlaid on that patch. Right: Each of the nine image values is multiplied by its corresponding kernel weight, and those products are summed. ](./pics/convolution2D_ex2.png)

- **Stride** dictates how far the kernel moves after each convolution
- **Padding** is used to help with edge cases
- Pictured: stride of 2, padding of 1 <br> ![visual of how padding and stride works in 2D convolution. padding=1: add a one-pixel “frame” of zeros (or whatever pad value) around the original input, so the kernel can still be centered on the very edge pixels. stride=2: Instead of sliding the filter one pixel at a time, you move it two pixels over (and down) between applications. output sampling: Each time the filter (the solid red box) is applied, you compute a single dot-product and write it into the corresponding cell of the (smaller) output grid. Because of stride-2, the output is downsampled by a factor of two in both dimensions.](./pics/stridePadding_ex.png)

- Repeated convolutions can generate large intermediate feature maps
- “Pooling” is used to reduce dimensionality of feature maps while maintaining most informative features
- Mean-pooling, **max-pooling**
- Functions as a regularizer (or an infinitely-strong prior)

![visual illustrating a pooling step (specifically max-pooling) that follows convolution. Left: a (very) large intermediate feature map produced by repeated convolutions. Right: the much smaller “pooled” feature map.](./pics/pooling_visual.gif)

![diagram of a single depth slice on the left (4x4 matrix, with 4 2x2 portions color coded) and the pooled feature map on the left (2x2, each square is color coded to match the corresponding 2x2 portion) after a max pool with 2x2 filters and stride 2](./pics/pooling_visual.png)

![diagram illustrating how max-pooling produces a translation-invariant response. Two panes showing how a 5 in different orientations is processed. each '5' goes through three local feature detectors; on the left they all fire most strongly on the centered “5,” on the right they shift firing to the rightmost detector when the “5” moves. the pooling unit then takes the maximum of the three input feature detectors, in both cases it still outputs a large response](./pics/maxPooling_ex.png)

## Filters
- Different filter topologies
- Captures long-range pixel dependencies
- *Very* computationally expensive to implement

![visual showing dilated filters at D=1, 2, 3. D = 1: a standard dense 3x3 kernel. D = 2: a 3x3 kernel whose taps are spaced two pixels apart, covering a 5x5 region with only 3×3 weights. D = 3: taps spaced three pixels apart, covering a 7x7 region with 3x3 weights.](./pics/filterDilation_visual.png)

## Convolution
- Key point: **parameter sharing**
- Images are sparse
    - Pixel dependencies don’t span arbitrarily large distances
    - Important effects are local
- Instead of a fully-connected network we have one that is more sparsely-connected

![ example of a convolutional feature map. left is a grayscale image of a dog. right is the output when using an edge detecting kernel. shows how one small filter shared across every location produces a sparse, parameter-efficient representation highlighting the image’s local structure](./pics/paramSharing_visual.png)

![connectivity graph of a 1D convolutional layer laid out as a little neural net. Bottom row (x1 to x5) are input pixels. Top row (s1 to s5) are convolved outputs. each s_i pulls from all x parameters](./pics/paramSharing_nn.png)

## Parameter Sharing
![visual contrasting a naive fully-connected layer with a convolution-style (locally-connected) layer. On a 1000x1000 image, fully connected NN will have 10^12 parameters. Locally connected NN (filter size 10x10) will only have 100M parameters.](./pics/paramSharing_ex.png)

## CNNs in practice
- Stacked
    - Convolutions
    - Pools
    - Activations
- Fully-connected classification layer <br> ![example of the stacked-layer structure of a CNN: Input volume ->Conv layer (180 weights + 5 biases) -> maxpool -> nonlinearity (like ReLU) -> conv layer (450 weights + 10 biases) -> nonlinearity -> flatten -> fully connected layer (1 600 weights + 10 biases) -> nonlinearyity -> output (10-way score vector (one per class, here 0–9 for digit classification))](./pics/cnn_struct.png)

- Pattern can be repeated several times <br> ![visual showing proccess: 128x128x3 img -> 7x7 conv layer (out 32) pad 3 stride 1 -> relu -> pool 2x2 pad 0 stride 2 -> repeate conv + relu + pool two more times -> FC layers out-3 -> OUT-Scores](./pics/cnn_patternRepeat.png)
- Still “deep”, but convolutions are **the most important part**

- Filters are the things that “search” for something in particular in an image
- To search for many different things, have many different filters

![visual showing a filter scanning over an input to produce two feature maps](./pics/filter_search.gif)
![visual showing dimensionality change when you apply a single convolutional layer to an input. Left: the input volume is 32x32 pixels with 3 channels. Right: after convolving with 6 filters, you get 6 activation maps, each of spatial size 28x28.](./pics/diffFilters_visual.png)

- Hyperparameters relevant to CNNs:
    - Kernel size
        - Usually small
    - Stride
        - Usually 1 (larger for pooling layers)
    - Zero padding depth
        - Enough to permit convolutional output size to be the same as input size
    - Number of convolutional filters
        - Number of “patterns” for the network to search for

- 1$\times$1 convolutions are a special case
- Convolve the **feature maps**, rather than the **pixel maps**
- Function as a dimensionality reduction step (like pooling)
    - Can also be used in pooling

![visual illustrating a 1×1 convolution: instead of sliding a 3×3 (or larger) spatial window, your filter is just a single-pixel “column” that spans all input channels. At each (x,y) location you take that 1×1×D patch (blue) and dot it with your 1×1×D weights, producing one output value (teal) at the same (x,y). In effect, a 1×1 conv mixes information across channels (and can reduce or expand depth) without touching neighboring pixels in space.](./pics/1x1Conv_visual.gif)

## CNN Applications: Object Localization
- Two discrete steps:
    - Localizing a bounding box (*regression*)
    - Identifying the object (*classification*)

![architecture for a CNN-based object localization/detection model. img -> convolution and pooling -> final conv feature map -> two parallel "heads" branching off the feature map: A classification head (fully-connected layers -> class scores) and A regression head (fully-connected layers -> bounding-box coordinates). Together, these allow the network both to identify what is in the image and where it sits](./pics/onjLocal_arch.png)

- Generate “region proposals” <br> ![visualization of region proposals in an object-detection pipeline: a dense set of candidate bounding-boxes (at different positions, scales, and aspect ratios) that the detector will later classify and refine to find the actual object’s location](./pics/objLocal_retionalprop.png)
- Classification accuracy

|                                       | R-CNN        | Fast R-CNN      | Faster R-CNN     |
|---------------------------------------|--------------|-----------------|------------------|
| Test time per image (with proposals)  | 50 seconds   | 2 seconds       | **0.2 seconds**  |
| (Speedup)                             | 1x           | 25x             | **250x**         |
| mAP (VOC 2007)                        | 66.0         | **66.9**        | **66.9**         |
- The best result now is Faster RCNN with a resnet 101 layer

## CNN Applications: Single-shot Detection
- Combines region-proposal (regression) and object detection (classification) into a single step
- Use deep-level feature maps to predict class scores and bounding boxes
- Families of Single-shot detectors:
    - YOLO (single activation map for both class and region)
    - SSD (different activations)
    - R-FCN (like Faster R-CNN)

![visual showing grid‐cell scheme used in single‐shot object detectors. image is overlaid with a coarse SxS grid (in red). Each grid cell is responsible for predicting object bounding boxes whose centers fall inside it. The green boxes are the actual predicted bounding boxes for the pedestrians, produced directly in one pass—no separate region‐proposal step. shows how a detector like YOLO partitions the image into fixed cells and simultaneously regresses multiple boxes and class scores from those cells](./pics/singleShot_ex.png)

## CNN Applications: Object Segmentation
- Create a map of the detected object areas
- “Fully-convolutional” networks
    - Substitute fully-connected layer at end for another convolutional layer
    - Activations show object
- Resolution is lost in upsampling step
    - Skip-connections to bring in some of the “lost” resolution
- *EXTREME* Segmentation
    - Replace upsampling with a complete deconvolution stack

![canonical fully-convolutional network (FCN) architecture for semantic segmentation. 1) A standard convolutional backbone (e.g. VGG) turns the input image into progressively smaller, deeper feature maps. 2) Instead of ending in dense (FC) layers, it uses a final 1×1 convolution to produce an low-resolution “score map” (21 channels here, one per class). 3) That coarse map is then upsampled (with learnable deconvolution/transpose-conv filters) back to the original image size. The result is a pixel-wise prediction map which is compared against the ground-truth segmentation.  shows how you convert a classification CNN into a dense, end-to-end trainable model that labels every pixel.](./pics/objSeg_visual.png)

![visual showing how you can turn a standard image classifier into a weak localization model by “convolutionalizing” its fully-connected layers to produce a class activation heatmap. Top row: a CNN + FC layers spits out a single “tabby cat” score—no idea where in the image the cat lives. convolutionalization: you replace those FC layers with equivalent 1×1 convolutions. Bottom row: that same network now outputs a low-resolution spatial map (the “tabby cat heatmap”) that lights up the regions most responsible for the “tabby cat” prediction.](./pics/localizationModel_imageClass.png)

![comparison of three fully-convolutional network (FCN) variants for semantic segmentation, showing how adding skip-connections at different depths trades off semantic strength for spatial detail. FCN-32s:  no skips—upsamples the very coarse final feature map by ×32. You get strong, high-level “semantics” but very blocky masks. FCN-16s:  injects a skip from the pool-4 layer (stride-16) before upsampling. You fuse somewhat finer features, so the mask is sharper than FCN-32s. FCN-8s: adds another skip from pool-3 (stride-8) as well. That restores even more spatial detail, producing the most finely resolved segmentation.](./pics/segmentation_comparison.png)

- "DeconvNet": *Super*-expensive to train <br> ![DeconvNet (encoder–decoder) architecture for pixel-wise semantic segmentation. Convolutional “Encoder”: A standard CNN (e.g. VGG-style) repeatedly applies convolutions + ReLUs and max-pooling, shrinking the spatial size from 224×224 down to a 7×7 “bottleneck.” Deconvolutional “Decoder”: It then mirrors that process: at each decoder stage you unpool (using the saved pooling switches to place activations back where they came from) and follow with learned deconvolution filters to upsample back through 14×14, 28×28, 56×56, 112×112 and finally reconstruct a 224×224 segmentation map. arrows show the skip-connections of pooling indices that guide the unpooling, letting the network recover fine spatial detail (important for small objects), at the cost of a very heavyweight, expensive-to-train model](./pics/deconvnet_arch.png)
- But results are excellent
    - Particularly for small objects <br> ![comparison of segmentation quality of a vanilla FCN versus a DeconvNet on two example images (ground truth also shown). DeconvNet has much sharper, more precise boundaries and better recovery of fine details](./pics/devconvnet_comparison.png)

```
The parameter sharing/ receptive field architecture unique to CNNs enables dramatically reduced parameter counts in neural architectures, exploiting the inherent sparsity of images. What is a disadvantage of this approach?
    a. CNNs tend to blur object boundaries, much like optical flow.
    b. CNNs cannot build up an internal representation of object hierarchy.
    c. CNNs struggle to identify objects in high-resolution images.
    d. CNNs have no notion of absolute object locality, or of relative positioning of multiple objects. // correct
```

## Conclusions
- CNNs are mostly “convolutions inside a deep network”
    - Main operator (i.e. **most important**) is the convolution
    - Exploits image sparsity: important features are **local**
- A couple new[ish] tricks include
    - Automatically learning the filters as part of the training process
    - Using pooling
    - 1$\times$1 convolutions
- Applications include
    - Object detection (is there an object)
    - Object localization and segmentation (where is the object)
    - Object classification (what is the object)
    - Zero- and single-shot detectors
