# Recurrent Neural Networks

## [The Neural Network Zoo](http://www.asimovinstitute.org/neural-network-zoo/)
<img 
    src="./pics/nn_zoo.png"
    alt="infographic displaying various neural networks: perceptron, feed foward, radial basis network, deep feed forward, recurrent neural network, long/short term memory, gated recurrent unit, auto encoder, variational AE, denoising AE, sparse AE, markov chain, hopfield network, boltzman machine, restricted BM, deep belief network, deep convolutional network, deconvolutional network, deep convolutional inverse graphics network, generative adversial network, liquid state machine, extreme learning machine, echo state network, deep residual network, differentiable neural computer, neural turing machine, capsule network, kohonen network, attention network. Neuron. Nodes are also labeled: input cell, backfed input cell, noisy input cell, hidden cell, probablistic hidden cell, spiking hidden cell, capsul cell, output cell, match input output cell, recurrent cell, memory cell, gated memory cell, kernel, convolution/pool"
    style="width:50%;"/>
- RNNs: recurrent neural network, long/short term memory, gated recurrent unit

## Modeling Sequences
- Input: $$ X = [\vec{x}_{1}, \vec{x}_{2}, \ldots, \vec{x}_{T}] $$
- Output: $$ X = [\vec{y}_{1}, \vec{y}_{2}, \ldots, \vec{y}_{N}] $$
- $T$ and $N$ not necessarily equal
- Dimensions of $X$ and $Y$ not necessarily equal
    - Language Translation
    - Weather and Climate forecasting
    - Automated Driving
    - Other "long distance" time series data

## Something we've seen before: Linear Dynamical Models
- Two main components (using notation from Hyndman 2006):
- Apperance Model: $$ y_{t} = Cx_{t} + u_{t} $$

- State Model: $$ x_{t} = Ax_{t-1} + Wv_{t} $$

## Autoregressive Models
- This is the definition of a 1st-order autoregressive (AR) process!
$$ x_{t} = Ax_{t-1} + Wv_{t} $$
- Each observation ($x_{t}$) is a function of previous observations, plus some noise
- **Markov model!**
- AR models can have higher orders than 1
- Each observation is dependent on the previous d observations: $$ x_{t} = A_{1}x_{t-1} + A_{2}x_{t-2} + \dots + A_{d}x_{t-d} + Wv_{t} $$

- Concrete, *a priori* definition of what is important
    - $n$th-order Markov process
    - $n+1$ terms and larger are explicitly ignored
- No concept of *attention*
    - All $n$ terms receive equal “attention” (computationally, if not also statistically)
    - Are you devoting equal time reading every word on this slide?
- Cannot handle *variable-length inputs*, nor *variable-length outputs*
    - Contrast with CNNs: all input images have to be the same size (usually)
    - Contrast with [insert deep network of choice]: all outputs are the same, given any input

## Attention
- Some things are more important than others

![visualization of the attention mechanism in a sequence-to-sequence (encoder–decoder) neural translation model. The top row of boxes (labeled “B”) are the encoder’s hidden states for each French source token. The bottom row of boxes (labeled “A”) are the decoder’s hidden states as it generates the English translation. The purple curves show the attention weights: when predicting each English word, the decoder “looks back” (attends) to certain French encoder states. Thicker/darker lines indicate stronger attention. ](./pics/attention_translation.png)

![set of attention visualizations for an image-captioning model. For each example, shows the original image and the model’s generated caption, with one word underlined and the corresponding attention map when the model produced that particular word. Brighter regions indicate where the model “looked” in the image to decide on that word.](./pics/attention_imgCaption.png)

## Recurrent Neural Networks
- In short, recurrent neural networks (RNNs) break the typical “directed acyclic” pedagogy of deep networks by introducing self-loops
    - Allows information to persist through multiple iterations
- We can get around problems introduced by loops by “unrolling” the loops <br> ![visual showing how a single recurrent cell with a “self-loop” can be unrolled over time. Left: one RNN cell (labeled A) at time t, taking input x_t and its own previous hidden state (loop) to produce a new hidden state h_t. Right: the same cell unfolded across timesteps 0...t. At each step, a fresh copy of A consumes x_0, x_1, x_2, ..., x_t (and the prior hidden state) to yield h_0, h_1, h_2, ..., h_t. Unrolling turns the cyclic graph into a feed-forward chain, enabling standard backpropagation through time.](./pics/rnnCell_unrolling.png)
    - This permits backprop to work as usual

![diagram showing the five canonical RNN “I/O patterns,” and how inputs (pink) and outputs (blue) can be arranged over time (green = the recurrent cell). One-to-one: A standard feed-forward network: one input -> one output (no recurrence). One-to-many: A single input (e.g. an image) seeds a recurrent chain that then generates a whole output sequence (e.g. image captioning). Many-to-one: A whole input sequence (e.g. words of a sentence) is read in step by step, and only the final hidden state produces one output (e.g. sentiment score, classification). Many-to-Many: Input and output sequences of the same length, aligned in time (e.g. part-of-speech tagging, where each word -> each tag). Many-to-Many: A full input sequence is first encoded, then a separate decoding chain produces an output sequence (e.g. machine translation).](./pics/rnn_listStruct.png)
- “List” structure intrinsically handles variable-length data
- Think: convolution, but over time instead of space

- Use the same “parameter sharing” as CNNs
    - And linear dynamical systems!
- $f$ maps each time point to the next
- Also updates internal state $h$

![contrasts a pure state-transition model with a full input-driven RNN unrolled over time. autonomous dynamical system: circles labeled s^(...) -> s^(t-1) -> s^(t) -> s^(t+1) -> s^(...) where arrows are labeled f. RNN: circles labeled h^(...) -> h^(t-1) -> h^(t) -> h^(t+1) -> h^(...) still use “f” to propagate state, but each h(t) also receives an external input x(t). Unrolling makes explicit how the hidden state evolves step by step, combining the previous state and the current input.](./pics/rnn_stateUpdate.png)

- Four main equations at each time point
    $$ \vec{a}^{(t)} + \vec{b} + W\vec{h}^{(t-1)} + U\vec{x}^{(t)} $$
    > $\vec{b}$: bias term <br>
    > $W$: weights for hidden-to-hidden conncetions <br>
    > $\vec{h}^{(t-1)}$: Internal RNN state <br>
    > $U$: Weights for input-to-hidden connections <br>
    > $\vec{x}^{(t)}$: Input
    
    $$ \vec{h}^{(t)} = \sigma(\vec{a}^{(t)}) $$
    > $\vec{h}^{(t)}$: Update internal state <br>
    > $\sigma$: Activation function

    $$ \vec{o}^{(t)} = \vec{c} + V\vec{h}^{(t)} $$
    > $\vec{c}$: bias term <br>
    > $V$: weights for hidden-to-output conncetions <br>
    > $\vec{h}^{(t)}$: internal RNN state <br>

    $$ \hat{\vec{y}}^{(t)} = \phi(\vec{o}^{(t)}) $$
    > $\hat{\vec{y}}^{(t)}$: predicted output <br>
    > $\phi$: Final layer activation

- RNNs are great for modeling sequences, but by themselves cannot capture *attention*
- Long-term dependencies require an explicit “memory”

## Long-term Dependencies
- RNNs *compose* the same activation function repeatedly
    - Think: recurrence relations
- Results in highly nonlinear behavior

![plot (y-axis: projection of output, x-axis: input coordinate) showing what happens when you re‑apply the same non‑linear state‑update function g over and over. There are 5 curves. Iteration 0 is just the initial linear projection. After even one or two iterations the curve begins to bend and saturate. By five iterations nearly all extreme inputs are pushed toward a narrow band, while mid‑range inputs wobble—evidence of the vanishing/exploding‑gradient problem and the tendency of simple RNNs to collapse information over long time lags. demonstrates how repeatedly composing a tanh/sigmoid‑style non‑linearity drives hidden activations toward fixed points, making it hard for a vanilla RNN to remember precise information far back in time.](./pics/rnn_longTerm.png)

- Put another way, recall the interal state update: $$ \vec{h}^{(t)} = W\vec{h}^{(t-1)} $$
- Where have we seen this before...
    $$ \vec{h}^{(t)} = (W^{t})^{T}\vec{h}^{(0)} $$
    $$ W = X \Lambda X^{T} $$
    $$ \vec{h}^{(t)} = X^{T}\Lambda^{t}X\vec{h}^{(0)} $$
- Eigenvalues are raised to the power $t$, decaying any eigenvalue $\lt 1$
- **Any component of $h^{(0)}$ not aligned with largest eigenvalue will be discarded**

- "I grew up in France... I speak fluent **French**." <br> ![RNN “unrolled” through time, but with a few nodes highlighted in red to show how early inputs have to travel through the recurrence to affect a later hidden state. The red halo around x_1 and x_2 marks two very early inputs. The red halo around h_{t+1} marks a much later hidden state. It visually emphasizes that any influence (or gradient) from those early inputs must pass through every intermediate RNN cell (the green boxes) to reach h_{t+1}, explaining why vanilla RNNs struggle to capture long-range dependencies](./pics/rnn_longTermvisual.png)

## Long-Short Term Memory
- Or “LSTM”
- A variant of the *gated* RNN
- Each hidden state comprises a **forget** gate
    - Determines what to “remember” and what to discard
    - Functions on self-loop input

![computational graph of a single LSTM cell, showing how its three gates control the flow of information into, through, and out of the cell state. There are 4 sigmoids at the bottom: input, input gate, forget gate, output gate. The input and input gate connect to a 'x' node which connects to a '+' node. The '+' node and forget gate connect to another 'x' node. This self loops back to the '+' node. Finally, the '+' and output gate connects to a 'x' node which goes to an output](./pics/LSTM_compGraph.png)

## LSTM versus “vanilla” RNN
- A “vanilla” RNN contains only a single activation
- LSTMs have four interacting layers in each step

![ unrolled recurrence over three timesteps, shown first for a vanilla RNN (top) and then for an LSTM (bottom). vanilla: three identical “A” blocks represent a plain RNN cell at t−1, t, and t+1. Each just applies one non-linearity (e.g. a tanh) to the input x_t and previous hidden state to get the next hidden state. three identical “A” blocks represent a plain RNN cell at t−1, t, and t+1. Each just applies one non-linearity (e.g. a tanh) to the input x_t and previous hidden state to get the next hidden state. LSTM: again three “A” blocks with shared parameters, but the middle one is expanded to reveal the four gated components of an LSTM cell (input gate, forget gate, cell-state update, output gate).  contrasts how an LSTM’s single timestep (bottom middle) involves multiple sigmoid/tanh gates and a cell-state accumulator, whereas a vanilla RNN timestep (top) is just one black-box activation.](./pics/vanillaRNNlstm_comparison.png)

## Encoder-Decoder Networks
- Maps input to output sequences
- Each mapping not necessarily of equal length!
- $C$ is a “semantic summary”
- Think: input “subspace”
- Have to ensure $C$ is of sufficient dimensionality to represent input space

![sequence-to-sequence (seq2seq) encoder–decoder architecture drawn as two unrolled RNNs. Encoder: an RNN reads the input sequence{x^(1) to x^(n_x)} step by step, passing its hidden state forward. The final hidden state is collapsed into a fixed-length context vector C. Decoder: a second RNN is initialized (or conditioned) on C and then unrolled to generate the output sequence {y^(1) to y^(n_y)}; At each timestep it takes as input its previous output (or a special start token) plus C to produce the next symbol. Solid arrows show the usual recurrent connections (hidden-to-hidden and hidden-to-output). The heavy arrow from the encoder’s last state into the decoder indicates that context vector C is fed into every decoder step. Dashed arrows typically denote teacher-forcing links (feeding ground-truth y^(t−1) during training).](./pics/encoderDecoder_visual.png)

## Deep Recurrent Networks
- Each recurrent state can feed into a series of hidden states
- Analogous to hidden markov models (HMMs) with attention and nearly infinite support for hidden states

![This illustration shows three ways to add depth to a recurrent network: on the left, two distinct recurrent states (each with its own self-loop) are chained in series before producing the output (multi-state rnn); in the center, a single recurrent state feeds into a small feed-forward “deep” output module (rnn + deep output); and on the right, two recurrent layers are stacked—each with its own hidden state and loop—before generating the final prediction (stacked RNN).](./pics/deepRNN_visual.png)

## Conclusions
- Recurrent neural networks
    - A generalization of convolution (or is a convolution a generalization of recurrence?): uses same **parameter-sharing** idea
    - Introduces self-loops, but over discrete intervals: loops can be “unrolled” so backpropagation can still be used as normal
    - Still have trouble with long-term dependencies, such as language translation (vanishing / exploding gradient)
- Long-short term memory
    - Introduce a series of gates within the self-loops
    - Gates determine what to remember, what to discard
    - No ill-conditioned gradients
- Attention + Encoder-Decoder Networks
    - Starting to see the foundations for modern Transformers
