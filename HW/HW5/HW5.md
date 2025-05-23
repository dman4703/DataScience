# Homework 5: Everything Old Is New Again

### 1. Neural Networks

In this question we’ll look at some of the basic properties of feed-forward neural networks.

First, consider the myriad activation functions available to neural networks. In this problem, we’ll only look at very simple ones, starting with a combination of linear activation functions and the hard threshold; specifically, where the output of a node y is either 1 or 0: $$ y = \begin{cases} 1 & \text{if} \;\; w_{0} + \sum_{i}w_{i}x_{i} \ge 0 \\ 0 & \text{otherwise}. \end{cases} $$

Which of the following functions can be exactly represented by a neural network with one hidden layer, and which uses linear and/or hard threshold activation functions? For each case, briefly justify your answer (sketching an example is fine).

##### a. Polynomials of degree one.

- Yes; the network’s forward pass is literally a linear polynomial
- If there needs to be a hidden layer, use a linear unit and composition keeps the function linear

##### b. Hinge loss $h(x) = \max(1-x, 0)$.

- One hidden layer of linear + hard-threshold units can give a flat line, a pure staircase, or a staircase sitting on a single slanted line, but it cannot make the shape of the hinge loss

##### c. Polynomials of degree two.

- cannot model the continuos change of this polynomial given the hard desicion boundaries that this NN can provide

##### d. Piecewise constant functions.

- yes, linear unit + hard threshold = piecwise constant

Consider the following XOR-like function in two-dimensional space: $$ f(x_{1}, x_{2}) = \begin{cases} 1 & x_{1}, x_{2} \ge 0 \;\; \text{or} \;\; x_{1}, x_{2} \lt 0 \\ -1 & \text{otherwise} \end{cases} $$

We want to represent this function with a neural network. For some reason, we decide we only want to use the threshold activation function for the hidden units and output unit: $$ h_{\theta}(v) = \begin{cases} 1 & v \ge \theta \\ -1 & \text{otherwise}  \end{cases} $$

##### e. Show that the smallest number of hidden layers needed to represent this XOR function is two. Give a neural network with two hidden layers of threshold functions that represent f, the XOR function. Again, you are welcome to provide a drawing, but that drawing must include values being propagated from each neuron. Alternatively, you could draw a table showing the values at each layer.

| layer                   | neuron                   | $(w,b)$                                          | output when $(x_1,x_2)$ is … |
| ----------------------- | ------------------------ | ------------------------------------------------ | ---------------------------- |
| **Input → Hidden 1**    | $h_1$                    | $(\,[1,0],\;0)$                                  | 1  1  -1  -1                 |
|                         | $h_2$                    | $(\,[0,1],\;0)$                                  | 1  -1  1  -1                 |
| **Hidden 1 → Hidden 2** | $g_1$ detects **both +** | $(\,[1,1],\;\!-1.5)$                             | 1  -1  -1  -1                |
|                         | $g_2$ detects **both -** | $(\,[\!-1,-1],\;\!-1.5)$                         | -1  -1  -1  1                |
| **Hidden 2 → Output**   | $y$                      | $(\,[1,1],\;0.5)$ (i.e. threshold $\theta=-0.5$) | **1  -1  -1  1**             |


### 2. Hierarchical Clustering

We spent a little bit of time in class discussing hierarchical clustering, of which–fun fact–graph-based segmentation algorithms (min-cut, norm-cut, and even spectral clustering variants) are often considered a part. We primarily discussed two variants: *top-down*, in which all data points start out in a single large cluster that is continually split until a stopping criterion is reached, and *bottom-up*, in which all data points start out in their own clusters that are continually merged until a stopping criterion is reached.

Here, we’ll explore the latter, also known as agglomerative clustering. The basic algorithm is as follows:
1. Start with each point in a cluster of its own
2. Until there is only one cluster
    1. Find closest pair of clusters
    2. Merge them
3. Return the tree of cluster-mergers

To convert this procedure into an implementation, one only needs to be able to quantify how “close” two clusters are. While not mentioned in detail, we did discuss metrics that define distance between two clusters, such as single-link, complete-link, and average-link.

In this problem, you’ll look at an alternative approach to quantifying the distance between two disjoint clusters, proposed by Joe H. Ward in 1963. We will call it **Ward’s metric**.

Ward’s metric simply says that the distance between two disjoint clusters, $X$ and $Y$, is how much the sum of squares will increase when we merge them. More formally: $$ \Delta(X, Y) = \sum_{i \in X \cup Y}\| \vec{x}_{i} - \vec{\mu}_{X \cup Y} \|^{2} - \sum_{i \in X}\| \vec{x}_{i} - \vec{\mu}_{X} \|^{2} - \sum_{i \in Y}\| \vec{x}_{i} - \vec{\mu}_{Y} \|^{2} $$ where $\vec{\mu}_{Z}$ is the centroid of cluster $Z$, and $\vec{x}_{i}$ is a data point in our corpus. Here, $\Delta(X, Y)$ is considered the *merging cost* of combining clusters $X$ and $Y$ into one cluster, $X \cup Y$. That is, on each iteration, the two clusters with the lowest *merging cost* is merged using Ward’s metric as a distance measure.

##### a. Can you reduce the formula given for $\Delta(X, Y)$ to a simpler form? Provide the simplified formula and the steps to get there. Your formula should be in terms of the cluster sizes (i.e., the number of points in a given cluster, denoted as $n_{X}$ and $n_{Y}$) and the distance $\| \vec{\mu}_{X} - \vec{\mu}_{Y} \|^{2}$ between cluster centroids $\vec{\mu}_{X}$ and $\vec{\mu}_{Y}$ (yes, **only** the $n_{X}$, $n_{Y}$, $\vec{\mu}_{X}$, and $\vec{\mu}_{Y}$ symbols should be in your final equation).

$$ \sum_{i \in X \cup Y}\| \vec{x}_{i} - \vec{\mu}_{X \cup Y} \|^{2} = \sum_{i \in X \cup Y}\| \vec{x}_{i} \|^{2} - n_{X \cup Y}\| \vec{\mu}_{X \cup Y} \|^{2} $$ 
$$ \sum_{i \in X}\| \vec{x}_{i} - \vec{\mu}_{X} \|^{2} = \sum_{i \in X}\| \vec{x}_{i} \|^{2} - n_{X}\| \vec{\mu}_{X} \|^{2} $$
$$ \sum_{i \in Y}\| \vec{x}_{i} - \vec{\mu}_{Y} \|^{2} = \sum_{i \in Y}\| \vec{x}_{i} \|^{2} - n_{Y}\| \vec{\mu}_{Y} \|^{2} $$
$$ \Delta(X, Y) = \sum_{i \in X \cup Y}\| \vec{x}_{i} \|^{2} - n_{X \cup Y}\| \vec{\mu}_{X \cup Y} \|^{2} - \sum_{i \in X}\| \vec{x}_{i} \|^{2} + n_{X}\| \vec{\mu}_{X} \|^{2} - \sum_{i \in Y}\| \vec{x}_{i} \|^{2} + n_{Y}\| \vec{\mu}_{Y} \|^{2} $$
$$ \Delta(X, Y) = - n_{X \cup Y}\| \vec{\mu}_{X \cup Y} \|^{2} + n_{X}\| \vec{\mu}_{X} \|^{2} + n_{Y}\| \vec{\mu}_{Y} \|^{2} $$
$$ \mu_{X \cup Y} = \frac{n_{X}\mu_{X}+n_{Y}\mu_{Y}}{n_{X}+n_{Y}} $$
$$ \Delta(X, Y) = \frac{n_{X}n_{Y}}{n_{X} + n_{Y}}\| \mu_{X} - \mu_{Y} \|^{2} $$

##### b. Assume you are given two *pairs* of clusters $P_{1}$ and $P_{2}$. The centers of the two clusters in the $P_{1}$ pair are farther apart than the pair of centers in $P_{2}$. Using Ward’s metric, does agglomerative clustering *always* choose to merge the two clusters in $P_{2}$? Why or why not? Justify your answer with a simple example.

- No, since Ward's metric also depends on cluster sizes, not just distance between centriods

| pair   | cluster sizes | centroid gap         | Ward cost                               |
| ------ | ------------- | -------------------- | --------------------------------------- |
| **P₁** | $n_X=n_Y=1$   | $\|\mu_X-\mu_Y\|=10$ | $\tfrac{1\cdot1}{1+1}\,10^{2}=50$       |
| **P₂** | $n_X=n_Y=100$ | $\|\mu_X-\mu_Y\|=5$  | $\tfrac{100\cdot100}{200}\,5^{2}=1 250$ |


### 3. Genetic Programming

In this part, you’ll re-implement your logistic regression code from the Homework 1 (we’ve gone full circle!) to use a simple genetic algorithm to learn the weights, instead of gradient descent. Provided is a skeleton script that sets the following:
1. a file containing training data
2. a file containing training labels
3. a file containing testing data

And optionally sets:
- `n`: a population size (default: 200)
- `s`: a per-generation survival rate (default: 0.3)
- `m`: a mutation rate (default: 0.05)
- `g`: a maximum number of generations (default: 50)
- `r`: a random seed (default: -1)

Your evolutionary algorithm for learning the weights should have a few core components:
- **Random population initialization.** You should initialize a full array of weights *randomly* (don’t use all 0s!); this counts as a single “person” in the full population. Consequently, initialize $n$ arrays of weights randomly for your full population. You’ll evaluate each of these weights arrays independently and pick the best-performing ones to carry on to the next generation.
- **Fitness function.** This is a way of evaluating how “good” your current solution is. Fortunately, we have this already: the objective function! You can use the weights to predict the training labels (as you did during gradient descent); the fitness for a set of weights is then the *average classification accuracy*.
- Reproduction. Once you’ve evaluated the fitness of your current population, you’ll use that information to evolve the “strongest.” You’ll first take the top $s$%–the $ns$ arrays of weights with the highest fitness scores–and set them aside as the “parents” of the next generation. Then, you’ll “breed” random pairs of these parents parents to produce “children” until you have $n$ arrays of weights again. The breeding is done by simply averaging the two sets of parent weights together.
- **Mutation.** Each individual weight has a mutation rate of $m$. Once you’ve computed the “child” weight array from two parents, you need to determine where and how many of the elements in the child array will mutate. First, flip a coin that lands on heads (i.e., indicates mutation) with probability $m$ (the mutation rate) for each weight $w_{i}$. Then, for each mutation, you’ll generate the new $w_{i}$ by sampling from a Gaussian distribution with mean and variance set to be the empirical mean and variance of *all* the $w_{i}$ weights of the *previous* generation. So if $W_{p}$ is the $n \times |\beta|$ matrix of the previous population of weights, then we can define $\mu_{i} = W_{p}[:, i]\text{.mean()}$ and $\sigma_{i}^{2}$ $=$ $W_{p}[:, i]\text{.var()}$ Using these quantities, we can then draw our new weight $w_{i} \sim \mathcal{N}(\mu_{i}, \sigma_{i}^{2})$.
- **Generations.** You’ll run the fitness evaluation, reproduction, and mutation repeatedly for $g$ generations, after which you’ll take the set of weights from the final population with the highest fitness and evaluate these weights against the testing dataset.
- The parent and child populations should be kept *distinct* during reproduction, and only the children should undergo mutation!

The data files (`train.data` and `test.data`) contains three numbers on each line: **\<document_id\> \<word_id\> \<count\>**

Each row of the data files contains the count of how often a given word (identified by ID) appears in certain documents (also identified by ID). The corresponding labels for the data has only one number per row in the file: the label, 1 or 0, of the document with ID corresponding to the row of the label in the label file. For example, a 0 on the 27th line of the label file means the document with ID 27 has the label 0.

After you’ve found your final weights and used them to make predictions on the test set, your code should out predicted labels (0 or 1) by itself on a single line, *one for each document*–this means a single line of output per unique document ID (or per line in one of the `.label` files) to an output file. For example, if the following `test.data` file has four unique document IDs in it, your program should print out four lines, each with a 1 or 0 on it, e.g.:
```
train_data_file  = "./data/train.data"
train_label_file = "./data/train.label"
test_data_file   = "./data/test.data"
...

--- ./data/testPredicted.label ---
0
0
1
1
```
Then, compare to the test labels and compute the accuracy and classification report.

**NOTE 1**: Evolutionary programs **will take longer** than logistic regression’s gradient descent. I strongly recommend staying under a population size of 300, with no more than about 300 generations. **Make liberal use of NumPy vectorized programming** to ensure your program is running as efficiently as possible.


```python
# Data exploration
files = [
    "./data/train.data",
    "./data/train.label",
    "./data/test_partial.data",
    "./data/test_partial.label"
]

for path in files:
    print(f"\n--- {path} ---")
    with open(path, "r") as f:
        for _ in range(5):
            line = f.readline()
            if not line:
                break
            print(line.rstrip())
```

    
    --- ./data/train.data ---
    21 1 1
    41 2 1
    72 2 1
    100 2 1
    138 2 1
    
    --- ./data/train.label ---
    0
    0
    0
    0
    0
    
    --- ./data/test_partial.data ---
    1 10 3
    1 21 4
    1 32 1
    1 36 1
    1 39 1
    
    --- ./data/test_partial.label ---
    0
    0
    0
    0
    0



```python
import numpy as np

train_data_file  = "./data/train.data"
train_label_file = "./data/train.label"
test_data_file   = "./data/test_partial.data"
test_label_file = "./data/test_partial.label"
output_file = "./data/testPredicted.label"

n = 200 # population size
s = 0.3 # per-generation survival rate
m = 0.05 # mutation rate
g = 50 # max number of generations
r = -1 # random seed
```


```python
def _load_X(path):
    # Load the data.
    mat = np.loadtxt(path, dtype = int)
    max_doc_id = mat[:, 0].max()
    max_word_id = mat[:, 1].max()
    X = np.zeros(shape = (max_doc_id, max_word_id))
    for (docid, wordid, count) in mat:
        X[docid - 1, wordid - 1] = count
    return X

def _load_data(data, labels):
    # Load the labels.
    y = np.loadtxt(labels, dtype = int)
    X = _load_X(data)

    # Return.
    return [X, y]
```


```python
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------------------------------------------ utilities --
def _sigmoid(z):
    """Vectorised logistic."""
    return 1.0 / (1.0 + np.exp(-z))

def _predict_labels(X, w):
    """0/1 predictions from weight vector w."""
    return (_sigmoid(X @ w) >= 0.5).astype(int)

def _fitness(pop, X, y):
    """Classification-accuracy fitness for every weight vector in pop (shape n×d)."""
    preds = _sigmoid(X @ pop.T) >= 0.5          # (n_samples, n_individuals)
    return (preds.T == y).mean(axis=1)          # (n_individuals,)

def _pad_features(X, d_target):
    """Right-pad feature matrix with zeros to reach d_target columns."""
    if X.shape[1] < d_target:
        pad = np.zeros((X.shape[0], d_target - X.shape[1]), dtype=X.dtype)
        X   = np.hstack([X, pad])
    return X

# ---------------------------------------------------------- load & harmonise --
X_train, y_train = _load_data(train_data_file,  train_label_file)
X_test , y_test  = _load_data(test_data_file ,  test_label_file)

d = max(X_train.shape[1], X_test.shape[1])      # global vocab / feature size
X_train = _pad_features(X_train, d)
X_test  = _pad_features(X_test , d)

# ---------------------------------------- GA initialisation (population W^0) --
rng = np.random.default_rng(None if r < 0 else r)
pop = rng.standard_normal(size=(n, d))          # N(0,1)  initial weights

# ----------------------------------------------------------- evolutionary loop --
num_parents = max(1, int(round(s * n)))

for gen in range(g):
    # -------- fitness evaluation
    fit = _fitness(pop, X_train, y_train)

    # -------- selection
    parent_idx = np.argsort(fit)[-num_parents:]     # top-s% survive
    parents    = pop[parent_idx]

    # -------- reproduction  (children = averaged random parent pairs)
    kids_needed = n - num_parents
    idx1 = rng.integers(0, num_parents, size=kids_needed)
    idx2 = rng.integers(0, num_parents, size=kids_needed)
    children = 0.5 * (parents[idx1] + parents[idx2])

    # -------- mutation  (children only)
    mu_prev    = pop.mean(axis=0)
    sigma_prev = pop.var(axis=0, ddof=0)
    mutate_mask = rng.random(size=children.shape) < m
    if mutate_mask.any():
        children[mutate_mask] = rng.normal(
            loc  = mu_prev[np.newaxis, :],
            scale= np.sqrt(sigma_prev + 1e-12)[np.newaxis, :],
            size = children.shape
        )[mutate_mask]

    # -------- next generation = parents (unmutated) ∪ mutated children
    pop = np.vstack([parents, children])

# ---------------------------------------------------- best individual & eval --
best_w   = pop[_fitness(pop, X_train, y_train).argmax()]

y_hat_tr = _predict_labels(X_train, best_w)
y_hat_ts = _predict_labels(X_test , best_w)

print(f"Train accuracy: {accuracy_score(y_train, y_hat_tr):.4f}")
print(f"Test  accuracy: {accuracy_score(y_test , y_hat_ts):.4f}\n")
print(classification_report(y_test, y_hat_ts, digits=3))

# ------------------------------------------------------- write predictions ----
with open(output_file, "w") as f_out:
    for p in y_hat_ts:
        f_out.write(f"{int(p)}\n")

print(f"\nPredictions written to {output_file}")
```

    /tmp/ipykernel_1321/3583838077.py:6: RuntimeWarning: overflow encountered in exp
      return 1.0 / (1.0 + np.exp(-z))


    Train accuracy: 0.8300
    Test  accuracy: 0.5500
    
                  precision    recall  f1-score   support
    
               0      0.556     0.500     0.526        50
               1      0.545     0.600     0.571        50
    
        accuracy                          0.550       100
       macro avg      0.551     0.550     0.549       100
    weighted avg      0.551     0.550     0.549       100
    
    
    Predictions written to ./data/testPredicted.label



```python

```
