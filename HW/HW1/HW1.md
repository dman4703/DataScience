# Homework 1: Machine Learning Review

### 1. Conditional Probability and the Chain Rule

Recall the definition of conditional probability: 
 $$ P(A \mid B) \;=\; \frac{P(A \cap B)}{P(B)} $$
where $\cap$ means "intersection."
##### a. Prove that $ P(A \cap B \cap C) = P(A \mid B, C)\,P(B \mid C)\,P(C) $

$$ P(A \cap (B \cap C)) $$
$$ = P(A \mid B, C)P(B \cap C) $$ 
$$ = P(A \mid B, C)\,P(B \mid C)\,P(C) $$

##### b. Derive Bayes’ Theorem from the law of conditional probability, and define each term in the equation with a 1-sentence description.

The definition of conditional probability: 
$$ P(A \mid B) \;=\; \frac{P(A, B)}{P(B)} $$
It is equally valid to say:
$$ P(B \mid A) \;=\; \frac{P(A, B)}{P(A)} $$
Solving for $ P(A, B) $ in the two equations allows $ P(A, B) $ to be expressed in two equivalent ways:
$$ P(A \mid B)P(B) = P(B \mid A)P(A) $$
Rearranging the above equation yields Bayes’ Theorem:
$$ P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)} $$

- $ P(A) $ and $ P(B) $ are the unconditional probabilities of those two events.
- $P(B \mid A)$ is the probability of $B$ given that we know $A$.
- $P(A \mid B)$ is the probability of $A$ given that we know $B$.

### 2. Total Probability
Let’s say I have two six-sided dice: one is fair, one is loaded. The loaded die has:
$$
P(x) =
\begin{cases}
\frac{1}{2}, & x = 6,\\
\frac{1}{10}, & x \neq 6.
\end{cases}
$$
In addition to the two dice, I have a coin which I flip to determine which dice to roll. If
the coin flip ends up heads I will roll the fair die, otherwise I’ll roll the loaded one. The
probability that the coin flip is heads is $ p \in [0, 1] $.

##### a. What is the expectation of the die roll, in terms of p?
*Hint*: Recall that the expected value $ E[X] $ of a discrete random variable $ X $ (e.g., a coin flip) can be computed as
$$ E[X] = \sum_{i} x_i \, P(X = x_i) $$

The probability that the coin flips head is $p$, that it flips tails is $(1-p)$.
Let $X$ be the die-roll outcome, and And $ E[X] $ be the expectation of the die roll.
$$ E[X] = E[X \mid Heads]P(Heads) + E[X \mid Tails]P(Tails) $$
$$ = (1 + 2 + 3 + 4 + 5 + 6)(\frac{1}{6})p $$
$$ + ((6)(\frac{1}{2}) + (1 + 2 + 3 + 4 + 5)(\frac{1}{10}))(1-p) $$
$$ = \frac{7}{2}p + \frac{9}{2}(1-p)$$
$$ = \frac{9}{2} - p$$

##### b. What is the variance of the die roll, in terms of p?
*Hint*: Recall that the variance $ Var(X) $ of a random variable X can be computed as $$ Var(X) = E[X^2] - (E[X])^2 $$

$$ E[X^2] = (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2)(\frac{1}{6})p + ((6^2)(\frac{1}{2}) + (1^2 + 2^2 + 3^2 + 4^2 + 5^2)(\frac{1}{10}))(1-p) $$
$$ = \frac{91}{6}p + \frac{47}{2}(1-p) = \frac{47}{2} - \frac{25}{3}p$$
$$ (E[X])^2 = (\frac{9}{2} - p)^2 = \frac{81}{4} - 9p + p^2 $$
$$ Var(X) = \frac{47}{2} - \frac{25}{3}p - \frac{81}{4} + 9p - p^2 $$
$$ = \frac{13}{4} + \frac{2}{3}p - p^2$$

### 3. Naive Bayes
Consider the learning function $ f(X) \to Y $, where class label $ Y \in \{T, F\} $ and $ X = \{x_1, x_2, \ldots, x_n \}$ , where $x_1$ is a boolean attribute and $ x_2, \ldots, x_n $ are continuous attributes.

##### a. Assuming the continuous attributes are modeled as Gaussians, give and briefly explain the total number of parameters that you would need to estimate in order to classify a future observation using a Naive Bayes (NB) classifier.
*Hint*: recall that a Naive Bayes classifier requires both the conditional probabilities $ P(X = x_i \mid Y) $ and the class prior probability $ P(Y) $

1. One parameter is needed for the class prior, $P(Y)$.
2. Two parameters are needed for boolean attribute $x_1$ (1 parameter per class)
3. For the continuous attributes:
   - There are $n - 1$ attributes
   - Each univariate Gaussian has two parameters (mean and variance)
   - Two parameters are needed for the two classes
   - Thus, $ (2)(n-1)(2) = 4n - 4 $ parameters are needed.

Summing each of these yields $ 4n - 1 $ total parameters.

##### b. How many more parameters would be required without the conditional independence assumption? No need for an exact number; an order of magnitude estimate will suffice.

Without conditional independence, you would need
- A mean vector of length $n-1$
- A full covariance matrix with $ \frac{(n-1)(n-1+1)}{2} $ entries

Yielding a bound of $O(n^2)$

### 4. Logistic Regression
In Logistic Regression (LR), we assume the observations are independent of each other (not *conditionally* independent, just independent).

##### a. Prove the decision boundary for Logistic Regression is linear. i.e., show that $ P(Y \mid X) $ has the form:
$$ w_0 + \sum_{i} w_i X_i $$
##### where $ Y \in \{0, 1 \} $, and the quantity of the sum in the above equation will determine whether LR predicts 1 or 0.
*Hint*: Recall that $$ P(Y = 0 \mid X) \;=\; \frac{1}{1 + \exp\!\bigl(w_0 + \sum_{i} w_i X_i\bigr)} $$ and that $ P(Y=0 \mid X) + P(Y = 1 \min X) = 1 $.

Let $p = P(Y | X) = P(Y=1 | X)$, the probability that $Y=1$ given the parameters $X$.<br>
Let $l$ denote the logit function. $l$ is defined as $l = ln($odds$)$.<br>
**odds** is the ratio of the probability $p$ to the probability $\neg p$:
$$ odds = \frac{p}{1-p} = \frac{P(Y=1 | X)}{P(Y=0 | X)}$$
Given the definition of $P(Y = 0 \mid X)$, the above can be simplified to:
$$ odds = \textrm{exp}(w_0 + \sum_i w_i X_i)$$
Plugging that in to the logit function yields the linear bound:
$$ l = ln(\textrm{exp}(w_0 + \sum_i w_i X_i)) = w_0 + \sum_i w_i X_i $$

##### b. *Briefly* describe one advantage and one disadvantage of LR compared to NB (two sentences total is plenty).

- Logistic regression learns the chance of each label directly from the data without treating features as independent, so it often gives more accurate predictions when features interact or don’t follow simple distributions
- However, it must run an iterative optimization (e.g. gradient descent) to find its weights, which is slower and can overfit on small datasets, whereas Naive Bayes just counts occurrences and has a closed-form solution

### 5. Coding
In this problem you will implement Logistic Regression (LR) for a document classification task.

##### a. Imagine a certain word is never observed during training, but appears in a testing set. What will happen when the NB classifier predicts the probability of the word? Explain. Will LR have the same problem? Why or why not?

- With a NB classifier, a new word would collapse the product of probabilities of all words to 0, meaning the model would not be able to distinguish between classes.
- LR uses weights to classify words into one of two classes. A new word would simply be unweighted, and does not zero out the output.

##### b. Implement LR.
This script should accept three arguments, in the following order:
1. a file containing training data
2. a file containing training labels
3. a file containing testing data

For training LR, we found a step size $\eta$ around 0.0001 worked well.<br><br>
The data files (train.data and test.data) contains three numbers on each line: **\<document_id\> \<word_id\> \<count\>**<br><br>
Each row of the data files contains the count of how often a word (identified by ID)
appears in a certain document. The corresponding label file for the training data has
only one number per row of the file: the label, 1 or 0, of the document in the same row of the data file. <br> <br>
For each line in the testing file, your code should print a predicted label (0 or 1) by itself on a single line.
For example, if the following test.data file has four lines (words) in it, your
program should print out four lines, each with either a 0 or a 1, e.g.
```
> python homework1.py train.data train.labels test.data
0
1
1
0
```
Don’t be alarmed if the training process of LR takes a few minutes; a good sanity checkis to make sure your weights are changing on each iteration (this can be a simple **print** statement). 
It is **highly recommended** that you use NumPy vectorized programming to train the weights efficiently.<br><br>
Once you’ve tuned your script so it trains correctly and spits out a reasonable testing
accuracy (should be substantially above random chance), give it a try on AutoLab! Just
follow the submission instructions, and check your score on the scoreboard. Good luck!


```python
# Data exploration
files = [
    "./data/train.label",
    "./data/test_partial.label",
    "./data/test_partial.data",
    "./data/train.data",
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

    
    --- ./data/train.label ---
    0
    0
    0
    0
    0
    
    --- ./data/test_partial.label ---
    1
    1
    1
    1
    1
    
    --- ./data/test_partial.data ---
    69 1 1
    68 2 1
    100 2 1
    122 2 1
    140 2 1
    
    --- ./data/train.data ---
    21 1 1
    41 2 1
    72 2 1
    100 2 1
    138 2 1



```python
'''
The data for this problem is drawn from the 20 Newsgroups data
set. The training and test sets each contain 200 documents, 100 from
comp.sys.ibm.pc.hardware (label 0) and 100 from comp.sys.mac.hardware
(label 1). Each document is represented as a vector of word
counts.

The data consists of four files: train.data, train.label, test.data
and test.label. The .data files contain word count matrices whose rows
correspond to document_ids and whose columns correspond to
word_ids. Each row of the .data files represents the number of times a
certain word appeared in a certain document, in the following three
column format:

<document_id> <word_id> <count>

The .label files simply list the class label for each document in
order. i.e., the first entry of train.label is the label for the first
document in train.data.

You are also given PARTIAL testing sets. The testing sets provided are to give you an idea of how your classifier will perform on the full dataset.
'''

import numpy as np
from scipy.sparse import coo_matrix

STEP_SIZE = 0.0001

train_data_file  = "./data/train.data"
train_label_file = "./data/train.label"
test_data_file   = "./data/test_partial.data"
test_label_file = "./data/test_partial.label"
```


```python
# Read in training data
# labels
y_train = np.loadtxt(train_label_file, dtype=int)

# data
train_triples  = np.loadtxt(train_data_file, dtype=int)
doc_ids_train  = train_triples[:, 0] - 1   # zero-based
word_ids_train = train_triples[:, 1] - 1
counts_train   = train_triples[:, 2]

n_train_docs = doc_ids_train.max() + 1
vocab_size   = word_ids_train.max() + 1    # derive V from train set

X_train = coo_matrix(
    (counts_train, (doc_ids_train, word_ids_train)),
    shape=(n_train_docs, vocab_size)
).tocsr()
```


```python
# Read in testing data
# 1) Load labels
y_test = np.loadtxt(test_label_file, dtype=int)

# 2) Load raw triples
test_triples  = np.loadtxt(test_data_file, dtype=int)
doc_ids_test  = test_triples[:, 0] - 1   # still 0-based, but possibly sparse
word_ids_test = test_triples[:, 1] - 1
counts_test   = test_triples[:, 2]

# 3) Build a compact mapping of document IDs → consecutive rows
unique_ids = np.unique(doc_ids_test)
id2row     = {orig: new for new, orig in enumerate(unique_ids)}
new_rows   = np.array([id2row[d] for d in doc_ids_test])

# 4) Build the sparse test‐matrix with exactly len(unique_ids) rows
X_test = coo_matrix(
    (counts_test, (new_rows, word_ids_test)),
    shape=(len(unique_ids), vocab_size)
).tocsr()
```


```python
# initialize parameters
w = np.zeros(vocab_size)
b = 0.0

# define the sigma function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```


```python
STEP_SIZE       = 0.0001
TOL_UPDATE      = 1e-6     # ℓ₂‐norm threshold on the weight update
TOL_LOSS_REL    = 1e-5     # relative change in loss threshold
MAX_ITER        = 100000   # hard cap on iterations

w = np.zeros(vocab_size)
b = 0.0

prev_loss = None
epoch     = 0

while True:
    # 1) forward pass
    z      = X_train.dot(w) + b
    y_hat  = sigmoid(z)

    # 2) compute gradients
    delta  = y_train - y_hat
    dw     = X_train.T.dot(delta)
    db     = delta.sum()

    # 2b) compute update norm
    weight_update = STEP_SIZE * dw
    update_norm   = np.linalg.norm(weight_update)

    # 3) compute current loss
    loss = -np.mean(y_train * np.log(y_hat) + (1-y_train) * np.log(1-y_hat))

    # 4) check stopping criteria
    cond_update = (update_norm < TOL_UPDATE)
    cond_loss   = (prev_loss is not None and 
                   abs(loss - prev_loss) / prev_loss < TOL_LOSS_REL)
    if cond_update or cond_loss or epoch >= MAX_ITER:
        print(f"Stopping at epoch {epoch}: "
              f"update_norm={update_norm:.2e}, "
              f"rel_loss_change={None if prev_loss is None else abs(loss-prev_loss)/prev_loss:.2e}")
        break

    # 5) parameter update
    w += weight_update
    b += STEP_SIZE * db

    # 6) logging
    if epoch % 1000 == 0:
        print(f"Epoch {epoch:4d} — loss {loss:.6f} — update_norm {update_norm:.2e}")

    prev_loss = loss
    epoch += 1
```

    Epoch    0 — loss 0.693147 — update_norm 3.23e-02
    Epoch 1000 — loss 0.101985 — update_norm 1.21e-03
    Epoch 2000 — loss 0.059219 — update_norm 6.99e-04
    Epoch 3000 — loss 0.041680 — update_norm 4.95e-04
    Epoch 4000 — loss 0.032069 — update_norm 3.83e-04
    Epoch 5000 — loss 0.026007 — update_norm 3.12e-04
    Epoch 6000 — loss 0.021841 — update_norm 2.63e-04
    Epoch 7000 — loss 0.018807 — update_norm 2.27e-04
    Epoch 8000 — loss 0.016502 — update_norm 2.00e-04
    Epoch 9000 — loss 0.014692 — update_norm 1.78e-04
    Epoch 10000 — loss 0.013236 — update_norm 1.61e-04
    Epoch 11000 — loss 0.012038 — update_norm 1.47e-04
    Epoch 12000 — loss 0.011037 — update_norm 1.35e-04
    Epoch 13000 — loss 0.010188 — update_norm 1.24e-04
    Epoch 14000 — loss 0.009459 — update_norm 1.16e-04
    Epoch 15000 — loss 0.008826 — update_norm 1.08e-04
    Epoch 16000 — loss 0.008272 — update_norm 1.01e-04
    Epoch 17000 — loss 0.007783 — update_norm 9.53e-05
    Epoch 18000 — loss 0.007348 — update_norm 9.00e-05
    Epoch 19000 — loss 0.006958 — update_norm 8.53e-05
    Epoch 20000 — loss 0.006608 — update_norm 8.10e-05
    Epoch 21000 — loss 0.006290 — update_norm 7.72e-05
    Epoch 22000 — loss 0.006002 — update_norm 7.37e-05
    Epoch 23000 — loss 0.005739 — update_norm 7.04e-05
    Epoch 24000 — loss 0.005498 — update_norm 6.75e-05
    Epoch 25000 — loss 0.005276 — update_norm 6.48e-05
    Epoch 26000 — loss 0.005071 — update_norm 6.23e-05
    Epoch 27000 — loss 0.004881 — update_norm 6.00e-05
    Epoch 28000 — loss 0.004705 — update_norm 5.78e-05
    Epoch 29000 — loss 0.004542 — update_norm 5.58e-05
    Epoch 30000 — loss 0.004389 — update_norm 5.40e-05
    Epoch 31000 — loss 0.004246 — update_norm 5.22e-05
    Epoch 32000 — loss 0.004112 — update_norm 5.06e-05
    Epoch 33000 — loss 0.003986 — update_norm 4.90e-05
    Epoch 34000 — loss 0.003868 — update_norm 4.76e-05
    Epoch 35000 — loss 0.003756 — update_norm 4.62e-05
    Epoch 36000 — loss 0.003651 — update_norm 4.49e-05
    Epoch 37000 — loss 0.003551 — update_norm 4.37e-05
    Epoch 38000 — loss 0.003457 — update_norm 4.26e-05
    Epoch 39000 — loss 0.003367 — update_norm 4.15e-05
    Epoch 40000 — loss 0.003282 — update_norm 4.04e-05
    Epoch 41000 — loss 0.003201 — update_norm 3.94e-05
    Epoch 42000 — loss 0.003124 — update_norm 3.85e-05
    Epoch 43000 — loss 0.003051 — update_norm 3.76e-05
    Epoch 44000 — loss 0.002981 — update_norm 3.67e-05
    Epoch 45000 — loss 0.002914 — update_norm 3.59e-05
    Epoch 46000 — loss 0.002850 — update_norm 3.51e-05
    Epoch 47000 — loss 0.002789 — update_norm 3.44e-05
    Epoch 48000 — loss 0.002730 — update_norm 3.36e-05
    Epoch 49000 — loss 0.002674 — update_norm 3.30e-05
    Epoch 50000 — loss 0.002620 — update_norm 3.23e-05
    Epoch 51000 — loss 0.002568 — update_norm 3.17e-05
    Epoch 52000 — loss 0.002518 — update_norm 3.10e-05
    Epoch 53000 — loss 0.002471 — update_norm 3.05e-05
    Epoch 54000 — loss 0.002424 — update_norm 2.99e-05
    Epoch 55000 — loss 0.002380 — update_norm 2.93e-05
    Epoch 56000 — loss 0.002337 — update_norm 2.88e-05
    Epoch 57000 — loss 0.002296 — update_norm 2.83e-05
    Epoch 58000 — loss 0.002256 — update_norm 2.78e-05
    Epoch 59000 — loss 0.002217 — update_norm 2.73e-05
    Epoch 60000 — loss 0.002180 — update_norm 2.69e-05
    Epoch 61000 — loss 0.002144 — update_norm 2.64e-05
    Epoch 62000 — loss 0.002109 — update_norm 2.60e-05
    Epoch 63000 — loss 0.002075 — update_norm 2.56e-05
    Epoch 64000 — loss 0.002043 — update_norm 2.52e-05
    Epoch 65000 — loss 0.002011 — update_norm 2.48e-05
    Epoch 66000 — loss 0.001980 — update_norm 2.44e-05
    Epoch 67000 — loss 0.001950 — update_norm 2.41e-05
    Epoch 68000 — loss 0.001921 — update_norm 2.37e-05
    Epoch 69000 — loss 0.001893 — update_norm 2.34e-05
    Epoch 70000 — loss 0.001866 — update_norm 2.30e-05
    Epoch 71000 — loss 0.001840 — update_norm 2.27e-05
    Epoch 72000 — loss 0.001814 — update_norm 2.24e-05
    Epoch 73000 — loss 0.001789 — update_norm 2.21e-05
    Epoch 74000 — loss 0.001764 — update_norm 2.18e-05
    Epoch 75000 — loss 0.001741 — update_norm 2.15e-05
    Epoch 76000 — loss 0.001718 — update_norm 2.12e-05
    Epoch 77000 — loss 0.001695 — update_norm 2.09e-05
    Epoch 78000 — loss 0.001673 — update_norm 2.06e-05
    Epoch 79000 — loss 0.001652 — update_norm 2.04e-05
    Epoch 80000 — loss 0.001631 — update_norm 2.01e-05
    Epoch 81000 — loss 0.001611 — update_norm 1.99e-05
    Epoch 82000 — loss 0.001591 — update_norm 1.96e-05
    Epoch 83000 — loss 0.001572 — update_norm 1.94e-05
    Epoch 84000 — loss 0.001553 — update_norm 1.92e-05
    Epoch 85000 — loss 0.001535 — update_norm 1.89e-05
    Epoch 86000 — loss 0.001517 — update_norm 1.87e-05
    Epoch 87000 — loss 0.001499 — update_norm 1.85e-05
    Epoch 88000 — loss 0.001482 — update_norm 1.83e-05
    Epoch 89000 — loss 0.001465 — update_norm 1.81e-05
    Epoch 90000 — loss 0.001449 — update_norm 1.79e-05
    Epoch 91000 — loss 0.001433 — update_norm 1.77e-05
    Epoch 92000 — loss 0.001417 — update_norm 1.75e-05
    Epoch 93000 — loss 0.001402 — update_norm 1.73e-05
    Epoch 94000 — loss 0.001387 — update_norm 1.71e-05
    Epoch 95000 — loss 0.001372 — update_norm 1.69e-05
    Epoch 96000 — loss 0.001358 — update_norm 1.68e-05
    Epoch 97000 — loss 0.001343 — update_norm 1.66e-05
    Epoch 98000 — loss 0.001330 — update_norm 1.64e-05
    Epoch 99000 — loss 0.001316 — update_norm 1.62e-05
    Stopping at epoch 100000: update_norm=1.61e-05, rel_loss_change=1.01e-05



```python
# --- Evaluation on test set ---

# 1) Compute predicted probabilities
z_test   = X_test.dot(w) + b
y_prob   = sigmoid(z_test)

# 2) Convert to binary predictions at threshold 0.5
y_pred   = (y_prob >= 0.5).astype(int)

# 3) Compute accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Test accuracy: {accuracy:.4f}")

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    log_loss
)

# 1) Confusion matrix and classification report (avoid warnings)
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(
    y_test, 
    y_pred, 
    digits=4, 
    zero_division=0            # fill precision/recall with 0 instead of warning
))
```

    Test accuracy: 0.5300
    Confusion matrix:
    [[ 0  0]
     [47 53]]
    
    Classification report:
                  precision    recall  f1-score   support
    
               0     0.0000    0.0000    0.0000         0
               1     1.0000    0.5300    0.6928       100
    
        accuracy                         0.5300       100
       macro avg     0.5000    0.2650    0.3464       100
    weighted avg     1.0000    0.5300    0.6928       100
    



```python
# Compare against scikit-learn’s LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    log_loss
)

# 1) Instantiate and train
clf = LogisticRegression(
    penalty='l2',       # ℓ2 regularization
    C=1.0,               # inverse regularization strength
    solver='lbfgs',      # good default for small-to-medium datasets
    max_iter=100000,
    random_state=0
)
clf.fit(X_train, y_train)

# 2) Make predictions
y_pred_prob = clf.predict_proba(X_test)[:, 1]   # probability for class “1”
y_pred      = clf.predict(X_test)

# 3) Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    digits=4,
    zero_division=0
))

```

    Accuracy: 0.5000
    
    Confusion matrix:
    [[ 0  0]
     [50 50]]
    
    Classification report:
                  precision    recall  f1-score   support
    
               0     0.0000    0.0000    0.0000         0
               1     1.0000    0.5000    0.6667       100
    
        accuracy                         0.5000       100
       macro avg     0.5000    0.2500    0.3333       100
    weighted avg     1.0000    0.5000    0.6667       100
    



```python

```
