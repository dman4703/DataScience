# Homework 1: Machine Learning Review

### 1. Conditional Probability and the Chain Rule

Recall the definition of conditional probability: 
 $$ P(A \mid B) \;=\; \frac{P(A \cap B)}{P(B)} $$
where $\cap$ means "intersection."
##### a. Prove that $ P(A \cap B \cap C) \;=\; P(A \mid B, C)\,P(B \mid C)\,P(C) $

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

import argparse
import numpy as np

STEP_SIZE = 0.0001

train_data_file  = "./data/train.data"
train_label_file = "./data/train.label"
test_data_file   = "./data/test_partial.data"
test_label_file = "./data/test_partial.data"
```


```python

```
