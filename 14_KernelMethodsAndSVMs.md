# Kernel Methods and Support Vector Machines

## Parametric Statistics
- Assume some functional form (Gaussian, Bernoulli, Multinomial, logistic, linear) for
    - $P(X_{i} \mid Y)$ and $P(Y)$ as in Naive Bayes
    - $P(Y \mid X)$ as in Logistic Regression
- Estimate parameters ($\mu$, $\sigma^{2}$, $\theta$, $w$, $\beta$) using MLE/MAP
    - Plug-n-chug
- Advantages: need relatively few data points to learn parameters
- Drawbacks: Strong assumptions rarely satisfied in practice

## Embeddings
- Again!
- MNIST, projected into 2D embedding space<br>
![visual showing MNIST projected into 2D embedding space, there are 9 color coded clusters](./pics/mnist_visual.png)
- What distribution do these follow?
    - **Highly nonlinear**

## Nonparametric Statistics
- Typically very few, if any, distributional assumptions
- Usually requires more data
- Let number of parameters scale with the data

- Today
    - Kernel density estimation
    - K-nearest neighbors classification
    - Kernel regression
    - Support Vector Machines (SVMs) $\rightarrow$ not exactly nonparametric, but kernels are involved!

## Density Estimation
- You’ve done this before—histograms!
- Partition feature space into distinct bins with specified widths and count number of observations $n_i$ in each bin: $$ \hat{p}(x) = \frac{n_{i}}{n\Delta_{i}}1_{x \in \mathrm{Bin}_{i}} $$
- Same width is often used for all bins
- Bin width acts as **smoothing parameter**

![visual showing the same data histogrammed with three different bin widths (Delta = 0.04, 0.08, 0.25), each overlaid with the true density curve. It illustrates how the bin-width Delta acts as a smoothing parameter. Small Delta = 0.04: noisy histogram, high variance. Medium Delta = 0.08: a nice balance, the bars track the true curve well. Large Delta = 0.25: very few bins, the histogram is oversmoothed and misses detail (high bias)](./pics/binWidth_visual.png)

## Effect of $\Delta$
- Number of bins = $1/\delta$
$$ \hat{p}(x) = \frac{n_{i}}{n\Delta_{i}}1_{x \in \mathrm{Bin}_{i}} = \frac{1}{\Delta}\frac{\sum_{j=1}^{n}1_{X_{j} \in \mathrm{Bin}_{x}}}{n} $$
- Bias of histogram density estimate: $$ \mathbb{E}[\hat{p}(x)] = \frac{1}{\Delta}P(X \in \mathrm{Bin}_{x}) = \frac{1}{\Delta}\int_{z \in \mathrm{Bin}_{x}}p(z)dz \approx \frac{p(x)\Delta}{\Delta} = p(x) $$
> $\frac{1}{\Delta}\int_{z \in \mathrm{Bin}_{x}}p(z)dz$ is approximatley equal to $\frac{p(x)\Delta}{\Delta}$ assuming density is roughly constant in each bin (roughly true, if $\Delta$ is small)

![plot that shades the single bin of width Delta around x and shows how its height approximates the area under the true density curve](./pics/bin_visual.png)

## Bias-Variance Trade-off
- Choice of # of bins
    - If $\Delta$ is small: $$ \mathbb{E}[\hat{p}(x)] \approx p(x) $$
        > $p(x)$ approximately constant per bin
    - If $\Delta$ is large: $$ \mathbb{E}[\hat{p}(x)] \approx \hat{p}(x) $$
        > More data per bin stabalizes estimate
- Bias: how close is mean of estimate to the truth
- Variance: how much does estimate vary around the mean
- Small $\Delta$, large #bins $\leftrightarrow$ "Small bias, Large Variance"
- Large $\Delta$, small #bins $\leftrightarrow$ "Large bias, Small Variance"

![classic “dartboard” illustration of Low Bias + Low Variance, Low Bias + High Variance, High Bias + Low Variance, High Bias + High Variance](./pics/biasVar_visual.png)

## Choice of number of bins
![plot of the mean‐squared error of the histogram‐density estimator against the number of bins for a fixed sample size n.](./pics/mse_plot.png)
- At very few bins (large $\Delta$) there is high bias meaning high MSE.
- As bins are added (decrease $\Delta$) bias falls faster than variance rises meaning MSE drops.
- Beyond the optimum bin count, variance (because each $n_i$ is small) dominates meaning MSE climbs again.

## Kernel Density Estimation
- Histograms are “blocky” estimates: $$ \hat{p}(x) = \frac{1}{\Delta}\frac{\sum_{j=1}^{n}1_{X_{j} \in \mathrm{Bin}_{x}}}{n} $$
- Kernel density estimate, aka “Parzen / moving window” method: $$ \hat{p}(x) = \frac{1}{\Delta}\frac{\sum_{j=1}^{n}1_{\| X_{j} - x \| \le \Delta}}{n} $$

![plots that contrast histogram vs. Parzen‐window density estimates using the same data. histogram density estimate is blocky. Parzen estimate is a smooth curve because the window “slides” continuously](./pics/movingWindow_comparison.png)

- More generally: $$ \hat{p}(x) = \frac{1}{\Delta}\frac{\sum_{j=1}^{n}K(\frac{X_{j} - x}{\Delta})}{n} $$
- $K$ is the kernel function
    - Much like kernels in Kernel PCA or SVMs: model a relationship between two data points
- Embodies any number of possible kernel functions

![visual illustration of how Kernel Density Estimation works: Red dashed curves are the individual “bumps” (the kernel function K) placed at each data point (marked by the blue ticks on the x-axis). Blue solid line is the sum (and normalization) of all those bumps, producing a smooth estimate of the underlying density. can see that where the data points are clustered more tightly, the overlapping kernels add up to a higher peak in the estimated density.](./pics/kde_visual.png)
- Place small “bumps” at each data point, determined by $K$
- Estimator itself consists of a [normalized] “sum of bumps”
- Where points are denser, density estimate will be higher

## Kernels
- Any function that satisfies
    $$ K(x) \ge 0 $$
    $$ \int K(x)dx = 1 $$
- SciPy has a ton
    - See `signal.get_window`

- Boxcar kernel<br>
![visual illustrating the uniform (“boxcar”) kernel and how it looks once you center and scale it around a data point](./pics/boxcar_visual.png)
    - Finite support: only need local points to compute estimate
- Gaussian kernel<br>
![the Gaussian kernel in two guises, standard, unit-variance Gaussian and same shape but now centered at a data point Xj and stretched by the bandwidth delta](./pics/gaussian_visual.png)
    - Infinite support: need all points to compute estimate. **But quite popular**.

- Deep theory associated with kernels and kernel functions
- Touched on in Kernel PCA lecture
- Foundational to Support Vector Machines and Deep Neural Networks
- Excerpt from Elements of Statistical Learning, Chpt. 5:
    - Regularization and reproducing Kernel Hilber Spaces: In this section we cast splines into the larger context of regularization methods and reproducing kernel Hilbert spaces. This section is quite technical and can be skipped by the disinterested or intimidated reader

## Choice of Kernel Bandwith
![kernel-density estimates of the same “Bart–Simpson” multimodal target with four different bandwidth choices. True density: actual underlying curve. bandwitdth too small: huge variance, the estimate is wildly spiky and noisy. bandwitdth just right: captures all the peaks without too much noise. bandwitdth too large: high bias, the estimate is overly flat and misses the fine structure.](./pics/bartSimpson_plot.png)

## Histogram vs KDE
![left: visual showing the same data histogrammed with three different bin widths (Delta = 0.04, 0.08, 0.25), each overlaid with the true density curve. right: visual showing kernel-density estimates at three different smoothness settings (h=0.005,0.07,0.2), each overlaid with the true density curve. illustrates how both histograms and KDE depend on a single smoothing parameter—and why choosing it well is crucial to capture the true shape without under- or over-smoothing](./pics/histogramKDE_comparison.png)

## KNN Density Estimation
- Recall
    - Histograms: $$ \hat{p}(x) = \frac{n_{i}}{n\Delta_{i}}1_{x \in \mathrm{Bin}_{i}} $$
    - KDE: $$ \hat{p}(x) = \frac{n_{x}}{n\Delta} $$
- Fix $\Delta$, estimate number of points within $\Delta$ of $x$ ($n_i$ or $n_x$) from the data
- Fix $n_{x} = k$, estimate $\Delta$ from data (volume of ball around $x$ with $k$ data points)
- **KNN Density Estimation**: $$ \hat{p}(x) = \frac{k}{n\Delta_{k,x}} $$

- $k$ acts as a smoother
- Not very popular for density estimation
    - Computationally expensive
    - Estimates are poor
- **But related version for classification is very popular**

![visual showing KNN density estimate for K=1, 5, 30 plotted against the true density. At k=1, there are extreme spikes- very high variance, under-smoothed. At k = 5, less spiky but still noisy- still under-smoothed. At k = 30, much smoother but now misses finer structure- over-smoothed](./pics/knn_visual.png)

## KNN Classification
![2D plot with three classes of points: sports, science, and arts. An unclassified 'test document' is shown on the plot, along with the radius Delta_{k,x} around it; k=4. The 4 closest neighbors in the radius come from science and sports.](./pics/knnClassification_visual.png)
- $k = 4$
```
What should we predict?
    a. Average
    b. Majority  // correct
```

- Optimal classifier: $$ f^{*}(x) = \arg\max_{y}P(y \mid x) = \arg\max_{y}P(x \mid y)P(y) $$
- KNN classifier: $$ \hat{f}_{kNN}(x) = \arg\max_{y}\hat p_{kNN}(x \mid y)\hat P(y) = \arg\max_{y}k_{y} $$

$$ \hat p_{kNN}(x \mid y)\hat P(y) = \frac{k_{y}}{n_{y}\Delta_{k,x}} $$
> where $n_{y}$ is the number of training points in class $y$<br>
> and where $k_{y}$ is the # of training points in class $y$ that lie within $\Delta_{k}$ ball

$$ \sum_{y}k_{y} = k $$
$$ \hat P(y) = \frac{n_{y}}{n} $$

![2D plot with three classes of points: sports, science, and arts. An unclassified 'test document' is shown on the plot. Visual illustrates what the test doc will be classified as when k=1, 2, 3, 5. k=1: sports, since 1 nearest neighbor is sports. k=2: not classified; note made that even value k not used in practice. k=3: science, since majority of 3 nearest neighbors are science. k=5: sports, since majority of 5 nearest neighbors are sports](./pics/knnClassification_ex.png)

## What is the best $k$?
- Bias-variance trade-off
- Large $k$ = predicted label is more stable
- Small $k$ = predicted label is more accurate
- **Similar to density estimation**

## KNN Decision Boundaries
![visual showing a 2d plot with red and blue points seperated by green jagged line marking the 1-NN decision boundary, and the corresponding Voronoi diagram—space partitioned into cells, each cell containing all the points whose nearest neighbor is that black training point.](./pics/1nnDesicionB_visual.png)
- 1-NN classification amounts to labeling by the Voronoi cell you fall into

![decision regions learned by KNN classifiers on the same 2D data for K=1, 3, 31. K=1: very jagged, high-variance boundary. K=3: the regions smooth out a bit, reducing variance at the cost of a little bias. k=31: oversmoothed, yielding very broad, almost linear decision bands (high bias, low variance)](./pics/knnOptimal_visual.png)
- **Guarantee**: For $n \to \infty$, error rate of 1-NN is never more than 2x optimal error rate

## Temperature Sensing
![plot of particles whose color signifies their temperature](./pics/avgTemp_visual.png)
- What is the temperature in the room? Average: $$ \hat T = \frac{1}{n}\sum_{i=1}^{n}Y_{i} $$

![plot of particles whose color signifies their temperature, with a particle X marked and a small radius (length h) of neighboring particlse also marked](./pics/localAvg_visual.png)
- At location x? "Local" Average: $$ \hat T = \frac{\sum_{i=1}^{n}Y_{i}1_{\| X_{i} - x \| \le h}}{\sum_{i=1}^{n}1_{\| X_{i} - x \| \le h}} $$

## Kernel Regression
- Or “local” regression
- Nadaraya-Watson Kernel Estimator: $$ \hat f_{n}(X) = \sum_{i=1}^{n}w_{i}Y_{i} $$ where $$ w_{i}(X) = \frac{K(\frac{X  -X_{i}}{h})}{\sum_{i=1}^{n}K(\frac{X  -X_{i}}{h})} $$
- Weight each training point on distance to test point
- Boxcar kernel yields local average

## Choice of kernel bandwidth
![noisy scatter of “power vs. multipole” data (the little dots) being smoothed by a Nadaraya–Watson kernel smoother with h=1, 10, 50, 200; heavy black curve in each is the fitted regression line. h=1: the bandwidth is far too small, so the fit chases every little wiggle (high variance, under-smoothed). h=10: still under-smoothed, but you see lots of spurious bumps. h=50: “just right”, the smoother captures the main peak and shoulders without overreacting to noise. h=200: too large, and the curve is overly flat (high bias, you’ve washed out the true shape)](./pics/kernelBW_visual.png)
- Choice of *kernel* is not terribly important!

## Kernel Regression as WLS
- Weighted Least Squares (WLS) has the form: $$ \min_{f}\sum_{i=1}^{n}w_{i}(f(X_{i})-Y_{i})^{2} $$
- Compare to Nadaraya-Watson form: $$ w_{i}(X) = \frac{K(\frac{X  -X_{i}}{h})}{\sum_{i=1}^{n}K(\frac{X  -X_{i}}{h})} $$
- Kernel regression corresponds to locally constant estimator obtained from [locally] weighted least squares
- Set $$ f(X_{i}) = \beta $$ **where $\beta$ is constant**

$$ \min_{f}\sum_{i=1}^{n}w_{i}(\beta - Y_{i})^{2} $$
> $\beta$: constant value

$$ w_{i}(X) = \frac{K(\frac{X  -X_{i}}{h})}{\sum_{i=1}^{n}K(\frac{X  -X_{i}}{h})} $$

$$ \frac{\partial J(\beta)}{\partial\beta} = 2\sum_{i=1}^{n}w_{i}(\beta - Y_{i}) = 0 $$
> $w_i$: individual weights have to sum to 1

$$ \to  \hat f_{n}(X) = \hat\beta = \sum_{i=1}^{n}w_{i}Y_{i} $$

## Support Vector Machines
![visual showing how an SVM picks the linear separator with the largest possible margin by hinging on just those boundary‐touching points: there is a linear decision boundary seperating two classes, The margin on either side (the short perpendicular ticks) measured to the nearest training points, Those nearest points are the support vectors that define the maximum-margin hyperplane](./pics/svm_ex.png)
- Linear classifiers—which is better?
- Pick the one with the **largest margin**

![visual showing linear‐classifier decision rule and how we measure its “confidence”](./pics/linearClassifier_visual.png)
- The black line is the hyperplane $w \cdot x + b = 0$ splitting space into two half-spaces:
    - $w \cdot x + b \lt 0$ which is one class
    - $w \cdot x + b \gt 0$ which is the other class
- For a labeled training point $(x_j, y_j)$ with $y_j \in \{\pm 1\}$, "confidence" = $(w \cdot x_j + b)y_j$
- SVMs choose $w$, $b$ to maximize the minimum of these confidences over all support vectors.

![geometric picture of the SVM margin, has the decision hyperplane w.x+b=0 as a black line, and two gray lines that are the “margin-boundaries”: w.x+b=+-a. The perpendicular distance from the central hyperplane to either of these gray planes is 2a/||w||](./pics/margin_visual.png)
- Maximize the margin
- Distance of closest example / data point from the decision boundary / hyperplane: $$ \text{margin} = \gamma = \frac{2a}{\| w \|} $$

![hard‐margin SVM’s primal optimization problem laid out as a quadratic program](./pics/svmQuad_visual.png)
- Rewrite the equation (drop $a$ in favor of 1): $$ \min_{w, b}w \cdot w \quad \text{s.t.} \quad (w \cdot x_{j} + b)y_{j} \ge 1 \; \forall j $$
- Solve via quadratic programming
- Data points along margin = **support vectors**

- What if the data aren’t linearly separable?
- Allow for “errors”: $$ \min_{w, b}w \cdot w + C \quad \text{s.t.} \quad (w \cdot x_{j} + b)y_{j} \ge 1 \; \forall j $$
- Maximize margin AND minimize mistakes
    -  $C$: tradeoff parameter (number of mistakes)

![soft-margin SVM in action, showing how the slack variables \xi_j work](./pics/softMargin_visual.png)
- What if the data *still* aren’t linearly separable?
- **"Soft" margin**
    - penalize misclassified data by how far it is from the margin
    $$ (w \cdot x_{j} + b)y_{j} \ge 1 - \xi_{j} \; \forall j $$
    $$ \xi_{j} \ge 0 \; \forall j $$
    - Misclassification penalty: $C$ $\xi_{j}$
    - Recover “hard” margin: Set $C = \infty$

## SVMs are great, but...
- Where is this going?
- **First**, SVMs were the “big thing” right before deep learning
    - Neural network research had been dead for 10+ years
    - SVMs were showing immense promise, especially with high-dim data
- **Second**, SVMs share a lot of theory with deep learning
    - Much of this theory found a second life in the Transformer architecture that powers all the modern large language models!

- (Excerpt) 12.2.1 Computing the Support vector classifier: The problem (12.7) is quadratic with linear inequality constraints, hence it is a convex optimization problem. We describe a quadratic programming solution using Lagrange multipliers. Computationally it is convenient to re-express (12.7) in the equivalent form.
- Start with core parameterization of SVM: $$ \min_{\beta, \beta_{0}}\frac{1}{2}\| \beta \|^{2} + C\sum_{i=1}^{N}\xi_{i} \quad \text{subject to} \quad \xi_{i} \ge 0,\; y_{i}(x_{i}^{T}\beta + \beta_{0}) \ge 1 - \xi_{i} \; \forall i$$
- Write "primal" objective (Lagrange) function: $$ L_{P} = \min_{\beta, \beta_{0}}\frac{1}{2}\| \beta \|^{2} + C\sum_{i=1}^{N}\xi_{i} - \sum_{i=1}^{N}\alpha_{i}[y_{i}(x_{i}^{T}\beta + \beta_{0}) - (1 - \xi_{i})] - \sum_{i=1}^{N}\mu_{i}\xi_{i} $$
- Differentiate with respect to $\beta$, $\beta_{0}$, and $\xi_{i}$ and set to 0:
    $$ \beta = \sum_{i=1}^{N}\alpha_{i}y_{i}x_{i} $$
    $$ 0 = \sum_{i=1}^{N}\alpha_{i}y_{i} $$
    $$ \alpha_{i} = C - \mu_{i} \; \forall i $$
- Substitute back into primal equation, and get the Lagrangian **Dual**: $$ L_{D} = \sum_{i=1}^{N}\alpha_{i} - \frac{1}{2}\sum_{i=1}^{N}\sum_{i'=1}^{N}\alpha_{i}\alpha_{i'}y_{i}y_{i'}x_{i}^{T}x_{i'} $$
- Notice anything?
    - **Kernel!**
$$ L_{D} = \sum_{i=1}^{N}\alpha_{i} - \frac{1}{2}\sum_{i=1}^{N}\sum_{i'=1}^{N}\alpha_{i}\alpha_{i'}y_{i}y_{i'}\langle h(x_{i}), h(x_{i'})\rangle $$
$$ K(x, x') = \langle h(x), h(x')\rangle $$

![composite figure is illustrating exactly how the choice of kernel and regularization parameter C shapes the SVM’s decision boundary (and thus its bias/variance trade‐off)](./pics/kernelBVT_visual.png)

## Summary
- Nonparametric places mild assumptions on data; good models for complex data
    - Usually requires storing & computing with full dataset
- Parametric models rely on very strong, simplistic assumptions
    - Once fitted, they are much more efficient with storage and computation
- Effects of bin width & kernel bandwidth
    - Bias-variance trade-off
- Kernel regression
    - Comparison to weighted least squares
- Support Vector Machines
    - Powerful “shallow” models
    - Dual formulation of objective allows for kernel functions

## Case Study: Newsgroups Classification
- 20 Newsgroups
- 61,118 words
- 18,774 documents
- Class label descriptions

| Category    | Newsgroups                                                                                  |
|-------------|---------------------------------------------------------------------------------------------|
| **Computing**   | comp.graphics<br>comp.os.ms-windows.misc<br>comp.sys.ibm.pc.hardware<br>comp.sys.mac.hardware<br>comp.windows.x |
| **Recreation**  | rec.autos<br>rec.motorcycles<br>rec.sport.baseball<br>rec.sport.hockey                   |
| **Science**     | sci.crypt<br>sci.electronics<br>sci.med<br>sci.space                                      |
| **Misc**        | misc.forsale                                                                              |
| **Politics**    | talk.politics.misc<br>talk.politics.guns<br>talk.politics.mideast                         |
| **Religion**    | talk.religion.misc<br>alt.atheism<br>soc.religion.christian                              |


- Training/Testing
    - 50%-50% randomly split
    - 10 runs
    - Report average results
- Evaluation Criteria: $$ \text{Accuracy} = \frac{\sum_{i \in \text{text set}}I(\text{predict}_{i} = \text{true label}_{i})}{\text{number of test samples}} $$
