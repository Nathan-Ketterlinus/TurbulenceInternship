## Need to:
- try to understand how neural network and deep learning works, make a slide presentation on it.
	1. could be used in conference talks later
	2. cannot present stuff from other people, word it yourself!

---
# From "But what is a neural network? | Deep learning chapter 1" by 3Blue1Brown
Many types of NNs
- Convolutional NN -> Good for image recognition
- Long short-term memory network -> good for speech recognition
- more!
- Multilayer perceptron -> "vanilla" NN, can be built on to make other models

we look at an example where we analyze a hand-drawing of a number to extract the number

Obv Neural Networks mimic the brain (as we understand it), but there's 2 big questions:
1. What are the "neurons"? 
	- A thing that holds a number (here between 0 & 1)
	- probably more accurate to think of each as a function that takes all previous layers as input and returns an int between 0 & 1![[Pasted image 20250606193354.png]]
	- each of these $28 \times 28 =784$ nodes are the first layer of our NN
	- Our last layer of the NN has 10 nodes, numbered 0-9 for our output
		- the value in each of these nodes (0 <= x <= 1) represents the likelihood our NN assigns to the input actually being a certain number
	- All layers in between are called "Hidden layers"
		- there's lots of flexibility in the number of hidden layers, the # of nodes in each layer, etc here![[Pasted image 20250606193854.png]]
		- activations in layer n *determine* activations in the next layer
2. How are these neurons connected?

### Why layers?
- consider our penultimate layer being used to represent larger structures, like the loop or vertical line in a 9
- but now how do we determine if there's a loop?
	- further subdivide the problem!
	- in the second layer, i magine that each node represents segments of these larger structures, like a quarter circle or generally straight line component
> Pixels > Edges > Patterns > Digits
![[Pasted image 20250606194625.png]]

but how do we verify that the network actually does this?
- assign a weight (just a number) to each edge between layers
- compute the weighted sum going into a node![[Pasted image 20250606195049.png]]
	- here, green -> >0, red -> <0
- activate when weighted sum is in an acceptable threshold (here its between 0 & 1)
- we can effectively compress the domains of our weighted sum like we want with the ==Sigmoid Function== (AKA logistic curve)![[Pasted image 20250606195318.png]]
	- ie take the Sigmoid ($\sigma$) of the weighted sum
- we can add a negative term to our Sigmoid input to add bias for being inactive (like -10)
	- this ensures the neuron is only activated meaningfully
- each 2nd layer neuron has 784 edges from first layer lmfaoo
	- need to define each edge weight & bias across all layers
	- these numbers get uncontrollable really quickly
	- need a way for the NN to "Learn" by itself
		- where learning here just means finding the right weights and balances automatically

### Notation
a node is represented by $a_{n}^{(m)}$
where:
-  $n$: what number node it is in the layer
- $m$: what layer the node is in

an edge is represented by $w_{m,n}$
where:
- $m$: matches the $m$ value of left node
- $n$: matches the $n$ value of left node

bias is represented by $b_{m}$
where:
- $m$: matches the $m$ value of left node

so the formula for the first node in layer 1 is:
$$a_{0}^{(1)} = \sigma( w_{0,0} a_{0}^{(0)} + w_{0,1} a_{1}^{(0)} + \dots + w_{0,n} a_{n}^{(0)} + b_{0})$$
there's a matrix vector multiplication shortcut here but its over my head D:![[Pasted image 20250606200732.png]]
this simplifies our definition into:
$a^{(1)}=\sigma(Wa^{(0)}+b)$

this simplifies and optimizes our code since many libraries optimize matrix multiplication!

NOTE!!!
in modern NNs, the sigmoid is largely deprecated in favor of the Rectified linear unit function, or ReLU!
its pretty simple though
generally learns much faster than sigmoid!
![[Pasted image 20250606201426.png]]


---
# from "Gradient Descent, how neural networks learn | DL2" by 3Blue1Brown

 how do NNs learn?
 generally:
 1. Give the system a whole bunch of training data, along with labels of what the correct outputs are
 2. Learning algorithm slowly but surely adjusts weights & biases in a way that improves success rate on training data.
 3. Hope that these tweaks are generalizable outside the training data.
	 1. test with novel inputs not in the training set

A simple way to get started is to initialize ALL weights and biases randomly!
obv this will spit back garbage, so what can we do?
- define a cost function
	- negative reinforcement to the network on bad output
	- for example:![[Pasted image 20250606213323.png]]
	- note the cost sum is small when the NN is confident, but high when not so confident
- consider the average cost across all inputs in the training set, and compare novel costs to this average
- now we introduce a mechanism that allows weights and biases to change
	- consider an example where the cost is only a function of a single input, instead of an infeasible number of inputs
	- how do we minimize the cost of this function?![[Pasted image 20250606213808.png]]
		- note $w$ is our single output, and $C(w)$, the output, is our cost
	- obv we can look at derivatives here!
	- but what if $C(w)$ is more complex, like this? (or $C(w)$ is actually $C(w_{0},\dots, C_{13002})$![[Pasted image 20250606214006.png]]
	- one approach is taking any output, and moving in the direction of the derivative that's negative until the derivative is 0!
		- like a ball rolling down the hill
			- can result in several local minima, but computing global minimum gets really complicated really quickly with so many inputs
		- make step sizes proportional to slope to prevent overshooting!
	- now consider $C(w_{1}, w_{2})$![[Pasted image 20250606214502.png]]
	- naturally, now we can use $-\nabla C(w_{1},w_{2})$ (the negative gradient) to find the fastest path down to a local minima
	- our algorithm to the local minima (called ==Gradient descent==):
		1. compute $\nabla C$
		2. small step in $-\nabla C$'s direction
		3. repeat
	- now we extend this idea to all 13,002  weights and biases
		- all 13,002 weights and biases: 
	$$\vec{W} = \begin{bmatrix}
2.25 \\
-1.57 \\
1.98 \\
\dots \\
-1.16 \\
3.82 \\
1.21
\end{bmatrix}
$$
		- how to nudge all weights and biases (add this to $\vec{W}$):
$$-\nabla C(\vec{W}) = \begin{bmatrix}
0.18\\
0.45 \\
-0.51 \\
\dots \\
0.40 \\
-0.32 \\
0.82
\end{bmatrix}
$$
> the algorithm for computing $-\nabla C(\vec{W})$ efficiently is called Backpropagation, covered in next section

machine "learning" is just minimizing our cost function!
- this means our cost function needs a smooth output so we can find a local minima!

an intuitive explanation for each number in $-\nabla C(\vec{W})$ is that it explains how much relevance/irrelevance each weight/bias carries

> state of the art accuracy in NNs is 99.79%

apparently all this content so far is stuff from the 80s lol

# From "Backpropagation, intuitively | DL3" by 3Blue1Brown

Backpropagation is the core algorithm behind how NNs learn
- its how we compute the gradient we use to nudge weight and bias values towards a local minima

note we can think of weights like this:
	say we have 2 edges with 2 weights, $3.20$ & $0.10$. Then we can say that the first edge is $32$ times more sensitive to changes than the second edge

### How should a single training set influence our NN?
say our NN is poorly trained, such that the output nodes are a mess, and the total cost is high. We can note how much we *want* each output node to change by (and in which direction)

recall our definition of a node's output:
$$a_{0}^{(1)} = \sigma( w_{0,0} a_{0}^{(0)} + w_{0,1} a_{1}^{(0)} + \dots + w_{0,n} a_{n}^{(0)} + b_{0})$$
this means we can change any of:
1. $w_{i}$ values: change the weight of any particular edge
	- obv, since our function multiplies node outputs ($a$) by edges ($w$), changing the weights of edges with a higher node value results in a more dramatic change than changing the weight of an edge with a lower node value. 
		- ie make changes that are proportional to corresponding $a_{i}$ value
	- "neurons that fire together wire together" - from Hebbian Theory
2. $b$ value: change the value of our bias
3. $a_{i}$ value: change the value of a node in a previous layer
	- pragmatically, this means that we change nodes with a positive outgoing edge to take a higher value, and negative outgoing edges to take a lower value
	- again, make changes proportional to the corresponding $w_{i}$ value

now, we can ==sum up== the desired changes each output node wants to make to previous nodes to get a list of ideal value nudges to make to layer $n-1$'s nodes. We can then recursively repeat this process to train our network on this particular example!

critically, though, we need to repeat this backpropagation across several examples (the more the merrier!) to get several sets of ideal changes to the edges of our NN

Now we can ==average== ALL of the ideal changes to a particular weight to get the direction we should nudge that weight!
- the averaged values for the set of all edges is basically our $-\nabla C$!!

as a note, its obv VERY computationally expensive to calculate 13,002+ nudges for EVERY training set, so commonly, we take a shortcut, called ==Stochastic gradient descent==:
- randomly shuffle all training data
- divide them into several "mini-batches" (ie you have 17,000 training sets, you make 170 mini-batches of 100 sets each)
- compute the gradient descent step for a single mini-batch, apply, then move to next mini-batch
while this is not a 1-to-1 of the ideal $-\nabla C$, it works as a great approximation while shaving off massive computational overhead ![[Pasted image 20250607195635.png]]
# From "Backpropagation calculus | DL4" by 3Blue1Brown
how do we think about the chain rule in the context of networks?

consider a SUPER simple NN, pictured below![[Pasted image 20250607200609.png]]
this network is determined by 3 weights and 3 biases
our goal is to understand how sensitive our cost function is to these variables

our notation is:
$a^{(L)}$: the activation value for the node in layer $L$
$y$: our desired output for our NN

thus in this case, our Cost function is equivalent to:
	$(a^{(L)}-y)^{2}$
for a concrete example, it could be something like:
	$(0.66-1.00)^{2}$

also recall this notation:
	$a^{(L)}=\sigma(w^{(L)}a^{(L-1)}+b^{(L)})$
for our purposes, we are redefining the input to the sigmoid as:
	$z^{(L)}=w^{(L)}a^{(L-1)}+b^{(L)}$
so now we can legally write:
	$a^{(L)}=\sigma(z^{(L)})$

so all of our terms are related as so:![[Pasted image 20250607201246.png]]
we acknowledge the recursive definition of $a^{(L-1)}$, but ignore it for now
![[Pasted image 20250607201337.png]]

we want to find out how much changing $w^{(L)}$ changes our cost function's ($C_{0}$) output, ie
$$\frac{\partial C_{0}}{\partial w^{(L)}}$$
but obviously, to see that change, we need to see how a change in $w^{(L)}$ changes $z^{(L)}$, then see how much THAT changes $a^{(L)}$, then see how much THAT finally changes $C_{0}$
- hello chain rule!
- more formally, we have:
$$\frac{\partial C_{0}}{\partial w^{(L)}} = \frac{\partial z^{(L)}}{\partial w^{(L)}} \cdot\frac{\partial a^{(L)}}{\partial z^{(L)}}\cdot\frac{\partial C_{0}}{\partial a^{(L)}}$$

 computing our relevant derivatives, we get:
 - $\frac{\partial C_{0}}{\partial a^{(L)}} = 2(a^{(L)}-y)$
	- from $C_{0} = (a^{(L)}-y)^{2}$
	- note its size is proportional to the difference between our intended output and our actual output!
- $\frac{\partial a^{(L)}}{\partial z^{(L)}} = \sigma'(z^{(L)})$
	- from $a^{(L)}= \sigma(z^{(L)})$
	- obv if you aren't using the sigmoid function, its the derivative of whatever squishing function you DID decide to use
- $\frac{\partial z^{(L)}}{\partial w^{(L)}} = a^{(L-1)}$
	- from $z^{(L)}=w^{(L)}a^{(L-1)}+b^{(L)}$
	- intuitively, this means that the impact that $w^{(L)}$ had on $a^{(L)}$ is directly dependent on the output of the node before it, $a^{(L-1)}$

note that this is only relevant for a SINGLE training example $C_{0}$.
To generalize to all training examples, we have the following summation:
$$\frac{\partial C}{\partial w^{(L)}} = \frac{1}{n} \sum_{k=0} ^{n-1} \frac{\partial C_{k}}{\partial w^{(L)}}$$
basically just average the result for ALL training examples lol

then, this term is just a single part of our fully defined gradient vector:
$$\nabla C = \begin{bmatrix}
\frac{\partial C}{\partial w^{(1)}} \\
\frac{\partial C}{\partial b^{(1)}} \\
\dots \\
\frac{\partial C}{\partial w^{(L)}} \\
\frac{\partial C}{\partial b^{(L)}}
\end{bmatrix}$$

### What about the bias?
our cost function's sensitivity to the bias term is calculated very similarly!
we only have to change one term!
$$\frac{\partial C_{0}}{\partial b^{(L)}} = \frac{\partial z^{(L)}}{\partial b^{(L)}} \cdot\frac{\partial a^{(L)}}{\partial z^{(L)}}\cdot\frac{\partial C_{0}}{\partial a^{(L)}} = \sigma'(z^{(L)})2(a^{(L)}-y)$$
but this term is always 1, so really we have:
$$\frac{\partial C_{0}}{\partial b^{(L)}} = \frac{\partial a^{(L)}}{\partial z^{(L)}}\cdot\frac{\partial C_{0}}{\partial a^{(L)}}$$

### What about $a^{(L-1)}$?
We also note how easy it is to find $\frac{\partial C_{0}}{\partial a^{(L-1)}}$:
$$\frac{\partial C_{0}}{\partial a^{(L-1)}} = \frac{\partial z^{(L)}}{\partial a^{(L-1)}} \cdot\frac{\partial a^{(L)}}{\partial z^{(L)}}\cdot\frac{\partial C_{0}}{\partial a^{(L)}} = w^{(L)}\sigma'(z^{(L)})2(a^{(L)}-y)$$

### Putting all our definitions together:
We can pretty easily extend these definitions to see how much any term in any previous layer impacts our cost function!!![[Pasted image 20250607203923.png]]
expanding this to an actual NN instead of our super simple one is also pretty trivial, it just takes a few more indices to keep track of!

$a^{(L)}$ becomes $a^{(L)}_{j}$, to denote which node it references in the layer!
$a^{(L-1)}$ becomes $a^{(L)}_{k}$
$C_{0}$ is now defined as the sum of differences across all nodes: 
$$C_{0}=\sum_{j=0}^{n_{L}-1}(a_{j}^{(L)}-y_{j})^{2}$$
to describe a weight between node $a^{(L)}_{j}$ and $a^{(L-1)}_{k}$, we have $w^{(L)}_{j,k}$
we also change our definition for $z^{(L)}$ to be:
$$z^{(L)}_{j} = w^{(L)}_{j,{0}} a^{(L-1)}_{0} + w^{(L)}_{j,{1}} a^{(L-1)}_{1} + \dots + w^{(L)}_{j,{n}} a^{(L-1)}_{n} + b_{j} ^{(L)}$$
so that we can have:
$$a^{(L)}_{j} =\sigma(z^{(L)}_{j})$$
all of these definitions are consistent with this graphic:
![[Pasted image 20250607204930.png]]

and thus, to find the effect of any arbitrary edge on our cost ($\frac{\partial C_{0}}{\partial w^{(L)}_{j,k}}$), we have:
$$\frac{\partial C_{0}}{\partial w^{(L)}_{j,k}} = \frac{\partial z^{(L)}_{j}}{\partial w^{(L)}_{j,k}} \cdot\frac{\partial a^{(L)}_{j}}{\partial z^{(L)}_{j}}\cdot\frac{\partial C_{0}}{\partial a^{(L)}_{j}}$$
note that we must change our definition for $a_{k}^{(L-1)}$ a little bit, though:
$$\frac{\partial C_{0}}{\partial a^{(L-1)}_{k}} = \sum_{j=0}^{n_{L}-1} \frac{\partial z^{(L)}_{j}}{\partial a^{(L-1)}_{k}} \cdot\frac{\partial a^{(L)}_{j}}{\partial z^{(L)}_{j}}\cdot\frac{\partial C_{0}}{\partial a^{(L)}_{j}}$$
> this is because the node $a_{k}^{(L-1)}$ influences the Cost function through multiple paths, not just one!!

as a quick visual summary: ![[Pasted image 20250607205701.png]]
---
# From "All Machine Learning algorithms explained in 17 min" by Infinite Codes
Classical Machine Learning is broken into two categories:
1. Supervised Learning
	- Uses pre-categorized data & is largely task driven
	- Has 2 main sub-categories!
		1. Classification
			- aim to predict discrete categorical variable (ie a label or a class)
				- "what number did I write?"
		2. Regression
			- aim to predict a continuous numeric target variable
				- like predicting the price of a house given any # of features
2. Unsupervised Learning
	- Uses unlabelled data & is largely data driven
	- has 3 main sub-categories!
		1. Clustering
		2. Association
		3. Dimensionality Reduction


Supervised Learning Algorithms include:
- Linear Regression
- Logistic Regression
- K Nearest Neighbors (KNN)
	- can be used for Classification OR Regression!
- Support Vector Machine (SVM)
	- can be used for Classification OR Regression!
- Naive Bayes Classifier
- Decision Tree
	- Bagging & Random Forests
	- Boosting & Strong Learners
- Neural Networks/ Deep Learning

Unsupervised Learning Algorithms include:
- K-means Clustering
- Dimensionality Reduction
	- Principle Component Analysis (PCA)


# Resources Used:
<iframe width="560" height="315" src="https://www.youtube.com/embed/aircAruvnKk?si=GOJXD48GrwkPEtAn" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<iframe width="560" height="315" src="https://www.youtube.com/embed/IHZwWFHWa-w?si=o3o6YlLD0tb0WuWn" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<iframe width="560" height="315" src="https://www.youtube.com/embed/Ilg3gGewQ5U?si=Hz8xoM5IrkV0OJtE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<iframe width="560" height="315" src="https://www.youtube.com/embed/tIeHLnjs5U8?si=TfiZEJPhWJdyDact" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<iframe width="560" height="315" src="https://www.youtube.com/embed/E0Hmnixke2g?si=_qENZI4hlmGBuREE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>