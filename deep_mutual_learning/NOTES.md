# Deep Mutual Learning

Mutual Learning is a different paradigm where a cohort of students learns together. Unlike distillation, in DML, there is no single teacher.

## Introduction
- Deep Neural Networks are:
    - Large in depth or width -> resulting in longer inference times
    - Contain a lot of parameters -> requiring more memory to store
- Small networks often have the same representation capacity as large networks.
- However, they are harder to train and find the right parameters to realize the desired function.
- In distillation, there is:
    - A large and powerful teacher model
    - A smaller student network that is taught to mimic the large teacher model
    - It turns out mimicking the large teacher model is easier than learning the objective function for a smaller network.
- Distillation is a *one-way* knowledge transfer.
- In DML, a pool of students *learns simultaneously*.
- Each student is trained with a combination of:
    - Supervised loss -> General objective
    - Mimicry loss -> Aligns each student's class posterior with the class probabilities of other students
- Each network starts from different initial conditions, so their estimates of the probabilities of the next classes vary. These secondary quantities provide extra information in DML (and even in distillation).
- Compared to independent training or having a static teacher, training in a cohort gives better results.
- Some observations are:
    - Many small networks can fit on the same GPU -> Better efficiency
    - We can have heterogeneous cohorts with a mix of big and small models.
    - Even large networks mutually trained in a cohort improve performance compared to independent training.

## Formulation
- If the DML cohort consists of two networks, the probability for one class of a particular network is given by its logits as an output of the softmax layer.
<insert softmax formula>
- For multi-class classification, we can use cross-entropy error.
<insert cross-entropy formula>
- For the mimicry error, we use KL-Divergence between the predictions of both networks.
- Therefore, the overall loss function is a sum of the supervised cross-entropy loss and the KL divergence of the other network with respect to the current one.

## Optimization
- This strategy is applied in each mini-batch-based model update step and throughout the whole training process.
- At each iteration, we compute the predictions of the two models and update both networks' parameters according to the predictions of the other.
- The optimization of both networks is conducted iteratively until convergence.

## Extension to Larger Student Cohorts
- If there are more than two networks, then for a given student network, the loss is the sum of the supervised cross-entropy loss and the "normalized" sum of all the KL divergences with respect to the current network.
- The coefficient of \(1/(K-1)\) is added to ensure that the teaching is primarily directed by the supervised learning of the true labels.
- An alternative is to take the ensemble of all the networks and create a single ensemble teacher (which is somewhat like distillation). However, while the teacher's posterior probabilities are more peaked at the true class, the posterior entropy over all classes is reduced. This is contradictory to the objectives of DML, which aim to provide robust solutions with high posterior entropy.

## Results
- It performs better than state-of-the-art models (obviously).

### Distillation
    - Having a pre-trained powerful teacher model obviously performs better.
    - However, if you have a powerful teacher that mutually trains with the smaller student, it yields better results.
    - Through mutual learning, the network that would play the role of the teacher becomes better than the pre-trained teacher via learning from interactions with an a-priori untrained student.
    - Using the same model as both teacher and student makes results worse (duh).

### Larger Cohort Sizes
    - The generalization ability of students is enhanced when learning together with an increasing number of peers.
    - Results become more stable as the number of networks in DML increases.
    - Ensemble predictions outperform individual network predictions, as expected.

### How and Why Does DML Work?
- The tendency toward finding robust minima can be enhanced by biasing deep networks toward solutions with higher posterior entropy.
- In practice, the training loss is 0 with minimal classification loss, but DML performs better on test data.
- Therefore, rather than finding a better minimum of the training loss, DML appears to help find a wider/more robust minimum that generalizes better to test data.
- Overall, networks in DML tend to aggregate their predictions of secondary probabilities, placing more mass on secondary probabilities altogether and assigning non-zero mass to more distinct secondary probabilities.
- In ensembles, the teaching signal of the ensemble compared to peer teaching is sharper on the true label. However, while the noise-averaging property of ensembling is effective for correct predictions, it is detrimental to providing a teaching signal where secondary class probabilities are the salient cue. High-entropy posteriors lead to more robust solutions during model training.
