![](images/cubegpt3-logo.webp)

# CubeGPT-3: Scaling up Transformer-based Rubik's Cube Solvers
A research report.

## Introduction

Solving Rubik’s Cubes is hard! There are on the order of 10^19 possible Rubik’s Cube states, and only one winning state. Any standard algorithms like A* search, which explore a significant fraction of possible states, are rendered useless. We have to prune trees extremely aggressively, which is where an estimator trained via machine learning comes in.

![](images/scrambled_to_solved.png)

All 3D renders in this project made using a 3D renderer I built, by hand.

So does the AI work? Yes! For instance, it finds the following 18-move solution to the cube pictured above:

https://github.com/user-attachments/assets/dd51b826-f864-45e7-99cc-ccca08a05a2e

### Previous Work

This project builds significantly off of the DeepCubeA and CayleyPy papers. It also builds off <a href=https://github.com/stanleytheli/cubegpt>my previous work on training transformers to solve Rubik’s Cubes, CubeGPT-2</a>. Essentially, the strongest model I tested in solving was CubeGPT-2g-PFT, an out-of-the-box transformer architecture running the GeLU activation trained over 25M samples. The strongest model I trained was CubeGPT-CLS-t154m-PFT, which was a classifier model trained on just over 200M samples. 

In this work, CubeGPT-3, I improve the model architecture and training pipeline, and devote more compute to larger-scale experiments, getting significantly closer to optimality. Experiments also reveal more about the nature of training Rubik's Cube models.

## Training

We update the architecture of CubeGPT using some more modern research and Rubik’s Cube symmetries. We also test ResNets to verify that the transformer architecture actually provides benefits.

![](images/iteration_loss.webp)

In order of being added:

* **GPU optimization**: This doesn’t directly improve the models, but speeds up experiments, allowing for faster experiments and hyperparameter tuning. The most significant optimization was moving the data pipeline onto the GPU, which sped up training ~2x and inference ~3x.
* **No Classifier Token**: The old transformer used a special token denoted the ‘cls-token’ to make the final prediction distribution. We now linear over the whole token sequence instead, which empirically works significantly better (possibly allows gradients to flow more thoroughly?)
* **Canonicalization**: Imagine rotating a Rubik’s Cube and then switching back all the colors. This looks different, but really it’s the exact same puzzle. 24 rotations of the cube gives 24 symmetries; we can mirror the cube as well to increase this to 48 symmetries. We enforce this on the model with a cheap preprocessing step. 
* **Gated MLP**: Old MLPs used to do f(x) = B(GeLU(Ax)) for matrices A, B. Gated MLPs do f(x) = C(Ax * SiLU(Bx)). Empirically shown to be better.
* **Mixture of Experts (MoE)**: Deepseek V3-style, using explicitly tuned expert biases for load balancing. Significantly increases information capacity while barely slowing down the forward pass.

Other GPU optimizations include moving matmuls to TF32 (degrades models surprisingly little while speeding up calculations ~1.5x) and simply upgrading hardware.

The MoE models actually train a bit slower, which is expected given that their gradients can only flow through one expert at a time. However, when we train an 8 expert model for longer, we indeed see it’s capable of reaching low loss. 

We also develop our training methods to encourage better convergence. Here's a long training run with an 8 expert model. Using initial batch size 256, it appears to saturate at 100 million samples; I call this CubeGPT-3-8A1-t100m. Increasing batch size to 4096 (also upgraded my hardware!) gives stabler gradients and measurements, letting us achieve a lower-loss convergence at 300 million samples; I call that model CubeGPT-3-8A1-t300m.

![](images/longer_training_run.webp)

When we account for a predicted residual loss value, and then smooth logarithmically (avg. over all samples within 2/3 - 3/2 of a given point), we find a beautiful power law. 

![](data_analysis/output/out_sam_f15.png)

Using these interventions, we see that CubeGPT-3 is already a step change in capabilities above our old best models, while maintaining similar inference speed due to holding the number of active parameters constant.

|Model|Training Examples (Millions)|Mean Squared Error|Within-3-Accuracy| Within-4-Accuracy|Accuracy (When rounding to nearest int)|
| --- | --- | --- | --- | --- | --- |
|CubeGPT-2g-PFT | 26.2 | 4.70 | 82.49% | 91.37% | 37.88% |
|CubeGPT-CLS-t154mPFT | 204.8 | 4.30 | 83.96% | 92.40% | 42.50% |
|CubeGPT-3-8A1-t100m | 102.4 |  4.06 | 84.6% | 92.9% | 44.6%
|CubeGPT-3-8A1-t300m | 307.2 |  **4.01** | **84.8%** | **93.0%** | **44.9%**
|CubeGPT-3-8A1-t100mPFT | 153.6 |  4.05 | 84.6% | 92.9% | 44.5%

Path finetuning does not have a clear effect on the metrics.

## Solving

While doing more experiments on solving, I found a small number of positions (est. <1%) that couldn't be solved at 4000 width by certain models. However, at 16000 width, this problem seems to disappear. The results broadly show that the CubeGPT-3 models broadly beat the CubeGPT-2 generation. Path finetuning appears to have a significant effect on solving, decreasing t100m's average solve length from 20.5 to 20.2. 

|Model| Width | Solve Rate | Average Moves | MAO |
| ---| --- | --- | --- | --- |
|CubeGPT-2g-PFT| 4000 | 99% | 21.1 | 3.4 |
|CubeGPT-3-8A1-t100m| 4000 | 99% | 20.5 | 2.8 |
|CubeGPT-3-8A1-t300m| 4000 | 100% | 20.3 | 2.6 |
|CubeGPT-3-8A1-t100mPFT| 4000 | 99% | 20.2 | 2.5 |
|CubeGPT-3-8A1-t100mPFT| 16000 | 100% | 19.6 | 1.9 |
|CubeGPT-3-8A1-t100mPFT| 64000 | 100% | 19.0 | 1.3 |


There are two big takeaways here. Firstly, the evals don't tell the whole story. PFT — which while working on CubeGPT-2 I dismissed as some neat benchmaxxing trick — is not just beneficial but seemingly necessary for solving. Secondly, increasing width is incredibly powerful. Interestingly, 8000 width seems to work worse than 4000 and 16000. It's possible that this has an intuitive reason rooted in beam search dynamics, but I'm not seeing why. 

## Future Work

My takeaway from this is that we need better evals. It would also be great to understand what about path finetuning particularly induces solving capability, and to see if we could purposefully boost that. My current interpretation is that training on unrelated samples gives models a global sense of Cube structure, but no local one --- there's no guarantee they'll rank nearby states sensibly. PFT induces that. 

I plan to update this project as I scale and investigate PFT further. I would go so far as to refer to training on non-paths as "pretraining" and PFT as "posttraining," but then I think the LARP would get ridiculous, even for me. There are also other ideas I plan to experiment with, such as recurrence from previous states.

## More Solve Videos

https://github.com/user-attachments/assets/d71554ae-2c3d-4c95-9c00-17c6c1f9edbc

https://github.com/user-attachments/assets/bf7bce2e-1663-4972-af68-6b5380861241
