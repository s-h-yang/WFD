# Weighted Flow Diffusion

Reproduces empirical results in our paper [Weighted flow diffusion for local graph clustering with node attributes: an algorithm and statistical guarantees](https://arxiv.org/abs/2301.13187).

- The notebook `synthetic.ipynb` reproduces Figure 1 and Figure 2 in the paper.
- To reproduce Tables 1, 3, 5 in the paper, run Julia script `experiment-{dataset}.jl`.
  This will produce two output text files. 
  `F1-{datasset}-base.txt` stores average F1 scores without using node attributes. 
  `F1-{dataset}-wfd.txt` stores average F1 scores using node attributes.
