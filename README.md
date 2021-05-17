# Recipie-for-Training-Neural-Nets

## Conde inspired from

(PyTorch) code inspired from: https://github.com/brando90/ultimate-utils/blob/081a3dfa5ec1310cc5b6191d57e6e061d50ebbad/ultimate-utils-proj-src/uutils/torch/__init__.py#L1508

## Citation

If you use this implementation consider citing us:

```
@software{brando2021recipie,
    author={Brando Miranda},
    title={Recipie-for-Training-Neural-Nets},
    url={https://github.com/brando90/Recipie-for-Training-Neural-Nets},
    year={2021}
}
```


## Inspired from Karpathy's (and others) discussions

Most common neural net mistakes: 
1) you didn't try to overfit a single batch first. 
2) you forgot to toggle train/eval mode for the net. 
3) you forgot to .zero_grad() (in pytorch) before .backward(). 
4) you passed softmaxed outputs to a loss that expects raw logits. ; others?🙂
5) you didn't use bias=False for your Linear/Conv2d layer when using BatchNorm, or conversely forget to include it for the output layer .This one won't make you silently fail, but they are spurious parameters
6) thinking view() and permute() are the same thing (& incorrectly using view)


---

references:
- https://twitter.com/karpathy/status/1013244313327681536
- https://pcc.cs.byu.edu/2017/10/02/practical-advice-for-building-deep-neural-networks/ 
- https://karpathy.github.io/2019/04/25/recipe/ 
- https://www.bigmarker.com/missinglink-ai/PyTorch-Code-to-Unpack-Andrej-Karpathy-s-6-Most-Common-NN-Mistakes 
- https://github.com/brando90/Recipie-for-Training-Neural-Nets
- https://www.youtube.com/watch?v=-SY4-GkDM8g
- https://www.youtube.com/watch?v=R_o6nUC1Nzo
- https://www.youtube.com/watch?v=jompe29_A74
