## Locally Biased Transformers Better Align with Human Reading Times

This repository contains the code supporting the findings of the paper "Locally Biased Transformers Better Align with Human Reading Times", published in the 13th edition of the Workshop on Cognitive Modeling and Computational Linguistics (CMCL 2024).

In the paper, we modify the self-attention mechanism of a transformer model to simulate a lossy context representation, biasing the model's predictions to give additional weight to the local linguistic context. We show that surprisal estimates from our locally-biased model generally provide a better fit to human psychometric data in three reading times datasets (MECO, Provo, Brown), underscoring the sensitivity of the human language processor to local linguistic information. However, we find no improvement over a GPT-2-based baseline in the UCL reading times corpus, and a decrease in model fit in the Natural Stories corpus.

Attention weights are initially computed using the standard dot-product attention ($W = QK^T$). Then, an exponential decay bias matrix $B$ is computed using an exponential decay function based on the absolute differences between positions in the sequence, scaled by a decay rate. Thus, the bias is computed as $B_{i, j} = e^{-\lambda |i - j|}$, where $i, j \in \{0, 1, \ldots, n-1\}$ indicate the position of two tokens in the sequence, $n$ specifies the sequence length, and $\lambda$ is the decay rate. As a final step, we blend together the original attention weights with the exponential decay bias with a weighted sum to obtain the final attention weights $A = (1 - \alpha) \cdot W + \alpha \cdot B$. As a last step, the softmax function is applied to $A$.

### Code

The script `hyperparam_search.py` uses a Tree-structured Parzen Estimator algorithm to identify the optimal $\alpha$ and $\lambda$ parameters. It searches for those optimal hyperparameters on the Provo corpus. 

The script `local_attention.py` uses the hyperparameters identified with the previous script to compute surprisal values used to predict reading times in the various corpora. 

### Data
The psychometric data is stored as a pickle file in `data.pkl`. The data we used was previously collected and released by other research groups, so if you use it, be sure to cite them. The appropriate references are provided in the article (§3.2).
