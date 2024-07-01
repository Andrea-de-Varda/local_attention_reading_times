## Locally Biased Transformers Better Align with Human Reading Times

This repository contains the code supporting the findings of the paper "Locally Biased Transformers Better Align with Human Reading Times", published in the 13th edition of the Workshop on Cognitive Modeling and Computational Linguistics (CMCL 2024).

In the paper, we modify the self-attention mechanism of a transformer model to simulate a lossy context representation, biasing the model's predictions to give additional weight to the local linguistic context. We show that surprisal estimates from our locally-biased model generally provide a better fit to human psychometric data in three reading times datasets (MECO, Provo, Brown), underscoring the sensitivity of the human language processor to local linguistic information. However, we find no improvement over a GPT-2-based baseline in the UCL reading times corpus, and a decrease in model fit in the Natural Stories corpus.

### Code
The script `gridsearch.py` 

### Data
The psychometric data is stored as a pickle file in `data.pkl`. The data we used was previously collected and released by other research groups, so if you use it, be sure to cite them. The appropriate references are provided in the article.
