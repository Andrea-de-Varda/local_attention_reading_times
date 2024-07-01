# pip install --target=/home/dev/anaconda3/lib/python3.9/site-packages --upgrade plotly

from transformers import GPT2Model, GPT2LMHeadModel, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block
import torch.nn as nn
from transformers import GPT2Tokenizer
import torch
import torch.nn.functional as F
import re
import statsmodels.api as sm
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from scipy.stats import pearsonr
import optuna
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pickle
import matplotlib

#######################
# surprisal functions #
#######################

def get_surprisal(prompt, toker, model):
    inputs = toker(prompt, return_tensors="pt")
    input_ids, output_ids = inputs["input_ids"], inputs["input_ids"][:, 1:]
    outputs = model(**inputs, labels=input_ids)
    logits = outputs.logits
    logprobs = torch.gather(F.log_softmax(logits, dim=2), 2, output_ids.unsqueeze(2))
    return [-item[0] for item in logprobs.tolist()[0]]

def tok_maker(a, toker, sep, cased = False):
    # Credit to Ben S. https://stackoverflow.com/questions/74458282/match-strings-of-different-length-in-two-lists-of-different-length
    plainseq = " ".join(a)
    b = [re.sub(sep, "", item) for item in toker.tokenize(plainseq)]
    c = []
    if cased:
        for element in a:
            temp_list = []
            while "".join(temp_list) != element:
                temp_list.append(b.pop(0))
            c.append(temp_list)
    else:
        for element in a:
            temp_list = []
            while "".join(temp_list) != element.lower():
                temp_list.append(b.pop(0))
            c.append(temp_list)
    return c

def get_surprisal_tokens(tokens, model, toker, sep, cased=False):
    s = get_surprisal(" ".join(tokens), toker, model)
    toks = tok_maker(tokens, toker, sep, cased)
    theindex = 0
    out = []
    for index, word in enumerate(toks[1:]):
        if len(word) == 1:
            surp = s[theindex]
            theindex += 1
            out.append(surp)
        else:
            surp = s[theindex:theindex+len(word)]
            theindex += len(word)
            out.append(sum(surp))
    return out

#################
# provo preproc #
#################

provo = pd.read_csv('Provo_Corpus-Eyetracking_Data.csv', sep=",") # not included in this repo!
provo = provo[['Text_ID','Word_Unique_ID','Word_Number', 'Sentence_Number', 'Word_In_Sentence_Number','Word', 'Word_Cleaned', 'Word_Length', 'IA_FIRST_FIXATION_DURATION','IA_FIRST_RUN_DWELL_TIME']]
provo['IA_FIRST_FIXATION_DURATION'] = provo['IA_FIRST_FIXATION_DURATION'].fillna(0)
provo['IA_FIRST_RUN_DWELL_TIME'] = provo['IA_FIRST_RUN_DWELL_TIME'].fillna(0)
provo = provo.groupby(["Sentence_Number", "Word_In_Sentence_Number", "Word", "Text_ID"], as_index=False, sort=False).agg({'IA_FIRST_FIXATION_DURATION':"mean", 'IA_FIRST_RUN_DWELL_TIME':"mean", 'Word_Number':"mean", 'Sentence_Number':"mean", "Word_Unique_ID":"max", "Word_In_Sentence_Number":"mean", "Word":"max", "Word_Cleaned":"max", "Word_Length":"mean", "Text_ID":"mean"})

provo.columns
provo["Word_Unique_ID"]
set(provo["Sentence_Number"])
set(provo["Text_ID"])

# in DF, the first word is missing! Take it from stimuli data
first_word = [s.split("\t")[2][1:-2].split()[0] for s in open("sentences/provo_sentences.txt").readlines()]
d_f_w = {num+1 : first_word[num] for num in range(55)}

# frequency
freq = pd.read_excel("frequency/subtlex.xlsx")
f = {row.Word : np.log(row.FREQcount) for index, row in freq.iterrows()}
minfreq = min(f.values())

def get_f(word):
    word = re.sub('[\.\,\:\-\?\!\)\(\"]', "", word)
    try:
        fr = f[word]
    except KeyError:
        fr = minfreq
    return fr

fp = []; provo_len = []; provo_freq = []; provo_sent = []; provo_word = []
for text in set(provo["Text_ID"]):
    temp = provo[provo["Text_ID"] == text]
    fp.extend(list(temp["IA_FIRST_RUN_DWELL_TIME"]))
    provo_len.extend(list(temp["Word_Length"]))
    provo_freq.extend([get_f(w) for w in list(temp["Word"])])
    provo_word.extend(list(temp["Word"]))
    sent = " ".join([d_f_w[text]]+list(temp["Word"]))
    provo_sent.append(sent)

fp = np.array(fp); provo_len = np.array(provo_len); provo_freq = np.array(provo_freq)

provo_sent = [re.sub("\n", "", s).split() for s in open("sentences/provo_sentences.txt").readlines()]


def model_evaluation(decay_rate, alpha):
    
    class CustomGPT2Attention(GPT2Attention):
        def __init__(self, config, layer_idx=None):
            super().__init__(config)
        def _attn(self, query, key, value, attention_mask=None, head_mask=None):
                attn_weights = torch.matmul(query, key.transpose(-1, -2))
                if self.scale_attn_weights:
                    attn_weights = attn_weights / torch.full(
                        [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
                    )
                ##############################################################################
                seq_len = query.size(-2) 
                indices = torch.arange(seq_len, device=query.device)
                exponential_decay_bias = torch.exp(-torch.abs(indices[None, :] - indices[:, None]) * decay_rate)
                #gauss_bias = torch.exp(-(indices[None, :] - indices[:, None]) ** 2 / (2 * (sigma ** 2)))
                #attn_weights = (1 - alpha) * attn_weights + alpha * (attn_weights * gauss_bias)
                attn_weights = (1 - alpha) * attn_weights + alpha * exponential_decay_bias
                ###############################################################################
                # Layer-wise attention scaling
                if self.scale_attn_by_inverse_layer_idx:
                    attn_weights = attn_weights / float(self.layer_idx + 1)
                if not self.is_cross_attention:
                    # if only "normal" attention layer implements causal mask
                    query_length, key_length = query.size(-2), key.size(-2)
                    causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
                    mask_value = torch.finfo(attn_weights.dtype).min
                    # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
                    # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
                    mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
                    attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
                if attention_mask is not None:
                    # Apply the attention mask
                    attn_weights = attn_weights + attention_mask
                attn_weights = nn.functional.softmax(attn_weights, dim=-1)
                # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
                attn_weights = attn_weights.type(value.dtype)
                attn_weights = self.attn_dropout(attn_weights)
                # Mask heads if we want to
                if head_mask is not None:
                    attn_weights = attn_weights * head_mask
                attn_output = torch.matmul(attn_weights, value)
                return attn_output, attn_weights

    class CustomGPT2Block(GPT2Block):
        def __init__(self, config, layer_idx=None, use_custom_attention=False):
            super().__init__(config)
            if use_custom_attention:
                self.attn = CustomGPT2Attention(config, layer_idx=layer_idx)
        
    class CustomGPT2Model(GPT2LMHeadModel):
        def __init__(self, config):
            super().__init__(config)
            self.transformer = GPT2Model(config)
            total_layers = config.n_layer
            self.transformer.h = nn.ModuleList(
            [CustomGPT2Block(config, use_custom_attention=(i == total_layers - 1)) for i in range(total_layers)]
        )
        

    toker = GPT2Tokenizer.from_pretrained("gpt2")
    custom_model = CustomGPT2Model.from_pretrained("gpt2")
    s_custom_provo = [s for sent in tqdm(provo_sent) for s in get_surprisal_tokens(sent, custom_model, toker, sep = "Ġ", cased=True)]
    X1 = np.column_stack((provo_len, provo_freq, s_custom_provo))
    X1 = sm.add_constant(X1)
    model_loc = sm.OLS(fp, X1)
    results_loc = model_loc.fit()
    score = -results_loc.llf
    
    print(f"\nNegative Log Likelihood = {score}")
    print(f"R2 = {results_loc.rsquared}")
    
    return score

def objective(trial):
    decay_rate = trial.suggest_float('decay_rate', 0.001, 100)
    alpha = trial.suggest_float('alpha', 0., 1.0)
    loss = model_evaluation(decay_rate, alpha)
    return loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Best parameters found
best_params = study.best_params
best_loss = study.best_value

print("Best parameters: ", best_params)
print("Best loss: ", best_loss) 

# with open('optim/optuna_study_GPT2_small_expdecay.pkl', 'wb') as f:
#     pickle.dump(study, f)

with open('optim/optuna_study_GPT2_small_expdecay.pkl', 'rb') as f:
    study = pickle.load(f)
    
############
# PLOTTING #
############

matplotlib.rcParams['font.family'] = ['serif']
matplotlib.rcParams['font.serif'] = ['Times New Roman']

# (1) OPTIM HISTORY
scores = [trial.value for trial in study.trials if trial.value is not None]
plt.figure(figsize=(10, 6))
plt.plot(scores, label='Score per Trial')
plt.xlabel('Trial')
plt.ylabel('Score')
plt.title('Optimization History')
plt.legend()
plt.show()

# (2) Hyperparameter Importance
decay = [trial.params['decay_rate'] for trial in study.trials if trial.value is not None]
alphas = [trial.params['alpha'] for trial in study.trials if trial.value is not None]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(decay, scores)
plt.xlabel('decay')
plt.ylabel('Score')
plt.title('decay vs. Score')
#############################
plt.subplot(1, 2, 2)
plt.scatter(alphas, scores)
plt.xlabel('$\\alpha$')
plt.ylabel('Score')
plt.title('$\\alpha$ vs. Score')
plt.tight_layout()
plt.show()

# (3) Contour Plot
# Create grid
decay_grid, alpha_grid = np.meshgrid(np.linspace(min(decay), max(decay), 100),
                                     np.linspace(min(alphas), max(alphas), 100))

# Interpolate
score_grid = griddata((decay, alphas), scores, (decay_grid, alpha_grid), method='cubic')

plt.figure(figsize=(3, 2.5), dpi = 300)
plt.contourf(decay_grid, alpha_grid, score_grid, levels=50, cmap='viridis')
plt.colorbar(label='Negative Log-Likelihood')
plt.xlabel('Decay')
plt.ylabel('$\\alpha$')
#plt.title('Contour of Score')
plt.show()

