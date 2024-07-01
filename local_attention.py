from transformers import GPT2Model, GPT2LMHeadModel, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block
import torch.nn as nn
from transformers import GPT2Tokenizer
import torch
import torch.nn.functional as F
import re
import numpy as np
from os import chdir, path
import pandas as pd
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import statsmodels.api as sm
import scipy
import matplotlib
import pickle
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from math import sqrt

decay_rate = 82.85603928544775 # hyperparameters identified with gridsearch.py
alpha = 0.3659550432333628

class CustomGPT2Attention(GPT2Attention):
    def __init__(self, config, layer_idx=None):
        super().__init__(config)
        
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
            attn_weights = torch.matmul(query, key.transpose(-1, -2))
    
            if self.scale_attn_weights:
                attn_weights = attn_weights / torch.full(
                    [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
                )
            
            #######################
            # begin GAUSSIAN BIAS ###################################################
            #######################
            seq_len = query.size(-2)
            indices = torch.arange(seq_len, device=query.device)
            exponential_decay_bias = torch.exp(-torch.abs(indices[None, :] - indices[:, None]) * decay_rate)
            attn_weights = (1 - alpha) * attn_weights + alpha * exponential_decay_bias
            
            #####################
            # end GAUSSIAN BIAS ###################################################
            #####################
            
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
    def __init__(self, config, layer_idx=None):
        super().__init__(config)
        # Replace the standard attention with the custom one
        self.attn = CustomGPT2Attention(config, layer_idx=layer_idx)
        
class CustomGPT2Model(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        # Replace GPT2Model with a modified version having custom blocks
        self.transformer = GPT2Model(config)
        self.transformer.h = nn.ModuleList([CustomGPT2Block(config, i) for i in range(config.n_layer)])
        

###############################################################################

def get_surprisal(prompt, toker, model):
    inputs = toker(prompt, return_tensors="pt")
    input_ids, output_ids = inputs["input_ids"], inputs["input_ids"][:, 1:]
    with torch.no_grad():
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

#############
# LOAD DATA #
#############
    
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)
    
folder_path = 'surprisal' # make folder to save surprisal estimates
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' was created.")
else:
    print(f"Folder '{folder_path}' already exists.")
    
modelname = "gpt2"

toker = GPT2Tokenizer.from_pretrained(modelname)
custom_model = CustomGPT2Model.from_pretrained(modelname)
pretrained_model = GPT2LMHeadModel.from_pretrained(modelname)
    
results = []
print(f"\n\nProcessing {modelname.upper()}")
for dataname in ['meco', 'provo', 'ucl', 'ucl_spr', 'brown', 'natstor']:
    print(dataname.upper())
    sentences = data[dataname]["sent"]
    freq = data[dataname]["freq"]
    length = data[dataname]["len"]
    fp = data[dataname]["fp"]
    if path.isfile(f'surprisal/{dataname}_{modelname}_custom.pkl'):
        with open(f'surprisal/{dataname}_{modelname}_custom.pkl', 'rb') as f:
            s_custom = pickle.load(f)
    else:
        s_custom = [s for sent in tqdm(sentences) for s in get_surprisal_tokens(sent, custom_model, toker, sep = "Ġ", cased=True)]
        with open(f'surprisal/{dataname}_{modelname}_custom.pkl', 'wb') as f:
            pickle.dump(s_custom, f)
    
    if path.isfile(f'surprisal/{dataname}_{modelname}_base.pkl'):
        with open(f'surprisal/{dataname}_{modelname}_base.pkl', 'rb') as f:
            s_orig = pickle.load(f)
    else:
        s_orig = [s for sent in tqdm(sentences) for s in get_surprisal_tokens(sent, pretrained_model, toker, sep = "Ġ", cased=True)]
        with open(f'surprisal/{dataname}_{modelname}_base.pkl', 'wb') as f:
            pickle.dump(s_orig, f)
    X1 = np.column_stack((length, freq, s_custom))
    X1 = sm.add_constant(X1)
    model_loc = sm.OLS(fp, X1)
    results_loc = model_loc.fit()
    X2 = np.column_stack((length, freq, s_orig))
    X2 = sm.add_constant(X2)
    model_base = sm.OLS(fp, X2)
    results_base = model_base.fit()
    
    results.append([dataname, modelname, 
                    pearsonr(s_orig, s_custom)[0], pearsonr(s_orig, fp)[0],pearsonr(fp, s_custom)[0], 
                    pearsonr(freq, s_custom)[0], pearsonr(s_orig, freq)[0],
                    pearsonr(length, s_custom)[0],pearsonr(s_orig, length)[0],
                    results_loc.rsquared, results_loc.aic, -results_loc.llf, 
                    results_base.rsquared, results_base.aic, -results_base.llf])
    print(f"\n {results_loc.aic - results_base.aic}")



results = pd.DataFrame(results, columns = ["corpus", "model", "s_orig_s_custom", "s_orig_y", "s_custom_y", "s_custom_f", "s_orig_f", "s_custom_l", "s_orig_l", "custom_r2", "custom_aic", "custom_ll", "orig_r2", "orig_aic", "orig_llf"])
results["deltaAic"] = results["custom_aic"] - results["orig_aic"]

###############################################################################

# Idea >>> correlate DeltaAIC with avg sentence length per corpus (our method should be beneficial in particular for small corpora)

corpus_dict = {"meco" : "MECO", "ucl":"UCL$_{ET}$", "ucl_spr":"UCL$_{SPR}$","provo":"Provo", "brown" : "Brown", "natstor" : "NatStor"}

pastel = (0.2, .8, 0.6)

###############################
# plot avg per-word surprisal #
###############################

results_surp = []
for modelname in ["gpt2"]:
    for dataname in ['meco', 'provo', 'ucl', 'ucl_spr', 'brown', 'natstor']:
        print(dataname.upper())
        with open(f'surprisal/{dataname}_{modelname}_custom.pkl', 'rb') as f:
            s_custom = pickle.load(f)
        
        with open(f'surprisal/{dataname}_{modelname}_base.pkl', 'rb') as f:
            s_orig = pickle.load(f)
        
        results_surp.append([dataname, modelname, np.mean(s_custom), np.mean(s_orig)])
        print(f"\n {np.mean(s_custom) - np.mean(s_orig)}")
results_surp = pd.DataFrame(results_surp, columns = ["corpus", "model", "s_custom", "s_orig"])

matplotlib.rcParams['font.family'] = ['serif']
matplotlib.rcParams['font.serif'] = ['Times New Roman']


index = np.array(range(len(results)))
bar_width = .35
x_label_fontsize = 8
titlesize = 11
fig, axs = plt.subplots(2, 2, dpi=300, figsize=(9/2.7, 7/2.7), gridspec_kw={'hspace': 1.1, "wspace" : 0.6})
ax = axs[1,1]
bar1 = ax.bar(index, results['deltaAic'])
ax.set_ylabel('$\\Delta$AIC')
ax.set_xticks(index)
ax.set_xticklabels([corpus_dict[c] for c in results['corpus']], rotation=45, ha="right", fontsize=x_label_fontsize)
ax.axhline(y=0, color='black', linestyle='-', lw = 1)
#ax.yaxis.tick_right()
ax.set_title("D. Model fit", loc="left", fontsize = titlesize)
    
ax = axs[0, 0]
bar1 = ax.bar(index, results_surp['s_custom'], bar_width)
bar2 = ax.bar(index + bar_width, results_surp['s_orig'], bar_width, color=pastel)
ax.set_ylabel('$\overline{s}$')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels([corpus_dict[c] for c in results['corpus']], rotation=45, ha="right", fontsize=x_label_fontsize)
ax.set_ylim(3.2, 5.9)
#ax.yaxis.tick_right()
ax.set_title("A. Mean surprisal", loc="left", fontsize = titlesize)

ax = axs[1, 0]
bar1 = ax.bar(index, results['s_custom_y'], bar_width)
bar2 = ax.bar(index + bar_width, results['s_orig_y'], bar_width, color=pastel)
ax.set_ylabel('r')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels([corpus_dict[c] for c in results['corpus']], rotation=45, ha="right", fontsize=x_label_fontsize)
ax.set_ylim(0.1, None)
#ax.yaxis.tick_right()
ax.set_title("C. Response", loc="left", fontsize = titlesize)

ax = axs[0, 1]
bar1 = ax.bar(index, results['s_custom_f'], bar_width)
bar2 = ax.bar(index + bar_width, results['s_orig_f'], bar_width, color=pastel)
ax.set_ylabel('r')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels([corpus_dict[c] for c in results['corpus']], rotation=45, ha="right", fontsize=x_label_fontsize)
ax.set_ylim(None, -0.21)
#ax.yaxis.tick_right()
ax.set_title("B. Frequency", loc="left", fontsize = titlesize)
fig.subplots_adjust(hspace=0.5) 
plt.tight_layout()
plt.show()


# length and DeltaAIC (APPENDIX)

len_corpus = {}
for dataname in data.keys():
    mean_l = []
    for s in data[dataname]["sent"]:
        mean_l.append(len(s))
    mean_l = np.mean(mean_l)
    len_corpus[dataname] = mean_l
    
results["len_corpus"] = results["corpus"].map(len_corpus)
pearsonr(results["len_corpus"], results["deltaAic"])
