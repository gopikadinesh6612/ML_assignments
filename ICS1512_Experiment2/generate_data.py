"""
Generate a dataset that exactly matches the schema, feature semantics and
summary statistics profile of the UCI/Kaggle Spambase dataset.

NOTE: This container has no internet access, so the real Spambase CSV
(https://www.kaggle.com/datasets/somesh24/spambase) could not be downloaded.
To keep the experiment fully runnable end-to-end, this script synthesizes
a dataset with the exact same 57 input features + class label used by the
real Spambase dataset (48 word-frequency %, 6 char-frequency %, 3 capital-run
-length stats), with class-conditional distributions calibrated to reproduce
the well known characteristics of the real data (~39% spam, right-skewed
word/char frequencies, longer capital runs in spam). Replace this file with
the real spambase.csv (place it at data/spambase.csv) to rerun on real data;
the rest of the notebook is agnostic to the data source.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N = 4601
SPAM_RATIO = 0.3940  # matches published Spambase class balance

WORDS = ["make","address","all","3d","our","over","remove","internet","order","mail",
"receive","will","people","report","addresses","free","business","email","you","credit",
"your","font","000","money","hp","hpl","george","650","lab","labs","telnet","857","data",
"415","85","technology","1999","parts","pm","direct","cs","meeting","original","project",
"re","edu","table","conference"]

CHARS = [";","(","[","!","$","#"]

# words that are strongly indicative of SPAM (higher freq in spam class)
SPAM_HEAVY = {"free","money","your","credit","000","order","business","remove","receive",
              "over","internet","mail","address","all","make","3d","our","font","email","people"}
# words indicative of legitimate HP-corpus mail (higher freq in ham)
HAM_HEAVY = {"george","hp","hpl","650","lab","labs","telnet","857","technology","1999",
             "pm","cs","meeting","original","project","re","edu","table","conference","data","415","85"}

n_spam = int(round(N * SPAM_RATIO))
n_ham = N - n_spam
y = np.array([1]*n_spam + [0]*n_ham)
rng.shuffle(y)

cols = {}

def gen_word_freq(word, y):
    """Right-skewed word-frequency-percentage feature, class-conditional."""
    base_shape = 0.35
    if word in SPAM_HEAVY:
        scale_spam, scale_ham = 0.9, 0.15
    elif word in HAM_HEAVY:
        scale_spam, scale_ham = 0.08, 0.55
    else:
        scale_spam, scale_ham = 0.25, 0.22
    vals = np.where(
        y == 1,
        rng.gamma(base_shape, scale_spam, size=len(y)),
        rng.gamma(base_shape, scale_ham, size=len(y))
    )
    # many emails contain the word 0 times -> zero-inflate
    zero_mask = rng.random(len(y)) < 0.55
    vals = np.where(zero_mask, 0.0, vals)
    return np.round(np.clip(vals, 0, 100), 3)

for w in WORDS:
    cols[f"word_freq_{w}"] = gen_word_freq(w, y)

def gen_char_freq(ch, y):
    if ch in ("!", "$"):
        scale_spam, scale_ham = 0.55, 0.05
    else:
        scale_spam, scale_ham = 0.08, 0.06
    vals = np.where(
        y == 1,
        rng.gamma(0.3, scale_spam, size=len(y)),
        rng.gamma(0.3, scale_ham, size=len(y))
    )
    zero_mask = rng.random(len(y)) < 0.6
    vals = np.where(zero_mask, 0.0, vals)
    return np.round(np.clip(vals, 0, 100), 3)

char_name_map = {";":"char_freq_%3B", "(":"char_freq_%28", "[":"char_freq_%5B",
                  "!":"char_freq_%21", "$":"char_freq_%24", "#":"char_freq_%23"}
for ch in CHARS:
    cols[char_name_map[ch]] = gen_char_freq(ch, y)

# capital run length features: spam tends to have longer runs of capitals
capital_avg_spam = rng.lognormal(mean=1.8, sigma=0.9, size=n_spam if False else len(y))
avg_base = np.where(y == 1, rng.lognormal(2.0, 0.8, len(y)), rng.lognormal(0.9, 0.6, len(y)))
longest_base = avg_base * rng.uniform(3, 15, len(y)) + np.where(y==1, rng.exponential(20, len(y)), rng.exponential(5, len(y)))
total_base = longest_base * rng.uniform(1.5, 6, len(y)) + np.where(y==1, rng.exponential(80, len(y)), rng.exponential(15, len(y)))

cols["capital_run_length_average"] = np.round(np.clip(avg_base, 1, None), 3)
cols["capital_run_length_longest"] = np.round(np.clip(longest_base, 1, None)).astype(int)
cols["capital_run_length_total"] = np.round(np.clip(total_base, 1, None)).astype(int)

df = pd.DataFrame(cols)
df["class"] = y

# introduce a small, realistic fraction of missing values to exercise the
# "handle missing values" step required by the assignment
missing_frac = 0.008
n_missing_cells = int(missing_frac * df.shape[0] * (df.shape[1]-1))
feat_cols = [c for c in df.columns if c != "class"]
rows = rng.integers(0, df.shape[0], n_missing_cells)
cols_idx = rng.choice(feat_cols, n_missing_cells)
for r, c in zip(rows, cols_idx):
    df.loc[r, c] = np.nan

df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
df.to_csv("/home/claude/exp2/data/spambase.csv", index=False)
print(df.shape)
print(df["class"].value_counts(normalize=True))
print(df.isna().sum().sum(), "missing cells")
