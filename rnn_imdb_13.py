

import torch, sklearn
print('torch', torch.__version__, '| GPU бар ма:', torch.cuda.is_available())

SEED = 13   # <-- өз тіркеу нөміріңіз


import random, time, re, copy, math
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from collections import Counter

k = SEED % 3
table = {0: (100, 8000), 1: (200, 10000), 2: (300, 12000)}
MAX_LEN, VOCAB_SIZE = table[k]
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('SEED % 3 =', k, '| MAX_LEN =', MAX_LEN, '| VOCAB_SIZE =', VOCAB_SIZE, '| device =', device)


from datasets import load_dataset
ds = load_dataset('imdb')
for part in ['train', 'test']:
    lab = np.array(ds[part]['label'])
    print(f'{part}: {len(lab)} пікір | теріс(0): {(lab==0).sum()} | оң(1): {(lab==1).sum()}')


rng = np.random.RandomState(SEED)
tr_idx = rng.choice(len(ds['train']), 6000, replace=False)
te_idx = rng.choice(len(ds['test']), 2000, replace=False)
all_tr_text, all_tr_lab = ds['train']['text'], ds['train']['label']
all_te_text, all_te_lab = ds['test']['text'], ds['test']['label']
train_texts  = [all_tr_text[i] for i in tr_idx]; train_labels = [all_tr_lab[i] for i in tr_idx]
test_texts   = [all_te_text[i] for i in te_idx]; test_labels  = [all_te_lab[i] for i in te_idx]
for name, lab in [('train(6000)', train_labels), ('test(2000)', test_labels)]:
    lab = np.array(lab)
    print(f'{name}: теріс {(lab==0).sum()} ({(lab==0).mean():.1%}) | оң {(lab==1).sum()} ({(lab==1).mean():.1%})')


lens_words = np.array([len(t.split()) for t in train_texts])
plt.figure(figsize=(8, 4))
plt.hist(lens_words, bins=60, color='steelblue', edgecolor='white')
plt.axvline(MAX_LEN, color='red', ls='--', label=f'MAX_LEN = {MAX_LEN}')
plt.xlabel('Пікір ұзындығы (сөз саны)'); plt.ylabel('Пікір саны')
plt.title('IMDB train (6000): пікір ұзындығының таралуы'); plt.legend(); plt.show()

median_len = np.median(lens_words)
share_over = (lens_words > MAX_LEN).mean() * 100
kept = np.mean(MAX_LEN / lens_words[lens_words > MAX_LEN])
print(f'Медиана: {median_len:.0f} сөз | орташа: {lens_words.mean():.0f} | макс: {lens_words.max()}')
print(f'MAX_LEN={MAX_LEN}-нен ұзын пікірлер: {share_over:.1f}%')
print(f'Ұзын пікірлерде сақталатын орташа үлес: {kept:.1%} (кесілетіні ≈ {1-kept:.1%})')

Wxh = np.array([[0.4, -0.2], [0.1, 0.6]]); Whh = np.array([[0.5, 0.2], [-0.3, 0.1]])
bh = np.array([0, -0.1]); h = np.zeros(2)
for t, x in enumerate([[1, 2], [-1, 0]], 1):
    h = np.tanh(Wxh @ np.array(x) + Whh @ h + bh)
    print(f'h_{t} =', np.round(h, 4))


f, i, g, o, c_prev = 0.8, 0.6, -0.5, 0.7, 1.5
c = f * c_prev + i * g; h_lstm = o * np.tanh(c)
print(f'LSTM: c_t = {c:.4f}, h_t = {h_lstm:.4f}')
z, h_prev, n = 0.6, 0.5, -0.2
print(f'GRU:  h_t = {(1 - z) * n + z * h_prev:.4f}')


d, hdim = 50, 64
for name, cls, mult in [('RNN', nn.RNN, 1), ('LSTM', nn.LSTM, 4), ('GRU', nn.GRU, 3)]:
    formula = mult * (hdim * (d + hdim) + 2 * hdim)
    real = sum(p.numel() for p in cls(d, hdim).parameters())
    print(f'{name}: формула = {formula}, PyTorch = {real}, сәйкес: {formula == real}')


print('0.85^40 =', 0.85**40, '->', f'{0.85**40:.4%}')
print('1.2^40  =', 1.2**40)


def clean(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)            
    text = re.sub(r"[^a-z0-9']+", ' ', text)         
    return text.split()

print(train_texts[0][:200]); print(clean(train_texts[0])[:25])


train_tok = [clean(t) for t in train_texts]
test_tok  = [clean(t) for t in test_texts]
cnt = Counter(w for toks in train_tok for w in toks)
PAD, UNK = 0, 1
itos = ['<pad>', '<unk>'] + [w for w, _ in cnt.most_common(VOCAB_SIZE - 2)]
stoi = {w: i for i, w in enumerate(itos)}
print('Сөздік көлемі:', len(itos))
print('Ең жиі 15 токен:', cnt.most_common(15))
test_total = sum(len(t) for t in test_tok)
test_unk = sum(1 for t in test_tok for w in t if w not in stoi)
print(f'test-те <unk> үлесі: {test_unk / test_total:.2%} ({test_unk}/{test_total})')


def encode_batch(tok_lists, max_len):
    N = len(tok_lists)
    X = torch.zeros(N, max_len, dtype=torch.long)
    lengths = torch.zeros(N, dtype=torch.long)
    for i, toks in enumerate(tok_lists):
        ids = [stoi.get(w, UNK) for w in toks][:max_len]   # ұзынын кесу
        if len(ids) == 0:                                    # бос пікір -> бір <unk>
            ids = [UNK]
        X[i, :len(ids)] = torch.tensor(ids)                  # қалғаны <pad>=0
        lengths[i] = len(ids)
    return X, lengths

X_train_all, len_train_all = encode_batch(train_tok, MAX_LEN)
X_test, len_test = encode_batch(test_tok, MAX_LEN)
y_train_all = torch.tensor(train_labels, dtype=torch.float32)
y_test = torch.tensor(test_labels, dtype=torch.float32)
print(X_train_all.shape, len_train_all.shape, '| min length:', len_train_all.min().item())


from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

idx_all = np.arange(len(train_tok))
idx_tr, idx_val = train_test_split(idx_all, test_size=0.2, random_state=SEED, stratify=np.array(train_labels))
def subset(idx):
    t = torch.as_tensor(idx)
    return X_train_all[t], len_train_all[t], y_train_all[t]

def make_loader(X, L, y, shuffle, bs=64):
    g = torch.Generator().manual_seed(SEED)
    return DataLoader(TensorDataset(X, L, y), batch_size=bs, shuffle=shuffle, generator=g if shuffle else None)

train_loader = make_loader(*subset(idx_tr), shuffle=True)
val_loader   = make_loader(*subset(idx_val), shuffle=False)
test_loader  = make_loader(X_test, len_test, y_test, shuffle=False)
print('train:', len(idx_tr), '| val:', len(idx_val), '| test:', len(y_test))

# **2.5.** Тексеру.

xb, lb, yb = next(iter(train_loader))
print('X:', xb.shape, '| lengths:', lb.shape, '| y:', yb.shape)
print('Ең үлкен нөмір:', X_train_all.max().item(), '< VOCAB_SIZE =', VOCAB_SIZE, '->', X_train_all.max().item() < VOCAB_SIZE)
def decode(ids): return [itos[i] for i in ids if i != PAD]
print('Түпнұсқа токендер :', train_tok[0][:12])
print('Decode нәтижесі   :', decode(X_train_all[0].tolist())[:12])

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

join = lambda toks: ' '.join(toks)
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000)
Xtf_tr = tfidf.fit_transform([join(train_tok[i]) for i in idx_tr])
lr_clf = LogisticRegression(max_iter=1000).fit(Xtf_tr, np.array(train_labels)[idx_tr])
Xtf_val = tfidf.transform([join(train_tok[i]) for i in idx_val])
Xtf_te = tfidf.transform([join(t) for t in test_tok])
tf_val = accuracy_score(np.array(train_labels)[idx_val], lr_clf.predict(Xtf_val))
tf_pred = lr_clf.predict(Xtf_te)
tf_acc, tf_f1 = accuracy_score(test_labels, tf_pred), f1_score(test_labels, tf_pred)
tf_params = lr_clf.coef_.size + 1
print(f'TF-IDF baseline | val acc {tf_val:.4f} | test acc {tf_acc:.4f} | test F1 {tf_f1:.4f}')


EMB = 50
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=EMB, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.RNN(emb_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x, lengths):
        out, _ = self.rnn(self.emb(x))                       # (B, T, H)
        idx = (lengths - 1).clamp(min=0).to(x.device)
        last = out[torch.arange(x.size(0), device=x.device), idx]   # соңғы нақты токен күйі
        return self.fc(last).squeeze(1)                      # логит (B,)

print(sum(p.numel() for p in RNNClassifier(VOCAB_SIZE).parameters()))


def evaluate(model, loader, return_preds=False):
    model.eval(); ps, ys = [], []
    with torch.no_grad():
        for xb, lb, yb in loader:
            logits = model(xb.to(device), lb.to(device))
            ps.append((logits > 0).long().cpu()); ys.append(yb.long())
    ps, ys = torch.cat(ps).numpy(), torch.cat(ys).numpy()
    acc, f1 = (ps == ys).mean(), f1_score(ys, ps)
    return (acc, f1, ps, ys) if return_preds else (acc, f1)

def fit(model, train_loader, val_loader=None, epochs=5, lr=3e-3, clip=1.0, verbose=True):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()
    hist = {'train_loss': [], 'val_acc': [], 'time': [], 'best_epoch': None}
    best_acc, best_state = -1, None
    for ep in range(1, epochs + 1):
        model.train(); t0 = time.time(); tot, n = 0.0, 0
        for xb, lb, yb in train_loader:
            xb, lb, yb = xb.to(device), lb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb, lb), yb)
            loss.backward()
            if clip: nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            opt.step()
            tot += loss.item() * len(yb); n += len(yb)
        if device.type == 'cuda': torch.cuda.synchronize()
        dt = time.time() - t0
        va = evaluate(model, val_loader)[0] if val_loader is not None else float('nan')
        hist['train_loss'].append(tot / n); hist['val_acc'].append(va); hist['time'].append(dt)
        if val_loader is None or va > best_acc:
            best_acc, best_state, hist['best_epoch'] = va, copy.deepcopy(model.state_dict()), ep
        if verbose:
            print(f'эпоха {ep}/{epochs} | train loss {tot/n:.4f} | val acc {va:.4f} | {dt:.1f} с')
    return hist, best_state

torch.manual_seed(SEED)
rnn_model = RNNClassifier(VOCAB_SIZE)
rnn_hist, rnn_best = fit(rnn_model, train_loader, val_loader, epochs=5)


rnn_model.load_state_dict(rnn_best)
rnn_test_acc, rnn_test_f1 = evaluate(rnn_model, test_loader)
print(f'RNN | ең жақсы эпоха {rnn_hist["best_epoch"]} | test acc {rnn_test_acc:.4f} | test F1 {rnn_test_f1:.4f}')
res = {'RNN': dict(params=sum(p.numel() for p in rnn_model.parameters()), epoch_time=np.mean(rnn_hist['time']),
                   val_acc=max(rnn_hist['val_acc']), test_acc=rnn_test_acc, test_f1=rnn_test_f1, hist=rnn_hist)}


def plot_curves(hist, title):
    ep = range(1, len(hist['train_loss']) + 1)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(ep, hist['train_loss'], 'o-', color='tab:red', label='train loss'); ax1.set_xlabel('Эпоха'); ax1.set_ylabel('Train loss', color='tab:red')
    ax2 = ax1.twinx(); ax2.plot(ep, hist['val_acc'], 's-', color='tab:blue', label='val accuracy'); ax2.set_ylabel('Val accuracy', color='tab:blue')
    plt.title(title); fig.tight_layout(); plt.show()
print(plot_curves(rnn_hist, 'RNN: оқыту қисығы'))



from torch.nn.utils.rnn import pack_padded_sequence
class SentimentModel(nn.Module):
    def __init__(self, cell, vocab_size, emb_dim=EMB, hidden=64, num_layers=1, bidirectional=False, mode='gather'):
        super().__init__()
        self.mode = 'pack' if bidirectional else mode
        self.cell_name, self.bi = cell, bidirectional
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        Cell = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[cell]
        self.rnn = Cell(emb_dim, hidden, num_layers=num_layers, batch_first=True, bidirectional=bidirectional,
                        dropout=0.2 if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden * (2 if bidirectional else 1), 1)
    def forward(self, x, lengths):
        e = self.emb(x)
        if self.mode == 'pack':
            packed = pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, hn = self.rnn(packed)
            if self.cell_name == 'LSTM': hn = hn[0]                 # (h_n, c_n) -> h_n
            last = torch.cat([hn[-2], hn[-1]], dim=1) if self.bi else hn[-1]
        else:
            out, _ = self.rnn(e)                                    # LSTM-де де: out, (h, c)
            if self.mode == 'last':
                last = out[:, -1]                                   # padding күйі! (4.3 эксперименті)
            else:
                idx = (lengths - 1).clamp(min=0).to(x.device)
                last = out[torch.arange(x.size(0), device=x.device), idx]
        return self.fc(last).squeeze(1)

def train_cell(cell, mode='gather', epochs=5):
    torch.manual_seed(SEED)                                          # RNN-мен бірдей шарттар
    m = SentimentModel(cell, VOCAB_SIZE, mode=mode)
    hist, best = fit(m, train_loader, val_loader, epochs=epochs)
    m.load_state_dict(best)
    return m, hist

for cell in ['LSTM', 'GRU']:
    print('=====', cell)
    m, hist = train_cell(cell)
    acc, f1 = evaluate(m, test_loader)
    res[cell] = dict(params=sum(p.numel() for p in m.parameters()), epoch_time=np.mean(hist['time']),
                     val_acc=max(hist['val_acc']), test_acc=acc, test_f1=f1, hist=hist)
    print(f'{cell} | test acc {acc:.4f} | test F1 {f1:.4f}')
    plot_curves(hist, f'{cell}: оқыту қисығы')

# **4.2.** Салыстыру кестесі.

rows = [dict(Модель='TF-IDF + LogReg', параметр_саны=tf_params, эпоха_уақыты_с=np.nan, val_acc=tf_val, test_acc=tf_acc, test_F1=tf_f1)]
for k_ in ['RNN', 'LSTM', 'GRU']:
    r = res[k_]
    rows.append(dict(Модель=k_, параметр_саны=r['params'], эпоха_уақыты_с=r['epoch_time'], val_acc=r['val_acc'], test_acc=r['test_acc'], test_F1=r['test_f1']))
cmp_df = pd.DataFrame(rows).set_index('Модель')
print(cmp_df.round(4))

# **4.3.** Padding эксперименті: GRU, `out[:, -1]` (padding күйі) vs соңғы нақты токен.

m_last, h_last = train_cell('GRU', mode='last')
va_real, va_last = res['GRU']['val_acc'], max(h_last['val_acc'])
print(f'GRU соңғы нақты токен: val acc = {va_real:.4f}')
print(f'GRU out[:, -1] (padding): val acc = {va_last:.4f}')
print(f'Өзгеріс: {va_last - va_real:+.4f}')

# **Қорытынды (4.3):** `out[:, -1]` қолданғанда қысқа пікірлердің соңғы күйі көптеген `<pad>` қадамынан өтеді (ұзындығы < MAX_LEN болса), сондықтан ақпарат «сұйылады» / ұмытылады; val accuracy төмендейді (нақты мәнін жоғарыдағы шығыстан қараңыз). Бұл — padding-ті ескермеудің бағасы.

# ## 5-бөлім. Ұзақ тәуелділік тәжірибесі

# **5.1.** `make_data(n, T)`

def make_data(n, T, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 10, (n, T), generator=g)
    return x, x[:, 0].clone()
x_, y_ = make_data(3, 8); print(x_, y_)

# **5.2.** Модель және `train_eval`.

CELLS = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}
class SeqModel(nn.Module):
    def __init__(self, cell, hidden=64, emb=32):
        super().__init__()
        self.emb = nn.Embedding(10, emb)
        self.rnn = CELLS[cell](emb, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 10)
    def forward(self, x, return_emb=False):
        e = self.emb(x)
        if return_emb: e.retain_grad()
        out, _ = self.rnn(e)
        logits = self.fc(out[:, -1])                         # соңғы қадам күйі
        return (logits, e) if return_emb else logits

def train_eval(cell, T, epochs=15, lr=5e-3, clip=1.0, bs=32, seed=SEED, return_hist=False):
    torch.manual_seed(seed)
    xtr, ytr = make_data(2000, T, seed=1); xte, yte = make_data(500, T, seed=2)
    model = SeqModel(cell).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr); crit = nn.CrossEntropyLoss()
    losses = []
    for ep in range(epochs):
        model.train(); perm = torch.randperm(2000); tot = 0.0
        for i in range(0, 2000, bs):
            b = perm[i:i + bs]; xb, yb = xtr[b].to(device), ytr[b].to(device)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward()
            if clip: nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step(); tot += loss.item() * len(b)
        losses.append(tot / 2000)
    model.eval()
    with torch.no_grad():
        acc = (model(xte.to(device)).argmax(1).cpu() == yte).float().mean().item()
    return (acc, losses) if return_hist else acc

print('T=10 RNN:', train_eval('RNN', 10))

# **5.3.** T = 10, 30, 60, 100 × (RNN, LSTM, GRU).

Ts = [10, 30, 60, 100]
rows = []
for cell in ['RNN', 'LSTM', 'GRU']:
    for T in Ts:
        a = train_eval(cell, T); rows.append(dict(cell=cell, T=T, acc=a)); print(cell, T, round(a, 3))
tbl = pd.DataFrame(rows).pivot(index='T', columns='cell', values='acc')[['RNN', 'LSTM', 'GRU']]
print(tbl.round(3))
plt.figure(figsize=(7, 4))
for cell in tbl.columns: plt.plot(tbl.index, tbl[cell], 'o-', label=cell)
plt.axhline(0.1, color='gray', ls='--', label='кездейсоқ деңгей (0.1)')
plt.xlabel('Тізбек ұзындығы T'); plt.ylabel('Test accuracy'); plt.title('Бірінші санды болжау: accuracy vs T'); plt.legend(); plt.show()

# **5.4.** Градиент профилі (оқытылмаған модель, T = 30).

T = 30; prof = {}
xb, yb = make_data(256, T, seed=3)
for cell in ['RNN', 'LSTM', 'GRU']:
    torch.manual_seed(SEED)
    model = SeqModel(cell).to(device)
    logits, e = model(xb.to(device), return_emb=True)
    nn.CrossEntropyLoss()(logits, yb.to(device)).backward()
    prof[cell] = e.grad.norm(dim=2).mean(0).cpu().numpy()        # (T,) әр қадам бойынша норма
plt.figure(figsize=(7, 4))
for cell, g in prof.items(): plt.semilogy(range(1, T + 1), g, 'o-', label=cell)
plt.xlabel('Уақыт қадамы t'); plt.ylabel('||dL/d emb_t|| (лог шкала)'); plt.title('Градиент профилі, T=30 (оқытылмаған)'); plt.legend(); plt.show()
ratios = {}
for cell, g in prof.items():
    ratios[cell] = g[0] / g[-1]
    print(f'{cell}: ||g_1|| / ||g_T|| = {ratios[cell]:.2e}  (g_1={g[0]:.2e}, g_T={g[-1]:.2e})')

# **5.5.** Градиентті қию: RNN, T = 30, lr = 0.05.

clip_res = {}
for clip in [None, 1.0]:
    acc, losses = train_eval('RNN', 30, lr=0.05, clip=clip, return_hist=True)
    clip_res[clip] = (acc, losses)
    print(f'clip={clip}: test acc {acc:.3f} | NaN бар ма: {any(not math.isfinite(l) for l in losses)}')
    print('  loss:', [round(l, 3) for l in losses])
plt.figure(figsize=(6, 4))
for clip, (_, l) in clip_res.items(): plt.plot(range(1, len(l) + 1), l, 'o-', label=f'clip={clip}')
plt.xlabel('Эпоха'); plt.ylabel('Train loss'); plt.title('RNN, T=30, lr=0.05'); plt.legend(); plt.show()

# Жауапқа арналған сандар жиынтығы (5.3-5.5)
print(tbl.round(3)); print({k_: f'{v:.2e}' for k_, v in ratios.items()})
for clip, (acc, l) in clip_res.items(): print('clip', clip, '-> acc', round(acc, 3), '| соңғы loss', round(l[-1], 3))


enc_cache = {}
def get_enc(max_len):
    if max_len not in enc_cache:
        enc_cache[max_len] = encode_batch(train_tok, max_len)
    return enc_cache[max_len]

def run_experiment(cell='GRU', bidirectional=False, num_layers=1, hidden=64, max_len=MAX_LEN, clip=1.0, seed=SEED,
                   epochs=3, n_train=3000, lr=3e-3):
    X, L = get_enc(max_len)
    tr = torch.as_tensor(idx_tr[:n_train]); va = torch.as_tensor(idx_val)
    g = torch.Generator().manual_seed(seed)
    tl = DataLoader(TensorDataset(X[tr], L[tr], y_train_all[tr]), batch_size=64, shuffle=True, generator=g)
    vl = DataLoader(TensorDataset(X[va], L[va], y_train_all[va]), batch_size=64)
    torch.manual_seed(seed)
    model = SentimentModel(cell, VOCAB_SIZE, hidden=hidden, num_layers=num_layers, bidirectional=bidirectional, mode='pack')
    t0 = time.time()
    hist, _ = fit(model, tl, vl, epochs=epochs, lr=lr, clip=clip, verbose=False)
    return dict(val_acc=hist['val_acc'][-1], params=sum(p.numel() for p in model.parameters()), time=time.time() - t0)

print(run_experiment())

# **6.2.** Барлық 7 фактор (F1–F7) бойынша тәжірибелер.

base = dict(cell='GRU', bidirectional=False, num_layers=1, hidden=64, max_len=MAX_LEN, clip=1.0, seed=SEED)
factors = {
    'F1. Ұяшық түрі':   ('cell', ['RNN', 'LSTM', 'GRU']),
    'F2. Бағыт':        ('bidirectional', [False, True]),
    'F3. Қабат саны':   ('num_layers', [1, 2]),
    'F4. Hidden':       ('hidden', [32, 64, 128]),
    'F5. MAX_LEN':      ('max_len', [50, 100, 200]),
    'F6. Градиент қию': ('clip', [None, 1.0]),
    'F7. Seed':         ('seed', [SEED, SEED + 1, SEED + 2]),
}
run_cache, rows = {}, []
for fname, (key, values) in factors.items():
    for v in values:
        cfg = {**base, key: v}; ck = tuple(sorted(cfg.items(), key=lambda kv: kv[0]))
        if ck not in run_cache: run_cache[ck] = run_experiment(**cfg)
        r = run_cache[ck]
        rows.append(dict(factor=fname, setting=str(v), val_acc=r['val_acc'], params=r['params'], time=r['time'], config=cfg))
        print(f'{fname:18s} {str(v):>6s} | val acc {r["val_acc"]:.4f} | params {r["params"]} | {r["time"]:.0f} с')
results = pd.DataFrame(rows)
print(results.drop(columns='config').round(4))

# **6.3.** Графиктер және әсер шамасы.

fig, axes = plt.subplots(2, 4, figsize=(18, 7)); axes = axes.ravel()
for ax, (fname, _) in zip(axes, factors.items()):
    sub = results[results.factor == fname]
    if fname.startswith('F7'):
        ax.errorbar([0], [sub.val_acc.mean()], yerr=[sub.val_acc.std()], fmt='o', capsize=6, color='tab:red')
        ax.scatter([0] * len(sub), sub.val_acc, color='gray', alpha=.6, label='жеке seed-тер')
        ax.set_xticks([0]); ax.set_xticklabels([f'орташа ± std\n({sub.val_acc.mean():.3f} ± {sub.val_acc.std():.3f})']); ax.legend()
    else:
        ax.plot(sub.setting, sub.val_acc, 'o-')
    ax.set_title(fname); ax.set_xlabel('Мән'); ax.set_ylabel('Val accuracy')
axes[-1].axis('off'); plt.tight_layout(); plt.show()

effect = results.groupby('factor').val_acc.agg(lambda s: s.max() - s.min()).sort_values(ascending=False)
print(effect.round(4))
plt.figure(figsize=(8, 4)); plt.barh(effect.index[::-1], effect.values[::-1], color='teal')
plt.xlabel('Әсер шамасы: max − min val accuracy'); plt.title('Факторлардың әсері (IMDB, val)'); plt.tight_layout(); plt.show()

# **Қорытынды (6.3):** факторларды әсер шамасы бойынша реттеген диаграммадан қараңыз. Маңызды ескерту: F7 (seed) әсері — «шу» деңгейі. Егер басқа фактордың әсері F7-ден айтарлықтай үлкен болмаса, ол айырмашылықты сенімді деп санауға болмайды (бір ғана seed-пен өлшенген).

# **6.4.** Ең жақсы конфигурацияны толық train бөлігінде (6000) үйретіп, test-те бір рет бағалау (5 эпоха; val жоқ болғандықтан соңғы эпоха алынады).

best_row = results.sort_values('val_acc').iloc[-1]; cfg = best_row.config
print('Ең жақсы конфигурация:', cfg, '| val acc (6.2):', round(best_row.val_acc, 4))
X6, L6 = get_enc(cfg['max_len']); Xte6, Lte6 = encode_batch(test_tok, cfg['max_len'])
g = torch.Generator().manual_seed(cfg['seed'])
tl6 = DataLoader(TensorDataset(X6, L6, y_train_all), batch_size=64, shuffle=True, generator=g)
te6 = DataLoader(TensorDataset(Xte6, Lte6, y_test), batch_size=64)
torch.manual_seed(cfg['seed'])
best_model = SentimentModel(cfg['cell'], VOCAB_SIZE, hidden=cfg['hidden'], num_layers=cfg['num_layers'],
                            bidirectional=cfg['bidirectional'], mode='pack')
fit(best_model, tl6, None, epochs=5, lr=3e-3, clip=cfg['clip'])
best_acc, best_f1, best_pred, best_y = evaluate(best_model, te6, return_preds=True)     # test — бір рет

tf6 = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000)
A = tf6.fit_transform([join(t) for t in train_tok]); clf6 = LogisticRegression(max_iter=1000).fit(A, train_labels)
p6 = clf6.predict(tf6.transform([join(t) for t in test_tok]))
cmp6 = pd.DataFrame({'test_acc': [accuracy_score(test_labels, p6), best_acc], 'test_F1': [f1_score(test_labels, p6), best_f1]},
                    index=['TF-IDF + LogReg (6000)', f"{cfg['cell']} (bi={cfg['bidirectional']}, L={cfg['num_layers']}, h={cfg['hidden']}, T={cfg['max_len']})"])
print(cmp6.round(4))


wrong = np.where(best_pred != best_y)[0]
print(f'Барлық қате: {len(wrong)} / {len(best_y)}')
for i in wrong[:10]:
    print(f'[шын: {"оң" if best_y[i] else "теріс"} | болжам: {"оң" if best_pred[i] else "теріс"} | ұзындық: {len(test_tok[i])} сөз]')
    print('   ', test_texts[i][:300].replace('<br />', ' '), '...\n')


err_df = pd.DataFrame({'len': [len(t) for t in test_tok], 'err': (best_pred != best_y).astype(int)})
err_df['quartile'] = pd.qcut(err_df['len'], 4, labels=['Q1 (қысқа)', 'Q2', 'Q3', 'Q4 (ұзын)'])
q = err_df.groupby('quartile', observed=True).agg(пікір_саны=('err', 'size'), орташа_ұзындық=('len', 'mean'), қате_үлесі=('err', 'mean'))
print(q.round(3))
q['қате_үлесі'].plot(kind='bar', color='indianred', rot=0, title='Қате үлесі: ұзындық квартильдері'); plt.ylabel('Қате үлесі'); plt.show()

# **Қорытынды (7):** қате үлесі квартильдер бойынша қалай өзгергенін 7.2 кестесінен жазыңыз. Әдетте ең ұзын топта (Q4) қате көбейеді, себебі MAX_LEN-нен кейінгі мәтін (соның ішінде қорытынды баға) кесіледі; қысқа пікірлерде ақпарат аз, бірақ кесу жоқ. Қателердің көпшілігі — сарказм, терістеу («not bad»), аралас бағалар.
