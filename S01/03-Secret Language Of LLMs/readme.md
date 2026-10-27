# 🔐 Secret Language of LLMs — Tokens, Tokenization & Context Window (Hinglish)

> **"LLM kabhi text nahi dekhta — wo sirf numbers dekta hai. Us number-bhasha ko bolna/likhna hi uski SECRET LANGUAGE hai."** 🔢🧠

Namaste dosto! 👋 Ye repo meri **"LLMs andar se kaise padhte hain?"** wali learning journey hai. Jab hum ChatGPT ko likhte hain *"Namaste dosto"* — humein lagta hai usne hamara **shabd** samjha. Sach ye hai: usne na koi word dekha, na koi sentence. Text pehle **tokens** mein tootta hai, phir har token ek **number (token ID)** ban jaata hai, aur sirf wo numbers model ke andar jaate hain. 🤯

Ye isi **secret language** ka decode hai — **tokens, tokenization, vocabulary, BPE, WordPiece, Unigram, fertility aur context window** — sab Hinglish mein, diagrams aur runnable code ke saath. Jaise koi senior aapko baitha ke samjhaye. 🚀

---

## 📚 Table of Contents

1. [Intro — Ye Repo Kya Hai? (Deep Knowledge)](#-intro--ye-repo-kya-hai-deep-knowledge)
2. [Tokens — LLM Ke Asli "Shabd"](#-1-tokens--llm-ke-asli-shabd)
3. [Subword Tokenization — Shabd Ko Chhote Tukdon Mein](#-2-subword-tokenization--shabd-ko-chhote-tukdon-mein)
4. [Vocabulary & Token ID — Bhasha Ka Number System](#-3-vocabulary--token-id--bhasha-ka-number-system)
5. [Byte-Pair Encoding (BPE)](#-4-byte-pair-encoding-bpe--frequency-wala-jo-merge-kare)
6. [WordPiece](#-5-wordpiece--bpe-ka-cousin-jo-probability-dekhta-hai)
7. [Unigram](#-6-unigram--ulti-se-shuru-hone-wala-smart-algorithm)
8. [Tokenization Fertility — Kis Bhasha Mein Kitne Tokens?](#-7-tokenization-fertility--kis-bhasha-mein-kitne-tokens)
9. [Context Window — Model Ki Chamakti Hui Yaadadaasht](#-8-context-window--model-ki-chamakti-hui-yaadadaasht)
10. [Diagram Explanations — Sab Kuch Ek Nazar Mein](#-9-diagram-explanations--sab-kuch-ek-nazar-mein)
11. [Real-Life Usages — Ye Sab Kahan Kaam Aata Hai](#-10-real-life-usages--ye-sab-kahan-kaam-aata-hai)
12. [Bonus: Tokenization Tips & Tricks](#-11-bonus-tokenization-tips--tricks)
13. [Resources — Padhne Ke Liye](#-12-resources--padhne-ke-liye)
14. [Contributing & Star](#-13-contributing--star)

---

## 🌟 Intro — Ye Repo Kya Hai? (Deep Knowledge)

### Pehle ye samjho: text model tak pahunchta kaise hai?

Koi bhi LLM **shabdon ko nahi**, sirf **numbers ko** padhta hai. Isiliye har LLM ke aage ek **tokenizer** baitha hota hai — wo ek **translator** hai jo human language ⇄ machine language ka kaam karta hai:

```
   AAPKA TEXT                    TOKENIZER (secret language ka gatekeeper)
 ┌─────────────────┐            ┌──────────────────────────────────────┐
 │ "Namaste dosto" │  ───────►  │  Step 1: split  → ["Nam","aste","   │
 │                 │            │            "d","osto"]               │
 └─────────────────┘            │  Step 2: lookup → [41288, 666, 281,  │
                                │                  9803]   (IDs)        │
                                └───────────────┬──────────────────────┘
                                                │ sirf ye numbers aage jaate hain
                                                ▼
                                ┌──────────────────────────────────────┐
                                │  EMBEDDING LAYER                     │
                                │  har ID → ek 4096-dim ka vector 📐   │
                                │  41288 → [0.02, -0.11, 0.77, ...]    │
                                └───────────────┬──────────────────────┘
                                                ▼
                                ┌──────────────────────────────────────┐
                                │  TRANSFORMER (attention ka jaadu)     │
                                │  → agla token predict 🎯             │
                                └──────────────────────────────────────┘
```

> 🔑 **Deep Knowledge:** Model ke liye `"Namaste"` naam ka koi shabd exist hi nahi karta — sirf **integer 41288** exist karta hai, aur uska **embedding vector** jo training ke dauran uski jagah (context) ke hisaab se shape leta rehta hai. Isiliye *"mere baare mein kya jaante ho"* poochhne ka matlab hai *"mere token IDs ke activations kya the"* poochhna. Ye **secret language ka pehla raaz** hai! 🤫

### Tokenization itna important kyun hai? (3 gehri wajahen)

```
┌────────────────────────────────────────────────────────────────────────┐
│ 1. COST       → APIs per TOKEN bill karti hain ($/1M tokens).          │
│                  Kam fertility = kam paisa. 💸                          │
│ 2. QUALITY    → Agar sahi tukde nahi bane, toh meaning hi bigad jaati  │
│                  hai — model ne kabhi "Baharatgarh" jaisa word dekha    │
│                  hi nahi hoga. ❌                                       │
│ 3. MEMORY     → Context window TOKENS mein measure hota hai.           │
│                  1 token ghalat = ek poora line ki jagah kha gaya. 🪟   │
└────────────────────────────────────────────────────────────────────────┘
```

**Isi liye:** *"Which model?"* se pehle aapko *"Which tokenizer?"* bhi jaanna chahiye — GPT, LLaMA, BERT, Mistral — sabka tokenizer alag hai, aur **alag tokenizer = alag worldview**. 🔭

**Prerequisites:** JavaScript (Node.js ≥ 18) basics aur thoda curiosity 🔥

```bash
# Repo clone karo aur shuru ho jao
git clone https://github.com/neerajgir/Namaste-AI.git
cd Namaste-AI

# Node v18+ chahiye; tokenizer experiments ke liye deps
node --version
npm init -y
npm pkg set type=module
npm install @huggingface/transformers
```

## 🔢 1. Tokens — LLM Ke Asli "Shabd"

**Token = text ka ek chhota tukda jo tokenizer ne apni vocabulary se chuna.** Ye character bhi ho sakta hai, poora shabd bhi, ya beech ka kuch bhi.

```
 Hum sochte hain:     "Tokenization ek shabd hai"    (5 words)
 Model dekhta hai:    [To, ken, ization, ek, sh, eb, d]  (7 tokens!)
                      └──┬──┘ └───┬────┘      └─┬─┘ └──┬──┘
                       alag alag pieces, har ek ka apna ID
```

**Rule of thumb (English ke liye):**

| Metric | GPT-3/4 ka anumaan |
|---|---|
| 1 token ≈ | **~4 characters** ya **~¾ word** (0.75 word) |
| 1 word ≈ | **~1.3 tokens** |
| 1 page (A4) ≈ | ~500 tokens |
| 1500 words ≈ | ~2000 tokens |

Code se dekhte hain:

```js
// token-vs-word.js — dekho kitne tukde bane
import { AutoTokenizer } from "@huggingface/transformers";

const tokenizer = await AutoTokenizer.from_pretrained("Xenova/gpt-4");

const text = "Namaste dosto, LLM ka asli swagat tokens mein hota hai!";
const { input_ids } = await tokenizer(text);

console.log("Tokens:", input_ids.tolist()[0].length);   // ~11 tokens
console.log("Words :", text.split(/\s+/).length);        // 8 words
console.log("IDs   :", input_ids.tolist()[0]);
// → tokens > words, kyunki bade shabd toot jaate hain 😅
```

> 🔑 **Deep Knowledge:** Token ki length **shabd nahi, FREQUENCY** decide karti hai. Jo tukde baar-baar corpus mein milte hain (like `"ing"`, `"the"`, `"ation"`) unhe tokenizer alag token bana deta hai, aur rare words chhote-chhote pieces mein tootte hain. **Ye ek compression problem hai — jo cheez baar-baar dikhe, wo lambi mile.** 🗜️

---

## ✂️ 2. Subword Tokenization — Shabd Ko Chhote Tukdon Mein

Pure word-level ya character-level dono extremes hain:

```
┌──────────────────────────────────────────────────────────────────────┐
│   EXTREME 1: CHARACTER         EXTREME 2: WORD          SWEET SPOT   │
│   ["c","a","t"]                ["cat"]                  ["cat"]      │
│   ✅ Har word mil jaayega      ✅ Meaning intact         ✅ Best of   │
│   ❌ Bahut saare tokens        ❌ OOV problem 😱         both worlds │
│      (= cost + context)        ❌ Vocab bahut bada                     │
│   Vocab: ~256                   Vocab: crores!            Vocab: 50K  │
│                                              SUBWORD TOKENIZATION ◄──┘
└──────────────────────────────────────────────────────────────────────┘
```

**Subword = common shabd poora, rare shabd ke tukde.** Isse **OOV (Out Of Vocabulary) problem** khatam ho jaati hai — jo shabd model ne training mein kabhi na dekha ho, wo uske chhote tukdon se ban jaayega.

```
 Training mein mila:  "unhappiness"  → ["un","happi","ness"]  ✅ samajh aa gaya
 Kabhi na suna shabd:  "unhappinessification" (rare!)
                       → ["un","happi","ness","ifi","cation"]  phir bhi kaam kiya!
```

> 🔑 **Deep Knowledge:** Subword ka asli faayda *fallback* nahi — **generalization** hai. "Happi" ko model ne lakho jagah dekha hai ("happy", "happiness", "happily"), toh uska **meaning embedding mein** baith gaya hai. Naye shabd mein yahi piece aaye, toh model **anumaan laga sakta hai** — bilkul LEGO bricks jaisa! 🧱

---

## 🗂️ 3. Vocabulary & Token ID — Bhasha Ka Number System

Tokenizer ka **vocabulary** ek dictionary hai — **token string ↔ integer ID**. Typical sizes:

| Model | Vocab Size | Algorithm |
|---|---|---|
| BERT | 30,522 | WordPiece |
| GPT-2 | 50,257 | BPE (byte-level) |
| GPT-3/4 | ~100K | tiktoken BPE |
| LLaMA 3 | 128,256 | BPE (tiktoken-style) |
| Mistral | 32,000 | BPE (SentencePiece) |
| T5 | 32,128 | Unigram |

```js
// vocab-lookup.js — shabd → ID ka safar
import { AutoTokenizer } from "@huggingface/transformers";

const tokenizer = await AutoTokenizer.from_pretrained("Xenova/bert-base-uncased");

const tokens = ["the", "cat", "sat", "on", "##mat"];   // WordPiece ka "##" prefix
const ids = tokens.map((t) => tokenizer.token_to_id(t));
console.log(ids);   // har token ka apna unique number 🪪

// ID → token wapas bhi mumkin (model ki output samajhne ke liye zaroori)
console.log(ids.map((id) => tokenizer.decode([id])));
```

```
  "the"  ──► ID 1996 ──┐
  "cat"  ──► ID 4937  ─┼──► Embedding matrix lookup ──► real-valued vectors
  "sat"  ──► ID 3948  ─┘    (vocab_size × hidden_dim)
                              e.g. 50257 × 4096 ≈ 206 MILLION params
                              — sirf embedding layer me! 🤯
```

> 🔑 **Deep Knowledge:** **Embedding matrix ka row-count = vocab size.** Isiliye vocab 128K karne ka matlab hai **128K × hidden_dim sirf is lookup table mein** — itna costly hai ki har model vocab ko carefully balance karta hai: **badi vocab = kam tokens per sentence (context bachta hai) par zyada parameters.** Ye ek economics ka game hai, na ki sirf linguistics ka. ⚖️

## 🔁 4. Byte-Pair Encoding (BPE) — Frequency Wala Jo Merge Kare

**BPE (1994 — Sennrich et al. ne 2016 mein NLP ke liye popular kiya)** ka idea bahut simple hai: **corpus ko characters se shuru karo, aur sabse frequent PAIR ko tab tak merge karo jab tak desired vocab size na ban jaaye.** GPT-2/3/4, LLaMA, Mistral — sab BPE use karte hain! ⭐

```
 STEP 0: Har character = ek token (byte-level toh 256 base tokens)
   corpus:  "low lower lowest lowest lowest"
   counts :  l o w _ l o w e r _ l o w e s t _ l o w e s t _ l o w e s t

 STEP 1: Sabse frequent pair = ("l","o") ×5  → merge → token "lo"
 STEP 2: Frequent pair = ("lo","w") ×5       → merge → token "low"
 STEP 3: Frequent pair = ("e","s")...        → merge → "es"
 ...            (repeat 50,000 baar for GPT-2)

 RESULT:
   "lowest" → ["low", "est"]     ✨ ek shabd = 2 tokens
```

**From-scratch mini BPE trainer (JS) — asli cheez ka chhota avtar:**

```js
// mini-bpe.js — 20 lines mein BPE ka dil
function getPairs(word) {
  const pairs = new Set();
  for (let i = 0; i < word.length - 1; i++) pairs.add(`${word[i]}|${word[i + 1]}`);
  return pairs;
}

function trainBPE(corpus, merges) {
  // shuruwat: har word = list of chars, last symbol = </w>
  let words = [...new Set(corpus.split(/\s+/))].map((w) => [...w, "</w>"]);
  const vocab = new Set(words.flat());
  const log = [];

  for (let i = 0; i < merges; i++) {
    const freq = {};
    for (const w of words)
      for (const p of getPairs(w)) freq[p] = (freq[p] || 0) + 1;

    const best = Object.entries(freq).sort((a, b) => b[1] - a[1])[0];
    if (!best) break;
    const [a, b] = best[0].split("|");

    // har jagah is pair ko ek naye token se replace karo
    words = words.map((w) => {
      const out = [];
      for (let j = 0; j < w.length; j++) {
        if (w[j] === a && w[j + 1] === b) { out.push(a + b); j++; }
        else out.push(w[j]);
      }
      return out;
    });
    vocab.add(a + b);
    log.push(`merge #${i + 1}: ("${a}","${b}") ×${best[1]} → "${a + b}"`);
  }
  return { vocab, log };
}

const { log } = trainBPE("low lower lowest lowest lowest", 3);
console.log(log.join("\n"));
// merge #1: ("l","o") ×5 → "lo"
// merge #2: ("e","r") ×1 ... (frequency ka khel 🔮)
```

**GPT-2 ka byte-level twist (asli game changer):**

```
 Purana BPE:  shuruwat WORD-level ──► rare word = <UNK> ❌ (OOV wapas!)
 GPT-2 BPE:   shuruwat BYTE-level ──► 256 base tokens, koi byte UNK nahi
              "❤️" bhi 4 bytes hain → kabhi UNK nahi hoga ✅
              (par haan, kabhi kabhi 4-5 tokens kha jaata hai 😬 — fertility!)
```

> 🔑 **Deep Knowledge:** BPE **greedy** hai — har step pe sirf aaj ki frequency dekhta hai, **kal ki possibility nahi.** Isiliye kuch merges baad mein "pachtava" wale hote hain. Doosra raaz: **token boundaries meaning ke boundary nahi hote** — `"unbelievable"` → `["un","bel","iev","able"]` mein `"iev"` ka koi meaning nahi, bas **data mein frequency hai.** Model ko grammar ya morphology ka "rule" kabhi nahi milta — **wo sirf frequency ke pattern seekhta hai.** 🔮

---

## 📎 5. WordPiece — BPE Ka Cousin Jo Probability Dekhta Hai

**BERT, DistilBERT, Electra** ka tokenizer. Dikhne mein BPE jaisa, par **merge ka criteria alag**:

| | BPE | WordPiece |
|---|---|---|
| Merge rule | **Sabse zyada frequency** wala pair | Wo pair jo **language-model likelihood** sabse zyada badhaye |
| Logic | "Kitni baar aaya?" | "Is merge se overall cost kitna ghati hai?" |
| Prefix mark | `Ġ` (space, GPT-2) | `##` (continuation) |
| Users | GPT-2/3/4, LLaMA, Mistral | BERT family |

```
 BPE sochta hai:      "in" + "play" → "inplay"  (agar frequency boli toh ho jaayega)
 WordPiece sochta hai: kya "inplay" banane se UNLIKELIHOOD kam hogi?
                       nahi? → reject ❌   (probability ka judge hai ye ⚖️)

 BERT tokens:  ["the", "play", "##ing", "##s"]
                 │      │        └── ## = "ye continuation hai, shabd ke andar"
                 │      └── apne aap mein bhi ek word ho sakta hai
                 └── shabd ki shuruaat (space ke baad)
```

> 🔑 **Deep Knowledge:** WordPiece ka formula `score = log P(new_word) − Σ log P(old_pieces)` hai — matlab **merge tabhi karo jab naya piece purane pieces se zyada probable ho.** Result: BPE ke muqable **zyada rare/meaningful pieces** bante hain, jo NLP tasks (NER, POS tagging) ke liye behtar hain. Par **speed BPE se thodi kam** — isi liye BERT ke baad industry BPE/Unigram ki taraf chali gayi. 🧭

## 🎲 6. Unigram — Ulti Se Shuru Hone Wala Smart Algorithm

**Unigram (SentencePiece, Kudo 2018)** ka sochne ka tarika **ulta** hai:

```
 BPE / WordPiece:   chhota vocab ──merge──► bada vocab   (BOTTOM-UP ➕)
 UNIGRAM:            BADA vocab (10K–1M) ──prune──► chhota vocab (TOP-DOWN ➖)

 ┌───────────────────────────────────────────────────────────────┐
 │ 1. Shuru mein: har possible subword ko rakh lo (bahut bada!)   │
 │ 2. Har piece ki probability nikalo (EM algorithm se)           │
 │ 3. Sabse KAM zaroori pieces hata do                            │
 │ 4. Naye corpus pe dobara score karo → repeat                  │
 │ 5. Jab desired vocab size bach jaaye → STOP ✋                 │
 └───────────────────────────────────────────────────────────────┘
```

**Sabse bada faayda — Subword Regularization (tokenization ka randomness = feature!):**

```
 Sentence: "the greyhound"
 Fixed BPE tokenization:   ["the", "grey", "hound"]     → hamesha yahi ❌
 Unigram (sampling):       p1: ["the", "grey", "hound"]  60%
                           p2: ["the", "gre", "y", "hound"] 30%  🎲
                           p3: ["the", "greyh", "ound"]   10%  🎲
 → Training ke dauran har epoch mein ALAG split = model zyada robust ✅
```

**Kaun use karta hai:** T5, ALBERT, XLNet, mBART, mT5, CamemBERT — yaani **multilingual aur research models** mein Unigram kaafi popular. SentencePiece library ka **default bhi Unigram** hai.

```js
// fertility-vs-bpe.js (concept demo) — Unigram style: ek word ke kitne splits?
const splits = {
  grehound: [
    ["grey", "hound", 0.6],
    ["gre", "y", "hound", 0.3],
    ["greyh", "ound", 0.1],
  ],
};

// training mein: probability ke hisaab se EK sample chuno (subword regularization)
function sampleSplit(options) {
  const r = Math.random();
  let acc = 0;
  for (const [toks, p] of options) {
    acc += p;
    if (r <= acc) return toks;
  }
  return options[0][0];
}
console.log("Aaj ka split:", sampleSplit(splits.grehound));
```

> 🔑 **Deep Knowledge:** Unigram ka EM (Expectation-Maximization) step **har piece ka "kitna contribute kiya"** ka posterior nikalta hai — matlab ye **frequency nahi, utility** dekhta hai (WordPiece ki tarah) — par **pruning** ki wajah se **exploration** bhi rakhta hai. Isiliye multilingual models (mT5 jaise jo 100+ bhashaen cover karte hain) Unigram choose karte hain: **random splits se rare languages ke liye behtar representations bante hain.** 🌍

---

## 🌾 7. Tokenization Fertility — Kis Bhasha Mein Kitne Tokens?

**Fertility = ek shabd (ya ek unit meaning) ko todne mein kitne tokens lage.** Jitna kam, utna sasta aur efficient.

```
 English:   "water"        → 1 token     🌾 fertility ≈ 1.0
 Hindi:     "पानी" (paani) → 5-6 tokens  🌾🌾🌾🌾🌾 fertility ≈ 5-6!
            ("नमस्ते" → 8+ tokens 😱)

 GPT-4 (o200k) example:
   "Namaste"     → ["Nam","aste"]           2 tokens / 1 word  = 1.0
   "Tokenization"→ ["Token","ization"]      2 tokens / 1 word  = 1.0
   "पानी"        → byte-fallback pieces      5+ tokens / 1 word = 5.0 ❌
```

**Fertility meter (chala ke dekho):**

```js
// fertility.js — apni language ki fertility measure karo
import { AutoTokenizer } from "@huggingface/transformers";

const tokenizer = await AutoTokenizer.from_pretrained("Xenova/gpt-4");

const samples = [
  ["English", "The quick brown fox jumps over the lazy dog"],
  ["Hinglish", "Yaar LLM ko tokens mein padhna hota hai, word mein nahi"],
  ["Hindi", "नमस्ते दोस्तों, टोकनाइज़ेशन बहुत महत्वपूर्ण है"],
  ["Emoji", "🔥🚀🎉🫠🧿✨🫡😂😅"],
];

let corpusTokens = 0, corpusWords = 0;
for (const [lang, text] of samples) {
  const { input_ids } = await tokenizer(text);
  const nTok = input_ids.tolist()[0].length;
  const nWord = text.split(/\s+/).length;
  corpusTokens += nTok; corpusWords += nWord;
  console.log(`${lang.padEnd(9)} | ${nTok} tokens | ${nWord} words | fertility=${(nTok / nWord).toFixed(2)}`);
}
console.log(`\nOverall fertility = ${(corpusTokens / corpusWords).toFixed(2)} tokens/word`);
// Hindi/Emoji line fertility 3-8 tak ja sakti hai — bill 3-8 guna zyada! 💸
```

**Fertility kyun matter karti hai — 3 seedhi wajahen:**

```
┌─────────────────────────────────────────────────────────────────┐
│ 💸 COST       1M tokens = $X → 2x fertility = 2x bill           │
│ 🪟 CONTEXT    128K window mein Hindi text jaldi bhar jaayegi      │
│ 📉 QUALITY    Har char = alag token → model ko "meaning" ke      │
│                bajay "bytes" ka pattern dekhna padta hai 😞       │
└─────────────────────────────────────────────────────────────────┘
```

> 🔑 **Deep Knowledge:** **2024–25 mein sabse bada tokenizer research topic yahi hai.** CIKM 2023 ka paper *"Tokenization is more than compression"* ne dikhaya ki multilingual models ki kharab fertility **seedha task performance** karti hai. Isiliye naye models (LLaMA 3, Command R, Aya) ne **multilingual data ko tokenizer training mein zyada diya** — aur Google ka **ByT5** jaise models **poora tokenizer hi hata kar bytes** par jaate hain (ek extreme!). **AI equity ka sawal bhi yahi hai: agar aapki bhasha mehngi hai, toh AI aapke liye sasta kyun ho?** 🌏⚖️

## 🪟 8. Context Window — Model Ki Chamakti Hui Yaadadaasht

**Context window = model ek baar mein kitne TOKENS dekh sakta hai** (apni input + output, mila kar). Ye uski "chamakti hui yaadadaasht" hai — isse bahar ka text uske liye **andhera** hai. 🌑

```
 Model ka view (context = 8192 tokens):

 [ SYSTEM PROMPT ][ Aapka purana chat history ][ Aapka naya sawal ]│
 ◄────────────── visible (attention yahan lagti hai) ──────────────►│
                                                                   ▼
                                                         ISKE AAGE = ANDHERA 🌑
                                                         (purani baat bhool gaya)
```

**Timeline — window kaise badi hoti gayi:**

| Model / Year | Context Window | ≈ Kitne shabd |
|---|---|---|
| GPT-2 (2019) | 2,048 | ~1,500 words |
| GPT-3.5 | 4K – 16K | ~12K words |
| GPT-4 (2023) | 8K – 128K | ~96K words |
| Claude 3.5 | 200K | ~150K words |
| Gemini 1.5/2.x | 1M – 2M+ | ~750K+ words (puri books!) |
| LLaMA 3.1 | 128K | ~96K words |

**Context badhane ke tarike (deep dive diagram):**

```
 1. Positional Encoding upgrade:
    sinusoidal (GPT-2) → learned → ALiBi → RoPE (Rotary) 🌀
    RoPE = position ko frequency domain mein ghumao → extrapolation behtar

 2. Attention complexity:  O(n²)  ⚠️
       n=4K   → 16M pairs      (aasan)
       n=1M   → 1 TRILLION pairs (GPU ro dega 😭)
    → FlashAttention ⚡ (memory mein smart), KV-Cache tricks 🪶

 3. Context Extension tricks:
    RoPE scaling (π/4 trick), YaRN, LongRoPE 📈
    → bina dobara training kiye window badh jaati hai

 4. Retrieval (RAG) 📚
    → window badhane ke bajay bahar ka relevant text ANDE se laao
```

```js
// context-window.js — window overflow ka demo
import { AutoTokenizer } from "@huggingface/transformers";

const tokenizer = await AutoTokenizer.from_pretrained("Xenova/gpt-4");
const MAX = 8192;   // model ka window (example)

const bigText = "Namaste! ".repeat(3000);   // ~27K tokens ka text 😅
const { input_ids } = await tokenizer(bigText);
const n = input_ids.tolist()[0].length;

console.log(`Tokens: ${n} | Window: ${MAX}`);
if (n > MAX) {
  console.log(`⛔ OVERFLOW — ${n - MAX} tokens bahar fekne padenge`);
  console.log("API yahan ya TRUNCATE karti hai ya ERROR deti hai — dono hi dukh dene wale hain 😢");
}

// Asli duniya mein isko SMARTLY handle karo:
// 1. Chunking (RAG) — 500-token ke tukde mein toda, sirf relevant chunk bheja 📚
// 2. Summarization — purani history ka summary + latest turn 🗜️
// 3. Sliding window — hamesha recent N tokens rakhe 🪟
```

> 🔑 **Deep Knowledge:** Window **badi hone se problem solve nahi hoti — "Lost in the Middle" 😶.** 2023 ka famous paper ne dikhaya: model ko agar answer document ke **beech** mein do, toh wo use **bhool jaata hai** — chahe 128K window ho! Attention beginning aur end ko zyada "dekti" hai (primacy + recency bias). Isiliye **RAG + prompt structure** aaj bhi zaroori hai — **zyada yaadadaasht ≠ zyada samajh.** Aur sabse gehra raaz: **position embeddings ke wajah se window "extrapolate" nahi hoti** — 100K pe train model 128K pe degrade ho sakta hai (needle test 🪡 se hi pata chalta hai!). 🧪

---

## 🗺️ 9. Diagram Explanations — Sab Kuch Ek Nazar Mein

**Diagram 1: Text ka poora safar (pipeline)**

```
 "Namaste dosto!" ──► 🔤 NORMALIZE (Unicode NFC, lowercasing…)
        ──► ✂️ PRE-TOKENIZE (space/punctuation se alag)
        ──► 🔁 SUBWORD (BPE/WordPiece/Unigram)
        ──► 🔢 TOKEN IDs [41288, 281, 0]
        ──► 📐 EMBEDDINGS (IDs → vectors)
        ──► 🧠 TRANSFORMER LAYERS (attention ×N)
        ──► 🎯 NEXT-TOKEN PROBS ──loop──► (sampling) ──► aapka jawab 💬
```

**Diagram 2: BPE vs WordPiece vs Unigram — ek nazar mein**

```
┌──────────────┬──────────────────┬──────────────────┬──────────────────┐
│              │      BPE         │    WordPiece     │    Unigram       │
├──────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Direction    │ Chhota → Bada ➕ │ Chhota → Bada ➕ │ Bada → Chhota ➖ │
│ Criteria     │ Frequency        │ Log-likelihood   │ Prune by loss    │
│ Determinism  │ Hamesha ek split │ Hamesha ek split │ Sampling possible│
│ Special mark │ Ġ (GPT-2)        │ ## (BERT)        │ ▁ (SentencePiece)│
│ Speed        │ ⚡⚡⚡ fast        │ ⚡⚡ thoda slow    │ ⚡⚡ fast (Viterbi)│
│ Users        │ GPT, LLaMA       │ BERT, Electra    │ T5, mT5, XLNet   │
└──────────────┴──────────────────┴──────────────────┴──────────────────┘
```

**Diagram 3: Fertility ka asar (bill + context dono)**

```
 Same meaning, alag cost:
 "I love you"        →  4 tokens  ████ 💸
 "मैं तुमसे प्यार करता हूँ" → 18 tokens ██████████████████ 💸💸💸
                                  ▲
                    4.5x zyada tokens = 4.5x zyada bill
                    aur context ka 4.5x zyada hissa kha gaya!
```

**Diagram 4: Context window + attention ka khel**

```
 Prompt: [P1 P2 P3 ... Pn]
            │   │   │        │
            ▼   ▼   ▼        ▼
        ┌─────────────────────────┐
        │   SELF-ATTENTION 🎯      │  har token sabko "dekhta" hai
        │   sab se judte hain O(n²)│  → isiliye lambi window = zyada GPU
        └─────────────────────────┘
```

## 🌍 10. Real-Life Usages — Ye Sab Kahan Kaam Aata Hai

| Domain | Kya hota hai | Kaunsa concept |
|---|---|---|
| **Billing/Pricing** | OpenAI/Anthropic/Google APIs **per 1M tokens** karti hain — aapka invoice seedha tokenizer se banta hai | Tokens + Fertility |
| **Content Moderation** | Toxic/PII content ab **token sequences** pe filter hota hai; kuch attacks tokens ke pattern todte hain | Token IDs |
| **Code Completion** | GitHub Copilot ko code bhi tokens mein milta hai — indentation bhi ek token! | Tokens |
| **Machine Translation** | Google Translate/Meta ne **subword regularization** se rare words ka translation sudhara | Unigram |
| **RAG / Chatbots** | Documents ko **token-aware chunks** (e.g., 500 tokens) mein toda jaata hai, warna context tut jaata hai | Context Window |
| **Search / SEO** | "People Also Ask" jaise features LLM ke token budget ke andar hi sab kuch fit karte hain | Context Window |
| **Multilingual Products** | Hindi/Regional language apps token budget + cost dono plan karte hain | Fertility |
| **Jailbreaks & Defense** | Unicode tricks, homoglyphs — tokenizer ke normalization se bache attacks | Tokenization |
| **TTS / Voice Assistants** | Text→speech alignment ke liye **character-level offsets** chahiye hote hain (Siri, Alexa) | Tokens↔Chars mapping |
| **Gaming / Real-time** | Edge pe chhote tokenizer (distilled) — kam latency, kam memory | Tokenizer size |

> 💡 **Pro insight:** Jab bhi koi AI product banate ho, **pehle tokenizer call karo** — uske bina na cost ka anumaan lagega, na context ke andar sab fit hoga. **Tokenizer = product ka pehla checkpoint.** 🚦

---

## 🎯 11. Bonus: Tokenization Tips & Tricks

```bash
# 1. Hamesha TOKENIZER se count karo, words se nahi (word ≈ 1.33 tokens hota hai)
#    Anumaan: tokens ≈ ceil(chars / 4)  ← emergency estimate 🚨

# 2. RAG chunking TEXT ke boundary pe karo, tokens ke beech mein nahi
#    (sentence/paragraph boundaries → meaning bachti hai)

# 3. API response mein max_tokens set karte waqt INPUT tokens bhi count karo
#    warna: input + output > window → silent truncation 😱

# 4. Multilingual app? Pehle fertility measure karo (upar wala fertility.js)
#    Hindi/Regional ke liye alag tokenizer wala model choose karo

# 5. Prompt mein "exact character count" mat pucho — model characters nahi gin sakta
#    (wo tokens dekhta hai, characters nahi! — isi liye "how many r's in strawberry" galat)

# 6. Fine-tune kar rahe ho? Apne target domain ka TOKEN DISTRIBUTION check karo —
#    agar bahut saare [UNK]/rare pieces dikhe, tokenizer retraining ka case hai
```

> 🔑 **Deep Knowledge:** **"Strawberry mein kitne R hain?" wali famous problem** isi ka side-effect hai! Model ko characters nahi, **tokens** milte hain — aur `"straw"` + `"berry"` mein `'r'` kahan hai, ye "dekh" nahi pata. Solution? Ya toh **character-level task ke liye tools use karo** (code interpreter), ya model ko **chain-of-thought** se tokens ke andar jhaankna sikhao. **Ye limitation intelligence ki nahi, TOKENIZATION ki hai.** 🍓

---

## 📚 12. Resources — Padhne Ke Liye

**Core reading (papers):**
- 📄 *Neural Machine Translation of Rare Words with Subword Units* (Sennrich et al., 2016) — BPE in NLP
- 📄 *SentencePiece* (Kudo & Richardson, 2018) — Unigram + language-agnostic tokenization
- 📄 *BERT* (Devlin et al., 2019) — WordPiece ka asli user
- 📄 *Tokenization is more than Compression* (2023) — fertility aur multilingualism
- 📄 *Lost in the Middle* (Liu et al., 2023) — context window ki asli limit

**Hands-on:**
- 🛠️ HuggingFace `tokenizers` library — Rust-backed, bahut fast ⚡
- 🛠️ tiktoken (OpenAI) — GPT ke exact tokenizers
- 🛠️ SentencePiece — Unigram/BPE ka reference implementation
- 🧪 OpenAI Tokenizer playground — visual mein tokens dekho
- 🧪 `@huggingface/transformers` (browser/Node) — upar wale sab examples

**Video + Blog:**
- 🎥 Andrej Karpathy — *Let's build the GPT Tokenizer* (YouTube, must-watch! 🔥)
- 📘 HuggingFace NLP Course — Tokenization chapter
- 📝 Chip Huyen, Lilian Weng ke blogs — production angle

**Practice karne wala kaam:**
- 🏋️ Apne kisi bhi purane prompt ko tokenizer se count karo → asli bill nikalo
- 🏋️ Hindi + English dono lines ki fertility compare karo → tweet karo 🐦
- 🏋️ Upar wala `mini-bpe.js` chalao aur merges ka output dekho

---

## 🤝 13. Contributing & Star

Ye repo meri learning journey hai — **aapka contribution welcome hai!** 🙏

- 🐛 Galat explanation / typo mile toh **issue kholo**
- ✍️ Koi topic add karna hai (UTF-8, BPE-dropout, tokenizer poisoning? 😈) → **PR bhejo**
- ❓ Doubt? **Discussion mein pucho**

```bash
git clone https://github.com/neerajgir/Namaste-AI.git
# branch banao → change karo → PR karo ✅
```

---

## ⭐ Star Karo Agar Helpful Laga!

> **"Tokenizer wahi hai jo bhasha ko machine ki bhasha mein bolta hai — aur jiski fertility jitni kam, uska AI utna sasta aur behtar."** 🚀
> Ab jab bhi kisi LLM ka bill dekho ya context bharta dekho — **samajh jaana ki tokens ka khel chal raha hai.**

Ye repo zinda hai — har hafte nayi cheezein add hoti rahengi. **Watch/follow** kar lo aur apni LLM internals ki learning journey shuru karo!

Made with ❤️ in Hinglish | © Namaste AI

---

**PS:** Agar aapne yahan tak padha — shabaash! 👏 Ab **jaake `fertility.js` chalao aur Hindi vs English ka asar khud dekho** — phir batao kitne tokens gaye! 🪙✨
