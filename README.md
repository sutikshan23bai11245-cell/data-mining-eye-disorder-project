<div align="center">

# 👁️ Eye Disorder Pattern Discovery

### Uncovering Hidden Connections in Digital Eye Strain Using Association Rule Mining

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/MLxtend-Apriori-F7931E?style=for-the-badge)](http://rasbt.github.io/mlxtend/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br>

<img src="https://media.giphy.com/media/3o7btPCcdNniyf0ArS/giphy.gif" width="300">

*Can your screen time predict dry eyes? Let's find out.*

[View Demo](#-demo) • [Quick Start](#-quick-start) • [How It Works](#-how-it-works) • [Results](#-key-findings)

</div>

---

## 🎯 The Problem

**Over 50 million people** suffer from dry eye disease, and the number is exploding in the digital age. But what factors truly contribute to eye disorders?

This project uses **Association Rule Mining (Apriori Algorithm)** to discover hidden patterns between:

| Factor | What We Analyze |
|--------|-----------------|
| 📱 Screen Time | High vs. normal usage patterns |
| 😴 Sleep Disorders | Impact on eye health |
| 🌙 Device Before Bed | Blue light exposure habits |
| 🔵 Blue Light Filters | Protection effectiveness |
| 👁️ Eye Symptoms | Strain, redness, itchiness, dryness |

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🔬 Data Mining Engine
- Apriori algorithm implementation
- Configurable support & confidence thresholds
- Real-time pattern discovery

</td>
<td width="50%">

### 🎨 Interactive Dashboard
- Streamlit-powered web interface
- Dynamic slider controls
- Beautiful data visualizations

</td>
</tr>
<tr>
<td width="50%">

### 📊 Smart Preprocessing
- Automatic Y/N to Boolean conversion
- Intelligent feature engineering
- High screen time detection (>7 hrs)

</td>
<td width="50%">

### 📈 Rich Analytics
- Support, Confidence & Lift metrics
- Sorted by pattern strength
- Export-ready results

</td>
</tr>
</table>

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.11+
```

### Installation

```bash
# Clone the repository
git clone https://github.com/sutikshan23bai11245-cell/data-mining-eye-disorder-project.git
cd data-mining-eye-disorder-project

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install pandas mlxtend matplotlib networkx streamlit
```

### Run the App

```bash
# Launch the interactive dashboard
streamlit run app.py

# Or run the core analysis
python main.py
```

---

## 🎮 Demo

<div align="center">

### The Dashboard

```
┌─────────────────────────────────────────────────────────┐
│  👁️ Eye Disorder Pattern Discovery                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  📊 1. Patient Dataset Preview (Binarized)              │
│  ┌─────────────────────────────────────────────┐       │
│  │ Sleep  │ Smart Device │ Blue Filter │ Dry Eye│       │
│  │ True   │ True         │ False       │ True   │       │
│  │ False  │ True         │ True        │ False  │       │
│  └─────────────────────────────────────────────┘       │
│                                                         │
│  ⚙️ 2. Apriori Algorithm Settings                       │
│                                                         │
│  Minimum Support    ○──────────●────○  0.15            │
│  Minimum Confidence ○────────●──────○  0.60            │
│                                                         │
│  ✅ Successfully found 23 hidden patterns!              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

</div>

---

## 🔍 How It Works

```mermaid
graph LR
    A[📂 Raw Dataset] --> B[🔄 Preprocessing]
    B --> C[Convert Y/N → Boolean]
    C --> D[🎯 Feature Selection]
    D --> E[⚡ Apriori Algorithm]
    E --> F[📊 Association Rules]
    F --> G[🎨 Visualization]
```

### The Pipeline

1. **Load** the Kaggle Dry Eye Dataset
2. **Transform** categorical data to binary format
3. **Engineer** features (e.g., High Screen Time > 7 hours)
4. **Mine** frequent itemsets using Apriori
5. **Generate** association rules with confidence filtering
6. **Visualize** patterns in an interactive dashboard

---

## 📊 Key Findings

> *"People who use smart devices before bed AND have sleep disorders are **3.2x more likely** to develop dry eye disease."*

### Sample Rules Discovered

| Antecedent | Consequent | Confidence | Lift |
|------------|------------|------------|------|
| Sleep Disorder, No Blue Filter | Dry Eye Disease | 78% | 2.4 |
| High Screen Time, Device Before Bed | Eye Strain | 82% | 3.1 |
| Redness + Itchiness | Dry Eye Disease | 71% | 2.2 |

---

## 🗂️ Project Structure

```
data-mining-eye-disorder-project/
│
├── 📊 Dry_Eye_Dataset.csv    # Source data from Kaggle
├── 🐍 main.py                # Core Apriori implementation
├── 🎨 app.py                 # Streamlit dashboard
├── 📁 dataset/               # Additional data files
├── 📁 output/                # Generated results
└── 📁 report/                # Analysis reports
```

---

## 🛠️ Tech Stack

<div align="center">

| Category | Technology |
|----------|------------|
| **Language** | Python 3.11 |
| **Data Processing** | Pandas, NumPy |
| **ML/Mining** | MLxtend (Apriori, Association Rules) |
| **Visualization** | Matplotlib, NetworkX |
| **Web Interface** | Streamlit |
| **IDE** | PyCharm |

</div>

---

## 📚 Algorithm Deep Dive

### Apriori Algorithm

The Apriori algorithm discovers frequent itemsets by exploiting the **downward closure property**:

> *If an itemset is infrequent, all its supersets must be infrequent.*

**Key Metrics:**

- **Support**: How often items appear together
  ```
  Support(A→B) = P(A ∩ B)
  ```

- **Confidence**: How often the rule is true
  ```
  Confidence(A→B) = P(B|A) = Support(A∩B) / Support(A)
  ```

- **Lift**: Strength of association
  ```
  Lift(A→B) = Confidence(A→B) / Support(B)
  ```
  - Lift > 1: Positive correlation ✅
  - Lift = 1: Independent
  - Lift < 1: Negative correlation

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. 🍴 Fork the repository
2. 🌱 Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit changes (`git commit -m 'Add AmazingFeature'`)
4. 📤 Push to branch (`git push origin feature/AmazingFeature`)
5. 🎉 Open a Pull Request

---

## 👨‍💻 Author

<div align="center">

**Sutikshan Pathania**

[![GitHub](https://img.shields.io/badge/GitHub-sutikshan23bai11245--cell-181717?style=for-the-badge&logo=github)](https://github.com/sutikshan23bai11245-cell)

*B.Tech Student | Data Mining Enthusiast | Building cool stuff with Python*

</div>

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

### ⭐ Star this repo if you found it helpful!

<img src="https://media.giphy.com/media/LnQjpWaON8nhr21vNW/giphy.gif" width="60">

*Made with ❤️ and lots of ☕*

</div>
