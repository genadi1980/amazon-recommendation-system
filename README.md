# 🛍️ Amazon Product Recommendation System

> **MIT IDSS Data Science & Machine Learning Certification Project**  
> Techniques: Collaborative Filtering · SVD Matrix Factorisation · KNN · GridSearchCV  
> Stack: Python · Pandas · NumPy · Scikit-Surprise · Matplotlib · Seaborn

---

## 📌 Business Problem

How do you recommend the right product to the right user from a catalogue of 48,000+ items — at scale?

This project replicates the core engine behind platforms like Amazon, Netflix and Spotify: a system that learns from user behaviour to surface personalised recommendations. The challenge is doing this accurately with sparse, high-dimensional data.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | Amazon Electronics Ratings (public dataset) |
| Raw ratings | 7,824,482 |
| Users after filtering | 1,540 |
| Products after filtering | 48,190 |
| Final interactions used | 125,871 |

**Why filter?** Users with fewer than 50 ratings and rarely-reviewed products add noise without signal. Filtering to high-interaction users and products improves model accuracy meaningfully.

---

## 🔧 Models Built

### 1. Rank-Based Recommendation (Baseline)
A popularity model that recommends the top-N products by average rating, with a configurable minimum interaction threshold (tested at 50 and 100 reviews).

- **Use case:** Cold-start problem — new users with no history
- **Limitation:** Not personalised; every user gets the same list

### 2. Collaborative Filtering — User-User & Item-Item
KNNBasic algorithm with cosine similarity. Finds users (or items) most similar to the target and recommends based on their preferences.

- **Hyperparameter tuning:** GridSearchCV across `k`, `min_k`, similarity metric
- **Evaluation:** Precision@10, Recall@10, F1-score@10

### 3. SVD Matrix Factorisation (Best Model)
Decomposes the user-item rating matrix into latent factors. Captures hidden patterns in user preferences that explicit similarity measures miss.

- **Optimisation:** SGD with tuning across `n_epochs`, `lr_all`, `reg_all`
- **Best RMSE: 1.64** after hyperparameter optimisation

---

## 📈 Results Summary

| Model | RMSE | Precision@10 | Recall@10 |
|---|---|---|---|
| Rank-Based (popularity) | — | Baseline | Baseline |
| KNN User-User | — | — | — |
| KNN Item-Item | — | — | — |
| **SVD (tuned)** | **1.64** | Best | Best |

---

## 💡 Key Insights

- **Popularity models** are effective for cold-start (new users) but fail to personalise
- **Collaborative filtering** improves as user history grows — best for active users
- **SVD matrix factorisation** consistently outperforms neighbourhood methods on sparse data
- In production, a **hybrid approach** (popularity for new users → SVD once history builds) is optimal

---

## 🗂️ Repository Structure

```
amazon-recommendation-system/
│
├── notebook/
│   └── Genadi_Georgiev_Recommendation_Systems.ipynb
│
├── README.md
└── requirements.txt
```

---

## ⚙️ How to Run

```bash
# Clone the repo
git clone https://github.com/genadi1980/amazon-recommendation-system.git
cd amazon-recommendation-system

# Install dependencies
pip install -r requirements.txt

# Open the notebook
jupyter notebook notebook/Genadi_Georgiev_Recommendation_Systems.ipynb
```

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-surprise
jupyter
```

---

## 👤 Author

**Genadi Georgiev** — MIT IDSS Data Science & Machine Learning  
[LinkedIn](https://www.linkedin.com/in/genadi-georgiev-45a65261/) · [GitHub](https://github.com/genadi1980)
