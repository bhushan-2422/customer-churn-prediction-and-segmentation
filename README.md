# Customer Churn Prediction & Segmentation — ML Project

---

## About This Project

This is an end-to-end Machine Learning project I built on a telecom company dataset during my AI/ML internship at Coding Blocks.

The goal was to solve a real business problem — predicting which customers are likely to cancel their subscription — and then go a step further by grouping customers into meaningful segments so the business knows exactly who to target and how.

The project covers two parts:

- **Part 1 — Churn Prediction** (Supervised Learning): Build a model that predicts whether a customer will churn
- **Part 2 — Customer Segmentation** (Unsupervised Learning): Group customers by behaviour and churn risk

---

## What is Churn?

Churn simply means a customer leaving. In telecom, that means cancelling a subscription.

It costs a company far more to acquire a new customer than to keep an existing one. So if you can predict who's about to leave, you get a chance to act before it's too late — offer a discount, improve their experience, or reach out proactively.

---

## Dataset

**Source:** Telco Customer Churn Dataset (Excel)  
**Size:** 7,043 customers, 33 columns

| Column | Description |
|--------|-------------|
| Tenure Months | How long the customer has been with the company |
| Monthly Charges | What they pay each month |
| Total Charges | Total amount paid so far |
| Contract | Month-to-month, One year, or Two year |
| Internet Service | DSL, Fiber optic, or None |
| Churn Value | 0 = stayed, 1 = churned (this is what we predict) |

---

## Methodology

### Step 1 — Exploratory Data Analysis (EDA)

Before building anything, I explored the data to understand what was going on. A few patterns stood out clearly:

- Customers who churn tend to have much shorter tenure — they leave early
- Higher monthly charges are linked to more churn
- Month-to-month customers churn the most; two-year contract holders almost never leave
- Fiber optic customers churn more than DSL users, despite paying more — which suggests a service quality issue
- Customers paying by electronic check churn significantly more than those on auto-pay

---

### Step 2 — Data Cleaning

A few problems needed fixing before any modelling:

- `Total Charges` was stored as text instead of a number — fixed with `pd.to_numeric(errors='coerce')`
- Some `Total Charges` values were blank — filled with 0
- `City` had over 1,500 unique values — encoding it would have added too many columns with little benefit, so it was dropped

Some columns were also removed entirely:

| Column | Why Removed |
|--------|-------------|
| CustomerID, Count | Just identifiers, nothing to learn from them |
| Country, State, City, Zip Code | Location data that doesn't help predict churn |
| Churn Label, Churn Score, CLTV, Churn Reason | These are generated after a customer churns — using them would be cheating (data leakage) |

---

### Step 3 — Feature Encoding

ML models need numbers, not text. Columns like "Contract" had values like "Month-to-month" which needed to be converted.

```python
pd.get_dummies(df, drop_first=True)
```

This converts all text categories into 0/1 columns. `drop_first=True` drops one column per category group since it's mathematically redundant.

Result: went from 20 columns to 31 after encoding.

---

### Step 4 — Train / Test Split

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
```

80% of the data (5,634 customers) was used to train the model. The remaining 20% (1,409 customers) was held back to test how well it performs on data it hasn't seen before.

---

## Part 1 — Churn Prediction

### A Quick Note on Decision Trees and Random Forest

A Decision Tree asks a series of yes/no questions — like "Is tenure less than 12 months?" — and follows the path of answers to reach a prediction. The **depth** of the tree controls how many questions it can ask.

Too shallow and it misses patterns. Too deep and it starts memorising the training data instead of learning from it — which makes it bad at predicting new customers.

Random Forest fixes this by building hundreds of trees, each trained on a slightly different version of the data, and taking a majority vote. Much more reliable than a single tree.

---

### Approach 1 — Basic Random Forest

```python
RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
```

| Metric | Score |
|--------|-------|
| Accuracy | 79% |
| Churn Recall | 50% |

The accuracy looked decent, but it was only catching half the actual churners. The reason was class imbalance — about 74% of customers in the dataset stayed, so the model learned to mostly predict "stay" and still got a decent accuracy score. Not useful for the business.

---

### Approach 2 — Balanced Random Forest ✅ Best Model

```python
RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
```

| Metric | Score |
|--------|-------|
| Accuracy | 79% |
| Churn Recall | **91%** |
| AUC Score | **0.84** |

Adding `class_weight='balanced'` tells the model to treat churners and non-churners more equally during training, even though there are fewer churners in the data.

**Why this is the best model:** For churn prediction, recall is what matters most — not accuracy. If the model misses a churner, that customer leaves with no intervention. If it gives a false alarm, the worst case is sending someone an offer they didn't need. At 91% recall, the model catches 91 out of 100 real churners.

---

### Approach 3 — Feature Importance + Selection

After training the balanced model, I checked which features it relied on most:

| Feature | Importance Score |
|---------|-----------------|
| Total Charges | 0.175 |
| Tenure Months | 0.170 |
| Monthly Charges | 0.148 |
| Contract (Two Year) | 0.057 |
| Dependents (Yes) | 0.048 |

The top three are all financial or time-related — makes sense. I removed the weakest features (like `Phone Service_Yes`) to see if a leaner model performed better. It didn't — recall dropped to 74%.

---

### Approach 4 — Hyperparameter Tuning

I tested 20 combinations of tree depth and number of trees:

| Max Depth | Churn Recall | Accuracy | Observation |
|-----------|-------------|----------|-------------|
| 5 | 82% | 74% | Too shallow — misses patterns |
| 10 | 75% | 78% | Best overall balance |
| 15 | 64% | 80% | Accuracy improves, but missing more churners |
| 20 | 53% | 79% | Overfitting — good on training data, bad in practice |

The deeper the tree, the more it overfits and the worse recall gets. Depth 10 was the sweet spot.

---

### Model Comparison

| Metric | Basic RF | Balanced RF | Feature Selected | Tuned RF |
|--------|----------|-------------|-----------------|----------|
| Accuracy | 79% | 79% | 78% | 78% |
| Churn Recall | 50% | **91%** | 74% | 75% |
| AUC Score | — | **0.84** | — | — |

The Balanced Random Forest wins on the metric that actually matters for this problem.

---

### ROC AUC Score

```
AUC: 0.84
```

| AUC Range | What it Means |
|-----------|--------------|
| 0.5 | Random guessing |
| 0.7 – 0.8 | Good |
| 0.8 – 0.9 | Very good ← our model |
| 0.9+ | Excellent |

An AUC of 0.84 means the model correctly identifies which of two customers is more likely to churn 84% of the time.

---

## Part 2 — Customer Segmentation

### Why Segment Customers?

Not all at-risk customers are the same. A new customer paying a lot is a very different situation from a long-term customer on a cheap plan. Segmentation helps the business send the right message to the right person.

This is unsupervised learning — there are no labels, the algorithm finds the natural groups on its own.

---

### Features Used for Clustering

- Tenure Months
- Monthly Charges
- Total Charges
- **Churn Probability** (from the model in Part 1)

Including churn probability connects both parts of the project — the segments are tied directly to retention risk, not just spending behaviour.

---

### Scaling the Data

```python
scaler = StandardScaler()
scaled_data = scaler.fit_transform(segmentation_data)
```

KMeans groups customers by distance. If Total Charges goes up to 8,000 and Tenure only goes to 72, the algorithm would treat Total Charges as far more important just because of the scale. Standardising fixes that.

---

### Choosing the Number of Clusters

I used the Elbow Method — testing K from 1 to 15 and looking at where improvement in cluster compactness starts to flatten. The curve bent clearly at **K = 3**.

---

### The Three Segments

| Cluster | Avg Tenure | Monthly Charges | Churn Probability | Label |
|---------|-----------|-----------------|-------------------|-------|
| 0 | 27 months | $38 | 7% | Budget Loyal Customers |
| 1 | 57 months | $90 | 13% | Premium Loyal Customers |
| 2 | 11 months | $74 | 71% | High Risk New Customers |

**Cluster 0 — Budget Loyal Customers**  
Been around for over two years, paying a modest amount, and almost never leave. They're happy with the service but undervalued from a revenue standpoint.  
→ Good candidates for personalised upgrade offers.

**Cluster 1 — Premium Loyal Customers**  
The company's most valuable customers. High spend, long tenure, low risk. Losing one of these hurts.  
→ Focus on loyalty perks, dedicated support, and keeping them feeling appreciated.

**Cluster 2 — High Risk New Customers**  
New to the service, paying a decent amount, but 71% predicted churn rate. They haven't built loyalty yet.  
→ Need immediate attention — better onboarding, early-stage discounts, proactive check-ins.

---

## Key Insights

**1. The first year is where most churn happens.**  
Customers under 12 months have a 71% churn probability. After the two-year mark, churn risk drops significantly. The onboarding experience is critical.

**2. High bills + short tenure = highest risk.**  
New customers paying a lot feel the financial pressure without having built attachment to the service yet. Early discounts or service guarantees for this group could make a real difference.

**3. Contract type is one of the strongest predictors.**  
Month-to-month customers churn far more than annual ones. Incentivising customers to commit longer — with a discount or bundle — would directly reduce churn.

**4. Fiber optic has a service problem.**  
These customers pay premium prices but churn more than DSL users. That's a sign of unmet expectations. Worth digging into complaint data specifically for this group.

**5. Auto-pay is a passive loyalty tool.**  
Electronic check users churn more. They're also not locked in to auto-renewal. A small discount for switching to auto-pay could quietly reduce churn in this group with minimal effort.

**6. The numbers matter.**  
Around 1,800 customers churn per year at ~$65/month average. That's roughly $117,000 in lost monthly revenue. If the model helps retain even 30% of predicted churners, that's about $35,000 saved per month.

**7. Budget Loyal customers are an untapped opportunity.**  
Paying only $38/month after 27 months is a good sign of satisfaction, not just price sensitivity. A well-timed upgrade offer to this group carries very low churn risk and meaningful revenue upside.

---

## Recommended Actions

| Segment | Problem | Action |
|---------|---------|--------|
| High Risk New Customers | 71% churn probability | Better onboarding, early retention offers |
| Premium Loyal Customers | Very costly to lose | Loyalty rewards, priority support |
| Budget Loyal Customers | Revenue potential being left on the table | Personalised upgrade campaigns |
| Month-to-month users | Easy to walk away | Discount to switch to annual plans |
| Fiber optic customers | High churn despite premium pricing | Investigate service quality issues |
| Electronic check users | Not locked into auto-renewal | Offer discount for switching to auto-pay |

---

## Technologies Used

| Tool | Purpose |
|------|---------|
| Python | Main programming language |
| Pandas | Data cleaning and manipulation |
| NumPy | Numerical operations |
| Matplotlib | Visualisations |
| Seaborn | EDA charts |
| Scikit-learn | ML models, preprocessing, evaluation |
| Google Colab | Development environment |

Scikit-learn components: `RandomForestClassifier`, `KMeans`, `StandardScaler`, `train_test_split`, `classification_report`, `roc_auc_score`

---

## Results at a Glance

| Metric | Result |
|--------|--------|
| Best Model | Balanced Random Forest |
| Accuracy | 79% |
| Churn Recall | 91% |
| AUC Score | 0.84 |
| Segments Found | 3 |
| High Risk Segment Churn Probability | 71% |
| Estimated Monthly Savings (30% retention) | ~$35,000 |

---

## What I Learned

Working on this project gave me hands-on experience with the full ML pipeline — not just fitting models, but understanding why certain decisions are made at each step.

A few things that really clicked for me:

- Why accuracy alone is misleading when the dataset is imbalanced
- How data leakage can silently make a model look great but fail in the real world
- That the "best" model depends entirely on the business problem, not just the numbers
- How unsupervised learning can produce insights that supervised learning misses

**Skills covered:** EDA, data cleaning, encoding, feature scaling, supervised classification, unsupervised clustering, class imbalance handling, hyperparameter tuning, model evaluation, and translating results into business recommendations.

---

*Built during my AI/ML internship at Coding Blocks.*
