"""
╔══════════════════════════════════════════════════════════════════╗
║         CUSTOMER CHURN PREDICTION - COMPLETE DA PROJECT          ║
║         Telecom Dataset | EDA + ML Models + Evaluation           ║
╚══════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)
from sklearn.inspection import permutation_importance

# ─────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────
PALETTE   = ['#2ECC71', '#E74C3C']          # green = stayed, red = churned
BG        = '#0F1117'
CARD      = '#1A1D27'
TEXT      = '#E8EAF0'
ACCENT    = '#7C83FD'
GRID_CLR  = '#2A2D3A'
sns.set_theme(style='darkgrid')
plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD,
    'axes.edgecolor': GRID_CLR, 'axes.labelcolor': TEXT,
    'xtick.color': TEXT, 'ytick.color': TEXT,
    'text.color': TEXT, 'grid.color': GRID_CLR,
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 13, 'axes.titleweight': 'bold',
    'axes.titlepad': 12, 'legend.framealpha': 0.15,
    'legend.facecolor': CARD,
})

# ══════════════════════════════════════════════════════════════════
# 1. SYNTHETIC TELECOM DATASET  (5 000 customers)
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  STEP 1 │ GENERATING TELECOM CHURN DATASET")
print("═"*65)

np.random.seed(42)
N = 5000

tenure        = np.random.exponential(scale=30, size=N).clip(1, 72).astype(int)
monthly_chg   = np.random.normal(65, 30, N).clip(20, 120)
total_chg     = monthly_chg * tenure * np.random.uniform(0.9, 1.1, N)
num_products  = np.random.choice([1,2,3,4], N, p=[0.4,0.3,0.2,0.1])
contract      = np.random.choice(['Month-to-month','One year','Two year'],
                                  N, p=[0.55, 0.25, 0.20])
internet_svc  = np.random.choice(['DSL','Fiber optic','No'], N, p=[0.34,0.44,0.22])
tech_support  = np.random.choice(['Yes','No'], N, p=[0.29, 0.71])
online_backup = np.random.choice(['Yes','No'], N, p=[0.34, 0.66])
senior        = np.random.choice([0, 1], N, p=[0.84, 0.16])
partner       = np.random.choice(['Yes','No'], N, p=[0.48, 0.52])
paperless     = np.random.choice(['Yes','No'], N, p=[0.59, 0.41])
payment       = np.random.choice(
    ['Electronic check','Mailed check','Bank transfer','Credit card'],
    N, p=[0.34, 0.23, 0.22, 0.21])
calls_to_support = np.random.poisson(1.2, N)

# Churn probability – depends on real-world-like factors
churn_prob = (
    0.05
    + 0.18 * (contract == 'Month-to-month')
    + 0.10 * (internet_svc == 'Fiber optic')
    - 0.12 * (tech_support == 'Yes')
    - 0.008 * tenure
    + 0.003 * (monthly_chg - 65)
    + 0.07  * (calls_to_support >= 3)
    + 0.05  * senior
    - 0.06  * (num_products >= 2)
    + 0.04  * (paperless == 'Yes')
    + np.random.normal(0, 0.05, N)
).clip(0.02, 0.95)

churn = (np.random.random(N) < churn_prob).astype(int)

df = pd.DataFrame({
    'CustomerID'      : [f'C{i:05d}' for i in range(N)],
    'Tenure'          : tenure,
    'MonthlyCharges'  : monthly_chg.round(2),
    'TotalCharges'    : total_chg.round(2),
    'NumProducts'     : num_products,
    'Contract'        : contract,
    'InternetService' : internet_svc,
    'TechSupport'     : tech_support,
    'OnlineBackup'    : online_backup,
    'SeniorCitizen'   : senior,
    'Partner'         : partner,
    'PaperlessBilling': paperless,
    'PaymentMethod'   : payment,
    'SupportCalls'    : calls_to_support,
    'Churn'           : churn
})

print(f"  Dataset shape : {df.shape}")
print(f"  Churn rate    : {df['Churn'].mean()*100:.1f}%")
print(f"  Churned       : {df['Churn'].sum()} | Stayed: {(df['Churn']==0).sum()}")
print(df.head(3).to_string(index=False))

df.to_csv('telecom_churn.csv', index=False)
print("\n  ✔ Dataset saved → telecom_churn.csv")


# ══════════════════════════════════════════════════════════════════
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  STEP 2 │ EXPLORATORY DATA ANALYSIS")
print("═"*65)

print("\n── Descriptive Statistics ──")
print(df.describe().round(2))
print(f"\n── Missing values ──\n{df.isnull().sum()}")

# ── EDA Figure 1 : Overview (4 plots) ──────────────────────────
fig = plt.figure(figsize=(20, 16), facecolor=BG)
fig.suptitle('CUSTOMER CHURN — EXPLORATORY DATA ANALYSIS',
             fontsize=18, fontweight='bold', color=TEXT, y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

# 1a. Churn distribution (donut)
ax0 = fig.add_subplot(gs[0, 0])
sizes = df['Churn'].value_counts()
wedges, texts, autotexts = ax0.pie(
    sizes, labels=['Stayed','Churned'], autopct='%1.1f%%',
    colors=PALETTE, startangle=90,
    wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=3),
    textprops=dict(color=TEXT, fontsize=11))
for at in autotexts: at.set_fontweight('bold')
ax0.set_title('Churn Distribution')

# 1b. Tenure by Churn (KDE)
ax1 = fig.add_subplot(gs[0, 1])
for val, lbl, clr in zip([0,1],['Stayed','Churned'], PALETTE):
    sub = df[df['Churn']==val]['Tenure']
    ax1.hist(sub, bins=30, alpha=0.65, color=clr, label=lbl, density=True)
ax1.set_xlabel('Tenure (months)'); ax1.set_ylabel('Density')
ax1.set_title('Tenure Distribution by Churn')
ax1.legend()

# 1c. Monthly Charges by Churn
ax2 = fig.add_subplot(gs[0, 2])
for val, lbl, clr in zip([0,1],['Stayed','Churned'], PALETTE):
    sub = df[df['Churn']==val]['MonthlyCharges']
    ax2.hist(sub, bins=30, alpha=0.65, color=clr, label=lbl, density=True)
ax2.set_xlabel('Monthly Charges ($)'); ax2.set_ylabel('Density')
ax2.set_title('Monthly Charges by Churn')
ax2.legend()

# 1d. Contract type vs Churn rate
ax3 = fig.add_subplot(gs[1, 0])
ct = df.groupby('Contract')['Churn'].mean().reset_index().sort_values('Churn', ascending=False)
bars = ax3.bar(ct['Contract'], ct['Churn']*100, color=[PALETTE[1], PALETTE[0], PALETTE[0]],
               edgecolor=BG, linewidth=1.5)
for b in bars:
    ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
             f"{b.get_height():.1f}%", ha='center', fontsize=9, color=TEXT)
ax3.set_xlabel('Contract Type'); ax3.set_ylabel('Churn Rate (%)')
ax3.set_title('Churn Rate by Contract Type')

# 1e. Internet Service vs Churn
ax4 = fig.add_subplot(gs[1, 1])
ct2 = df.groupby('InternetService')['Churn'].mean().reset_index().sort_values('Churn', ascending=False)
clrs = [PALETTE[1] if v > 0.22 else PALETTE[0] for v in ct2['Churn']]
bars2 = ax4.bar(ct2['InternetService'], ct2['Churn']*100, color=clrs, edgecolor=BG, linewidth=1.5)
for b in bars2:
    ax4.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
             f"{b.get_height():.1f}%", ha='center', fontsize=9, color=TEXT)
ax4.set_xlabel('Internet Service'); ax4.set_ylabel('Churn Rate (%)')
ax4.set_title('Churn Rate by Internet Service')

# 1f. Support Calls vs Churn
ax5 = fig.add_subplot(gs[1, 2])
ct3 = df.groupby('SupportCalls')['Churn'].mean().reset_index()
ax5.bar(ct3['SupportCalls'], ct3['Churn']*100,
        color=ACCENT, edgecolor=BG, linewidth=1.5)
ax5.set_xlabel('# Support Calls'); ax5.set_ylabel('Churn Rate (%)')
ax5.set_title('Support Calls vs Churn Rate')

# 1g. Correlation heatmap (numeric)
ax6 = fig.add_subplot(gs[2, :])
num_cols = ['Tenure','MonthlyCharges','TotalCharges','NumProducts','SupportCalls','SeniorCitizen','Churn']
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, ax=ax6, linewidths=0.5,
            cbar_kws={'shrink': 0.6})
ax6.set_title('Correlation Matrix — Numeric Features')

plt.savefig('fig1_eda.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  ✔ EDA figure saved → fig1_eda.png")


# ══════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  STEP 3 │ FEATURE ENGINEERING & PREPROCESSING")
print("═"*65)

df_ml = df.drop('CustomerID', axis=1).copy()

# New features
df_ml['ChargesPerMonth']    = df_ml['TotalCharges'] / df_ml['Tenure']
df_ml['HighCallsFlag']      = (df_ml['SupportCalls'] >= 3).astype(int)
df_ml['LongTenureFlag']     = (df_ml['Tenure'] >= 24).astype(int)

cat_cols = ['Contract','InternetService','TechSupport','OnlineBackup',
            'Partner','PaperlessBilling','PaymentMethod']
le = LabelEncoder()
for col in cat_cols:
    df_ml[col] = le.fit_transform(df_ml[col])

print(f"  Final feature count : {df_ml.shape[1]-1}")
print(f"  Features : {list(df_ml.drop('Churn',axis=1).columns)}")

X = df_ml.drop('Churn', axis=1)
y = df_ml['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\n  Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


# ══════════════════════════════════════════════════════════════════
# 4. MODEL TRAINING
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  STEP 4 │ MODEL TRAINING")
print("═"*65)

models = {
    'Logistic Regression' : LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'Random Forest'       : RandomForestClassifier(n_estimators=200, max_depth=10,
                                                    class_weight='balanced', random_state=42, n_jobs=-1),
    'Gradient Boosting'   : GradientBoostingClassifier(n_estimators=200, learning_rate=0.08,
                                                        max_depth=5, random_state=42),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    X_fit = X_train_sc if name == 'Logistic Regression' else X_train
    X_ev  = X_test_sc  if name == 'Logistic Regression' else X_test

    cv_scores = cross_val_score(model, X_fit, y_train, cv=cv, scoring='roc_auc')
    model.fit(X_fit, y_train)

    y_pred      = model.predict(X_ev)
    y_prob      = model.predict_proba(X_ev)[:, 1]

    results[name] = {
        'model'    : model,
        'X_ev'     : X_ev,
        'y_pred'   : y_pred,
        'y_prob'   : y_prob,
        'accuracy' : accuracy_score(y_test, y_pred),
        'recall'   : recall_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'f1'       : f1_score(y_test, y_pred),
        'roc_auc'  : roc_auc_score(y_test, y_prob),
        'cv_auc'   : cv_scores.mean(),
        'cv_std'   : cv_scores.std(),
    }
    print(f"\n  [{name}]")
    print(f"    Accuracy  : {results[name]['accuracy']:.4f}")
    print(f"    Recall    : {results[name]['recall']:.4f}")
    print(f"    Precision : {results[name]['precision']:.4f}")
    print(f"    F1-Score  : {results[name]['f1']:.4f}")
    print(f"    ROC-AUC   : {results[name]['roc_auc']:.4f}")
    print(f"    CV-AUC    : {results[name]['cv_auc']:.4f} ± {results[name]['cv_std']:.4f}")


# ══════════════════════════════════════════════════════════════════
# 5. MODEL EVALUATION VISUALISATIONS
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  STEP 5 │ MODEL EVALUATION VISUALISATIONS")
print("═"*65)

model_names = list(results.keys())
colors_mdl  = ['#7C83FD', '#FFA07A', '#98FB98']

# ── Figure 2 : Metrics + ROC + Confusion Matrices ───────────────
fig2 = plt.figure(figsize=(22, 18), facecolor=BG)
fig2.suptitle('MODEL EVALUATION DASHBOARD', fontsize=18,
               fontweight='bold', color=TEXT, y=0.99)
gs2 = gridspec.GridSpec(3, 3, figure=fig2, hspace=0.5, wspace=0.38)

# ── 2a. Grouped bar – metrics comparison ────────────────────────
ax_m = fig2.add_subplot(gs2[0, :2])
metrics = ['accuracy','recall','precision','f1','roc_auc']
metric_labels = ['Accuracy','Recall','Precision','F1','ROC-AUC']
x = np.arange(len(metrics))
width = 0.25
for i, (name, clr) in enumerate(zip(model_names, colors_mdl)):
    vals = [results[name][m] for m in metrics]
    bars = ax_m.bar(x + i*width, vals, width, label=name, color=clr,
                    edgecolor=BG, linewidth=1.5)
    for b, v in zip(bars, vals):
        ax_m.text(b.get_x()+b.get_width()/2, b.get_height()+0.003,
                  f"{v:.3f}", ha='center', fontsize=7.5, color=TEXT)
ax_m.set_xticks(x + width); ax_m.set_xticklabels(metric_labels)
ax_m.set_ylim(0, 1.05); ax_m.set_ylabel('Score')
ax_m.set_title('Model Performance Comparison'); ax_m.legend()

# ── 2b. CV AUC with error bars ──────────────────────────────────
ax_cv = fig2.add_subplot(gs2[0, 2])
cv_means = [results[n]['cv_auc'] for n in model_names]
cv_stds  = [results[n]['cv_std'] for n in model_names]
ax_cv.bar(range(len(model_names)), cv_means, yerr=cv_stds,
          color=colors_mdl, edgecolor=BG, linewidth=1.5,
          capsize=6, error_kw=dict(ecolor=TEXT, lw=2))
ax_cv.set_xticks(range(len(model_names)))
ax_cv.set_xticklabels([n.split()[0] for n in model_names])
ax_cv.set_ylim(0.5, 1.05); ax_cv.set_ylabel('CV ROC-AUC')
ax_cv.set_title('5-Fold Cross-Validation AUC')
for i,(m,s) in enumerate(zip(cv_means,cv_stds)):
    ax_cv.text(i, m+s+0.01, f"{m:.3f}", ha='center', fontsize=9, color=TEXT, fontweight='bold')

# ── 2c. ROC Curves ──────────────────────────────────────────────
ax_roc = fig2.add_subplot(gs2[1, :2])
ax_roc.plot([0,1],[0,1],'--', color='gray', lw=1, label='Random Classifier')
for name, clr in zip(model_names, colors_mdl):
    fpr, tpr, _ = roc_curve(y_test, results[name]['y_prob'])
    auc = results[name]['roc_auc']
    ax_roc.plot(fpr, tpr, color=clr, lw=2, label=f"{name} (AUC={auc:.3f})")
ax_roc.fill_between([0,1],[0,1], alpha=0.04, color='gray')
ax_roc.set_xlabel('False Positive Rate'); ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC Curves — All Models'); ax_roc.legend(loc='lower right')

# ── 2d. Precision-Recall trade-off visual (threshold analysis for best model) ──
best_name = max(results, key=lambda n: results[n]['roc_auc'])
ax_thr = fig2.add_subplot(gs2[1, 2])
thresholds = np.linspace(0.1, 0.9, 80)
precs, recs, f1s = [], [], []
for t in thresholds:
    yp = (results[best_name]['y_prob'] >= t).astype(int)
    precs.append(precision_score(y_test, yp, zero_division=0))
    recs.append(recall_score(y_test, yp, zero_division=0))
    f1s.append(f1_score(y_test, yp, zero_division=0))
ax_thr.plot(thresholds, precs, color='#FFA07A', lw=2, label='Precision')
ax_thr.plot(thresholds, recs,  color='#98FB98', lw=2, label='Recall')
ax_thr.plot(thresholds, f1s,   color=ACCENT,    lw=2, label='F1', ls='--')
ax_thr.set_xlabel('Decision Threshold'); ax_thr.set_ylabel('Score')
ax_thr.set_title(f'Threshold Analysis\n({best_name.split()[0]})')
ax_thr.legend()

# ── 2e-g. Confusion Matrices ─────────────────────────────────────
for i, (name, clr) in enumerate(zip(model_names, colors_mdl)):
    ax_cm = fig2.add_subplot(gs2[2, i])
    cm = confusion_matrix(y_test, results[name]['y_pred'])
    disp = ConfusionMatrixDisplay(cm, display_labels=['Stayed','Churned'])
    disp.plot(ax=ax_cm, colorbar=False, cmap='Blues')
    ax_cm.set_title(f'Confusion Matrix\n{name}')
    for text in ax_cm.texts: text.set_fontsize(12)

plt.savefig('fig2_evaluation.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  ✔ Evaluation figure saved → fig2_evaluation.png")


# ── Figure 3 : Feature Importance ───────────────────────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(22, 7), facecolor=BG)
fig3.suptitle('FEATURE IMPORTANCE — ALL MODELS', fontsize=16,
               fontweight='bold', color=TEXT)

feat_names = X.columns.tolist()

for ax, (name, clr) in zip(axes3, zip(model_names, colors_mdl)):
    model = results[name]['model']
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        # Logistic Regression: use absolute coefficients
        importances = np.abs(model.coef_[0])

    idx = np.argsort(importances)[::-1][:12]
    top_feats = [feat_names[i] for i in idx]
    top_vals  = importances[idx]

    colors_bar = [clr if v > np.median(top_vals) else '#3A3D52' for v in top_vals]
    ax.barh(top_feats[::-1], top_vals[::-1], color=colors_bar[::-1], edgecolor=BG)
    ax.set_xlabel('Importance Score')
    ax.set_title(name)
    ax.tick_params(labelsize=8)

plt.tight_layout(rect=[0,0,1,0.94])
plt.savefig('fig3_feature_importance.png', dpi=150,
             bbox_inches='tight', facecolor=BG)
plt.close()
print("  ✔ Feature importance figure saved → fig3_feature_importance.png")


# ── Figure 4 : Business Insights (EDA deep-dive) ────────────────
fig4, axes4 = plt.subplots(2, 3, figsize=(20, 12), facecolor=BG)
fig4.suptitle('BUSINESS INSIGHTS — CHURN DRIVERS', fontsize=16,
               fontweight='bold', color=TEXT)

# 4a. Box: Tenure by Churn
ax = axes4[0,0]
data0 = [df[df['Churn']==0]['Tenure'].values, df[df['Churn']==1]['Tenure'].values]
bp = ax.boxplot(data0, labels=['Stayed','Churned'], patch_artist=True,
                medianprops=dict(color='white', lw=2))
for patch, c in zip(bp['boxes'], PALETTE): patch.set_facecolor(c)
ax.set_ylabel('Tenure (months)'); ax.set_title('Tenure vs Churn')

# 4b. Box: Monthly Charges by Churn
ax = axes4[0,1]
data1 = [df[df['Churn']==0]['MonthlyCharges'].values, df[df['Churn']==1]['MonthlyCharges'].values]
bp2 = ax.boxplot(data1, labels=['Stayed','Churned'], patch_artist=True,
                 medianprops=dict(color='white', lw=2))
for patch, c in zip(bp2['boxes'], PALETTE): patch.set_facecolor(c)
ax.set_ylabel('Monthly Charges ($)'); ax.set_title('Monthly Charges vs Churn')

# 4c. Payment Method churn rate
ax = axes4[0,2]
pm = df.groupby('PaymentMethod')['Churn'].mean().sort_values(ascending=True)
clrs_pm = [PALETTE[1] if v > 0.25 else PALETTE[0] for v in pm.values]
ax.barh(pm.index, pm.values*100, color=clrs_pm, edgecolor=BG)
ax.set_xlabel('Churn Rate (%)'); ax.set_title('Churn Rate by Payment Method')
for i, v in enumerate(pm.values):
    ax.text(v*100+0.3, i, f"{v*100:.1f}%", va='center', fontsize=9)

# 4d. Senior Citizen vs churn
ax = axes4[1,0]
sc = df.groupby('SeniorCitizen')['Churn'].mean()
ax.bar(['Non-Senior','Senior'], sc.values*100,
       color=[PALETTE[0], PALETTE[1]], edgecolor=BG, linewidth=1.5)
for i, v in enumerate(sc.values):
    ax.text(i, v*100+0.5, f"{v*100:.1f}%", ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('Churn Rate (%)'); ax.set_title('Senior Citizen vs Churn Rate')

# 4e. Num Products vs Churn
ax = axes4[1,1]
np2 = df.groupby('NumProducts')['Churn'].mean()
clrs_np = [PALETTE[1] if v > 0.25 else PALETTE[0] for v in np2.values]
ax.bar(np2.index, np2.values*100, color=clrs_np, edgecolor=BG, linewidth=1.5)
for i, (x2, v) in enumerate(zip(np2.index, np2.values)):
    ax.text(x2, v*100+0.5, f"{v*100:.1f}%", ha='center', fontsize=10)
ax.set_xlabel('# Products'); ax.set_ylabel('Churn Rate (%)')
ax.set_title('Number of Products vs Churn')

# 4f. Scatter: Tenure vs Monthly Charges (coloured by churn)
ax = axes4[1,2]
stayed  = df[df['Churn']==0]
churned = df[df['Churn']==1]
ax.scatter(stayed['Tenure'],  stayed['MonthlyCharges'],
           c=PALETTE[0], alpha=0.25, s=10, label='Stayed')
ax.scatter(churned['Tenure'], churned['MonthlyCharges'],
           c=PALETTE[1], alpha=0.35, s=12, label='Churned')
ax.set_xlabel('Tenure (months)'); ax.set_ylabel('Monthly Charges ($)')
ax.set_title('Tenure vs Monthly Charges by Churn')
ax.legend(markerscale=2)

plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig('fig4_business_insights.png', dpi=150,
             bbox_inches='tight', facecolor=BG)
plt.close()
print("  ✔ Business insights figure saved → fig4_business_insights.png")


# ══════════════════════════════════════════════════════════════════
# 6. FINAL SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  STEP 6 │ FINAL RESULTS SUMMARY")
print("═"*65)

summary = pd.DataFrame({
    'Model'     : model_names,
    'Accuracy'  : [f"{results[n]['accuracy']:.4f}"  for n in model_names],
    'Recall'    : [f"{results[n]['recall']:.4f}"    for n in model_names],
    'Precision' : [f"{results[n]['precision']:.4f}" for n in model_names],
    'F1-Score'  : [f"{results[n]['f1']:.4f}"        for n in model_names],
    'ROC-AUC'   : [f"{results[n]['roc_auc']:.4f}"   for n in model_names],
    'CV-AUC'    : [f"{results[n]['cv_auc']:.4f}±{results[n]['cv_std']:.4f}" for n in model_names],
})
print(summary.to_string(index=False))

best = max(results, key=lambda n: results[n]['roc_auc'])
print(f"\n  🏆 Best Model  : {best}")
print(f"     ROC-AUC    : {results[best]['roc_auc']:.4f}")
print(f"     Recall     : {results[best]['recall']:.4f}  ← crucial for churn (catch churners)")

summary.to_csv('model_results.csv', index=False)
print("\n  ✔ Results table saved → model_results.csv")

print("\n" + "═"*65)
print("  ✅  ALL STEPS COMPLETE — Project files generated!")
print("═"*65)
print("""
  Output files
  ├── telecom_churn.csv          ← 5 000-row dataset
  ├── model_results.csv          ← metric comparison table
  ├── fig1_eda.png               ← EDA overview (7 charts)
  ├── fig2_evaluation.png        ← ROC · metrics · confusion matrices
  ├── fig3_feature_importance.png← feature importances (3 models)
  └── fig4_business_insights.png ← business-level churn drivers
""")
