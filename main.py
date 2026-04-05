import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import networkx as nx

# 1. Load the real Kaggle Dataset
file_name = 'Dry_Eye_Dataset.csv'
print(f"Loading {file_name}...")
df = pd.read_csv(file_name)

# 2. Data Preprocessing (Converting to True/False for Apriori)
print("Cleaning and formatting data...")

# Create a new column: If screen time is > 7 hours, mark as True
df['High_Screen_Time'] = df['Average screen time'] > 7.0

# Select only the relevant categorical columns for eye health
columns_to_keep = [
    'Sleep disorder', 'Smart device before bed', 'Blue-light filter',
    'Discomfort Eye-strain', 'Redness in eye', 'Itchiness/Irritation in eye',
    'Dry Eye Disease', 'High_Screen_Time'
]
subset_df = df[columns_to_keep].copy()


# Convert 'Y' to True and 'N' to False
def map_yes_no(x):
    if x == 'Y': return True
    if x == 'N': return False
    return x  # Keeps True/False from High_Screen_Time intact


basket_sets = subset_df.map(map_yes_no)

# 3. Run Apriori Algorithm
print("Mining Hidden Medical Patterns...")
# Look for patterns happening in at least 15% of people
frequent_itemsets = apriori(basket_sets, min_support=0.15, use_colnames=True)

# Find rules with at least 60% confidence
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.60)
rules = rules.sort_values('lift', ascending=False)

# 4. Print the top rules
print("\n=== TOP REAL-WORLD MEDICAL PATTERNS FOUND ===")
for index, row in rules.head(5).iterrows():
    antecedents = ", ".join(list(row['antecedents']))
    consequents = ", ".join(list(row['consequents']))
    print(f"IF a patient has [{antecedents}], THEN they likely have [{consequents}]")
    print(f"  - Confidence: {row['confidence']:.2f} | Lift: {row['lift']:.2f}\n")


# 5. Draw the Network Graph
def draw_network_graph(rules_df):
    plt.figure(figsize=(10, 6))
    G = nx.DiGraph()

    for index, row in rules_df.head(15).iterrows():
        antecedents = ", ".join(list(row['antecedents']))
        consequents = ", ".join(list(row['consequents']))
        G.add_edge(antecedents, consequents, weight=row['lift'])

    pos = nx.spring_layout(G, k=2.0)  # Spread out the nodes
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', edge_color='gray',
            node_size=2500, font_size=9, font_weight='bold', arrows=True)

    plt.title("Eye Disorder Association Rules (Kaggle Dataset)")
    plt.show()


draw_network_graph(rules)