import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

st.title("👁️ Mining Patterns in Eye Disorders of Young Adults")
st.write("Using real-world clinical data from Kaggle to discover associations between lifestyle and eye health.")


# Load and Preprocess Data
@st.cache_data
def load_and_prep_data():
    df = pd.read_csv('Dry_Eye_Dataset.csv')
    df['High_Screen_Time'] = df['Average screen time'] > 7.0
    cols = ['Sleep disorder', 'Smart device before bed', 'Blue-light filter',
            'Discomfort Eye-strain', 'Redness in eye', 'Itchiness/Irritation in eye',
            'Dry Eye Disease', 'High_Screen_Time']

    subset = df[cols].copy()

    def map_yes_no(x):
        if x == 'Y': return True
        if x == 'N': return False
        return x

    return subset.map(map_yes_no)


basket_sets = load_and_prep_data()

st.subheader("1. Patient Dataset Preview (Binarized)")
st.dataframe(basket_sets.head())

st.subheader("2. Apriori Algorithm Settings")
col1, col2 = st.columns(2)
with col1:
    min_sup = st.slider("Minimum Support (Frequency)", 0.05, 0.50, 0.15)
with col2:
    min_conf = st.slider("Minimum Confidence (Accuracy)", 0.10, 1.00, 0.60)

# Run Algorithm
frequent_itemsets = apriori(basket_sets, min_support=min_sup, use_colnames=True)

if not frequent_itemsets.empty:
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)

    if not rules.empty:
        rules = rules.sort_values('lift', ascending=False)
        st.success(f"Successfully found {len(rules)} hidden patterns!")

        # Clean up text for the web
        rules["antecedents"] = rules["antecedents"].apply(lambda x: ', '.join(list(x)))
        rules["consequents"] = rules["consequents"].apply(lambda x: ', '.join(list(x)))

        display_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
        st.dataframe(display_rules, use_container_width=True)
    else:
        st.warning("No strong rules found. Try lowering the Confidence slider.")
else:
    st.warning("No frequent patterns found. Try lowering the Support slider.")