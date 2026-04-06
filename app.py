import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules

# ============== PAGE CONFIG ==============
st.set_page_config(
    page_title="Eye Disorder Pattern Discovery",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============== CUSTOM CSS ==============
st.markdown("""
<style>
    /* Main background and text */
    .main {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0;
    }

    .sub-header {
        text-align: center;
        color: #a0a0a0;
        font-size: 1.2rem;
        margin-top: -10px;
        margin-bottom: 30px;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e1e30 0%, #2d2d44 100%);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(102, 126, 234, 0.3);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        color: #888;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Section headers */
    .section-header {
        color: #667eea;
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 30px;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
    }

    /* Info boxes */
    .info-box {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        padding: 15px 20px;
        border-radius: 0 8px 8px 0;
        margin: 15px 0;
    }

    /* Success animation */
    .success-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    /* DataFrame styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    /* Slider styling */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)


# ============== LOAD DATA ==============
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

    return subset.map(map_yes_no), df


basket_sets, raw_df = load_and_prep_data()

# ============== HEADER ==============
st.markdown('<h1 class="main-header">👁️ Eye Disorder Pattern Discovery</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Uncovering hidden connections between lifestyle habits and eye health using Association Rule Mining</p>',
    unsafe_allow_html=True)

# Divider
st.markdown("---")

# ============== SIDEBAR ==============
with st.sidebar:
    st.markdown("## ⚙️ Algorithm Settings")
    st.markdown("")

    min_sup = st.slider(
        "🎯 Minimum Support",
        min_value=0.05,
        max_value=0.50,
        value=0.15,
        step=0.01,
        help="How frequently an itemset must appear in the dataset"
    )

    st.markdown("")

    min_conf = st.slider(
        "📊 Minimum Confidence",
        min_value=0.10,
        max_value=1.00,
        value=0.60,
        step=0.05,
        help="How often the rule has been found to be true"
    )

    st.markdown("---")

    st.markdown("### 📖 Quick Guide")
    st.markdown("""
    <div class="info-box">
    <strong>Support:</strong> Frequency of pattern<br>
    <strong>Confidence:</strong> Rule accuracy<br>
    <strong>Lift:</strong> Correlation strength<br><br>
    <em>Lift > 1 = Positive correlation</em>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### 🔬 Powered by Apriori Algorithm")
    st.markdown("Made with ❤️ by Sutikshan")

# ============== KEY METRICS ==============
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(raw_df):,}</div>
        <div class="metric-label">Total Patients</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    dry_eye_pct = (basket_sets['Dry Eye Disease'].sum() / len(basket_sets) * 100)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{dry_eye_pct:.1f}%</div>
        <div class="metric-label">Dry Eye Cases</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    high_screen_pct = (basket_sets['High_Screen_Time'].sum() / len(basket_sets) * 100)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{high_screen_pct:.1f}%</div>
        <div class="metric-label">High Screen Time</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    sleep_disorder_pct = (basket_sets['Sleep disorder'].sum() / len(basket_sets) * 100)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{sleep_disorder_pct:.1f}%</div>
        <div class="metric-label">Sleep Disorders</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# ============== DATA PREVIEW ==============
with st.expander("📋 View Patient Dataset (Binarized)", expanded=False):
    st.dataframe(
        basket_sets.head(10).style.applymap(
            lambda
                x: 'background-color: rgba(102, 126, 234, 0.2)' if x == True else 'background-color: rgba(239, 68, 68, 0.1)'
        ),
        use_container_width=True
    )
    st.caption(f"Showing 10 of {len(basket_sets)} records")

# ============== RUN APRIORI ==============
st.markdown('<div class="section-header">🔍 Discovered Patterns</div>', unsafe_allow_html=True)

frequent_itemsets = apriori(basket_sets, min_support=min_sup, use_colnames=True)

if not frequent_itemsets.empty:
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)

    if not rules.empty:
        rules = rules.sort_values('lift', ascending=False)

        # Success message
        st.markdown(f"""
        <div class="success-box">
            <span style="font-size: 2rem;">🎉</span><br>
            <span style="font-size: 1.5rem; font-weight: 600; color: #10b981;">
                Successfully discovered {len(rules)} hidden patterns!
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        # Clean up text for display
        rules_display = rules.copy()
        rules_display["antecedents"] = rules_display["antecedents"].apply(lambda x: ', '.join(list(x)))
        rules_display["consequents"] = rules_display["consequents"].apply(lambda x: ', '.join(list(x)))

        # Two column layout for results
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.markdown("#### 📊 Association Rules Table")
            display_rules = rules_display[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
            display_rules.columns = ['If This... (Antecedent)', 'Then This... (Consequent)', 'Support', 'Confidence',
                                     'Lift']

            # Format the dataframe
            st.dataframe(
                display_rules.style.format({
                    'Support': '{:.3f}',
                    'Confidence': '{:.2%}',
                    'Lift': '{:.2f}'
                }).background_gradient(subset=['Lift'], cmap='Purples'),
                use_container_width=True,
                height=400
            )

        with col_right:
            st.markdown("#### 🏆 Top 5 Strongest Rules")
            top_rules = rules_display.head(5)

            for idx, row in top_rules.iterrows():
                lift_color = "#10b981" if row['lift'] > 1.5 else "#667eea"
                st.markdown(f"""
                <div style="background: rgba(102, 126, 234, 0.1); padding: 12px; border-radius: 8px; margin-bottom: 10px; border-left: 3px solid {lift_color};">
                    <strong style="color: #667eea;">{row['antecedents']}</strong><br>
                    <span style="color: #888;">→</span> <strong style="color: #764ba2;">{row['consequents']}</strong><br>
                    <span style="font-size: 0.85rem; color: #888;">
                        Lift: <strong style="color: {lift_color};">{row['lift']:.2f}</strong> | 
                        Conf: {row['confidence']:.0%}
                    </span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("")

        # ============== VISUALIZATIONS ==============
        st.markdown('<div class="section-header">📈 Visual Analytics</div>', unsafe_allow_html=True)

        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            # Scatter plot: Support vs Confidence
            fig_scatter = px.scatter(
                rules_display,
                x='support',
                y='confidence',
                size='lift',
                color='lift',
                hover_data=['antecedents', 'consequents'],
                color_continuous_scale='Viridis',
                title='Support vs Confidence (Size = Lift)'
            )
            fig_scatter.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#888',
                title_font_color='#667eea'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with viz_col2:
            # Bar chart: Top rules by lift
            top_10 = rules_display.head(10).copy()
            top_10['rule'] = top_10['antecedents'].str[:20] + ' → ' + top_10['consequents'].str[:15]

            fig_bar = px.bar(
                top_10,
                x='lift',
                y='rule',
                orientation='h',
                color='confidence',
                color_continuous_scale='Purples',
                title='Top 10 Rules by Lift Score'
            )
            fig_bar.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#888',
                title_font_color='#667eea',
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # ============== INSIGHTS ==============
        st.markdown('<div class="section-header">💡 Key Insights</div>', unsafe_allow_html=True)

        # Find most impactful rules
        dry_eye_rules = rules_display[rules_display['consequents'].str.contains('Dry Eye')]

        if not dry_eye_rules.empty:
            top_dry_eye = dry_eye_rules.iloc[0]
            insight_col1, insight_col2 = st.columns(2)

            with insight_col1:
                st.markdown(f"""
                <div class="info-box">
                    <strong style="color: #667eea;">🔬 Strongest Dry Eye Predictor:</strong><br><br>
                    People with <strong>{top_dry_eye['antecedents']}</strong> are 
                    <strong style="color: #10b981;">{top_dry_eye['lift']:.1f}x more likely</strong> 
                    to develop Dry Eye Disease.
                </div>
                """, unsafe_allow_html=True)

            with insight_col2:
                st.markdown(f"""
                <div class="info-box">
                    <strong style="color: #667eea;">📊 Pattern Strength:</strong><br><br>
                    This association appears in <strong>{top_dry_eye['support'] * 100:.1f}%</strong> of all patients
                    and is accurate <strong>{top_dry_eye['confidence'] * 100:.1f}%</strong> of the time.
                </div>
                """, unsafe_allow_html=True)

    else:
        st.warning("⚠️ No strong rules found. Try lowering the **Confidence** slider in the sidebar.")

else:
    st.warning("⚠️ No frequent patterns found. Try lowering the **Support** slider in the sidebar.")

# ============== FOOTER ==============
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>Built with Streamlit • Powered by MLxtend Apriori • Data from Kaggle</p>
    <p style="font-size: 0.8rem;">© 2026 Eye Disorder Pattern Discovery | Sutikshan Pathania</p>
</div>
""", unsafe_allow_html=True)