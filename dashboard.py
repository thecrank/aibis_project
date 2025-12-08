import streamlit as st
import pandas as pd
import numpy as np
import shap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
import joblib
import os

st.set_page_config(page_title="Credit Risk AI Dashboard", layout="wide", page_icon="🏦", initial_sidebar_state="collapsed")


@st.cache_resource
def load_saved_project():
    try:
        data = joblib.load('aibis_projekt_daten.pkl')
        return data['model'], data['X_test'], data['y_test'], data['X_full']
    except FileNotFoundError:
        return None, None, None, None


@st.cache_resource
def load_cluster_brain():
    try:
        return joblib.load('cluster_brain.pkl')
    except FileNotFoundError:
        return None

model, X_test, y_test, X_full = load_saved_project()
cluster_brain = load_cluster_brain()

# Initialize SHAP Explainer for Risk (cached)
explainer = shap.TreeExplainer(model)

# --- HEADER ---
st.title("🏦 Passau Finance AI - Credit Risk Dashboard")
st.markdown("---")

# --- TOP NAVIGATION WITH TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Executive Overview", "🧩 Customer Segmentation", "🕵️‍♂️ Loan Officer Interface", "📝 Manual Prediction"])

# Determine which page we're on based on active tab
# We'll use the tab context managers below

# ==========================================
# PAGE 1: EXECUTIVE OVERVIEW
# ==========================================
with tab1:
    st.subheader("📊 Strategic Risk Portfolio Overview")
    st.markdown("Performance monitoring of the deployed Credit Risk AI Model.")
    st.caption("**Model:** Random Forest | **Threshold:** 0.30")

    # 1. KPIs
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.3).astype(int)
    
    cm = confusion_matrix(y_test, y_pred)
    recall = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    accuracy = accuracy_score(y_test, y_pred)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Recall (Risk Detection)", f"{recall:.1%}", "Target: >80%")
    col2.metric("Prevented Defaults", f"{cm[1, 1]} / {cm[1, 1]+cm[1, 0]}", "Loans identified")
    col3.metric("Model Accuracy", f"{accuracy:.1%}", "Trade-off for safety")
    
    st.markdown("---")

    # 2. Interactive Plots
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Confusion Matrix")
        st.write("Visualizing True Positives vs. False Alarms.")
        
        # Interactive Plotly Confusion Matrix
        labels = ['Good', 'Bad']
        z = cm
        z_text = [[str(y) for y in x] for x in z]
        
        fig_cm = go.Figure(data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            text=z_text,
            texttemplate="%{text}",
            textfont={"size": 20},
            colorscale='Blues',
            showscale=False,
            hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
        ))
        
        fig_cm.update_layout(
            xaxis_title='Predicted Label',
            yaxis_title='True Label',
            height=400,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_right:
        st.subheader("ROC Curve")
        st.write("Trade-off Analysis (Precision vs. Recall).")
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Interactive Plotly ROC Curve
        fig_roc = go.Figure()
        
        # ROC Curve
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.2f})',
            line=dict(color='darkorange', width=2),
            hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
        ))
        
        # Diagonal line
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='navy', dash='dash'),
            showlegend=True
        ))
        
        # Selected threshold point
        idx_03 = (np.abs(thresholds - 0.3)).argmin()
        fig_roc.add_trace(go.Scatter(
            x=[fpr[idx_03]], y=[tpr[idx_03]],
            mode='markers',
            name='Threshold 0.3',
            marker=dict(color='red', size=12),
            hovertemplate=f'Threshold: 0.3<br>FPR: {fpr[idx_03]:.3f}<br>TPR: {tpr[idx_03]:.3f}<extra></extra>'
        ))
        
        fig_roc.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(x=0.6, y=0.1)
        )
        
        st.plotly_chart(fig_roc, use_container_width=True)

    # 3. Global SHAP - Interactive
    st.subheader("Global Feature Importance (XAI)")
    st.write("Which factors drive the model's decisions globally?")
    
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        shap_values_risk = shap_values[1]
    else:
        shap_values_risk = shap_values[:, :, 1]
    
    # Calculate mean absolute SHAP values
    shap_importance = np.abs(shap_values_risk).mean(axis=0)
    feature_names = X_test.columns
    
    # Sort by importance
    indices = np.argsort(shap_importance)[::-1][:15]  # Top 15 features
    
    fig_shap = go.Figure(go.Bar(
        x=shap_importance[indices],
        y=[feature_names[i] for i in indices],
        orientation='h',
        marker=dict(
            color=shap_importance[indices],
            colorscale=[[0, '#1a1a2e'], [0.25, '#16213e'], [0.5, '#0f3460'], [0.75, '#533483'], [1, '#e94560']],
            showscale=True,
            colorbar=dict(title="Importance"),
            line=dict(color='rgba(255,255,255,0.2)', width=1)
        ),
        hovertemplate='%{y}<br>Importance: %{x:.4f}<extra></extra>'
    ))
    
    fig_shap.update_layout(
        xaxis_title='Mean |SHAP Value|',
        yaxis_title='Feature',
        height=500,
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis=dict(autorange="reversed")
    )
    
    st.plotly_chart(fig_shap, use_container_width=True)

# ==========================================
# PAGE 2: CUSTOMER SEGMENTATION 
# ==========================================
with tab2:
    st.subheader("🧩 Customer Market Segments")
    st.markdown("Unsupervised Analysis of the customer base using **K-Means**.")
    st.caption("**Model:** K-Means Clustering | **Data:** PCA Reduced")
    
    if cluster_brain is None:
        st.error("⚠️ 'cluster_brain.pkl' not found.")
        st.info("Please run the 'colleague_clustering.ipynb' notebook to generate the file.")
    else:
        # Daten entpacken
        plot_df = cluster_brain['pca_data']
        shap_values_global = cluster_brain['shap_values_global']
        raw_features = cluster_brain['raw_features']
        
        # Load original data with features for distribution plots
        clusters = plot_df['Cluster'].values
        feature_names = ['duration_months', 'credit_amount', 'age_years', 'installment_rate_percent', 
                        'present_residence_since', 'number_existing_credits', 'number_people_liable']
        feature_df = X_full[feature_names].iloc[:len(clusters)].copy()
        feature_df['Cluster'] = clusters
        feature_df.columns = ['Duration (months)', 'Credit Amount', 'Age (years)', 'Installment Rate (%)', 
                             'Residence Since', 'Existing Credits', 'People Liable', 'Cluster']
        
        t1, t2, t3, t4 = st.tabs(["2D Cluster Map", "3D Cluster Map", "Segment Distribution", "Cluster Meaning (SHAP)"])
        
        with t1:
            st.subheader("Customer Segments Map (2D)")
            st.markdown("Projection of customers into 2D space.")
            
            # Interactive 2D Scatter Plot
            fig_cluster_2d = px.scatter(
                plot_df, 
                x="PC1", 
                y="PC2", 
                color="Cluster",
                color_continuous_scale="viridis",
                hover_data={'PC1': ':.2f', 'PC2': ':.2f', 'Cluster': True},
                title="Identified Customer Groups"
            )
            
            fig_cluster_2d.update_traces(marker=dict(size=8, opacity=0.7))
            fig_cluster_2d.update_layout(
                height=600,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig_cluster_2d, use_container_width=True)
            
            st.info("""
            **Interpretation:**
            * **Cluster 0:** Standard Customers
            * **Cluster 1:** High Value / Investors
            * **Cluster 2:** Savers (Conservative)
            """)
        
        with t2:
            st.subheader("Customer Segments Map (3D)")
            st.markdown("Interactive 3D visualization - rotate and zoom to explore!")
            
            # Create 3D visualization
            if 'PC3' in plot_df.columns:
                # Use actual PC3 if available
                fig_cluster_3d = px.scatter_3d(
                    plot_df,
                    x='PC1',
                    y='PC2',
                    z='PC3',
                    color='Cluster',
                    color_continuous_scale='viridis',
                    hover_data={'PC1': ':.2f', 'PC2': ':.2f', 'PC3': ':.2f', 'Cluster': True},
                    title="3D Customer Segmentation (PCA Components)"
                )
                
                fig_cluster_3d.update_traces(marker=dict(size=5, opacity=0.7))
                fig_cluster_3d.update_layout(
                    height=700,
                    margin=dict(l=0, r=0, t=40, b=0),
                    scene=dict(
                        xaxis_title='PC1',
                        yaxis_title='PC2',
                        zaxis_title='PC3'
                    )
                )
            else:
                # Create 3D view using PC1, PC2, and cluster-based Z-axis
                st.info("💡 3D visualization created using PC1, PC2, and cluster separation on Z-axis")
                
                # Create a copy with Z-axis based on cluster
                plot_df_3d = plot_df.copy()
                # Add jitter to Z-axis based on cluster for visual separation
                np.random.seed(42)
                plot_df_3d['Z'] = plot_df_3d['Cluster'] * 2 + np.random.normal(0, 0.3, len(plot_df_3d))
                
                fig_cluster_3d = px.scatter_3d(
                    plot_df_3d,
                    x='PC1',
                    y='PC2',
                    z='Z',
                    color='Cluster',
                    color_continuous_scale='viridis',
                    hover_data={'PC1': ':.2f', 'PC2': ':.2f', 'Z': ':.2f', 'Cluster': True},
                    title="3D Customer Segmentation (Enhanced View)"
                )
                
                fig_cluster_3d.update_traces(marker=dict(size=5, opacity=0.7))
                fig_cluster_3d.update_layout(
                    height=700,
                    margin=dict(l=0, r=0, t=40, b=0),
                    scene=dict(
                        xaxis_title='PC1',
                        yaxis_title='PC2',
                        zaxis_title='Cluster Separation'
                    )
                )
            
            st.plotly_chart(fig_cluster_3d, use_container_width=True)
            
            # Add interaction tips
            st.markdown("""
            **🎮 Interaction Tips:**
            - **Rotate**: Click and drag to rotate the 3D view
            - **Zoom**: Scroll to zoom in/out
            - **Pan**: Right-click and drag to pan
            - **Reset**: Double-click to reset the view
            """)
            
        with t3:
            st.subheader("📊 Customer Distribution by Segment")
            st.markdown("Overview of customer counts across different segments.")
            
            # Calculate customer counts per cluster
            cluster_counts = plot_df['Cluster'].value_counts().sort_index()
            total_customers = len(plot_df)
            
            # Calculate percentages
            cluster_percentages = (cluster_counts / total_customers * 100).round(1)
            
            # Create DataFrame for plotting
            dist_df = pd.DataFrame({
                'Cluster': [f'Cluster {i}' for i in cluster_counts.index],
                'Count': cluster_counts.values,
                'Percentage': cluster_percentages.values
            })
            
            # Create interactive bar chart
            fig_dist = go.Figure()
            
            # Add bars with premium gradient colors
            premium_colors = ['#667eea', '#764ba2', '#f093fb']
            colors = premium_colors[:len(dist_df)]
            
            fig_dist.add_trace(go.Bar(
                x=dist_df['Cluster'],
                y=dist_df['Count'],
                text=[f"{count}<br>({pct}%)" for count, pct in zip(dist_df['Count'], dist_df['Percentage'])],
                textposition='outside',
                marker=dict(
                    color=colors,
                    line=dict(color='rgba(255,255,255,0.3)', width=2),
                    pattern=dict(shape="")
                ),
                hovertemplate='<b>%{x}</b><br>Customers: %{y}<br>Percentage: %{customdata}%<extra></extra>',
                customdata=dist_df['Percentage']
            ))
            
            fig_dist.update_layout(
                title="Customer Count per Segment",
                xaxis_title="Segment",
                yaxis_title="Number of Customers",
                height=500,
                margin=dict(l=20, r=20, t=60, b=20),
                showlegend=False
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Add summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Customers", f"{total_customers:,}")
            with col2:
                largest_cluster = cluster_counts.idxmax()
                st.metric("Largest Segment", f"Cluster {largest_cluster}", f"{cluster_counts.max()} customers")
            with col3:
                smallest_cluster = cluster_counts.idxmin()
                st.metric("Smallest Segment", f"Cluster {smallest_cluster}", f"{cluster_counts.min()} customers")
            
            # Add detailed table
            st.markdown("### Detailed Breakdown")
            
            # Create detailed table
            detailed_df = pd.DataFrame({
                'Segment': [f'Cluster {i}' for i in cluster_counts.index],
                'Customer Count': cluster_counts.values,
                'Percentage': [f"{pct}%" for pct in cluster_percentages.values],
                'Description': ['Standard Customers', 'High Value / Investors', 'Savers (Conservative)'][:len(cluster_counts)]
            })
            
            st.dataframe(detailed_df, use_container_width=True, hide_index=True)
            
        with t4:
            st.subheader("What defines the clusters?")
            st.markdown("SHAP Summary Plot - showing feature impact distribution for the selected cluster.")
            
            # WICHTIG: Sicherstellen, dass Cluster als int vorliegen für die Auswahl
            unique_clusters = sorted(plot_df['Cluster'].unique())
            cluster_id_view = st.selectbox("Select Cluster to Analyze:", unique_clusters)
            
            # Get SHAP values for selected cluster
            if isinstance(shap_values_global, list):
                sv_plot = shap_values_global[cluster_id_view]
            else:
                sv_plot = shap_values_global[:, :, cluster_id_view]
            
            # Create interactive SHAP summary plot
            st.markdown("**Interactive SHAP Summary Plot** - Each dot represents a customer")
            
            # Use the feature_names already defined above (line 210-211)
            # and get the actual feature values from X_full
            n_samples = len(plot_df)
            feature_values = X_full[feature_names].iloc[:n_samples].values
            
            # Now feature_values should be 2D with shape (n_samples, 7)
            # Calculate mean absolute SHAP values for sorting
            shap_importance = np.abs(sv_plot).mean(axis=0)
            top_indices = np.argsort(shap_importance)[::-1][:15]
            
            # Create interactive beeswarm plot
            fig_shap = go.Figure()
            
            for i, feat_idx in enumerate(top_indices):
                feat_name = feature_names[feat_idx]
                shap_vals = sv_plot[:, feat_idx]
                feat_vals = feature_values[:, feat_idx]
                
                # Normalize feature values for color (0-1)
                if feat_vals.max() != feat_vals.min():
                    feat_vals_norm = (feat_vals - feat_vals.min()) / (feat_vals.max() - feat_vals.min())
                else:
                    feat_vals_norm = np.zeros_like(feat_vals)
                
                # Add jitter for beeswarm effect
                np.random.seed(42 + i)  # Different seed per feature for consistency
                y_jitter = np.random.normal(0, 0.12, len(shap_vals))
                y_positions = np.full(len(shap_vals), i) + y_jitter
                
                # Add scatter trace
                fig_shap.add_trace(go.Scatter(
                    x=shap_vals,
                    y=y_positions,
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=feat_vals_norm,
                        colorscale='RdBu_r',  # Red=High, Blue=Low (reversed)
                        showscale=(i == 0),  # Show colorbar only once
                        colorbar=dict(
                            title=dict(
                                text="Feature<br>Value",
                                side="right"
                            ),
                            tickmode="array",
                            tickvals=[0, 1],
                            ticktext=["Low", "High"],
                            len=0.6,
                            y=0.5,
                            x=1.02
                        ),
                        cmin=0,
                        cmax=1,
                        line=dict(width=0.5, color='rgba(255,255,255,0.4)'),
                        opacity=0.8
                    ),
                    hovertemplate=(
                        f'<b>{feat_name}</b><br>' +
                        'SHAP: %{x:.4f}<br>' +
                        'Value: %{customdata:.2f}<br>' +
                        '<extra></extra>'
                    ),
                    customdata=feat_vals,
                    showlegend=False,
                    name=feat_name
                ))
            
            # Update layout
            fig_shap.update_layout(
                xaxis=dict(
                    title='SHAP Value (impact on cluster assignment)',
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor='rgba(255,255,255,0.3)',
                    gridcolor='rgba(255,255,255,0.1)',
                    color='white'
                ),
                yaxis=dict(
                    title='',
                    tickmode='array',
                    tickvals=list(range(len(top_indices))),
                    ticktext=[feature_names[i] for i in top_indices],
                    autorange='reversed',
                    color='white'
                ),
                height=600,
                margin=dict(l=180, r=80, t=20, b=60),
                hovermode='closest',
                plot_bgcolor='rgba(20,20,30,0.95)',
                paper_bgcolor='rgba(15,15,25,1)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig_shap, use_container_width=True)
            
            # Add explanation
            st.info("""
            **How to read this plot:**
            - Each dot represents one customer
            - **X-axis**: SHAP value (positive = pushes toward this cluster, negative = pushes away)
            - **Color**: Feature value (Red = High, Blue = Low)
            - **Position**: Features ranked by importance (top = most important)
            - **Hover** over dots to see exact values
            
            **Example**: If you see red dots on the right for "credit_amount", it means high credit amounts 
            strongly push customers into this cluster.
            """)

# ==========================================
# PAGE 3: LOAN OFFICER INTERFACE
# ==========================================
with tab3:
    st.subheader("🕵️‍♂️ Loan Application Assessment")
    st.markdown("AI-supported decision making for individual loan applications.")
    st.caption("**Model:** Random Forest | **Threshold:** 0.30")

    # Select Customer
    st.info("Select a test case below to simulate a real-time prediction.")
    customer_id = st.selectbox("Select Applicant ID:", options=range(len(X_test)))
    
    # Get Data
    customer_data = X_test.iloc[[customer_id]]
    
    # Live Prediction
    risk_prob = model.predict_proba(customer_data)[0][1]
    threshold = 0.3
    is_risk = risk_prob >= threshold
    
    st.markdown("### AI Recommendation")
    
    c1, c2 = st.columns([2, 1])
    with c1:
        if is_risk:
            st.error(f"🔴 **REJECT APPLICATION**")
            st.markdown(f"High risk of default detected.")
        else:
            st.success(f"🟢 **APPROVE APPLICATION**")
            st.markdown(f"Applicant meets creditworthiness criteria.")
            
    with c2:
        # Interactive Risk Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Default Risk %", 'font': {'size': 16}},
            delta={'reference': threshold * 100, 'increasing': {'color': "red"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkred" if is_risk else "darkgreen"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, threshold * 100], 'color': 'lightgreen'},
                    {'range': [threshold * 100, 100], 'color': 'lightcoral'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold * 100
                }
            }
        ))
        
        fig_gauge.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("---")
    
    # Explainability - Interactive Waterfall
    st.subheader("Why did the AI decide this?")
    st.write("Local explanation of positive (red) and negative (blue) factors:")
    
    # Prepare Waterfall Plot
    shap_values_single = explainer.shap_values(customer_data)
    if isinstance(shap_values_single, list):
        sv = shap_values_single[1][0]
        ev = explainer.expected_value[1]
    else:
        sv = shap_values_single[0, :, 1]
        ev = explainer.expected_value[1]
    
    # Sort by absolute SHAP value and take top features
    indices_waterfall = np.argsort(np.abs(sv))[::-1][:10]
    
    # Create waterfall data
    feature_names_waterfall = [X_full.columns[i] for i in indices_waterfall]
    shap_vals_waterfall = [sv[i] for i in indices_waterfall]
    feature_vals_waterfall = [customer_data.iloc[0, i] for i in indices_waterfall]
    
    # Build waterfall chart
    cumulative = ev
    x_data = ['Base Value']
    y_data = [ev]
    text_data = [f'{ev:.3f}']
    colors_data = ['lightgray']
    
    for i, (fname, sval, fval) in enumerate(zip(feature_names_waterfall, shap_vals_waterfall, feature_vals_waterfall)):
        x_data.append(f'{fname}<br>= {fval:.2f}')
        y_data.append(sval)
        text_data.append(f'{sval:+.3f}')
        colors_data.append('salmon' if sval > 0 else 'lightblue')
        cumulative += sval
    
    x_data.append('Final Prediction')
    y_data.append(cumulative)
    text_data.append(f'{cumulative:.3f}')
    colors_data.append('gold')
    
    fig_waterfall = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute"] + ["relative"] * len(feature_names_waterfall) + ["total"],
        x=x_data,
        y=y_data,
        text=text_data,
        textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "lightblue"}},
        increasing={"marker": {"color": "salmon"}},
        totals={"marker": {"color": "gold"}}
    ))
    
    fig_waterfall.update_layout(
        title="SHAP Waterfall - Top 10 Contributing Features",
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis={'tickangle': -45}
    )
    
    st.plotly_chart(fig_waterfall, use_container_width=True)
    
    # Show Raw Data
    with st.expander("View Applicant's Raw Data"):
        st.dataframe(customer_data.T)

# ==========================================
# PAGE 4: MANUAL PREDICTION
# ==========================================
with tab4:
    st.subheader("📝 Manual Credit Risk Assessment")
    st.markdown("Enter applicant details manually to get a risk prediction.")
    
    with st.form("prediction_form"):
        st.markdown("### 1. Financial Data")
        c1, c2, c3 = st.columns(3)
        with c1:
            chk_acct = st.selectbox("Checking Account Status", 
                ["A11: < 0 DM", "A12: 0 <= ... < 200 DM", "A13: >= 200 DM / Salary assignments", "A14: no checking account"])
            duration = st.number_input("Duration (months)", min_value=1, max_value=100, value=12)
            cred_hist = st.selectbox("Credit History", 
                ["A30: no credits taken/all paid back duly", "A31: all credits at this bank paid back duly", "A32: existing credits paid back duly till now", "A33: delay in paying off in the past", "A34: critical account/other credits existing"])
        with c2:
            amt = st.number_input("Credit Amount", min_value=100, value=1000)
            savings = st.selectbox("Savings Account/Bonds", 
                ["A61: < 100 DM", "A62: 100 <= ... < 500 DM", "A63: 500 <= ... < 1000 DM", "A64: >= 1000 DM", "A65: unknown/no savings account"])
            employment = st.selectbox("Present Employment Since", 
                ["A71: unemployed", "A72: < 1 year", "A73: 1 <= ... < 4 years", "A74: 4 <= ... < 7 years", "A75: >= 7 years"])
        with c3:
            install_rate = st.slider("Installment Rate (%)", 1, 4, 3)
            personal_status = st.selectbox("Personal Status & Sex", 
                ["A91: male : divorced/separated", "A92: female : divorced/separated/married", "A93: male : single", "A94: male : married/widowed", "A95: female : single"])
            guarantors = st.selectbox("Other Debtors / Guarantors", 
                ["A101: none", "A102: co-applicant", "A103: guarantor"])

        st.markdown("### 2. Personal & Asset Data")
        c4, c5, c6 = st.columns(3)
        with c4:
            residence = st.slider("Present Residence Since (years)", 1, 4, 2)
            property_type = st.selectbox("Property", 
                ["A121: real estate", "A122: building society savings agreement/life insurance", "A123: car or other", "A124: unknown / no property"])
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=30)
        with c5:
            other_install = st.selectbox("Other Installment Plans", 
                ["A141: bank", "A142: stores", "A143: none"])
            housing = st.selectbox("Housing", 
                ["A151: rent", "A152: own", "A153: for free"])
            credits_count = st.number_input("Number of Existing Credits", min_value=1, value=1)
        with c6:
            job = st.selectbox("Job", 
                ["A171: unemployed/unskilled - non-resident", "A172: unskilled - resident", "A173: skilled employee / official", "A174: management/ self-employed/ highly qualified employee/ officer"])
            liable = st.number_input("Number of People Liable", min_value=1, value=1)
            telephone = st.selectbox("Telephone", ["A191: none", "A192: yes, registered under the customers name"])
            foreign = st.selectbox("Foreign Worker", ["A201: yes", "A202: no"])
            
        st.markdown("### 3. Purpose")
        purpose = st.selectbox("Purpose", 
            ["A40: car (new)", "A41: car (used)", "A42: furniture/equipment", "A43: radio/television", "A44: domestic appliances", "A45: repairs", "A46: education", "A47: (vacation - does not exist?)", "A48: retraining", "A49: business", "A410: others"])

        submit_btn = st.form_submit_button("🔮 Predict Risk")

    if submit_btn:
        # Create a dictionary with default False/0
        input_data = {col: 0 for col in X_full.columns}
        
        # Fill Numerical
        input_data['duration_months'] = duration
        input_data['credit_amount'] = amt
        input_data['installment_rate_percent'] = install_rate
        input_data['present_residence_since'] = residence
        input_data['age_years'] = age
        input_data['number_existing_credits'] = credits_count
        input_data['number_people_liable'] = liable
        
        # Fill Categorical (Map selection to column name)
        def set_cat(selection, prefix):
            code = selection.split(":")[0]
            col_name = f"{prefix}_{code}"
            if col_name in input_data:
                input_data[col_name] = 1
        
        set_cat(chk_acct, "checking_account_status")
        set_cat(cred_hist, "credit_history")
        set_cat(purpose, "purpose")
        set_cat(savings, "savings_account_bonds")
        set_cat(employment, "present_employment_since")
        set_cat(personal_status, "personal_status_sex")
        set_cat(guarantors, "other_debtors_guarantors")
        set_cat(property_type, "property")
        set_cat(other_install, "other_installment_plans")
        set_cat(housing, "housing")
        set_cat(job, "job")
        set_cat(telephone, "telephone")
        set_cat(foreign, "foreign_worker")
        
        # Create DataFrame
        df_input = pd.DataFrame([input_data])
        
        # Predict
        risk_prob_manual = model.predict_proba(df_input)[0][1]
        is_risk_manual = risk_prob_manual >= 0.3
        
        # Display Result
        st.markdown("---")
        st.subheader("Prediction Result")
        
        mc1, mc2 = st.columns([2, 1])
        with mc1:
            if is_risk_manual:
                st.error(f"🔴 **REJECT APPLICATION**")
                st.markdown(f"High risk of default detected ({risk_prob_manual:.1%}).")
            else:
                st.success(f"🟢 **APPROVE APPLICATION**")
                st.markdown(f"Applicant meets creditworthiness criteria ({risk_prob_manual:.1%}).")
                
        with mc2:
            st.metric("Default Probability", f"{risk_prob_manual:.1%}", delta=f"Threshold: 30%")
            
        # SHAP for manual
        st.markdown("#### Key Drivers")
        shap_values_manual = explainer.shap_values(df_input)
        if isinstance(shap_values_manual, list):
            sv_man = shap_values_manual[1][0]
            ev_man = explainer.expected_value[1]
        else:
            sv_man = shap_values_manual[0, :, 1]
            ev_man = explainer.expected_value[1]
            
        # Waterfall Plot
        indices_man = np.argsort(np.abs(sv_man))[::-1][:10]
        feature_names_man = [X_full.columns[i] for i in indices_man]
        shap_vals_man = [sv_man[i] for i in indices_man]
        feature_vals_man = [df_input.iloc[0, i] for i in indices_man]
        
        cumulative = ev_man
        x_data = ['Base Value']
        y_data = [ev_man]
        text_data = [f'{ev_man:.3f}']
        colors_data = ['lightgray']
        
        for i, (fname, sval, fval) in enumerate(zip(feature_names_man, shap_vals_man, feature_vals_man)):
            x_data.append(f'{fname}<br>= {fval:.2f}')
            y_data.append(sval)
            text_data.append(f'{sval:+.3f}')
            colors_data.append('salmon' if sval > 0 else 'lightblue')
            cumulative += sval
        
        x_data.append('Final Prediction')
        y_data.append(cumulative)
        text_data.append(f'{cumulative:.3f}')
        colors_data.append('gold')
        
        fig_waterfall_man = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute"] + ["relative"] * len(feature_names_man) + ["total"],
            x=x_data,
            y=y_data,
            text=text_data,
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "lightblue"}},
            increasing={"marker": {"color": "salmon"}},
            totals={"marker": {"color": "gold"}}
        ))
        
        fig_waterfall_man.update_layout(
            title="SHAP Waterfall - Top 10 Contributing Features",
            height=500,
            margin=dict(l=20, r=20, t=60, b=20),
            xaxis={'tickangle': -45}
        )
        
        st.plotly_chart(fig_waterfall_man, use_container_width=True)

