import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="GB Bicycle Accidents", 
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    accidents = pd.read_csv('Accidents.csv')
    bikers = pd.read_csv('Bikers.csv')
    
    df = pd.merge(accidents, bikers, on='Accident_Index', how='inner')
    
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce').dt.hour
    df['Weekday'] = df['Date'].dt.day_name()
    
    return df

df = load_data()

st.title('Great Britain Bicycle Accidents Analysis')
st.markdown(f"**Dataset**: {len(df):,} accidents from {df['Year'].min()} to {df['Year'].max()}")

with st.sidebar:
    st.header("Filters")
    
    year_range = st.slider(
        "Years",
        min_value=int(df['Year'].min()),
        max_value=int(df['Year'].max()),
        value=(2010, 2018),
        step=1
    )
    
    severity = st.multiselect(
        "Severity",
        options=df['Severity'].unique(),
        default=df['Severity'].unique()
    )
    
    gender_filter = st.multiselect(
        "Gender",
        options=df['Gender'].dropna().unique(),
        default=df['Gender'].dropna().unique()
    )
    
    weather = st.multiselect(
        "Weather",
        options=df['Weather_conditions'].dropna().unique()[:10],
        default=df['Weather_conditions'].dropna().unique()[:5]
    )
    
    age_groups = st.multiselect(
        "Age Group",
        options=df['Age_Grp'].dropna().unique(),
        default=df['Age_Grp'].dropna().unique()
    )

filtered = df[
    (df['Year'] >= year_range[0]) & 
    (df['Year'] <= year_range[1]) &
    (df['Severity'].isin(severity)) &
    (df['Gender'].isin(gender_filter)) &
    (df['Weather_conditions'].isin(weather)) &
    (df['Age_Grp'].isin(age_groups))
]

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Accidents", f"{len(filtered):,}")
with col2:
    fatal = len(filtered[filtered['Severity'] == 'Fatal'])
    st.metric("Fatal", f"{fatal:,}")
with col3:
    serious = len(filtered[filtered['Severity'] == 'Serious'])
    st.metric("Serious", f"{serious:,}")
with col4:
    fatality_rate = (fatal / len(filtered) * 100) if len(filtered) > 0 else 0
    st.metric("Fatality Rate", f"{fatality_rate:.2f}%")
with col5:
    avg_casualties = filtered['Number_of_Casualties'].mean() if len(filtered) > 0 else 0
    st.metric("Avg Casualties", f"{avg_casualties:.2f}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Trends", "⚡ Risk Factors", "🗺️ Location", "👥 Demographics", "📊 Statistics"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        yearly = filtered.groupby('Year').size().reset_index(name='count')
        fig_year = px.line(yearly, x='Year', y='count', 
                          title='Yearly Trend',
                          markers=True)
        fig_year.update_traces(line_color='#1f77b4', line_width=3)
        fig_year.update_layout(height=350)
        st.plotly_chart(fig_year, use_container_width=True)
        
    with col2:
        monthly = filtered.groupby('Month').size().reset_index(name='count')
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly['month_name'] = monthly['Month'].apply(lambda x: month_names[x-1])
        
        fig_month = px.bar(monthly, x='month_name', y='count',
                          title='Seasonal Pattern',
                          color='count',
                          color_continuous_scale='Reds')
        fig_month.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_month, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        hourly = filtered.groupby('Hour').size().reset_index(name='count')
        fig_hour = px.area(hourly, x='Hour', y='count',
                          title='24-Hour Distribution',
                          color_discrete_sequence=['#ff7f0e'])
        fig_hour.update_layout(height=350)
        st.plotly_chart(fig_hour, use_container_width=True)
    
    with col4:
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday = filtered['Weekday'].value_counts().reindex(day_order)
        
        colors = ['#636EFA' if d not in ['Saturday', 'Sunday'] else '#EF553B' for d in day_order]
        fig_week = go.Figure(data=[
            go.Bar(x=day_order, y=weekday.values, marker_color=colors)
        ])
        fig_week.update_layout(title='Weekly Pattern', height=350)
        st.plotly_chart(fig_week, use_container_width=True)
    
    st.subheader("Long-term Analysis")
    
    yearly_severity = filtered.groupby(['Year', 'Severity']).size().unstack(fill_value=0)
    
    fig_trend = go.Figure()
    colors_map = {'Fatal': '#d62728', 'Serious': '#ff7f0e', 'Slight': '#2ca02c'}
    
    for severity_type in yearly_severity.columns:
        fig_trend.add_trace(go.Scatter(
            x=yearly_severity.index,
            y=yearly_severity[severity_type],
            mode='lines+markers',
            name=severity_type,
            line=dict(color=colors_map.get(severity_type, 'gray'), width=2)
        ))
    
    fig_trend.update_layout(
        title='Severity Trends Over Time',
        xaxis_title='Year',
        yaxis_title='Number of Accidents',
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)

    with col1:
        severity_dist = filtered['Severity'].value_counts()
        fig_sev = px.pie(
            values=severity_dist.values,
            names=severity_dist.index,
            title='Severity Distribution',
            color_discrete_map={'Fatal': '#d62728', 'Serious': '#ff7f0e', 'Slight': '#2ca02c'}
        )
        fig_sev.update_layout(height=400)
        st.plotly_chart(fig_sev, use_container_width=True)

    with col2:
        weather_top = filtered['Weather_conditions'].value_counts().head(8)
        fig_weather = px.bar(
            x=weather_top.values,
            y=weather_top.index,
            orientation='h',
            title='Weather Conditions',
            color=weather_top.values,
            color_continuous_scale='Blues'
        )
        fig_weather.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_weather, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        light = filtered['Light_conditions'].value_counts().head(6)
        fig_light = px.bar(
            x=light.index,
            y=light.values,
            title='Light Conditions',
            color=light.values,
            color_continuous_scale='Sunset'
        )
        fig_light.update_layout(height=350, showlegend=False, xaxis_tickangle=45)
        st.plotly_chart(fig_light, use_container_width=True)

    with col4:
        road_cond = filtered['Road_conditions'].value_counts().head(6)
        fig_road_cond = px.bar(
            x=road_cond.index,
            y=road_cond.values,
            title='Road Conditions',
            color=road_cond.values,
            color_continuous_scale='Greys'
        )
        fig_road_cond.update_layout(height=350, showlegend=False, xaxis_tickangle=45)
        st.plotly_chart(fig_road_cond, use_container_width=True)

    st.subheader("Risk Matrix")

    risk_data = filtered.groupby(['Weather_conditions', 'Light_conditions'])['Severity'].apply(
        lambda x: (x == 'Fatal').sum()
    ).unstack(fill_value=0)

    if len(risk_data) > 0 and len(risk_data.columns) > 0:
        risk_data = risk_data.head(10).iloc[:, :8]

        fig_heat = px.imshow(
            risk_data,
            labels=dict(x="Light Conditions", y="Weather Conditions", color="Fatal Accidents"),
            color_continuous_scale='YlOrRd',
            title="Fatal Accident Heatmap: Weather vs Light Conditions"
        )
        fig_heat.update_layout(height=500)
        st.plotly_chart(fig_heat, use_container_width=True)

with tab3:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        road_type = filtered['Road_type'].value_counts().head(5)
        fig_road = px.bar(
            y=road_type.index,
            x=road_type.values,
            orientation='h',
            title='Top 5 Road Types',
            color=road_type.values,
            color_continuous_scale='Viridis'
        )
        fig_road.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_road, use_container_width=True)
    
    with col2:
        speed = filtered['Speed_limit'].value_counts().sort_index()
        fig_speed = px.bar(
            x=speed.index,
            y=speed.values,
            title='Speed Limit Distribution',
            color=speed.values,
            color_continuous_scale='Reds'
        )
        fig_speed.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_speed, use_container_width=True)
    
    with col3:
        vehicles = filtered['Number_of_Vehicles'].value_counts().sort_index().head(10)
        fig_vehicles = px.bar(
            x=vehicles.index,
            y=vehicles.values,
            title='Vehicles Involved',
            color=vehicles.values,
            color_continuous_scale='Purples'
        )
        fig_vehicles.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_vehicles, use_container_width=True)
    
    st.subheader("Day of Week Analysis")
    
    day_stats = filtered.groupby('Day').agg({
        'Accident_Index': 'count',
        'Number_of_Casualties': 'sum',
        'Severity': lambda x: (x == 'Fatal').sum()
    }).rename(columns={'Accident_Index': 'Total_Accidents', 'Severity': 'Fatal_Count'})
    
    day_stats['Fatality_Rate'] = (day_stats['Fatal_Count'] / day_stats['Total_Accidents']) * 100
    
    fig_day_analysis = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Accidents by Day of Month', 'Fatality Rate by Day')
    )
    
    fig_day_analysis.add_trace(
        go.Bar(x=day_stats.index, y=day_stats['Total_Accidents'], name='Accidents'),
        row=1, col=1
    )
    
    fig_day_analysis.add_trace(
        go.Scatter(x=day_stats.index, y=day_stats['Fatality_Rate'], 
                  mode='lines+markers', name='Fatality Rate (%)', line=dict(color='red')),
        row=1, col=2
    )
    
    fig_day_analysis.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_day_analysis, use_container_width=True)

with tab4:
    col1, col2 = st.columns(2)
    
    with col1:
        age_dist = filtered['Age_Grp'].value_counts()
        fig_age = px.bar(
            x=age_dist.index,
            y=age_dist.values,
            title='Age Group Distribution',
            color=age_dist.values,
            color_continuous_scale='Viridis'
        )
        fig_age.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        gender_dist = filtered['Gender'].value_counts()
        fig_gender = px.pie(
            values=gender_dist.values,
            names=gender_dist.index,
            title='Gender Distribution',
            color_discrete_sequence=['#4FC3F7', '#F48FB1', '#B0BEC5']
        )
        fig_gender.update_layout(height=400)
        st.plotly_chart(fig_gender, use_container_width=True)
    
    st.subheader("Demographics vs Severity")
    
    col3, col4 = st.columns(2)
    
    with col3:
        age_severity = pd.crosstab(filtered['Age_Grp'], filtered['Severity'], normalize='index') * 100
        
        fig_age_sev = go.Figure()
        for col in age_severity.columns:
            fig_age_sev.add_trace(go.Bar(
                x=age_severity.index,
                y=age_severity[col],
                name=col
            ))
        
        fig_age_sev.update_layout(
            barmode='stack',
            title='Severity by Age Group (%)',
            yaxis_title='Percentage',
            xaxis_title='Age Group',
            height=400
        )
        st.plotly_chart(fig_age_sev, use_container_width=True)
    
    with col4:
        gender_severity = pd.crosstab(filtered['Gender'], filtered['Severity'])
        
        fig_gender_sev = px.bar(
            gender_severity.T,
            title='Accidents by Gender and Severity',
            barmode='group',
            height=400
        )
        st.plotly_chart(fig_gender_sev, use_container_width=True)
    
    st.subheader("Time Patterns by Demographics")
    
    demo_time = filtered.groupby(['Hour', 'Gender']).size().unstack(fill_value=0)
    
    fig_demo_time = go.Figure()
    for gender in demo_time.columns:
        fig_demo_time.add_trace(go.Scatter(
            x=demo_time.index,
            y=demo_time[gender],
            mode='lines',
            name=gender,
            stackgroup='one'
        ))
    
    fig_demo_time.update_layout(
        title='Hourly Distribution by Gender',
        xaxis_title='Hour of Day',
        yaxis_title='Number of Accidents',
        height=400
    )
    st.plotly_chart(fig_demo_time, use_container_width=True)

with tab5:
    st.subheader("Statistical Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Top Risk Factors**")
        
        fatal_weather = filtered[filtered['Severity'] == 'Fatal']['Weather_conditions'].value_counts().head()
        total_weather = filtered['Weather_conditions'].value_counts()
        fatal_pct = (fatal_weather / total_weather.reindex(fatal_weather.index)) * 100
        
        risk_df = pd.DataFrame({
            'Condition': fatal_pct.index,
            'Fatal %': fatal_pct.values.round(2)
        })
        st.dataframe(risk_df, hide_index=True)
    
    with col2:
        st.write("**Time Patterns**")
        
        patterns = {
            'Peak Hour': f"{filtered.groupby('Hour').size().idxmax()}:00",
            'Peak Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][filtered.groupby('Month').size().idxmax()-1],
            'Peak Day': filtered['Weekday'].value_counts().idxmax(),
            'Safest Day': filtered['Weekday'].value_counts().idxmin()
        }
        
        pattern_df = pd.DataFrame(list(patterns.items()), columns=['Metric', 'Value'])
        st.dataframe(pattern_df, hide_index=True)
    
    with col3:
        st.write("**Casualty Metrics**")
        
        metrics = {
            'Total Casualties': filtered['Number_of_Casualties'].sum(),
            'Avg per Accident': filtered['Number_of_Casualties'].mean(),
            'Max Casualties': filtered['Number_of_Casualties'].max(),
            'Avg Vehicles': filtered['Number_of_Vehicles'].mean()
        }
        
        metrics_df = pd.DataFrame([
            ['Total Casualties', f"{metrics['Total Casualties']:,}"],
            ['Avg per Accident', f"{metrics['Avg per Accident']:.2f}"],
            ['Max Casualties', f"{metrics['Max Casualties']}"],
            ['Avg Vehicles', f"{metrics['Avg Vehicles']:.2f}"]
        ], columns=['Metric', 'Value'])
        
        st.dataframe(metrics_df, hide_index=True)
    
    st.subheader("Trend Analysis")
    
    years_for_trend = filtered.groupby('Year').agg({
        'Accident_Index': 'count',
        'Number_of_Casualties': 'sum',
        'Number_of_Vehicles': 'mean',
        'Severity': lambda x: (x == 'Fatal').sum()
    }).rename(columns={'Accident_Index': 'Accidents', 'Severity': 'Fatal', 'Number_of_Vehicles': 'Avg_Vehicles'})
    
    years_for_trend['Fatality_Rate'] = (years_for_trend['Fatal'] / years_for_trend['Accidents']) * 100
    
    fig_multi = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Total Accidents', 'Total Casualties', 'Fatal Accidents', 'Fatality Rate (%)')
    )
    
    fig_multi.add_trace(
        go.Scatter(x=years_for_trend.index, y=years_for_trend['Accidents'], 
                  mode='lines+markers', name='Accidents', line=dict(color='blue')),
        row=1, col=1
    )
    
    fig_multi.add_trace(
        go.Scatter(x=years_for_trend.index, y=years_for_trend['Number_of_Casualties'], 
                  mode='lines+markers', name='Casualties', line=dict(color='green')),
        row=1, col=2
    )
    
    fig_multi.add_trace(
        go.Scatter(x=years_for_trend.index, y=years_for_trend['Fatal'], 
                  mode='lines+markers', name='Fatal', line=dict(color='red')),
        row=2, col=1
    )
    
    fig_multi.add_trace(
        go.Scatter(x=years_for_trend.index, y=years_for_trend['Fatality_Rate'], 
                  mode='lines+markers', name='Rate %', line=dict(color='orange')),
        row=2, col=2
    )
    
    fig_multi.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig_multi, use_container_width=True)
    
    st.info(f"""
    **Key Insights:**
    - Dataset spans {df['Year'].max() - df['Year'].min() + 1} years with {len(df):,} total accidents
    - Peak accident times: {patterns['Peak Hour']} hour, {patterns['Peak Month']} month
    - {patterns['Peak Day']} has the most accidents, {patterns['Safest Day']} is safest
    - Most affected age group: {filtered['Age_Grp'].value_counts().idxmax()}
    - Gender distribution: {filtered['Gender'].value_counts().idxmax()} ({(filtered['Gender'].value_counts().max()/len(filtered)*100):.1f}%)
    """)

st.sidebar.markdown("---")
st.sidebar.info("""
**Dashboard Features:**
- Real-time filtering
- 5 analysis categories
- 20+ interactive visualizations
- Statistical insights
""")

st.sidebar.markdown("---")
st.sidebar.caption("Data Source: GB Bicycle Accidents 1979-2018")