import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

accidents = pd.read_csv(r'C:\Users\HP\Downloads\Assignment1\Accidents.csv')
bikers = pd.read_csv(r'C:\Users\HP\Downloads\Assignment1\Bikers.csv')

df = pd.merge(accidents, bikers, on='Accident_Index', how='inner')

df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce').dt.hour
df['Weekday'] = df['Date'].dt.day_name()

print(f"Dataset shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"Date range: {df['Year'].min()} to {df['Year'].max()}")

print("\n" + "="*70)
print("DESCRIPTIVE STATISTICS")
print("="*70)

print("\nNumerical Variables:")
print(df[['Number_of_Vehicles', 'Number_of_Casualties', 'Speed_limit', 'Year']].describe())

print("\nSeverity Distribution:")
severity_dist = df['Severity'].value_counts()
for sev, count in severity_dist.items():
    pct = (count / len(df)) * 100
    print(f"  {sev}: {count:,} ({pct:.1f}%)")

print("\nGender Distribution:")
print(df['Gender'].value_counts())

print("\nAge Group Distribution:")
print(df['Age_Grp'].value_counts())

print("\nWeather Conditions:")
print(df['Weather_conditions'].value_counts())

print("\nRoad Surface Conditions:")
print(df['Road_conditions'].value_counts())

print("\nLight Conditions:")
print(df['Light_conditions'].value_counts())

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Bicycle Accidents in Great Britain (1979-2018): Exploratory Analysis', 
             fontsize=16, fontweight='bold', y=0.995)

yearly = df.groupby('Year').size()
axes[0, 0].plot(yearly.index, yearly.values, marker='o', linewidth=2.5, 
                color='#2E86AB', markersize=5)
axes[0, 0].fill_between(yearly.index, yearly.values, alpha=0.3, color='#2E86AB')
axes[0, 0].set_title('Annual Bicycle Accidents Trend', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('Year')
axes[0, 0].set_ylabel('Number of Accidents')
axes[0, 0].grid(alpha=0.3)
axes[0, 0].axhline(yearly.mean(), color='red', linestyle='--', alpha=0.5, label='Average')
axes[0, 0].legend()

severity_order = ['Fatal', 'Serious', 'Slight']
severity_counts = df['Severity'].value_counts().reindex(severity_order)
colors = ['#E63946', '#F77F00', '#06A77D']
bars = axes[0, 1].bar(severity_order, severity_counts.values, color=colors, 
                       edgecolor='black', linewidth=1.2)
axes[0, 1].set_title('Accident Severity Distribution', fontsize=13, fontweight='bold')
axes[0, 1].set_ylabel('Count')
for bar in bars:
    height = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom', fontweight='bold')

hourly = df.groupby('Hour').size()
axes[1, 0].bar(hourly.index, hourly.values, color='#A8DADC', edgecolor='#457B9D', linewidth=1)
axes[1, 0].set_title('Accidents by Hour of Day', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Hour (24-hour format)')
axes[1, 0].set_ylabel('Number of Accidents')
axes[1, 0].set_xticks(range(0, 24, 2))
axes[1, 0].grid(axis='y', alpha=0.3)

peak_hours = hourly.nlargest(3)
for hour in peak_hours.index:
    axes[1, 0].bar(hour, hourly[hour], color='#E63946', edgecolor='#457B9D', linewidth=1)

weather_counts = df['Weather_conditions'].value_counts().head(6)
explode = [0.05 if i == 0 else 0 for i in range(len(weather_counts))]
axes[1, 1].pie(weather_counts.values, labels=weather_counts.index, autopct='%1.1f%%',
                startangle=90, colors=sns.color_palette("Set2", len(weather_counts)),
                explode=explode)
axes[1, 1].set_title('Weather Conditions During Accidents', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("KEY INSIGHTS & RECOMMENDATIONS")
print("="*70)

print("\n1. SEVERITY ANALYSIS:")
fatal_count = (df['Severity'] == 'Fatal').sum()
serious_count = (df['Severity'] == 'Serious').sum()
fatal_pct = fatal_count / len(df) * 100
serious_pct = serious_count / len(df) * 100
print(f"   Fatal: {fatal_count:,} ({fatal_pct:.2f}%)")
print(f"   Serious: {serious_count:,} ({serious_pct:.2f}%)")
print(f"   Combined Fatal/Serious: {(fatal_pct + serious_pct):.2f}%")
print(f"   → Priority: Implement targeted interventions to reduce severe accidents")

print("\n2. TEMPORAL TREND:")
peak_year = yearly.idxmax()
lowest_year = yearly.idxmin()
recent_5yr = yearly.tail(5).mean()
early_5yr = yearly.head(5).mean()
trend_change = ((recent_5yr - early_5yr) / early_5yr) * 100
print(f"   Peak: {peak_year} ({yearly.max():,} accidents)")
print(f"   Lowest: {lowest_year} ({yearly.min():,} accidents)")
print(f"   40-year trend: {trend_change:+.1f}% change")
print(f"   → {'Strengthen safety measures' if trend_change > 0 else 'Maintain current policies'}")

print("\n3. TIME OF DAY PATTERNS:")
print(f"   Peak hours: {', '.join([f'{h}:00' for h in peak_hours.index])}")
print(f"   Highest risk: {peak_hours.idxmax()}:00 ({peak_hours.max():,} accidents)")
morning_rush = hourly.loc[7:9].sum()
evening_rush = hourly.loc[16:18].sum()
print(f"   Morning rush (7-9 AM): {morning_rush:,} accidents")
print(f"   Evening rush (4-6 PM): {evening_rush:,} accidents")
print(f"   → Deploy targeted patrols during peak commuting hours")

print("\n4. WEATHER & ROAD CONDITIONS:")
clear_weather_pct = (df['Weather_conditions'] == 'Clear').sum() / len(df) * 100
dry_road_pct = (df['Road_conditions'] == 'Dry').sum() / len(df) * 100
print(f"   Clear weather: {clear_weather_pct:.1f}% of accidents")
print(f"   Dry roads: {dry_road_pct:.1f}% of accidents")
print(f"   → Focus on driver behavior rather than infrastructure alone")

print("\n5. DEMOGRAPHIC INSIGHTS:")
male_pct = (df['Gender'] == 'Male').sum() / len(df) * 100
top_age = df['Age_Grp'].value_counts().index[0]
top_age_pct = df['Age_Grp'].value_counts().iloc[0] / len(df) * 100
print(f"   Male cyclists: {male_pct:.1f}% of accidents")
print(f"   Most affected age group: {top_age} ({top_age_pct:.1f}%)")
print(f"   → Design targeted education campaigns for high-risk demographics")

print("\n" + "="*70)