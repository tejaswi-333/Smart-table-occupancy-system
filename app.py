import streamlit as st
import requests
import pymongo
import time
from datetime import datetime, timezone
import pandas as pd
from pymongo import MongoClient
import altair as alt

# --- NEW ML IMPORTS ---
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
# --- END NEW ML IMPORTS ---

 
# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Smart Table Occupancy",
    page_icon="🪑",
    layout="wide"
)

# --- 2. GLOBAL VARIABLES & CONSTANTS (UPDATED) ---
TOTAL_TABLES = 2 # <-- CHANGED from 10 to 2
TABLE_SENSORS = {
    "Table 1": "http://192.168.137.2/status",
    "Table 2": "http://192.168.137.2/table2", # <-- ADDED Table 2
}
DB_NAME = "smart_tables"
COLLECTION_STATUS = "table_status"
COLLECTION_LOG = "occupancy_log"

# --- 3. CUSTOM CSS STYLING ---
st.markdown("""
<style>
/* Main title */
h1 {
    font-weight: 600;
}

/* --- Analytics Metrics Box Styling --- */
.analytics-metric-box {
    background-color: #EEEEEE;
    border: 1px solid #E0E0E0;
    padding: 15px 20px;
    border-radius: 12px;
    color: #333333;
}
[data-theme="dark"] .analytics-metric-box {
    background-color: #333333; /* Dark gray background */
    border: 1px solid #444444;
    color: #FFFFFF; /* White text */
}
.analytics-metric-label {
    font-size: 0.9rem;
    color: #555555;
    margin-bottom: 5px;
}
.analytics-metric-value {
    font-size: 2rem;
    font-weight: 600;
    color: #333333;
}
[data-theme="dark"] .analytics-metric-label {
    color: #A0A0A0;
}
[data-theme="dark"] .analytics-metric-value {
    color: #FFFFFF;
}

/* --- Live Dashboard Metric Box Styling --- */
div[data-testid="stMetric"] {
    padding: 15px 20px;
    border-radius: 12px;
}

/* --- Table Grid Cards --- */
.table-card {
    background-color: #FFFFFF;
    border: 1px solid #E0E0E0;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    text-align: left;
    height: 210px;
    margin-bottom: 15px;
}
[data-theme="dark"] .table-card {
    background-color: #262730;
    border: 1px solid #31333F;
    box-shadow: none;
}
.table-card h3 {
    margin-top: 0;
    margin-bottom: 5px;
    font-size: 1.25rem;
    color: #888888;
}
.table-card .table-number {
    font-size: 3.5rem;
    font-weight: 600;
    line-height: 1.1;
    margin-bottom: 15px;
    color: #111111;
}
[data-theme="dark"] div.table-card div.table-number {
    color: #FAFAFA !important;
}

# /* --- Duration Text --- */
# .table-card .duration-text {
#     font-size: 0.9rem;
#     color: #555555;
#     margin-top: 12px;
# }
# [data-theme="dark"] div.table-card div.duration-text {
#     color: #B0B0B0 !important;
# }

/* --- Status Pills (Unchanged) --- */
.status-pill {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: 500;
}
.status-occupied {
    background-color: #FCE8E6;
    color: #C7221F;
}
[data-theme="dark"] .status-occupied {
    background-color: #5C2524;
    color: #F8A9A8;
}
.status-unoccupied {
    background-color: #E6F4EA;
    color: #137333;
}
[data-theme="dark"] .status-unoccupied {
    background-color: #21432A;
    color: #A8DAB5;
}
.status-offline {
    background-color: #E8EAED;
    color: #5F6368;
}
[data-theme="dark"] .status-offline {
    background-color: #3C4043;
    color: #BDC1C6;
}
</style>
""", unsafe_allow_html=True)


# --- 4. MONGODB CONNECTION ---
@st.cache_resource
def init_connection():
    """Connect to MongoDB. Will be cached for the session."""
    try:
        uri = st.secrets["MONGO_URI"]
        client = MongoClient(uri)
        client.admin.command('ping')
        db = client[DB_NAME]
        return db, db[COLLECTION_STATUS], db[COLLECTION_LOG]
    except Exception as e:
        st.error(f"Failed to connect to MongoDB. Check 'secrets.toml' and network rules.")
        st.exception(e)
        return None, None, None

db, status_collection, log_collection = init_connection()


# --- 5. CORE HELPER FUNCTIONS ---

def get_live_status(url):
    """Pings a single ESP32 sensor for its JSON status."""
    try:
        response = requests.get(url, timeout=2.0)
        if response.status_code == 200:
            status = response.json().get("status")
            if status == "Occupied":
                return "Occupied"
            return "Unoccupied"
    except requests.RequestException:
        pass
    return "Offline"

def update_table_status(table_id, current_status):
    """
    Updates the 'table_status' collection and logs completed events.
    """
    if status_collection is None or log_collection is None:
        return

    last_doc = status_collection.find_one({"table_id": table_id})
    last_status = last_doc.get("status") if last_doc else "Unoccupied"
    
    now_utc = datetime.now(timezone.utc)
    
    update_doc = {
        "$set": {
            "status": current_status,
            "last_updated": now_utc
        }
    }
    
    if current_status == "Occupied" and last_status != "Occupied":
        update_doc["$set"]["occupied_start_time"] = now_utc
        
    if current_status != "Occupied" and last_status == "Occupied":
        if last_doc and last_doc.get("occupied_start_time"):
            start_time = last_doc["occupied_start_time"]
            end_time = now_utc
            
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
            
            duration = (end_time - start_time).total_seconds()
            
            if duration > 10: 
                log_collection.insert_one({
                    "table_id": table_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_seconds": duration
                })
        
        update_doc["$set"]["occupied_start_time"] = None

    status_collection.update_one(
        {"table_id": table_id},
        update_doc,
        upsert=True
    )

def format_duration(seconds):
    """Converts a duration in seconds into a MM:SS or HH:MM:SS string."""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    if hours > 0:
        return f"{hours:02}:{minutes:02}:{seconds:02}"
    else:
        return f"{minutes:02}:{seconds:02}"

def convert_df_to_csv(df):
    """Converts a Pandas DataFrame to a CSV string for download."""
    return df.to_csv(index=False).encode('utf-8')


# --- 6. DATA RETRIEVAL FUNCTIONS (Cached) ---

@st.cache_data(ttl=5)
def get_all_table_data():
    """Main data function for the live dashboard."""
    for table_id, url in TABLE_SENSORS.items():
        status = get_live_status(url)
        update_table_status(table_id, status)
        
    table_data = []
    occupied_count = 0
    now_utc = datetime.now(timezone.utc)
    
    all_db_docs = {doc['table_id']: doc for doc in status_collection.find()}
                
    # --- UPDATED: Loop from 1 to TOTAL_TABLES (which is 2) ---
    for i in range(1, TOTAL_TABLES + 1):
        table_id = f"Table {i}"
        doc = all_db_docs.get(table_id)
        
        status = "Unoccupied"
        duration_str = ""
        
        if doc:
            status = doc.get("status", "Unoccupied")
            start_time = doc.get("occupied_start_time")
            
            if status == "Occupied" and start_time:
                if start_time.tzinfo is None:
                    start_time = start_time.replace(tzinfo=timezone.utc)
                
                duration_seconds = (now_utc - start_time).total_seconds()
                duration_str = format_duration(duration_seconds)
                
        if status == "Occupied":
            occupied_count += 1
            
        table_data.append({
            "id": i,
            "name": table_id,
            "status": status,
            "duration_str": duration_str
        })
            
    unoccupied_count = TOTAL_TABLES - occupied_count
    
    return table_data, occupied_count, unoccupied_count

@st.cache_data(ttl=10)
def get_hourly_analytics():
    """
    Queries MongoDB to get the COUNT of occupancy events per hour,
    converted to IST (Asia/Kolkata) timezone.
    """
    if log_collection is None:
        return pd.DataFrame()

    ist_timezone = "Asia/Kolkata"

    pipeline = [
        {
            "$project": {
                "hour": {
                    "$hour": {
                        "date": "$start_time",
                        "timezone": ist_timezone
                    }
                }
            }
        },
        {"$group": {"_id": "$hour", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}}
    ]
    
    try:
        results = list(log_collection.aggregate(pipeline))
        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results).rename(columns={"_id": "hour"})
        all_hours = pd.DataFrame({"hour": range(24)})
        df = pd.merge(all_hours, df, on="hour", how="left").fillna(0)
        df['count'] = df['count'].astype(int)
        return df
    except Exception as e:
        st.error(f"Error running hourly analytics query: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=10)
def get_summary_analytics():
    """
    Queries the log collection for total event count and average duration.
    """
    if log_collection is None:
        return 0, 0.0

    try:
        total_events = log_collection.count_documents({})
        pipeline = [
            {"$group": {"_id": None, "avg_duration_seconds": {"$avg": "$duration_seconds"}}}
        ]
        result = list(log_collection.aggregate(pipeline))
        
        if not result or 'avg_duration_seconds' not in result[0]:
            avg_duration_seconds = 0
        else:
            avg_duration_seconds = result[0]['avg_duration_seconds']
        
        avg_duration_minutes = (avg_duration_seconds / 60)
        return total_events, avg_duration_minutes
    except Exception as e:
        st.error(f"Error getting summary analytics: {e}")
        return 0, 0.0


# --- TIMEZONE-AWARE HELPER FUNCTION ---
def make_datetime_aware(dt_series):
    """
    Robustly converts a Pandas Series of datetimes to timezone-aware UTC.
    Handles mixed naive and aware datetimes.
    """
    def convert_dt(dt):
        if dt.tzinfo is None:
            return dt.tz_localize('UTC') # Add UTC timezone if naive
        return dt.tz_convert('UTC') # Convert to UTC if already aware
    
    dt_series = pd.to_datetime(dt_series)
    return dt_series.apply(convert_dt)


# --- UPDATED DATA FUNCTION (WITH TIMEZONE FIX) ---
@st.cache_data(ttl=10)
def get_log_data_as_df():
    """
    Fetches all log data as a Pandas DataFrame for display.
    Converts times to IST (Asia/Kolkata) for the table.
    """
    if log_collection is None:
        return pd.DataFrame()
    
    try:
        cursor = log_collection.find({}, sort=[("start_time", -1)])
        df = pd.DataFrame(list(cursor))
        
        if df.empty:
            return pd.DataFrame()
            
        ist_tz = timezone(pd.Timedelta(hours=5, minutes=30))
            
        df['start_time'] = make_datetime_aware(df['start_time'])
        df['end_time'] = make_datetime_aware(df['end_time'])
        
        df['start_time_ist'] = df['start_time'].dt.tz_convert(ist_tz)
        df['end_time_ist'] = df['end_time'].dt.tz_convert(ist_tz)
        
        df['start_time'] = df['start_time_ist'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df['end_time'] = df['end_time_ist'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        df['duration_minutes'] = (df['duration_seconds'] / 60).round(2)
        df = df[['table_id', 'start_time', 'end_time', 'duration_minutes']]
        
        return df
    except Exception as e:
        st.error(f"Error fetching raw log data: {e}")
        return pd.DataFrame()


# --- NEW SYNTHETIC DATA FUNCTION ---
def generate_synthetic_data(num_events=500):
    """
    Generates a DataFrame of fake, realistic-looking occupancy start times.
    All timestamps are timezone-aware (UTC).
    """
    base_date = datetime.now(timezone.utc) - pd.Timedelta(days=90)
    random_days = np.random.randint(0, 90, num_events)
    hours_lunch = np.random.normal(loc=13, scale=2, size=int(num_events * 0.4))
    hours_dinner = np.random.normal(loc=19, scale=2.5, size=int(num_events * 0.6))
    random_hours = np.concatenate([hours_lunch, hours_dinner])
    random_hours = np.clip(random_hours, 0, 23)
    np.random.shuffle(random_hours)
    random_minutes = np.random.randint(0, 60, num_events)
    random_seconds = np.random.randint(0, 60, num_events)
    
    timestamps = []
    for i in range(num_events):
        event_time = base_date + pd.Timedelta(days=int(random_days[i]))
        event_time = event_time.replace(
            hour=int(random_hours[i]), 
            minute=int(random_minutes[i]), 
            second=int(random_seconds[i]),
            microsecond=0
        )
        timestamps.append(event_time)
        
    df = pd.DataFrame({'start_time': timestamps})
    return df

# --- MODIFIED ML DATA FUNCTIONS (WITH TIMEZONE FIX) ---

@st.cache_data(ttl=600) # Cache for 10 minutes
def get_training_data():
    """
    Fetches all log data and engineers features for training.
    If real data is sparse, it generates synthetic data.
    """
    if log_collection is not None:
        cursor = log_collection.find({}, {"_id": 0, "start_time": 1})
        df = pd.DataFrame(list(cursor))
    else:
        df = pd.DataFrame(columns=['start_time'])

    MIN_EVENTS_TO_TRAIN = 20
    
    if df.empty or len(df) < MIN_EVENTS_TO_TRAIN:
        st.info("Generating synthetic data to train the model...")
        df = generate_synthetic_data(num_events=500)
    
    try:
        ist_tz = timezone(pd.Timedelta(hours=5, minutes=30))
        df['start_time'] = make_datetime_aware(df['start_time'])
        df['start_time'] = df['start_time'].dt.tz_convert(ist_tz)
        
        df['hour_of_day'] = df['start_time'].dt.hour
        df['day_of_week'] = df['start_time'].dt.dayofweek # 0=Monday, 6=Sunday
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        hourly_counts = df.groupby(['day_of_week', 'hour_of_day', 'is_weekend']).size().reset_index(name='event_count')

        try:
            labels = ['Low', 'Medium', 'High']
            hourly_counts['busyness_level'] = pd.qcut(hourly_counts['event_count'], q=3, labels=labels, duplicates='drop')
        except ValueError:
            try:
                labels = ['Low', 'High']
                hourly_counts['busyness_level'] = pd.qcut(hourly_counts['event_count'], q=2, labels=labels, duplicates='drop')
            except Exception:
                return pd.DataFrame(), None

        X = hourly_counts[['hour_of_day', 'day_of_week', 'is_weekend']]
        y = hourly_counts['busyness_level']
        
        return X, y
        
    except Exception as e:
        st.error(f"Error preparing training data: {e}")
        return pd.DataFrame(), None

@st.cache_resource
def train_prediction_model(X, y):
    """
    Trains and caches the Random Forest model.
    """
    try:
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        if len(label_encoder.classes_) < 2:
            return None, None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        return model, label_encoder
        
    except Exception as e:
        st.warning(f"Failed to train model: {e}")
        return None, None

# --- END MODIFIED ML FUNCTIONS ---


# --- 7. DASHBOARD LAYOUT (UPDATED) ---

st.title("Smart Table Occupancy")

if db is None:
    st.error("Application is not connected to the database. Please check connection.")
else:
    tab1, tab2, tab3, tab4 = st.tabs(["🪑 Live Dashboard", "📊 Analytics", "🗃️ Raw Data Log", "🔮 Peak Predictor"])

    # --- Tab 1: Live Dashboard ---
    with tab1:
        table_data, occupied_count, unoccupied_count = get_all_table_data()

        # --- Top Metric Row (Custom HTML) ---
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(f"""<div data-testid="stMetric" style="background-color: #2684FF; border-color: #2684FF; color: white;"><div data-testid="stMetricLabel">TOTAL TABLES</div><div data-testid="stMetricValue">{TOTAL_TABLES}</div></div>""", unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""<div data-testid="stMetric" style="background-color: #E24C4B; border-color: #E24C4B; color: white;"><div data-testid="stMetricLabel">OCCUPIED</div><div data-testid="stMetricValue">{occupied_count}</div></div>""", unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"""<div data-testid="stMetric" style="background-color: #29A347; border-color: #29A347; color: white;"><div data-testid="stMetricLabel">UNOCCUPIED</div><div data-testid="stMetricValue">{unoccupied_count}</div></div>""", unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)

        # --- Table Grid (UPDATED) ---
        # Create 2 columns for 2 tables
        all_cols = st.columns(2)
        
        for i, table in enumerate(table_data):
            if i < len(all_cols):
                col = all_cols[i]
                
                status_text = "Occupied" if table['status'] == "Occupied" else "Unoccupied"
                status_class = "status-occupied" if table['status'] == "Occupied" else "status-unoccupied"
                if table['status'] == "Offline":
                    status_text = "Offline"
                    status_class = "status-offline"

                duration_html = ""
                if table['status'] == "Occupied" and table['duration_str']:
                    duration_html = f"""<div class="duration-text">Duration: <strong>{table['duration_str']}</strong></div>"""

                with col:
                    st.markdown(f"""
                        <div class="table-card">
                            <h3>{table['name']}</h3>
                            <div class="table-number">{table['id']}</div>
                            <div class="status-pill {status_class}"> ● {status_text}</div>
                            {duration_html}
                        </div>
                    """, unsafe_allow_html=True)
    
    # --- Tab 2: Analytics ---
    with tab2:
        st.subheader("Occupancy Analytics")
        
        if st.button("Refresh Analytics"):
            st.cache_data.clear()
            st.cache_resource.clear() # Clear model cache too
            st.rerun()
        
        st.write("---")
        total_events, avg_minutes = get_summary_analytics()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(
                f"""
                <div class="analytics-metric-box">
                    <div class="analytics-metric-label">Total Occupancy Events Logged</div>
                    <div class="analytics-metric-value">{total_events:d}</div>
                </div>
                """, unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                f"""
                <div class="analytics-metric-box">
                    <div class="analytics-metric-label">Average Occupancy Duration</div>
                    <div class="analytics-metric-value">{avg_minutes:.1f} minutes</div>
                </div>
                """, unsafe_allow_html=True
            )
            
        st.write("---")

        
        hourly_data = get_hourly_analytics()
        
        if hourly_data.empty or hourly_data['count'].sum() == 0:
            st.info("No occupancy log data has been recorded yet. Occupy a table (and then leave it) to log an event.")
        else:
            st.write("#### Occupancy Events by Hour of Day (IST)")
            st.write("This chart shows the **total number of times** a table became occupied, grouped by your local hour (0-23) in which the event started.")
            
            chart = alt.Chart(hourly_data).mark_bar().encode(
                x=alt.X('hour:O', title='Hour of Day (0-23)', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('count', title='Number of Occupancy Events'),
                tooltip=[
                    alt.Tooltip('hour:O', title='Hour'),
                    alt.Tooltip('count', title='Event Count')
                ]
            ).interactive()
            
            st.altair_chart(chart, width='stretch')

    # --- Tab 3: Raw Data Log ---
    with tab3:
        st.subheader("Historical Occupancy Log")
        
        log_df = get_log_data_as_df()
        
        if log_df.empty:
            st.info("No occupancy log data has been recorded yet.")
        else:
            st.write("This table shows all completed occupancy events, sorted by most recent. Timestamps are in IST.")
            
            csv_data = convert_df_to_csv(log_df)
            
            st.download_button(
                label="Download Data as CSV",
                data=csv_data,
                file_name="occupancy_log_ist.csv",
                mime="text/csv",
            )
            
            st.dataframe(log_df, use_container_width=True)

    # --- Tab 4: ML PREDICTOR ---
    with tab4:
        st.subheader("🔮 Peak Time Predictor")
        
        X, y = get_training_data()
        
        if X.empty or y is None:
            st.error("Error: Could not load or generate training data.")
        else:
            model, encoder = train_prediction_model(X, y)
            
            if model and encoder:
                st.write("Select a day and hour to predict its busyness level based on historical data.")
                
                day_map = {
                    0: "Monday", 1: "Tuesday", 2: "Wednesday", 
                    3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"
                }
                
                pred_col1, pred_col2 = st.columns(2)
                with pred_col1:
                    day_name = st.selectbox("Select a Day:", options=day_map.values())
                with pred_col2:
                    selected_hour = st.slider("Select an Hour (0-23):", 0, 23, 13)
                
                selected_day_of_week = list(day_map.keys())[list(day_map.values()).index(day_name)]
                selected_is_weekend = 1 if selected_day_of_week in [5, 6] else 0
                
                input_data = pd.DataFrame({
                    'hour_of_day': [selected_hour],
                    'day_of_week': [selected_day_of_week],
                    'is_weekend': [selected_is_weekend]
                })
                
                prediction_encoded = model.predict(input_data)
                prediction_label = encoder.inverse_transform(prediction_encoded)[0]
                
                st.write("#### Prediction:")
                if prediction_label == "High":
                    st.error(f"**{prediction_label} Busyness**")
                    st.write(f"Based on past data, **{day_name} at {selected_hour}:00** is predicted to be a **peak time**.")
                elif prediction_label == "Medium":
                    st.warning(f"**{prediction_label} Busyness**")
                    st.write(f"Based on past data, **{day_name} at {selected_hour}:00** is predicted to have **moderate** traffic.")
                else: # 'Low'
                    st.success(f"**{prediction_label} Busyness**")
                    st.write(f"Based on past data, **{day_name} at {selected_hour}:00** is predicted to be **not busy**.")
            
            else:
                st.info("Gathering more data... The prediction model will appear here once enough historical data is logged (e.g., ~10-15 events).")

# --- 8. AUTO-REFRESH CONTROL ---
st.sidebar.title("Controls")
if st.sidebar.checkbox("Enable Auto-Refresh (every 3 seconds)"): 
    st.cache_data.clear()
    time.sleep(3)
    st.rerun()


# TO RUN THE APP: python -m streamlit run app.py