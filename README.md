# 🪑 Smart Table Occupancy System

An end-to-end IoT-based system that monitors and manages real-time table occupancy in restaurants using sensors, cloud storage, and a live dashboard.

---

## 📌 Project Overview

The Smart Table Occupancy System automatically detects whether a table is occupied or available using PIR sensors connected to an ESP32. The data is stored in a cloud database and visualized using a real-time Streamlit dashboard. This system helps restaurants optimize seating utilization and analyze usage patterns.

---

## ✨ Features

- Real-time table occupancy detection  
- Live dashboard with grid-based table view  
- Automatic logging of occupancy duration  
- Auto-refresh every 5 seconds  
- Cloud storage using MongoDB Atlas  
- Modern UI with light and dark mode styling  

---

## 🚀 Getting Started

### 🔧 Prerequisites

**Hardware**
- ESP32 Development Board  
- PIR Motion Sensor (HC-SR501)  
- Breadboard  
- Jumper Wires  

**Software**
- Arduino IDE (ESP32 board installed)  
- Python 3.9 or above  

**Database**
- MongoDB Atlas (Free Tier)

---

## ⚙️ Installation & Setup

### 1️⃣ Hardware Setup (ESP32)

1. Open `firmware/smart_table.ino` in Arduino IDE  
2. Update WiFi credentials in the code  
3. Connect PIR Sensor:
   - VCC → 5V  
   - GND → GND  
   - OUT → GPIO 2  
4. Upload the code  
5. Note the IP address from Serial Monitor  

---

### 2️⃣ Database Setup

1. Create a MongoDB Atlas account  
2. Create database: `SmartTableDB`  
3. Create collections:
   - `table_status`
   - `occupancy_log`  
4. Allow Network Access from `0.0.0.0/0`  

---

### 3️⃣ Dashboard Setup

```bash
git clone <your-repository-url>
cd smart-table-occupancy
pip install streamlit pymongo requests pandas
```

---

### 4️⃣ Configure Secrets

Create `.streamlit/secrets.toml`:

```toml
MONGO_URI = "mongodb+srv://<username>:<password>@cluster.mongodb.net/"
```

---

### 5️⃣ Run the App

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```bash
smart-table-occupancy/
├── firmware/
│   └── smart_table.ino
├── .streamlit/
│   └── secrets.toml
├── app.py
├── requirements.txt
├── README.md
└── assets/
```

---

## 🧠 How It Works

1. PIR sensor detects human motion  
2. ESP32 updates table status  
3. ESP32 exposes a JSON API  
4. Streamlit fetches data every 5 seconds  
5. MongoDB stores live and historical data  
6. Dashboard displays table status  

---

## 🛠️ Technologies Used

- ESP32, PIR Sensor  
- Arduino (C++)  
- Python, Streamlit  
- MongoDB Atlas  
- REST API  

---

## 📚 Learning Outcomes

- End-to-end IoT system design  
- Real-time data synchronization  
- Cloud database integration  
- Web dashboard development  

---

## 📄 License

This project is for academic and educational purposes only.
