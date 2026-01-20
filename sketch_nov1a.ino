/*
 * ESP32 Simple Web Server for *TWO* Table Occupancy Sensors
 *
 * This code does four things:
 * 1. Connects to your "tejureddy" WiFi.
 * 2. Reads *two* PIR sensors constantly.
 * 3. Controls *two* corresponding LEDs.
 * 4. Starts a web server with two different endpoints:
 * - /status (for Table 1)
 * - /table2 (for Table 2)
 */

// Include the built-in libraries
#include <WiFi.h>
#include <WebServer.h>

// --- 1. CONFIGURE YOUR SETTINGS ---

// Set your WiFi network credentials
const char* WIFI_SSID = " ";     // Your WiFi Name
const char* WIFI_PASSWORD = " "; // Your WiFi Password

// Define the pins for Table 1 (Existing)
const int pirPin1 = 27;  // GPIO 27 for PIR sensor 1
const int ledPin1 = 26;  // GPIO 26 for LED 1

// Define the pins for Table 2 (New)
const int pirPin2 = 25;  // GPIO 25 for PIR sensor 2
const int ledPin2 = 23;  // GPIO 23 for LED 2

// --- END OF CONFIGURATION ---

// Create a web server object on port 80
WebServer server(80);

// Variables to store the current sensor states
volatile int pirState1 = LOW;
volatile int pirState2 = LOW;

/**
 * @brief This function handles requests for Table 1
 */
void handleGetTable1Status() {
  Serial.println("Server: Received a request for /status (Table 1)");
  
  String jsonResponse = "";
  if (pirState1 == HIGH) {
    jsonResponse = "{\"status_code\": 200, \"status\": \"Occupied\"}";
  } else {
    jsonResponse = "{\"status_code\": 200, \"status\": \"Available\"}";
  }
  server.send(200, "application/json", jsonResponse);
}

/**
 * @brief This function handles requests for Table 2
 */
void handleGetTable2Status() {
  Serial.println("Server: Received a request for /table2 (Table 2)");
  
  String jsonResponse = "";
  if (pirState2 == HIGH) {
    jsonResponse = "{\"status_code\": 200, \"status\": \"Occupied\"}";
  } else {
    jsonResponse = "{\"status_code\": 200, \"status\": \"Available\"}";
  }
  server.send(200, "application/json", jsonResponse);
}


/**
 * @brief Standard Arduino setup function
 */
void setup() {
  Serial.begin(115200);
  delay(100);

  // Set the pin modes
  pinMode(pirPin1, INPUT);
  pinMode(ledPin1, OUTPUT);
  pinMode(pirPin2, INPUT);
  pinMode(ledPin2, OUTPUT);
  
  // Set the LEDs to LOW (off) initially
  digitalWrite(ledPin1, LOW);
  digitalWrite(ledPin2, LOW);

  // --- 1. Connect to WiFi ---
  Serial.println();
  Serial.print("Connecting to WiFi network: ");
  Serial.println(WIFI_SSID);
  
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("\nWiFi Connected!");
  Serial.print("ESP32 IP Address: ");
  Serial.println(WiFi.localIP()); // <-- This is the IP for your app
  Serial.println("------------------------------------");

  // --- 2. Calibrate Sensor ---
  // This delay calibrates both sensors at the same time
  Serial.println("Calibrating PIR sensors... Please wait (20 seconds).");
  digitalWrite(ledPin1, HIGH); // Turn on LEDs during calibration
  digitalWrite(ledPin2, HIGH);
  delay(20000); // 20-second delay for calibration
  digitalWrite(ledPin1, LOW);  // Turn off LEDs
  digitalWrite(ledPin2, LOW);
  Serial.println("Sensors ready!");
  Serial.println("------------------------------------");

  // --- 3. Start the Web Server ---
  
  // When the server gets a request for "/status", call "handleGetTable1Status"
  server.on("/status", HTTP_GET, handleGetTable1Status);
  
  // When the server gets a request for "/table2", call "handleGetTable2Status"
  server.on("/table2", HTTP_GET, handleGetTable2Status);

  // Start the server
  server.begin();
  Serial.println("HTTP server started.");
  Serial.print("Test Table 1: http://");
  Serial.print(WiFi.localIP());
  Serial.println("/status");
  Serial.print("Test Table 2: http://");
  Serial.print(WiFi.localIP());
  Serial.println("/table2");
  Serial.println("------------------------------------");
}

/**
 * @brief Standard Arduino loop function
 */
void loop() {
  // --- Task 1: Check Sensor 1 ---
  pirState1 = digitalRead(pirPin1);
  if (pirState1 == HIGH) {
    digitalWrite(ledPin1, HIGH);
  } else {
    digitalWrite(ledPin1, LOW);
  }

  // --- Task 2: Check Sensor 2 ---
  pirState2 = digitalRead(pirPin2);
  if (pirState2 == HIGH) {
    digitalWrite(ledPin2, HIGH);
  } else {
    digitalWrite(ledPin2, LOW);
  }

  // --- Task 3: Handle any incoming web requests ---
  server.handleClient();
}
