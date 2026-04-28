#include <Arduino.h>
#include <math.h>
#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEServer.h>
#include <FastLED.h>

// =====================================================
// BLE CONFIG
// =====================================================
const char* BLE_DEVICE_NAME = "ServoBLE";

#define SERVICE_UUID "11111111-2222-3333-4444-555555555555"
#define HR_CHAR_UUID "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
#define STATUS_CHAR_UUID "ffffffff-1111-2222-3333-444444444444"

BLECharacteristic* pHrChar = nullptr;
BLECharacteristic* pStatusChar = nullptr;
String currentPatternName = "UNKNOWN";

// =====================================================
// PINS
// =====================================================
const int SERVO_PIN  = 16;   // ESP32 PWM-capable pin
const int BUTTON_PIN = 5;    // Optional stop button
const unsigned long SERIAL_BAUD = 115200;

// hr_echo LED matrix
const int LED_DATA_PIN = 18;
const int NUM_LEDS = 64;
const int LED_BRIGHTNESS = 40;

// =====================================================
// SERVO / ACTUATOR SETTINGS (conservative defaults)
// =====================================================
int NEUTRAL_PULSE_US = 1498;
int INHALE_PULSE_US  = 1300;
int EXHALE_PULSE_US  = 1676;
const int UP_PULSE_US  = 900;
const int DOWN_PULSE_US = 2100;
const int SERVO_MIN_US_SAFE = 900;
const int SERVO_MAX_US_SAFE = 2100;

// =====================================================
// ESP32 LEDC PWM (Arduino-ESP32 core 3.x)
// =====================================================
const int SERVO_LEDC_CH       = 0;
const int SERVO_PWM_FREQ      = 50;
const int SERVO_PWM_RES_BITS  = 16;

static inline uint32_t usToDuty(int us) {
  const uint32_t maxDuty = (1UL << SERVO_PWM_RES_BITS) - 1;
  return (uint32_t)((((uint64_t)us) * maxDuty) / 20000UL);
}

static inline void servoWriteUs(int us) {
  if (us < SERVO_MIN_US_SAFE) us = SERVO_MIN_US_SAFE;
  if (us > SERVO_MAX_US_SAFE) us = SERVO_MAX_US_SAFE;
  ledcWrite((uint8_t)SERVO_PIN, usToDuty(us));
}

// =====================================================
// HR ECHO LED STATE
// =====================================================
#define HR_MIN_BPM  50.0f
#define HR_MAX_BPM  120.0f
#define HUE_WARM    0
#define HUE_COOL    160
#define LERP_SPEED  0.02f
#define LED_FRAME_MS 30

CRGB leds[NUM_LEDS];
float currentHue = HUE_WARM;
float targetHue = HUE_WARM;
unsigned long lastLedFrameMs = 0;
bool ledsEnabled = true;

float bpmToHue(float bpm) {
  float t = (bpm - HR_MIN_BPM) / (HR_MAX_BPM - HR_MIN_BPM);
  t = constrain(t, 0.0f, 1.0f);
  return HUE_WARM + t * (HUE_COOL - HUE_WARM);
}

void clearLedMatrix() {
  FastLED.clear(true);
}

void setLedTargetFromHr(float hr) {
  targetHue = bpmToHue(hr);
}

void updateLedMatrixNonBlocking() {
  unsigned long now = millis();
  if (now - lastLedFrameMs < LED_FRAME_MS) return;
  lastLedFrameMs = now;

  if (!ledsEnabled) {
    return;
  }

  currentHue += LERP_SPEED * (targetHue - currentHue);
  fill_solid(leds, NUM_LEDS, CHSV((uint8_t)currentHue, 220, 255));
  FastLED.show();
}

void setLedIdleWarm() {
  targetHue = HUE_WARM;
}

// =====================================================
// HEART RATE STATE (raw float input from Raspberry Pi)
// =====================================================
float heartRateBpm = 0.0f;
bool heartRateUpdated = false;

// =====================================================
// BREATHING TIMING (milliseconds)
// =====================================================
unsigned long inhaleMs = 5500;
unsigned long inhaleHoldMs = 0;
unsigned long exhaleMs = 5500;
unsigned long exhaleHoldMs = 0;
bool useSmoothWave = false;

// =====================================================
// TIMEOUT LOGIC
// =====================================================
unsigned long lastHrTime = 0;
const unsigned long HR_TIMEOUT = 5000;
bool timeoutDefaultActive = false;
const unsigned long PATTERN_CHANGE_UP_MS = 4000;
const unsigned long PATTERN_CHANGE_HOLD_MS = 500;
const unsigned long STARTUP_UP_MS = 0;
const unsigned long SHUTDOWN_DOWN_MS = 0;
const unsigned long SHUTDOWN_PAUSE_MS = 500;
bool pendingPatternSwitch = false;
bool pendingShutdown = false;
bool hrTimedOutToDefault = false;
bool applyPendingAfterOverride = false;
bool holdAfterOverrideActive = false;
bool applyPendingAfterHold = false;
unsigned long holdAfterOverrideEndMs = 0;
unsigned long pendingInhaleMs = 5500;
unsigned long pendingInhaleHoldMs = 0;
unsigned long pendingExhaleMs = 5500;
unsigned long pendingExhaleHoldMs = 0;
String pendingPatternName = "DEFAULT";
bool pendingInRecoveryMode = false;
int pendingRecoveryCyclesLeft = 0;
bool shutdownPauseActive = false;
unsigned long shutdownPauseEndMs = 0;
int pendingLastHrZone = 0;
int switchHandledCycleCounter = 0;

// =====================================================
// INPUT THROTTLE
// =====================================================
unsigned long lastCommandMs = 0;
const unsigned long CMD_MIN_INTERVAL_MS = 100;

// =====================================================
// SERIAL INPUT BUFFER
// =====================================================
String serialBuffer = "";
bool manualSerialHrLatched = false;

// =====================================================
// BUTTON
// =====================================================
int prevButtonState = HIGH;

// =====================================================
// RECOVERY MODE
// =====================================================
int lastHrZone = 0;
bool inRecoveryMode = false;
int recoveryCyclesLeft = 0;

// =====================================================
// EMERGENCY STOP
// =====================================================
volatile bool emergencyStop = false;

// =====================================================
// MANUAL OVERRIDE
// =====================================================
bool overrideActive = false;
unsigned long overrideEndMs = 0;
int overridePulseUs = 1500;
bool startDefaultAfterOverride = false;
bool startupPositioned = false;
bool systemRunning = false;
bool startupMoveActive = false;
unsigned long manualMoveDurationMs = 0;
unsigned long lastCountdownPrintMs = 0;
String manualMoveLabel = "";

// =====================================================
// BREATH ENGINE
// =====================================================
enum BreathState { INHALE, INHALE_HOLD, EXHALE, EXHALE_HOLD };
BreathState breathState = INHALE;
unsigned long stateStartMs = 0;
unsigned long lastServoUpdateMs = 0;
const unsigned long SERVO_UPDATE_MS = 20;
int cycleCounter = 0;
int lastCycleCounter = 0;

// =====================================================
// FORWARD DECLARATIONS
// =====================================================
void processIncomingValue(String input, bool fromSerial = false);
void resetHrStateOnStop();
void updateBreathingFromHeartRate(int hr);
void updateBreathingNonBlocking();
void resetBreathMachine();
void startBreathState(BreathState s);
void setPatternMs(unsigned long inhale, unsigned long inhaleHold,
                  unsigned long exhale, unsigned long exhaleHold);
void setDefaultPattern();
void setPatternFromHeartRate(int hr);
bool isFourSevenEightPattern(const String& patternName);
void buildPendingPatternFromHeartRate(int hr);
void applyPendingPatternIfAny();

// =====================================================
// UTIL
// =====================================================
void sendBleStatus(const String& msg) {
  if (pStatusChar == nullptr) return;
  pStatusChar->setValue(msg.c_str());
  pStatusChar->notify();
  Serial.print("BLE notify: ");
  Serial.println(msg);
}

int getHrZone(int hr) {
  if (hr <= 60)  return 1;
  if (hr <= 80)  return 2;
  if (hr <= 100) return 3;
  return 4;
}

static inline float smoothWave01(float p) {
  return 0.5f - 0.5f * cosf(2.0f * PI * p);
}

void startBreathState(BreathState s) {
  breathState = s;
  stateStartMs = millis();
  if (s == INHALE_HOLD || s == EXHALE_HOLD) {
    servoWriteUs(NEUTRAL_PULSE_US);
  }
  lastServoUpdateMs = 0;
}

void resetBreathMachine() {
  breathState = INHALE;
  stateStartMs = millis();
  lastServoUpdateMs = 0;
}

// =====================================================
// HR -> BREATH MAP
// =====================================================
void setPatternMs(unsigned long inhale, unsigned long inhaleHold,
                  unsigned long exhale, unsigned long exhaleHold) {
  inhaleMs = inhale;
  inhaleHoldMs = inhaleHold;
  exhaleMs = exhale;
  exhaleHoldMs = exhaleHold;
}

void setDefaultPattern() {
  currentPatternName = "5.5 inhale, 5.5 exhale";
  setPatternMs(5500, 0, 5500, 0);
}

void setPatternFromHeartRate(int hr) {
  if (hr > 60 && hr < 80) {
    currentPatternName = "5.5 inhale, 5.5 exhale";
    setPatternMs(5500, 0, 5500, 0);
  } else if (hr >= 50 && hr < 60) {
    currentPatternName = "4 inhale, 7 hold, 8 exhale";
    setPatternMs(4000, 7000, 8000, 0);
  } else if (hr > 80 && hr < 100) {
    currentPatternName = "4 inhale, 6 exhale";
    setPatternMs(4000, 0, 6000, 0);
  } else if (hr > 100) {
    currentPatternName = "4 inhale, 4 hold, 4 exhale, 4 hold";
    setPatternMs(4000, 4000, 4000, 4000);
  } else {
    currentPatternName = "DEFAULT";
    setDefaultPattern();
  }
}

void setRecoveryMode() {
  currentPatternName = "Recovery 5.5 inhale, 2 hold, 5.5 exhale, 2 hold";
  setPatternMs(5500, 2000, 5500, 2000);
  inRecoveryMode = true;
  recoveryCyclesLeft = 3;
}

bool isFourSevenEightPattern(const String& patternName) {
  return patternName == "4 inhale, 7 hold, 8 exhale";
}

void buildPendingPatternFromHeartRate(int hr) {
  int zone = getHrZone(hr);
  pendingLastHrZone = zone;
  pendingInRecoveryMode = false;
  pendingRecoveryCyclesLeft = 0;

  if (lastHrZone == 4 && zone <= 2) {
    pendingPatternName = "Recovery 5.5 inhale, 2 hold, 5.5 exhale, 2 hold";
    pendingInhaleMs = 5500;
    pendingInhaleHoldMs = 2000;
    pendingExhaleMs = 5500;
    pendingExhaleHoldMs = 2000;
    pendingInRecoveryMode = true;
    pendingRecoveryCyclesLeft = 3;
    return;
  }

  if (hr > 60 && hr < 80) {
    pendingPatternName = "5.5 inhale, 5.5 exhale";
    pendingInhaleMs = 5500;
    pendingInhaleHoldMs = 0;
    pendingExhaleMs = 5500;
    pendingExhaleHoldMs = 0;
  } else if (hr >= 50 && hr < 60) {
    pendingPatternName = "4 inhale, 7 hold, 8 exhale";
    pendingInhaleMs = 4000;
    pendingInhaleHoldMs = 7000;
    pendingExhaleMs = 8000;
    pendingExhaleHoldMs = 0;
  } else if (hr > 80 && hr < 100) {
    pendingPatternName = "4 inhale, 6 exhale";
    pendingInhaleMs = 4000;
    pendingInhaleHoldMs = 0;
    pendingExhaleMs = 6000;
    pendingExhaleHoldMs = 0;
  } else if (hr > 100) {
    pendingPatternName = "4 inhale, 4 hold, 4 exhale, 4 hold";
    pendingInhaleMs = 4000;
    pendingInhaleHoldMs = 4000;
    pendingExhaleMs = 4000;
    pendingExhaleHoldMs = 4000;
  } else {
    pendingPatternName = "5.5 inhale, 5.5 exhale";
    pendingInhaleMs = 5500;
    pendingInhaleHoldMs = 0;
    pendingExhaleMs = 5500;
    pendingExhaleHoldMs = 0;
  }
}

void applyPendingPatternIfAny() {
  if (!pendingPatternSwitch) return;
  currentPatternName = pendingPatternName;
  setPatternMs(pendingInhaleMs, pendingInhaleHoldMs, pendingExhaleMs, pendingExhaleHoldMs);
  inRecoveryMode = pendingInRecoveryMode;
  recoveryCyclesLeft = pendingRecoveryCyclesLeft;
  lastHrZone = pendingLastHrZone;
  pendingPatternSwitch = false;
}

void updateBreathingFromHeartRate(int hr) {
  int zone = getHrZone(hr);

  if (lastHrZone == 4 && zone <= 2) {
    setRecoveryMode();
  } else if (!inRecoveryMode) {
    setPatternFromHeartRate(hr);
  }

  lastHrZone = zone;

  Serial.print("HR=");
  Serial.print(hr);
  Serial.print(" zone=");
  Serial.print(zone);
  Serial.print(" inhale ");
  Serial.print(inhaleMs / 1000.0f, 1);
  if (inhaleHoldMs) { Serial.print("+"); Serial.print(inhaleHoldMs / 1000.0f, 1); }
  Serial.print(" exhale ");
  Serial.print(exhaleMs / 1000.0f, 1);
  if (exhaleHoldMs) { Serial.print("+"); Serial.print(exhaleHoldMs / 1000.0f, 1); }
  if (inRecoveryMode) {
    Serial.print(" [RECOVERY left=");
    Serial.print(recoveryCyclesLeft);
    Serial.print("]");
  }
  Serial.println();
}

void resetHrStateOnStop() {
  heartRateBpm = 0.0f;
  heartRateUpdated = false;
  manualSerialHrLatched = false;
  setDefaultPattern();
  inRecoveryMode = false;
  recoveryCyclesLeft = 0;
  lastHrZone = 0;
  overrideActive = false;
  overridePulseUs = NEUTRAL_PULSE_US;
  startDefaultAfterOverride = false;
  timeoutDefaultActive = false;
  pendingPatternSwitch = false;
  pendingShutdown = false;
  hrTimedOutToDefault = false;
  applyPendingAfterOverride = false;
  holdAfterOverrideActive = false;
  applyPendingAfterHold = false;
  holdAfterOverrideEndMs = 0;
  shutdownPauseActive = false;
  shutdownPauseEndMs = 0;
  setLedIdleWarm();
  resetBreathMachine();
}

void updateBreathingNonBlocking() {
  unsigned long now = millis();

  if (emergencyStop) {
    servoWriteUs(NEUTRAL_PULSE_US);
    return;
  }

  if (overrideActive) {
    if (manualMoveDurationMs > 0) {
      if (lastCountdownPrintMs == 0 || now - lastCountdownPrintMs >= 1000) {
        long remainingMs = (long)(overrideEndMs - now);
        if (remainingMs < 0) remainingMs = 0;
        int remainingSec = (remainingMs + 999) / 1000;
        Serial.print(manualMoveLabel);
        Serial.print(" countdown: ");
        Serial.println(remainingSec);
        lastCountdownPrintMs = now;
      }
    }

    if (now >= overrideEndMs) {
      overrideActive = false;
      servoWriteUs(NEUTRAL_PULSE_US);
      manualMoveDurationMs = 0;
      lastCountdownPrintMs = 0;
      manualMoveLabel = "";

      if (applyPendingAfterOverride) {
        applyPendingAfterOverride = false;
        holdAfterOverrideActive = true;
        applyPendingAfterHold = true;
        holdAfterOverrideEndMs = now + PATTERN_CHANGE_HOLD_MS;
        return;
      }
      if (startDefaultAfterOverride) {
        startDefaultAfterOverride = false;
        systemRunning = true;
        setDefaultPattern();
        resetBreathMachine();
      } else {
        if (startupMoveActive) {
          startupMoveActive = false;
          startupPositioned = true;
          Serial.println("STARTUP positioning complete. Type 1 or GO to start the program.");
        }
        servoWriteUs(NEUTRAL_PULSE_US);
      }
    } else {
      servoWriteUs(overridePulseUs);
    }
    return;
  }

  if (holdAfterOverrideActive) {
    servoWriteUs(NEUTRAL_PULSE_US);
    if (now >= holdAfterOverrideEndMs) {
      holdAfterOverrideActive = false;
      if (applyPendingAfterHold) {
        applyPendingAfterHold = false;
        applyPendingPatternIfAny();
      }
      resetBreathMachine();
    }
    return;
  }

  if (now - lastServoUpdateMs < SERVO_UPDATE_MS) return;
  lastServoUpdateMs = now;

  unsigned long durMs = 0;
  int dir = 0;
  switch (breathState) {
    case INHALE:      durMs = inhaleMs;      dir = -1; break;
    case INHALE_HOLD: durMs = inhaleHoldMs;  dir =  0; break;
    case EXHALE:      durMs = exhaleMs;      dir = +1; break;
    case EXHALE_HOLD: durMs = exhaleHoldMs;  dir =  0; break;
  }
  unsigned long elapsed = now - stateStartMs;

  if (durMs == 0 || elapsed >= durMs) {
    switch (breathState) {
      case INHALE:
        startBreathState(inhaleHoldMs > 0 ? INHALE_HOLD : EXHALE);
        break;
      case INHALE_HOLD:
        startBreathState(EXHALE);
        break;
      case EXHALE:
        startBreathState(exhaleHoldMs > 0 ? EXHALE_HOLD : INHALE);
        if (exhaleHoldMs == 0) cycleCounter++;
        break;
      case EXHALE_HOLD:
        startBreathState(INHALE);
        cycleCounter++;
        break;
    }
    return;
  }

  int pulse = NEUTRAL_PULSE_US;
  if (dir == 0) {
    pulse = NEUTRAL_PULSE_US;
  } else if (!useSmoothWave) {
    pulse = (dir < 0) ? INHALE_PULSE_US : EXHALE_PULSE_US;
  } else {
    float p = (durMs > 0) ? (float)elapsed / (float)durMs : 1.0f;
    if (p < 0) p = 0;
    if (p > 1) p = 1;
    float w = smoothWave01(p);
    if (dir < 0) {
      pulse = NEUTRAL_PULSE_US - (int)((NEUTRAL_PULSE_US - INHALE_PULSE_US) * w);
    } else {
      pulse = NEUTRAL_PULSE_US + (int)((EXHALE_PULSE_US - NEUTRAL_PULSE_US) * w);
    }
  }

  servoWriteUs(pulse);
}

void processIncomingValue(String input, bool fromSerial) {
  input.trim();
  input.toUpperCase();
  Serial.print("PROCESSING INPUT: ");
  Serial.println(input);

  if (input == "STOP" || input == "0") {
    Serial.println("STOP received");
    emergencyStop = true;
    resetHrStateOnStop();
    return;
  }

  if (input == "GO" || input == "1") {
    Serial.println("GO received");
    emergencyStop = false;
    if (!startupPositioned) {
      Serial.println("Type STARTUP first to move into position.");
      return;
    }
    systemRunning = true;
    pendingPatternSwitch = false;
    applyPendingAfterOverride = false;
    holdAfterOverrideActive = false;
    applyPendingAfterHold = false;
    pendingShutdown = false;
    setDefaultPattern();
    resetBreathMachine();
    Serial.println("Program started.");
    return;
  }

  if (input == "STARTUP") {
    Serial.println("STARTUP received");
    resetHrStateOnStop();
    emergencyStop = false;
    systemRunning = false;
    startupPositioned = false;
    startupMoveActive = true;
    startDefaultAfterOverride = false;
    overrideActive = true;
    overridePulseUs = UP_PULSE_US;
    overrideEndMs = millis() + STARTUP_UP_MS;
    return;
  }

  if (input == "SHUTDOWN") {
    Serial.println("SHUTDOWN received");
    if (!systemRunning) {
      Serial.println("System is already idle.");
      return;
    }
    if (!hrTimedOutToDefault || currentPatternName != "5.5 inhale, 5.5 exhale") {
      Serial.println("Shutdown is only allowed after HR timeout while on the default pattern.");
      return;
    }
    pendingShutdown = true;
    Serial.println("Shutdown queued. Will shut down at the end of the current default cycle.");
    return;
  }

  float bpm = input.toFloat();

  if (input.startsWith("UP")) {
    String numPart = input.substring(2);
    numPart.trim();
    int seconds = numPart.toInt();
    if (seconds < 1 || seconds > 50) {
      Serial.println("Invalid UP duration. Use 'UP 1' to 'UP 50'.");
      return;
    }
    Serial.print("UP: interrupting current state and moving INHALE direction for ");
    Serial.print(seconds);
    Serial.println("s");
    emergencyStop = false;
    pendingPatternSwitch = false;
    pendingShutdown = false;
    applyPendingAfterOverride = false;
    holdAfterOverrideActive = false;
    applyPendingAfterHold = false;
    startDefaultAfterOverride = false;
    overrideActive = true;
    overridePulseUs = UP_PULSE_US;
    overrideEndMs = millis() + (unsigned long)seconds * 1000UL;
    manualMoveDurationMs = (unsigned long)seconds * 1000UL;
    lastCountdownPrintMs = 0;
    manualMoveLabel = "UP";
    return;
  }

  if (input.startsWith("DOWN")) {
    String numPart = input.substring(4);
    numPart.trim();
    int seconds = numPart.toInt();
    if (seconds < 1 || seconds > 50) {
      Serial.println("Invalid DOWN duration. Use 'DOWN 1' to 'DOWN 50'.");
      return;
    }
    Serial.print("DOWN: interrupting current state and moving EXHALE direction for ");
    Serial.print(seconds);
    Serial.println("s");
    emergencyStop = false;
    pendingPatternSwitch = false;
    pendingShutdown = false;
    applyPendingAfterOverride = false;
    holdAfterOverrideActive = false;
    applyPendingAfterHold = false;
    startDefaultAfterOverride = false;
    overrideActive = true;
    overridePulseUs = DOWN_PULSE_US;
    overrideEndMs = millis() + (unsigned long)seconds * 1000UL;
    manualMoveDurationMs = (unsigned long)seconds * 1000UL;
    lastCountdownPrintMs = 0;
    manualMoveLabel = "DOWN";
    return;
  }

  if (bpm <= 0.0f || bpm >= 250.0f) {
    Serial.print("Invalid HR input: ");
    Serial.println(input);
    return;
  }

  if (heartRateBpm > 0.0f && fabsf(bpm - heartRateBpm) > 30.0f) {
    Serial.print("Spike rejected: ");
    Serial.println(bpm, 1);
    return;
  }

  heartRateBpm = bpm;
  lastHrTime = millis();
  emergencyStop = false;
  manualSerialHrLatched = fromSerial;
  hrTimedOutToDefault = false;
  timeoutDefaultActive = false;
  setLedTargetFromHr(heartRateBpm);

  if (!systemRunning) {
    heartRateUpdated = false;
    Serial.print("LED-only HR update: ");
    Serial.println(heartRateBpm, 1);
    if (startupPositioned) {
      Serial.println("System positioned and idle. Type 1 or GO to start the servo program.");
    } else {
      Serial.println("Servo program idle. LED matrix is active; type STARTUP to position the servo.");
    }
    return;
  }

  String previousPattern = currentPatternName;
  heartRateUpdated = true;

  if (!pendingPatternSwitch) {
    buildPendingPatternFromHeartRate((int)roundf(heartRateBpm));
    bool switchedPatterns = (pendingPatternName != previousPattern);
    if (switchedPatterns) {
      pendingPatternSwitch = true;
      sendBleStatus("VALUE=" + String(heartRateBpm, 1) + ",QUEUED=" + pendingPatternName);
    } else {
      pendingPatternSwitch = false;
      lastHrZone = pendingLastHrZone;
      sendBleStatus("VALUE=" + String(heartRateBpm, 1) + ",PATTERN=" + currentPatternName);
    }
  } else {
    sendBleStatus("VALUE=" + String(heartRateBpm, 1) + ",WAITING_FOR_CYCLE_END");
  }

  Serial.print("DISPLAY HR: ");
  Serial.println(heartRateBpm, 1);
}

class HrCharacteristicCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic* pCharacteristic) override {
    String value = String(pCharacteristic->getValue().c_str());
    value.trim();
    if (value.length() == 0) return;
    Serial.print("BLE write: ");
    Serial.println(value);
    sendBleStatus("RECEIVED=" + value);
    processIncomingValue(value, false);
  }
};

void readHeartRateFromSerial() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (serialBuffer.length() > 0) {
        String input = serialBuffer;
        serialBuffer = "";
        input.trim();

        if (input.length() == 0) {
          continue;
        }

        String upperInput = input;
        upperInput.toUpperCase();

        bool isCommand =
          upperInput == "STARTUP" ||
          upperInput == "GO" ||
          upperInput == "STOP" ||
          upperInput == "SHUTDOWN" ||
          upperInput == "1" ||
          upperInput == "0" ||
          upperInput.startsWith("UP") ||
          upperInput.startsWith("DOWN");

        Serial.print("Serial input: ");
        Serial.println(input);

        if (isCommand) {
          processIncomingValue(input, true);
        } else {
          float value = input.toFloat();

          if (value == 0.0f && input != "0.0" && input != "0") {
            Serial.print("ERROR: Invalid float: ");
            Serial.println(input);
          } else {
            Serial.print("ACK: ");
            Serial.println(value, 1);
            processIncomingValue(String(value, 1), true);
          }
        }
      }
    } else {
      serialBuffer += c;
      if (serialBuffer.length() > 64) serialBuffer = "";
    }
  }
}


void setup() {
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  Serial.begin(SERIAL_BAUD);
  Serial.println("Breathing Servo + hr_echo (ESP32, non-blocking) starting...");

  bool ok = ledcAttachChannel((uint8_t)SERVO_PIN, (uint32_t)SERVO_PWM_FREQ,
                              (uint8_t)SERVO_PWM_RES_BITS, (int8_t)SERVO_LEDC_CH);
  if (!ok) {
    Serial.println("ERROR: ledcAttachChannel failed!");
  }
  servoWriteUs(NEUTRAL_PULSE_US);

  FastLED.addLeds<WS2812, LED_DATA_PIN, RGB>(leds, NUM_LEDS).setCorrection(TypicalLEDStrip);
  FastLED.setBrightness(LED_BRIGHTNESS);
  setLedIdleWarm();
  fill_solid(leds, NUM_LEDS, CHSV((uint8_t)currentHue, 220, 255));
  FastLED.show();

  BLEDevice::init(BLE_DEVICE_NAME);
  BLEServer* pServer = BLEDevice::createServer();
  BLEService* pService = pServer->createService(SERVICE_UUID);

  pHrChar = pService->createCharacteristic(
    HR_CHAR_UUID,
    BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_READ
  );

  pStatusChar = pService->createCharacteristic(
    STATUS_CHAR_UUID,
    BLECharacteristic::PROPERTY_NOTIFY | BLECharacteristic::PROPERTY_READ
  );

  pHrChar->setCallbacks(new HrCharacteristicCallbacks());
  pHrChar->setValue("0");
  pStatusChar->setValue("READY");

  pService->start();
  BLEAdvertising* pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  BLEDevice::startAdvertising();

  Serial.println("BLE advertising as ServoBLE");

  setDefaultPattern();
  resetBreathMachine();
  systemRunning = false;
  startupPositioned = false;
  startDefaultAfterOverride = false;
}

void loop() {
  readHeartRateFromSerial();

  int reading = digitalRead(BUTTON_PIN);
  if (prevButtonState == HIGH && reading == LOW) {
    Serial.println("STOP button pressed!");
    emergencyStop = true;
    resetHrStateOnStop();
  }
  prevButtonState = reading;

  if (shutdownPauseActive) {
    if (millis() < shutdownPauseEndMs) {
      servoWriteUs(NEUTRAL_PULSE_US);
    } else {
      shutdownPauseActive = false;
      overrideActive = true;
      overridePulseUs = DOWN_PULSE_US;
      overrideEndMs = millis() + SHUTDOWN_DOWN_MS;
    }
    updateLedMatrixNonBlocking();
    return;
  }

  if (!systemRunning && !overrideActive && !holdAfterOverrideActive) {
    servoWriteUs(NEUTRAL_PULSE_US);
    updateLedMatrixNonBlocking();
    return;
  }

  bool timedOut = (systemRunning && !manualSerialHrLatched && lastHrTime != 0 && (millis() - lastHrTime) > HR_TIMEOUT);
  if (timedOut && !timeoutDefaultActive) {
    Serial.println("HR timeout -> default 5.5/5.5");
    setDefaultPattern();
    inRecoveryMode = false;
    recoveryCyclesLeft = 0;
    lastHrZone = 0;
    timeoutDefaultActive = true;
    hrTimedOutToDefault = true;
    pendingPatternSwitch = false;
    pendingShutdown = false;
    applyPendingAfterOverride = false;
    holdAfterOverrideActive = false;
    applyPendingAfterHold = false;
    lastHrTime = 0;
    heartRateBpm = 0.0f;
        emergencyStop = false;
    setLedIdleWarm();
    resetBreathMachine();
  }

  if (heartRateUpdated) {
    heartRateUpdated = false;
    if (heartRateBpm > 0.0f) updateBreathingFromHeartRate((int)roundf(heartRateBpm));
  }

  updateBreathingNonBlocking();

  if (cycleCounter != switchHandledCycleCounter) {
    switchHandledCycleCounter = cycleCounter;

    if (pendingShutdown && currentPatternName == "5.5 inhale, 5.5 exhale") {
      Serial.println("Completed default cycle -> shutting down");
      resetHrStateOnStop();
      emergencyStop = false;
      systemRunning = false;
      startupPositioned = false;
      startDefaultAfterOverride = false;
      pendingShutdown = false;
      shutdownPauseActive = true;
      shutdownPauseEndMs = millis() + SHUTDOWN_PAUSE_MS;
      servoWriteUs(NEUTRAL_PULSE_US);
      updateLedMatrixNonBlocking();
      return;
    }

    if (pendingPatternSwitch) {
      if (isFourSevenEightPattern(currentPatternName)) {
        Serial.println("Completed 4/7/8 cycle -> UP for 4s, then start queued pattern");
        overrideActive = true;
        overridePulseUs = INHALE_PULSE_US;
        overrideEndMs = millis() + PATTERN_CHANGE_UP_MS;
        applyPendingAfterOverride = true;
      } else {
        Serial.println("Completed cycle -> starting queued pattern");
        applyPendingPatternIfAny();
        resetBreathMachine();
      }
    }
  }

  if (inRecoveryMode && cycleCounter != lastCycleCounter) {
    lastCycleCounter = cycleCounter;
    recoveryCyclesLeft--;
    if (recoveryCyclesLeft <= 0) {
      inRecoveryMode = false;
      if (heartRateBpm > 0.0f) updateBreathingFromHeartRate((int)roundf(heartRateBpm));
      else setDefaultPattern();
    }
  }

  updateLedMatrixNonBlocking();
}
