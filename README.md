# OFDM SIMO Smart Estimator

This project is a Python-based simulation of an OFDM communication system designed to evaluate and compare different channel estimation techniques. It implements both Single-Input Single-Output (SISO) and Single-Input Multiple-Output (SIMO) architectures, focusing on the performance of adaptive Kalman filtering versus standard Least Squares (LS) estimation under various fading conditions.

## Project Overview

The simulation models a complete PHY layer communication chain including random signal generation, channel coding, OFDM transmission, channel effects (AWGN and Rayleigh fading), and reception. It is designed to test signal integrity in dynamic environments, such as "walking" or "running" scenarios, where channel conditions drift over time.

### Key Features

* **Modulation Schemes:** Supports QPSK, 16-QAM, and 64-QAM with Gray coding.
* **Channel Estimation:**
  * **Least Squares (LS):** Standard static estimation for block pilots.
  * **Kalman Filter:** Adaptive tracking algorithm to estimate time-varying channels.
  * **Comb Pilots:** Uses linear interpolation for fast-fading environments.
* **SIMO Diversity:** Implements Maximum Ratio Combining (MRC) for a 1x2 receiver antenna configuration.
* **Error Correction:** Integrated Hamming (7,4) Forward Error Correction (FEC) for robust data transmission.
* **Diagnostics:** Includes tools to generate Bit Error Rate (BER) waterfall curves, phase tracking plots, and constellation diagrams.

## Project Structure

* **`main.py`**: The primary script for running audio transmission experiments. It processes an input `.wav` file, simulates the channel, and saves the received audio for subjective evaluation.
* **`system_test.py`**: A diagnostic suite that generates performance plots (BER waterfalls, bit rate efficiency) and verifies system stability.
* **`transmitter.py`**: Handles the transmit chain, including symbol mapping, pilot insertion (Block/Comb), and OFDM modulation (IFFT/CP).
* **`receiver.py`**: Handles the receive chain, including synchronization, channel estimation (LS/Kalman), SIMO combining, and decoding.

## Dependencies

* Python 3.x
* NumPy
* Matplotlib
* SciPy (for audio file I/O)

## Usage

To run the full audio transmission simulation:
```bash
python main.py
