import numpy as np
import scipy.io.wavfile as wav
import transmitter as tx
import receiver as rx
from math import sqrt
import time
import os
from datetime import datetime

# GLOBAL CONFIG 
FFT_SIZE = 1024
CP_SIZE = 256
SYMBOLS_PER_PACKET = 100
INPUT_FILE = "test_audio.wav"

# experiment config 
SNR_DB = 30  # High SNR to highlight Fading differences
FADING_SEVERITY = 'High' # Options: 'Low', 'High'

def get_channel_params(severity):
    if severity == 'Low':
        return {'L_tap': 5, 'alpha': 1.0} # Static
    else:
        return {'L_tap': 50, 'alpha': 0.9048} # Fast Drift

def apply_channel(signal, snr_db, severity_params, channel_type):
    if channel_type == 'awgn':
        sig_pwr = np.mean(np.abs(signal)**2)
        n0 = sig_pwr / (10**(snr_db/10))
        return signal + (np.random.randn(len(signal))+1j*np.random.randn(len(signal))) * sqrt(n0/2)

    # Fading
    L = severity_params['L_tap']
    alpha = severity_params['alpha']
    
    h = (np.random.randn(L) + 1j*np.random.randn(L)) / sqrt(2*L)
    faded = np.zeros_like(signal, dtype=complex)
    
    sym_len = FFT_SIZE + CP_SIZE    
    num_syms = len(signal) // sym_len
    
    for i in range(num_syms):
        chunk = signal[i*sym_len : (i+1)*sym_len]
        conv_res = np.convolve(chunk, h, mode='full')
        faded[i*sym_len : (i+1)*sym_len] = conv_res[:len(chunk)]
        
        # Drift Update
        noise = (np.random.randn(L) + 1j*np.random.randn(L)) / sqrt(2*L)
        h = alpha * h + sqrt(1 - alpha**2 + 1e-12) * noise
        
    sig_pwr = np.mean(np.abs(faded)**2)
    n0 = sig_pwr / (10**(snr_db/10))
    return faded + (np.random.randn(len(signal))+1j*np.random.randn(len(signal))) * sqrt(n0/2)

def run_simulation_pass(all_bits, mod_scheme, method_name, channel_type, simo, pilot_pattern='Block', save_dir=None, fs=44100):
    # Setup Transmitter capacity
    bps = 2 if mod_scheme==0 else (4 if mod_scheme==1 else 6)
    
    if pilot_pattern == 'Block':
        data_subcarriers = FFT_SIZE
    else:
        data_subcarriers = FFT_SIZE - (FFT_SIZE // 4)

    raw_bits_per_sym = data_subcarriers * bps * 4 // 7
    packet_size_bits = raw_bits_per_sym * SYMBOLS_PER_PACKET
    
    rem = len(all_bits) % packet_size_bits
    padded_bits = all_bits if rem==0 else np.concatenate((all_bits, np.zeros(packet_size_bits - rem, dtype=int)))
    total_packets = len(padded_bits) // packet_size_bits
    
    received_bits_all = []
    severity_params = get_channel_params(FADING_SEVERITY)
    
    for i in range(total_packets):
        chunk_bits = padded_bits[i*packet_size_bits : (i+1)*packet_size_bits]
        
        # Tx
        tx_sig, _ = tx.main(chunk_bits, mod_scheme, FFT_SIZE, CP_SIZE, pilot_pattern=pilot_pattern)
        
        # Channel & Rx
        if not simo:
            rx_sig = apply_channel(tx_sig, SNR_DB, severity_params, channel_type)
            rx_method = 'None' if channel_type == 'awgn' else method_name
            
            rx_bits_chunk = rx.receive(rx_sig, mod_scheme, FFT_SIZE, CP_SIZE, 
                                   method=rx_method, alpha=severity_params['alpha'],
                                   pilot_pattern=pilot_pattern)
        else:
            # SIMO Case
            rx1 = apply_channel(tx_sig, SNR_DB, severity_params, channel_type)
            rx2 = apply_channel(tx_sig, SNR_DB, severity_params, channel_type)
            
            rx_bits_chunk = rx.receive_simo(rx1, rx2, mod_scheme, FFT_SIZE, CP_SIZE,
                                           method=method_name, alpha=severity_params['alpha'],
                                           pilot_pattern=pilot_pattern)
       
        if len(rx_bits_chunk) > len(chunk_bits):
            rx_bits_chunk = rx_bits_chunk[:len(chunk_bits)]
        received_bits_all.append(rx_bits_chunk)

    final_rx_bits = np.concatenate(received_bits_all)
    L = min(len(all_bits), len(final_rx_bits))
    
    # save output audio if needed
    if save_dir:
        rx_bytes = np.packbits(final_rx_bits[:L])
        rx_audio = np.frombuffer(rx_bytes, dtype=np.int16)
        
        # filename
        mod_name = ['QPSK', '16QAM', '64QAM'][mod_scheme]
        sim_type = "SIMO" if simo else ("MISO" if False else "SISO") # Placeholder for MISO (note implemeted yet)
        fname = f"{mod_name}_{method_name}_{pilot_pattern}_{'SIMO' if simo else 'SISO'}.wav"
        
        wav.write(os.path.join(save_dir, fname), fs, rx_audio)

    return np.sum(all_bits[:L] != final_rx_bits[:L]) / L

def generate_full_report():
    print(f"Loading Audio...")
    try:
        fs, data = wav.read(INPUT_FILE)
        
        if len(data.shape) > 1 and data.shape[1] == 2:
            print("Stereo detected. Converting to Mono...")
            data = data.mean(axis=1).astype(data.dtype)
            
        if len(data) > fs*3: data = data[:fs*3] 
    except:
        print("Error: test_audio.wav not found."); return

    if data.dtype != np.int16: data = (data * 32767).astype(np.int16)
    all_bits = np.unpackbits(np.frombuffer(data.tobytes(), dtype=np.uint8))
    
    # DIRECTORY SETUP
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"experiment_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving audio outputs to: {save_dir}/")
    
    print(f"\n{'='*60}")
    print(f"FINAL PROJECT REPORT: ADAPTIVE KALMAN OFDM and COMP PILOTS")
    print(f"{'='*60}")
    print(f"Condition: {FADING_SEVERITY} Severity Fading (SNR={SNR_DB}dB)")
    print(f"{'='*60}\n")

    schemes = ['QPSK', '16-QAM', '64-QAM']
    
    for scheme_idx, scheme_name in enumerate(schemes):
        print(f"Testing {scheme_name}...", end='\r')
        t0 = time.time()
        
        # AWGN
        ber_awgn = run_simulation_pass(all_bits, scheme_idx, 'None', 'awgn', False, 'Block', save_dir, fs)
        
        # Blind
        ber_blind = run_simulation_pass(all_bits, scheme_idx, 'None', 'fading', False, 'Block', save_dir, fs)
        
        # LS
        ber_ls = run_simulation_pass(all_bits, scheme_idx, 'LS', 'fading', False, 'Block', save_dir, fs)
        
        # Kalman (Block)
        ber_kalman = run_simulation_pass(all_bits, scheme_idx, 'Kalman', 'fading', False, 'Block', save_dir, fs)
        
        # Kalman (Comb)
        ber_comb = run_simulation_pass(all_bits, scheme_idx, '--', 'fading', False, 'Comb', save_dir, fs)

        # 1x2 SIMO (Block) - Fails in Fast Fading
        ber_simo_block = run_simulation_pass(all_bits, scheme_idx, 'Kalman', 'fading', True, 'Block', save_dir, fs) 

        # 1x2 SIMO (Comb) - our runner-up best
        ber_simo_comb = run_simulation_pass(all_bits, scheme_idx, '--', 'fading', True, 'Comb', save_dir, fs)
        
        dt = time.time() - t0
        
        print(f"\n--- Results for {scheme_name} (Time: {dt:.1f}s) ---")
        print(f"{'Method':<20} | {'BER':<10} | {'Notes'}")
        print("-" * 50)
        print(f"{'AWGN Baseline':<20} | {ber_awgn:.5f}    | Ideal Hardware")
        print(f"{'Blind Fading':<20} | {ber_blind:.5f}    | Raw Channel Impact")
        print(f"{'Standard LS':<20} | {ber_ls:.5f}    | Static Estimation")
        print(f"{'Kalman (Block)':<20} | {ber_kalman:.5f}    | Adaptive Tracking")
        print(f"{'SIMO (Block)':<20} | {ber_simo_block:.5f}    | Fails Fast Fading")
        print(f"{'SIMO (Comb)':<20} | {ber_simo_comb:.5f}    | Best Performance")

if __name__ == "__main__":
    generate_full_report()