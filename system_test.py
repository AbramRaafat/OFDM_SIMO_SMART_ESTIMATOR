import numpy as np
import matplotlib.pyplot as plt
import transmitter as tx
import receiver as rx
from math import sqrt


FFT_SIZE = 1024
CP_SIZE = 256
L_TAP = 50


SCENARIOS = {
    'walking': 0.998,   # Slow drift (Kalman works well)
    'running': 0.9048      # Fast drift (Kalman fails, Comb required)
}


def qam_slicer_helper(symbol, scheme):
    if scheme == 0: return (np.sign(symbol.real) + 1j*np.sign(symbol.imag)) / sqrt(2)
    norm = sqrt(10) if scheme == 1 else sqrt(42)
    limit = 3 if scheme == 1 else 7
    re = np.clip(2 * np.round((symbol.real * norm + 1) / 2) - 1, -limit, limit)
    im = np.clip(2 * np.round((symbol.imag * norm + 1) / 2) - 1, -limit, limit)
    return (re + 1j*im) / norm

def run_ber_simulation(mod_scheme, snr_db, num_syms, method, pilot_pattern, alpha):
    """
    Runs the full Tx-Channel-Rx chain and returns BER.
    """
    #  Capacity Calculation (Dynamic based on Pilot Pattern)
    bps = 2 if mod_scheme==0 else (4 if mod_scheme==1 else 6)
    
    if pilot_pattern == 'Comb':
        data_subcarriers = FFT_SIZE - (FFT_SIZE // 4)
    else:
        data_subcarriers = FFT_SIZE

    # Total bits (taking Hamming 7,4 overhead into account)
    n_bits = data_subcarriers * bps * 4 // 7 * num_syms
    
    #  Transmit
    bits = tx.generate_random_signal(n_bits)
    tx_sig, _ = tx.main(bits, mod_scheme, FFT_SIZE, CP_SIZE, pilot_pattern=pilot_pattern)
    
    # Trim to correct length
    expected_len = num_syms * (FFT_SIZE + CP_SIZE)
    if pilot_pattern == 'Block': expected_len += (FFT_SIZE + CP_SIZE)
    tx_sig = tx_sig[:len(tx_sig)]

    #  Channel (Fading + Drift)
    h = (np.random.randn(L_TAP)+1j*np.random.randn(L_TAP))/sqrt(2*L_TAP)
    faded_sig = np.zeros_like(tx_sig, dtype=complex)
    sym_len = FFT_SIZE + CP_SIZE
    
    num_tx_syms = len(tx_sig) // sym_len
    for i in range(num_tx_syms):
        chunk = tx_sig[i*sym_len : (i+1)*sym_len]
        conv = np.convolve(chunk, h, mode='full')[:len(chunk)]
        faded_sig[i*sym_len : (i+1)*sym_len] = conv
        
        # Drift Update
        h = alpha * h + sqrt(1-alpha**2)*(np.random.randn(L_TAP)+1j*np.random.randn(L_TAP))/sqrt(2*L_TAP)

    # Noise
    sig_pwr = np.mean(np.abs(faded_sig)**2)
    n0 = sig_pwr/(10**(snr_db/10))
    rx_sig = faded_sig + (np.random.randn(len(faded_sig))+1j*np.random.randn(len(faded_sig)))*sqrt(n0/2)

    #  Receive
    rx_bits = rx.receive(rx_sig, mod_scheme, FFT_SIZE, CP_SIZE, 
                         method=method, alpha=alpha, pilot_pattern=pilot_pattern)
    
    #  BER
    L_min = min(len(bits), len(rx_bits))
    ber = np.sum(bits[:L_min] != rx_bits[:L_min]) / L_min
    return ber

def run_diagnostic_simulation(mod_scheme, snr_db, num_syms, alpha):
    """
    Runs a simulation but manually extracts internal Kalman states 
    (H_est, Constellation) for plotting. Assumes Block Pilot.
    """
    # Setup
    bps = 2 if mod_scheme==0 else (4 if mod_scheme==1 else 6)
    n_bits = FFT_SIZE * bps * num_syms
    bits = tx.generate_random_signal(n_bits)
    tx_sig, _ = tx.main(bits, mod_scheme, FFT_SIZE, CP_SIZE, pilot_pattern='Block')
    tx_sig = tx_sig[:(num_syms+1)*(FFT_SIZE+CP_SIZE)] # +1 for preamble

    # Channel
    h = (np.random.randn(L_TAP)+1j*np.random.randn(L_TAP))/sqrt(2*L_TAP)
    faded_sig = np.zeros_like(tx_sig, dtype=complex)
    sym_len = FFT_SIZE + CP_SIZE
    
    true_H_history = []
    
    for i in range(num_syms+1):
        H_freq = np.fft.fft(h, n=FFT_SIZE)
        if i > 0: true_H_history.append(H_freq[10]) # Record data symbols only

        chunk = tx_sig[i*sym_len : (i+1)*sym_len]
        conv = np.convolve(chunk, h, mode='full')[:len(chunk)]
        faded_sig[i*sym_len : (i+1)*sym_len] = conv
        h = alpha * h + sqrt(1-alpha**2)*(np.random.randn(L_TAP)+1j*np.random.randn(L_TAP))/sqrt(2*L_TAP)

    # Noise
    sig_pwr = np.mean(np.abs(faded_sig)**2)
    n0 = sig_pwr/(10**(snr_db/10))
    rx_sig = faded_sig + (np.random.randn(len(faded_sig))+1j*np.random.randn(len(faded_sig)))*sqrt(n0/2)


    time_no_cp = rx.Remove_CP(rx_sig, FFT_SIZE, CP_SIZE)
    rx_freq = rx.FFT_Block(time_no_cp)
    
    # Init Kalman
    Y_pre = rx_freq[0, :]
    np.random.seed(42)
    pre_bits = np.random.randint(0, 2, FFT_SIZE * 2)
    X_pre = np.zeros(FFT_SIZE, dtype=complex)
    for i in range(FFT_SIZE):
        X_pre[i] = ((-1 if pre_bits[2*i]==0 else 1) + 1j*(-1 if pre_bits[2*i+1]==0 else 1))/sqrt(2)
        
    H_hat = Y_pre / X_pre
    P = np.ones(FFT_SIZE) * 0.1
    Q = 1 - alpha**2 + 1e-6
    R = 0.1
    
    est_H_history = []
    constellation = []
    
    for i in range(1, num_syms+1):
        Y_data = rx_freq[i, :]
        H_pred = alpha * H_hat
        P_pred = alpha**2 * P + Q
        X_est = Y_data / H_pred
        constellation.extend(X_est)
        
        # Kalman Update
        X_dec = qam_slicer_helper(X_est, mod_scheme)
        K = (P_pred * np.conj(X_dec)) / (np.abs(X_dec)**2 * P_pred + R)
        H_hat = H_pred + K * (Y_data - X_dec * H_pred)
        P = (1 - K * X_dec) * P_pred
        
        est_H_history.append(H_hat[10])
        
    return np.array(true_H_history), np.array(est_H_history), np.array(constellation)



def plot_waterfalls():
    snrs = [0, 10, 20, 30]
    schemes = ['QPSK', '16-QAM', '64-QAM']
    
    for scenario_name, alpha_val in SCENARIOS.items():
        print(f"Generating Waterfalls for {scenario_name} (Alpha={alpha_val})...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, scheme_name in enumerate(schemes):
            print(f"  Simulating {scheme_name}...")
            ax = axes[i]
            
            # 1. LS (Block)
            ber_ls = [run_ber_simulation(i, snr, 50, 'LS', 'Block', alpha_val) for snr in snrs]
            ax.semilogy(snrs, ber_ls, 'r--x', label='LS (Block)')
            
            # 2. Kalman (Block)
            ber_kal = [run_ber_simulation(i, snr, 50, 'Kalman', 'Block', alpha_val) for snr in snrs]
            ax.semilogy(snrs, ber_kal, 'g-o', label='Kalman (Block)')
            
            # 3. Comb (Interpolation)
            ber_comb = [run_ber_simulation(i, snr, 50, 'Kalman', 'Comb', alpha_val) for snr in snrs]
            ax.semilogy(snrs, ber_comb, 'b-^', linewidth=2, label='Comb (Interp)')
            
            ax.set_title(f"{scheme_name}")
            ax.set_xlabel("SNR (dB)"); ax.set_ylabel("BER")
            ax.grid(True, which='both', alpha=0.5)
            ax.legend()
            ax.set_ylim(1e-5, 1.0)

        plt.suptitle(f"BER Performance: {scenario_name} Scenario (Alpha={alpha_val})")
        plt.tight_layout()
        plt.savefig(f"fig_waterfall_{scenario_name}.png")
        print(f"Saved fig_waterfall_{scenario_name}.png")

def plot_diagnostics():
    print("Generating Diagnostics (Tracking & Constellation)...")
    
    # Run a Pedestrian scenario for clean tracking (QPSK)
    true_H, est_H, _ = run_diagnostic_simulation(0, 25, 200, alpha=0.998)
    
    # 1. Phase Tracking Plot
    plt.figure(figsize=(10,4))
    plt.plot(np.angle(true_H), 'k', linewidth=1.5, label='True Channel Phase')
    plt.plot(np.angle(est_H), 'r--', linewidth=1.5, label='Kalman Estimate')
    plt.title("Kalman Phase Tracking (Pedestrian Scenario, Subcarrier 10)")
    plt.xlabel("Symbol Time"); plt.ylabel("Phase (Rad)")
    plt.legend(); plt.grid(True)
    plt.savefig("fig_tracking.png")
    print("Saved fig_tracking.png")
    
    #  Constellation Stability Plot (16-QAM)
    _, _, const = run_diagnostic_simulation(1, 30, 500, alpha=0.998)
    
    plt.figure(figsize=(6,6))
    # Plot Start (First 150)
    start_pts = const[:150]
    plt.scatter(start_pts.real, start_pts.imag, c='blue', s=10, alpha=0.6, label='Packet Start (t=0)')
    # Plot End (Last 150)
    end_pts = const[-150:]
    plt.scatter(end_pts.real, end_pts.imag, c='red', s=10, marker='x', alpha=0.8, label='Packet End (t=500)')
    
    plt.title("16-QAM Stability Check (Kalman Corrected)")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.xlim(-4,4); plt.ylim(-4,4)
    plt.xlabel("In-Phase"); plt.ylabel("Quadrature")
    plt.savefig("fig_constellation.png")
    print("Saved fig_constellation.png")

def plot_bitrate_loss():
    print("Generating Bit Rate Analysis...")
    
    # 64-QAM Baseline (FFT 1024)
    raw_rate = FFT_SIZE * 6 # 6144 bits per symbol
    
    # 1. CP Loss
    rate_cp = raw_rate * (FFT_SIZE / (FFT_SIZE + CP_SIZE))
    
    # 2. Coding Loss (Hamming 7,4)
    rate_coded = rate_cp * (4/7)
    
    # 3. Pilot Loss (Comb vs Block)
    rate_comb = rate_coded * 0.75  # Lose 1/4 subcarriers
    rate_block = rate_coded * (100/101) # Lose 1 symbol per 100
    
    labels = ['Raw Physics', 'After CP', 'After Hamming', 'Final (Block)', 'Final (Comb)']
    values = [raw_rate, rate_cp, rate_coded, rate_block, rate_comb]
    colors = ['gray', 'orange', 'red', 'green', 'blue']
    
    plt.figure(figsize=(10,6))
    bars = plt.bar(labels, values, color=colors)
    
    # Add text labels
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 100, f"{int(yval)}", ha='center', va='bottom')
        
    plt.title("Bit Rate Efficiency Analysis (Effective Bits per OFDM Symbol)")
    plt.ylabel("Bits per Symbol")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("fig_bitrate_loss.png")
    print("Saved fig_bitrate_loss.png")

if __name__ == "__main__":
    plot_bitrate_loss()
    plot_diagnostics()
    plot_waterfalls()
    print("\nAll plots generated successfully.")