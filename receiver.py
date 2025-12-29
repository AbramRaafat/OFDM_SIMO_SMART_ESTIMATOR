import numpy as np 
from math import sqrt 

# Parity Check Matrix
H_mat = np.array([
    [1,0,1,0,1,0,1],
    [0,1,1,0,0,1,1],
    [0,0,0,1,1,1,1]
])


def Remove_CP(signal, fft_size, cp_size):
    sym_len = fft_size + cp_size
    num_syms = len(signal) // sym_len
    signal_matrix = signal.reshape(num_syms, sym_len)
    return signal_matrix[:, cp_size:]

def FFT_Block(time_domain_matrix):
    return np.fft.fft(time_domain_matrix, axis=1)

def qam_slicer_helper(symbol, scheme):
    if scheme == 0: return (np.sign(symbol.real) + 1j*np.sign(symbol.imag)) / sqrt(2)
    norm = sqrt(10) if scheme == 1 else sqrt(42)
    limit = 3 if scheme == 1 else 7
    re = np.clip(2 * np.round((symbol.real * norm + 1) / 2) - 1, -limit, limit)
    im = np.clip(2 * np.round((symbol.imag * norm + 1) / 2) - 1, -limit, limit)
    return (re + 1j*im) / norm

def symbol_unmap(signal, scheme):
    bits = []
    if scheme == 0: # QPSK
        signal *= sqrt(2)
        for s in signal:
            bits.extend([0 if s.real < 0 else 1, 0 if s.imag < 0 else 1])
    elif scheme == 1: # 16-QAM
        signal *= sqrt(10)
        for s in signal:
            bits.extend([0 if s.real < 0 else 1, 1 if abs(s.real) < 2 else 0])
            bits.extend([0 if s.imag < 0 else 1, 1 if abs(s.imag) < 2 else 0])
    elif scheme == 2: # 64-QAM
        signal *= sqrt(42)
        for s in signal:
            bits.extend([0 if s.real < 0 else 1, 1 if abs(s.real) < 4 else 0])
            bits.append(1 if (2 < abs(s.real) < 6) else 0)
            bits.extend([0 if s.imag < 0 else 1, 1 if abs(s.imag) < 4 else 0])
            bits.append(1 if (2 < abs(s.imag) < 6) else 0)
    return np.array(bits)

def channel_decode(signal):
    n_blocks = signal.size // 7
    decoded_bits = np.zeros(n_blocks * 4, dtype=int)
    
    # The Parity Check Matrix (Standard Hamming 7,4)
    # H * r^T = Syndrome
    H = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1]
    ])

    for i in range(n_blocks):
        # Get the 7-bit codeword
        r = signal[i*7 : i*7+7]
        
        # Calculate Syndrome (Matrix Multiply modulo 2)
        # s will be a vector like [1, 0, 1]
        s = H.dot(r) % 2
        
        #  Convert binary syndrome to integer index (error position)
        # s=[p1, p2, p3] -> index. Note: In Hamming(7,4), the syndrome 
        # binary value usually maps to the column index of H that failed.
        # For this specific H matrix construction:
        # s=[0,0,0] -> No Error
        # s=[1,1,0] -> Error in bit 0, etc.
        
        error_idx = s[0]*1 + s[1]*2 + s[2]*4  # Binary to Decimal
        
        # Flip the bit if there's an error
        if error_idx > 0:
            # Map syndrome index to array index (This depends on H construction)
            # For your specific H_mat structure:
            # col 1 (idx 0) = [1,0,0] -> s=1
            # col 2 (idx 1) = [0,1,0] -> s=2
            # col 3 (idx 2) = [1,1,0] -> s=3 
            # ...
            # Simple fix: r[error_idx - 1] ^= 1
             r[error_idx - 1] ^= 1
            
        # Extract Data Bits (Indices 2, 4, 5, 6)
        decoded_bits[i*4 : i*4+4] = [r[2], r[4], r[5], r[6]]

    return decoded_bits


def receive(signal, scheme, fft_size=1024, cp_size=256, 
            method='Kalman', alpha=0.9999, pilot_pattern='Block'):
            
    time_no_cp = Remove_CP(signal, fft_size, cp_size)
    freq_domain = FFT_Block(time_no_cp)
    num_syms = freq_domain.shape[0]
    
    equalized_stream = []
    
    if pilot_pattern == 'Block':
        # Initialization using Preamble (Symbol 0)
        Y_pre = freq_domain[0, :]
        
        if method == 'None':
            H_hat = np.ones(fft_size, dtype=complex)
        else:
            np.random.seed(42) 
            pre_bits = np.random.randint(0, 2, fft_size * 2)
            X_pre = np.zeros(fft_size, dtype=complex)
            for i in range(fft_size):
                r = -1 if pre_bits[2*i]==0 else 1
                im = -1 if pre_bits[2*i+1]==0 else 1
                X_pre[i] = (r + 1j*im)/sqrt(2)
            H_hat = Y_pre / X_pre 
        
        # Kalman Covariances
        P = np.ones(fft_size) * 0.1 
        Q = (1 - alpha**2) + 1e-6
        R = 0.1
        
        # Loop over Data Symbols
        for i in range(1, num_syms):
            Y_data = freq_domain[i, :]
            
            if method == 'None':
                H_pred = H_hat 
            else:
                H_pred = alpha * H_hat 
                
            P_pred = (alpha**2) * P + Q
            X_est = Y_data / H_pred
            
            if method == 'Kalman':
                X_dec = qam_slicer_helper(X_est, scheme)
                K = (P_pred * np.conj(X_dec)) / (np.abs(X_dec)**2 * P_pred + R)
                H_hat = H_pred + K * (Y_data - X_dec * H_pred)
                P = (1 - K * X_dec) * P_pred
                
            equalized_stream.append(X_est)


    elif pilot_pattern == 'Comb':
        PILOT_SPACING = 4
        # Indices: 0, 4, 8, ...
        pilot_indices = np.arange(0, fft_size, PILOT_SPACING)
        
        # Data Indices mask
        data_mask = np.ones(fft_size, dtype=bool)
        data_mask[pilot_indices] = False
        
        # Fixed Pilot Value (1+1j)/sqrt(2)
        X_pilot = (1+1j)/sqrt(2)
        
        for i in range(num_syms):
            Y_sym = freq_domain[i, :]
            
            # LS Estimate at Pilot locations
            Y_pilots = Y_sym[pilot_indices]
            H_pilots = Y_pilots / X_pilot
            
            # Manual Linear Interpolation
            # Create an empty channel array
            H_interp = np.zeros(fft_size, dtype=complex)
            
            # Fill exact pilot spots
            H_interp[pilot_indices] = H_pilots
            
            # Fill gaps between pilots
            for j in range(len(pilot_indices) - 1):
                idx_start = pilot_indices[j]
                idx_end = pilot_indices[j+1]
                
                H_start = H_pilots[j]
                H_end = H_pilots[j+1]
                
                slope = (H_end - H_start) / (idx_end - idx_start)
                
                for k in range(1, PILOT_SPACING):
                    H_interp[idx_start + k] = H_start + slope * k
            # CIRCULAR INTERPOLATION FOR TRAILING EDGE 
            # Interpolate between Last Pilot (1020) and First Pilot (0)
            idx_last = pilot_indices[-1] # 1020
            H_last = H_pilots[-1]
            H_first = H_pilots[0] # Wrap around to 0
            
            # Slope wrapping around (Virtual distance is 4)
            slope_wrap = (H_first - H_last) / PILOT_SPACING
            
            for k in range(1, PILOT_SPACING):
                if idx_last + k < fft_size:
                    H_interp[idx_last + k] = H_last + slope_wrap * k

            
            # Equalize ONLY Data Subcarriers
            Y_data = Y_sym[data_mask]
            H_data = H_interp[data_mask]
            
            X_est = Y_data / H_data
            equalized_stream.append(X_est)

    # Flatten and Decode
    equalized_signal = np.array(equalized_stream).flatten()
    demapped_bits = symbol_unmap(equalized_signal, scheme)
    decoded_bits = channel_decode(demapped_bits)
    
    return decoded_bits


def receive_simo(signal1, signal2, scheme,
                 fft_size=1024, cp_size=256,
                 method='Kalman', alpha=0.9999, pilot_pattern='Block'):

    # 1. Prepare Inputs
    td1 = Remove_CP(signal1, fft_size, cp_size)
    td2 = Remove_CP(signal2, fft_size, cp_size)
    Y1 = FFT_Block(td1)
    Y2 = FFT_Block(td2)
    num_syms = Y1.shape[0]

    equalized_stream = []


    if pilot_pattern == 'Block':
        # Preamble Init
        np.random.seed(42)
        pre_bits = np.random.randint(0, 2, fft_size * 2)
        X_pre = np.zeros(fft_size, dtype=complex)
        for i in range(fft_size):
            r = -1 if pre_bits[2*i] == 0 else 1
            im = -1 if pre_bits[2*i+1] == 0 else 1
            X_pre[i] = (r + 1j*im) / sqrt(2)

        # Initial Estimates
        H1_hat = Y1[0, :] / X_pre
        H2_hat = Y2[0, :] / X_pre

        # Independent Kalman States
        P1 = np.ones(fft_size) * 0.1; P2 = np.ones(fft_size) * 0.1
        Q = (1 - alpha**2) + 1e-6;    R = 0.1

        for i in range(1, num_syms):
            Y1_data = Y1[i, :]; Y2_data = Y2[i, :]

            # Prediction
            H1_pred = alpha * H1_hat; H2_pred = alpha * H2_hat
            P1_pred = (alpha**2) * P1 + Q; P2_pred = (alpha**2) * P2 + Q

            # MRC Combining
            num = (np.conj(H1_pred) * Y1_data + np.conj(H2_pred) * Y2_data)
            den = (np.abs(H1_pred)**2 + np.abs(H2_pred)**2 + 1e-12)
            X_est = num / den
            equalized_stream.append(X_est)

            # Dual Kalman Update
            if method == 'Kalman':
                X_dec = qam_slicer_helper(X_est, scheme)
                # Update Branch 1
                K1 = (P1_pred * np.conj(X_dec)) / (np.abs(X_dec)**2 * P1_pred + R)
                H1_hat = H1_pred + K1 * (Y1_data - X_dec * H1_pred)
                P1 = (1 - K1 * X_dec) * P1_pred
                # Update Branch 2
                K2 = (P2_pred * np.conj(X_dec)) / (np.abs(X_dec)**2 * P2_pred + R)
                H2_hat = H2_pred + K2 * (Y2_data - X_dec * H2_pred)
                P2 = (1 - K2 * X_dec) * P2_pred
            else:
                H1_hat = H1_pred; H2_hat = H2_pred


    elif pilot_pattern == 'Comb':
        PILOT_SPACING = 4
        pilot_indices = np.arange(0, fft_size, PILOT_SPACING)
        data_mask = np.ones(fft_size, dtype=bool)
        data_mask[pilot_indices] = False
        X_pilot = (1+1j)/sqrt(2) # Fixed Pilot

        for i in range(num_syms):
            Y1_sym = Y1[i, :]; Y2_sym = Y2[i, :]

            # LS Estimate on Pilots (Both Antennas)
            H1_pilots = Y1_sym[pilot_indices] / X_pilot
            H2_pilots = Y2_sym[pilot_indices] / X_pilot

            # Manual Interpolation (Both Antennas)
            H1_interp = np.zeros(fft_size, dtype=complex)
            H2_interp = np.zeros(fft_size, dtype=complex)
            
            # Fill Pilots
            H1_interp[pilot_indices] = H1_pilots
            H2_interp[pilot_indices] = H2_pilots

            # Fill Gaps (Standard)
            for j in range(len(pilot_indices) - 1):
                idx_start, idx_end = pilot_indices[j], pilot_indices[j+1]
                
                slope1 = (H1_pilots[j+1] - H1_pilots[j]) / (idx_end - idx_start)
                slope2 = (H2_pilots[j+1] - H2_pilots[j]) / (idx_end - idx_start)
                
                for k in range(1, PILOT_SPACING):
                    H1_interp[idx_start + k] = H1_pilots[j] + slope1 * k
                    H2_interp[idx_start + k] = H2_pilots[j] + slope2 * k

            idx_last = pilot_indices[-1]
            H1_last, H2_last = H1_pilots[-1], H2_pilots[-1]
            H1_first, H2_first = H1_pilots[0], H2_pilots[0] # Wrap
            
            slope1_wrap = (H1_first - H1_last) / PILOT_SPACING
            slope2_wrap = (H2_first - H2_last) / PILOT_SPACING
            
            for k in range(1, PILOT_SPACING):
                if idx_last + k < fft_size:
                    H1_interp[idx_last + k] = H1_last + slope1_wrap * k
                    H2_interp[idx_last + k] = H2_last + slope2_wrap * k

            # MRC Combining on DATA subcarriers only
            Y1_data = Y1_sym[data_mask]
            Y2_data = Y2_sym[data_mask]
            H1_data = H1_interp[data_mask]
            H2_data = H2_interp[data_mask]

            num = (np.conj(H1_data) * Y1_data + np.conj(H2_data) * Y2_data)
            den = (np.abs(H1_data)**2 + np.abs(H2_data)**2 + 1e-12)
            X_est = num / den
            
            equalized_stream.append(X_est)

    # Decode
    equalized_signal = np.array(equalized_stream).flatten()
    demapped_bits = symbol_unmap(equalized_signal, scheme)
    decoded_bits = channel_decode(demapped_bits)

    return decoded_bits