import numpy as np 
from math import sqrt 

# Global Pilot Spacing for Comb Pilots
PILOT_SPACING = 4 # 1 pilot every 4 subcarriers

def generate_random_signal(nbits):
    return np.random.randint(0, 2, size=nbits)

def symbol_map(signal, scheme):
    '''
    Maps bits to symbols using Gray Coding logic inside the original structure.
    '''
    # QPSK
    if scheme == 0:
        if signal.size % 2 != 0:
            signal = np.concatenate((signal, [0] * (2 - signal.size % 2)))
        symbol_mapped_signal = np.zeros(signal.size // 2, dtype=complex)
        for i in range(signal.size // 2):
            bits = signal[i*2:i*2+2]
            re = -1 if bits[0] == 0 else 1
            im = -1 if bits[1] == 0 else 1
            symbol_mapped_signal[i] = (re + 1j*im) / sqrt(2)
            
    # 16 QAM
    elif scheme == 1:
        if signal.size % 4 != 0:
            signal = np.concatenate((signal, [0] * (4 - signal.size % 4)))
        symbol_mapped_signal = np.zeros(signal.size // 4, dtype=complex)
        for i in range(signal.size // 4):
            bits = signal[i*4:i*4+4]
            re = (-3 if bits[1]==0 else -1) if bits[0]==0 else (3 if bits[1]==0 else 1)
            im = (-3 if bits[3]==0 else -1) if bits[2]==0 else (3 if bits[3]==0 else 1)
            symbol_mapped_signal[i] = (re + 1j*im) / sqrt(10)
            
    # 64 QAM
    elif scheme == 2:
        if signal.size % 6 != 0:
            signal = np.concatenate((signal, [0] * (6 - signal.size % 6)))
        symbol_mapped_signal = np.zeros(signal.size // 6, dtype=complex)
        for i in range(signal.size // 6):
            bits = signal[i*6:i*6+6]
            def get_amp(b2, b1, b0):
                if b2==0: return -7 if b1==0 and b0==0 else (-5 if b1==0 else (-3 if b0==1 else -1))
                else:     return  7 if b1==0 and b0==0 else ( 5 if b1==0 else ( 3 if b0==1 else  1))
            
            re = get_amp(bits[0], bits[1], bits[2])
            im = get_amp(bits[3], bits[4], bits[5])
            symbol_mapped_signal[i] = (re + 1j*im) / sqrt(42)
            
    else:
        raise ValueError(f"Unknown Scheme: {scheme}")
            
    return symbol_mapped_signal

def channel_encode(signal):
    n_blocks = signal.size // 4
    channel_encoded_signal = np.zeros(n_blocks * 7, dtype=int)
    for i in range(n_blocks):
        part = signal[i*4:i*4+4]
        channel_encoded_signal[i*7:i*7+7] = (
            part[0] ^ part[1] ^ part[3],
            part[0] ^ part[2] ^ part[3],
            part[0],
            part[1] ^ part[2] ^ part[3],
            part[1],
            part[2],
            part[3]
        )
    return channel_encoded_signal

PILOT_SPACING = 4 # 1 pilot every 4 subcarriers

def pilot_insertion(signal, fft_size, pilot_pattern='Block'):
    np.random.seed(42) 
    
    if pilot_pattern == 'Block':
        preamble_bits = np.random.randint(0, 2, fft_size * 2)
        preamble = np.zeros(fft_size, dtype=complex)
        for i in range(fft_size):
            r = -1 if preamble_bits[2*i]==0 else 1
            im = -1 if preamble_bits[2*i+1]==0 else 1
            preamble[i] = (r + 1j*im)/sqrt(2)
            
        num_data_syms = int(np.ceil(len(signal) / fft_size))
        padding = num_data_syms * fft_size - len(signal)
        if padding > 0:
            signal = np.concatenate((signal, np.zeros(padding)))
        
        frame = np.vstack((preamble, signal.reshape(num_data_syms, fft_size)))
        return frame.flatten(), fft_size

    elif pilot_pattern == 'Comb':
        pilot_indices = np.arange(0, fft_size, PILOT_SPACING)
        num_pilots = len(pilot_indices)
        num_data = fft_size - num_pilots
        

        p_val = (1+1j)/sqrt(2)
        
        num_syms = int(np.ceil(len(signal) / num_data))
        padding = num_syms * num_data - len(signal)
        if padding > 0:
            signal = np.concatenate((signal, np.zeros(padding)))
        
        data_matrix = signal.reshape(num_syms, num_data)
        full_symbols = np.zeros((num_syms, fft_size), dtype=complex)
        
        data_mask = np.ones(fft_size, dtype=bool)
        data_mask[pilot_indices] = False
        
        full_symbols[:, pilot_indices] = p_val
        full_symbols[:, data_mask] = data_matrix
        
        return full_symbols.flatten(), 0 # No preamble length needed
    else:
        raise ValueError(f"Unknown pilot_pattern: {pilot_pattern}")

def ifft_block(signal, fft_size):
    num_symbols = len(signal) // fft_size
    signal_matrix = signal.reshape(num_symbols, fft_size)
    return np.fft.ifft(signal_matrix, axis=1)

def add_cp(time_domain_matrix, cp_size):
    cp = time_domain_matrix[:, -cp_size:]
    return np.hstack((cp, time_domain_matrix)).flatten()

def main(signal, scheme, fft_size=1024, cp_size=256, pilot_pattern='Block'):
    padding_len = 0
    # Adjust padding logic if needed, but the simple one usually works
    if len(signal) % 4 != 0:
        padding_len = 4 - (len(signal) % 4)
        signal = np.concatenate((signal, np.zeros(padding_len, dtype=int)))
        
    encoded = channel_encode(signal)
    symbols = symbol_map(encoded, scheme)
    
    # Pass the pattern to insertion
    framed_signal, _ = pilot_insertion(symbols, fft_size, pilot_pattern)
    
    # Ensure framed_signal is properly shaped for IFFT
    if len(framed_signal) % fft_size != 0:
        padding = fft_size - (len(framed_signal) % fft_size)
        framed_signal = np.concatenate((framed_signal, np.zeros(padding, dtype=complex)))
    
    time_dom = ifft_block(framed_signal, fft_size)
    tx_signal = add_cp(time_dom, cp_size)
    
    return tx_signal, padding_len