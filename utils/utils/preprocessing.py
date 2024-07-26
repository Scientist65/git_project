import pandas as pd
import numpy as np
from scipy.integrate import trapz

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    I_columns = [col for col in data.columns if col.startswith('I')]
    I_array = data[I_columns].values
    Potential = data['V'].values

    # Define potential ranges
    potential_ranges = {
        'Cd(II)': (-0.85, -0.7),
        'Pb(II)': (-0.59, -0.47),
        'Cu(II)': (-0.15, -0.03)
    }

    def find_max_in_range(potential_range, I_data):
        min_potential, max_potential = potential_range
        indices_in_range = np.where((Potential >= min_potential) & (Potential <= max_potential))[0]
        I_in_range = I_data[indices_in_range, :]
        max_I_indices = np.argmax(I_in_range, axis=0)
        max_I_values = I_in_range[max_I_indices, np.arange(I_in_range.shape[1])]
        max_potential_values = Potential[indices_in_range][max_I_indices]
        return max_potential_values, max_I_values

    def calculate_peak_area(potential_range, I_data, baseline):
        min_potential, max_potential = potential_range
        indices_in_range = np.where((Potential >= min_potential) & (Potential <= max_potential))[0]
        I_in_range = I_data[indices_in_range, :]
        peak_areas = []
        for i in range(I_in_range.shape[1]):
            I_in_range_baseline_subtracted = np.maximum(I_in_range[:, i] - baseline, 0)
            area = trapz(I_in_range_baseline_subtracted, Potential[indices_in_range])
            peak_areas.append(area)
        return peak_areas

    features = []

    for metal_ion, potential_range in potential_ranges.items():
        max_potentials, max_currents = find_max_in_range(potential_range, I_array)
        baseline_indices = np.where((Potential >= potential_range[0]) & (Potential <= potential_range[1]))[0]
        baseline_start = I_array[baseline_indices[:int(len(baseline_indices) * 0.1)], :]
        baseline_end = I_array[baseline_indices[-int(len(baseline_indices) * 0.1):], :]
        baseline = np.mean(np.concatenate((baseline_start, baseline_end)))
        peak_areas = calculate_peak_area(potential_range, I_array, baseline)
        for j in range(I_array.shape[1]):
            features.append([max_potentials[j], max_currents[j], peak_areas[j]])

    features = np.array(features).reshape(-1, 3)

    vcd, Icd, acd = features[0]
    vpb, Ipb, apb = features[1]
    vcu, Icu, acu = features[2]

    df = pd.DataFrame({
        'Icd': [Icd], 'acd': [acd], 'vcd': [vcd],
        'Ipb': [Ipb], 'apb': [apb], 'vpb': [vpb],
        'Icu': [Icu], 'acu': [acu], 'vcu': [vcu]
    })

    df['Icd_squared'] = df['Icd'] ** 2
    df['vcd_squared'] = df['vcd'] ** 2
    df['Icd_acd'] = df['Icd'] * df['acd']
    df['Icd_vcd'] = df['Icd'] * df['vcd']
    df['Icu_squared'] = df['Icu'] ** 2
    df['vcu_squared'] = df['vcu'] ** 2
    df['Icu_acu'] = df['Icu'] * df['acu']
    df['Icu_vcu'] = df['Icu'] * df['vcu']
    df['Ipb_squared'] = df['Ipb'] ** 2
    df['vpb_squared'] = df['vpb'] ** 2
    df['Ipb_apb'] = df['Ipb'] * df['apb']
    df['Ipb_vpb'] = df['Ipb'] * df['vpb']

    return df

def load_scaler_params(scaler_path):
    scaler_params = np.load(scaler_path)
    return scaler_params['min'], scaler_params['max']

def scale_data(X, scaler_min, scaler_max):
    return (X - scaler_min) / (scaler_max - scaler_min)
