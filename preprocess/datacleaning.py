import numpy as np
import argparse
import pandas as pd




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Prepartion')
    parser.add_argument('--dataset', default='../data/bmw_data.csv', help='`bmv`')
    args = parser.parse_args()

    dataset = args.dataset

    data = pd.read_csv(dataset)
    print(data.shape)
    data['time_stamp'] = pd.to_timedelta(data['time_stamp']).dt.total_seconds()

    print(data.describe())
    data['section'] = pd.Categorical(data['section'])
    df_temp = pd.get_dummies(data['section'], prefix='section')
    data = pd.concat([data, df_temp], axis=1)

    data = data.applymap(lambda x: 1 if x == True else x)
    data = data.applymap(lambda x: 0 if x == False else x)
    #new_data = pd.concat([data['component_number'], data['time_stamp'], data['section_1'], data['section_2'], data['section_3'], data['section_4'],
                          #data['amperage [A]'], data['voltage [V]'], data['wire_speed [m/min]'], data['OK']], axis=1)

    new_data = pd.concat([data['component_number'], data['section'], data['amperage [A]'], data['voltage [V]'], data['wire_speed [m/min]'], data['OK']], axis=1)

    print(new_data.describe())

    new_data.to_csv('../data/bmw_cleaned.csv', index=False)

    print("read the data")

#later the data will be converted to tuple(comp. ID, seq. data, outcome)