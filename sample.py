from collections import Counter
import random
import argparse
import math
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", default="hdfs_xu", help="path to input files", type=str,
                    choices=['adfa_verazuo', 'hdfs_xu', 'hdfs_loghub', 'bgl_loghub', 'bgl_cfdr', 'openstack_loghub',
                             'openstack_parisakalaki', 'hadoop_loghub', 'thunderbird_cfdr', 'awsctd_djpasco'])
parser.add_argument("--grouping_method", default='sequence_identifier', help='choose grouping method for sampling', type=str,
                    choices=['sequence_identifier', 'time_window', 'sliding_window'])
parser.add_argument("--train_ratio", default=0.01, help="fraction of normal data used for training", type=float)
parser.add_argument("--sample_ratio", default=1.0, help="fraction of data sampled from normal and anomalous events", type=float)
# time window
parser.add_argument("--time_window", default=3600,
                    help="size of the fixed time window in seconds (setting this parameter replaces session-based with window-based grouping)",
                    type=float)
# sliding window
parser.add_argument("--window_size", default=20, help='window size for sliding window', type=int)
parser.add_argument("--step_size", default=4, help='step size for sliding window', type=int)

params = vars(parser.parse_args())
source = params["data_dir"]
train_ratio = params["train_ratio"]
tw = params["time_window"]
sample_ratio = params["sample_ratio"]
grouping_method = params["grouping_method"]
window_size = params["window_size"]
step_size = params["step_size"]

if source in ['adfa_verazuo', 'hdfs_xu', 'hdfs_loghub', 'openstack_loghub', 'openstack_parisakalaki', 'hadoop_loghub',
              'awsctd_djpasco'] and grouping_method == 'time_window':
    # Only BGL and Thunderbird should be used with time-window based grouping
    print('WARNING: Using time-window grouping, even though session-based grouping is recommended for this data set.')


def map_window(window):
    if set(window) == {'Normal'}:
        return 'Normal'
    return 'Anomaly'


def do_sample(source, train_ratio):
    header = True
    cnt = 0
    sequences_extracted = {}
    tw_groups = {}  # Only used for time-window based grouping
    tw_labels = {}  # Only used for time-window based grouping
    labels = {}
    train_file_name = source + '/' + source.split('_')[0] + '_train'
    test_norm_file_name = source + '/' + source.split('_')[0] + '_test_normal'
    test_abnormal_file_name = source + '/' + source.split('_')[0] + '_test_abnormal'

    if grouping_method == 'sliding_window':
        with (open(source + '/parsed.csv') as extracted, open(train_file_name, 'w+') as train,
              open(test_norm_file_name, 'w+') as test_norm, open(test_abnormal_file_name, 'w+') as test_abnormal):
            print('Read in parsed sequences ...')
            counter_normal = 0
            current_window = []
            current_label_window = []

            for line in extracted:
                if header:
                    header = False
                    colnames = line.strip('\n').split(';')
                    continue
                parts = line.strip('\n').split(';')

                if 'eventlabel' in colnames:
                    current_label_window.append(parts[colnames.index('eventlabel')])
                else:
                    current_label_window.append(parts[colnames.index('label')])

                current_window.append(parts[colnames.index('event_type')])

                if len(current_window) == window_size:
                    cnt += 1

                    if cnt % 1000000 == 0:
                        print(f'{cnt} sliding windows processed')

                    value_str = ' '.join(map(str, current_window))
                    if set(current_label_window) == {'Normal'}:
                        test_norm.write(f'{cnt},{value_str}\n')
                        counter_normal += 1
                    else:
                        test_abnormal.write(f'{cnt},{value_str}\n')

                    current_window = current_window[step_size:]
                    current_label_window = current_label_window[step_size:]

            print(f'Processing complete, found {counter_normal} normal and '
                  f'{cnt - counter_normal} anomalous sequences')

        df_normal = pd.read_csv(test_norm_file_name, header=None)
        df_normal = df_normal.rename(columns={0: "id"})
        df_abnormal = pd.read_csv(test_abnormal_file_name, header=None)
        df_abnormal = df_abnormal.rename(columns={0: "id"})

        # sample dataset if sample_ratio is set
        if sample_ratio < 1:
            df_normal = df_normal.sample(frac=sample_ratio)
            df_abnormal = df_abnormal.sample(frac=sample_ratio)
            print(f'Sampled {len(df_normal)} normal and {len(df_abnormal)} anomalous sequences')

        # get train ratio
        num_train_logs = math.ceil(train_ratio * len(df_normal))
        print('Randomly selecting ' + str(num_train_logs) + ' sequences from ' + str(
            len(df_normal)) + ' normal sequences for training')

        train_seq_id_list = random.sample(list(df_normal['id']), num_train_logs)
        df_training = df_normal[df_normal['id'].isin(train_seq_id_list)]
        df_normal = df_normal[~df_normal.index.isin(df_training.index)]

        df_training.to_csv(train_file_name, sep=',', index=False, header=False, mode='w')
        df_abnormal.to_csv(test_abnormal_file_name, sep=',', index=False, header=False, mode='w')
        df_normal.to_csv(test_norm_file_name, sep=',', index=False, header=False, mode='w')

    else:
        with (open(source + '/parsed.csv') as extracted, open(train_file_name, 'w+') as train,
              open(test_norm_file_name, 'w+') as test_norm, open(test_abnormal_file_name, 'w+') as test_abnormal):
            print('Read in parsed sequences ...')
            for line in extracted:
                if header:
                    header = False
                    colnames = line.strip('\n').split(';')
                    continue
                parts = line.strip('\n').split(';')
                event_id = parts[colnames.index('event_type')]
                if grouping_method == 'time_window':
                    # Print processing status
                    cnt += 1
                    if cnt % 1000000 == 0:
                        print(str(cnt) + ' lines processed, ' + str(len(tw_groups)) + ' time windows found so far')
                    # Use label of the event
                    label = parts[colnames.index('eventlabel')]
                    # Group events by occurrence time
                    time = float(parts[colnames.index('time')])
                    time_group = math.floor(time / tw) * tw
                    if time_group not in tw_groups:
                        tw_groups[time_group] = [event_id]
                    else:
                        tw_groups[time_group].append(event_id)
                    if time_group not in tw_labels:
                        tw_labels[time_group] = label
                    if label != "Normal":
                        # If any event in the time window is anomalous, consider the entire time window as anomalous
                        tw_labels[time_group] = label
                elif grouping_method == 'sequence_identifier':
                    # Print processing status
                    cnt += 1
                    if cnt % 1000000 == 0:
                        num_seq_anom = 0
                        for lbl, seqs in sequences_extracted.items():
                            if lbl == 'Normal':
                                continue
                            num_seq_anom += len(seqs)
                        num_normal = 0
                        if 'Normal' in sequences_extracted:
                            num_normal = len(sequences_extracted['Normal'])
                        print(str(cnt) + ' lines processed, ' + str(num_normal) + ' normal and ' + str(
                            num_seq_anom) + ' anomalous sequences found so far')
                    # Use label of the entire sequence
                    label = parts[colnames.index('label')]
                    # Group events by sequence identifier
                    seq_id = parts[colnames.index('seq_id')]
                    if label not in sequences_extracted:
                        sequences_extracted[label] = {}
                    if seq_id not in sequences_extracted[label]:
                        sequences_extracted[label][seq_id] = [event_id]
                    else:
                        sequences_extracted[label][seq_id].append(event_id)
            if grouping_method == 'time_window':
                # After processing all lines it is known which label applies to which time window
                for time_group, event_sequence in tw_groups.items():
                    if tw_labels[time_group] not in sequences_extracted:
                        sequences_extracted[tw_labels[time_group]] = {}
                    sequences_extracted[tw_labels[time_group]][time_group] = event_sequence
            num_seq_anom = 0
            for lbl, seqs in sequences_extracted.items():
                if lbl == 'Normal':
                    continue
                num_seq_anom += len(seqs)
            num_normal = 0
            if 'Normal' in sequences_extracted:
                num_normal = len(sequences_extracted['Normal'])
            print('Processing complete, found ' + str(num_normal) + ' normal and ' + str(num_seq_anom) + ' anomalous sequences')
            if sample_ratio < 1:
                sampled_sequences = {}
                num_sampled_anom = 0
                for lbl, seqs in sequences_extracted.items():
                    sampled_seq_list = random.sample(list(sequences_extracted[lbl].keys()), math.ceil(sample_ratio * len(seqs)))
                    sampled_sequences[lbl] = {}
                    for selected_seq in sampled_seq_list:
                        sampled_sequences[lbl][selected_seq] = sequences_extracted[lbl][selected_seq]
                    if lbl != "Normal":
                        num_sampled_anom += len(sampled_seq_list)
                sequences_extracted = sampled_sequences
                print(
                    'Sampled ' + str(len(sequences_extracted['Normal'])) + ' normal and ' + str(num_sampled_anom) + ' anomalous sequences')
            num_train_logs = math.ceil(train_ratio * len(sequences_extracted['Normal']))
            print('Randomly selecting ' + str(num_train_logs) + ' sequences from ' + str(
                len(sequences_extracted['Normal'])) + ' normal sequences for training')
            train_seq_id_list = random.sample(list(sequences_extracted['Normal'].keys()), num_train_logs)
            print('Write vector files ...')
            for label, seq_id_dict in sequences_extracted.items():
                if label == 'Normal':
                    for seq_id, event_list in seq_id_dict.items():
                        if seq_id in train_seq_id_list:
                            train.write(str(seq_id) + ',' + ' '.join([str(event) for event in event_list]) + '\n')
                        else:
                            test_norm.write(str(seq_id) + ',' + ' '.join([str(event) for event in event_list]) + '\n')
                elif label == "Anomaly":
                    for seq_id, event_list in seq_id_dict.items():
                        test_abnormal.write(str(seq_id) + ',' + ' '.join([str(event) for event in event_list]) + '\n')
                else:
                    with open(source + '/' + source.split('_')[0] + '_test_abnormal_' + label, 'w+') as test_label:
                        for seq_id, event_list in seq_id_dict.items():
                            test_label.write(str(seq_id) + ',' + ' '.join([str(event) for event in event_list]) + '\n')
                            test_abnormal.write(str(seq_id) + ',' + ' '.join([str(event) for event in event_list]) + '\n')


if __name__ == "__main__":
    do_sample(source, train_ratio)
