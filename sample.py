from collections import Counter
import random
import argparse
import math
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", default="hdfs_xu", help="path to input files", type=str, choices=['adfa_verazuo', 'hdfs_xu', 'hdfs_loghub', 'bgl_loghub', 'bgl_cfdr', 'openstack_loghub', 'openstack_parisakalaki', 'hadoop_loghub', 'thunderbird_cfdr', 'awsctd_djpasco'])
parser.add_argument("--grouping_method", default='sequence_identifier', help='choose grouping method for sampling', type=str, choices=['sequence_identifier', 'time_window', 'sliding_window'])
parser.add_argument("--train_ratio", default=0.01, help="fraction of normal data used for training", type=float)
parser.add_argument("--sample_ratio", default=1.0, help="fraction of data sampled from normal and anomalous events", type=float)
# time window
parser.add_argument("--time_window", default=3600, help="size of the fixed time window in seconds (setting this parameter replaces session-based with window-based grouping)", type=float)
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


if source in ['adfa_verazuo', 'hdfs_xu', 'hdfs_loghub', 'openstack_loghub', 'openstack_parisakalaki', 'hadoop_loghub', 'awsctd_djpasco'] and tw is not None:
    # Only BGL and Thunderbird should be used with time-window based grouping
    print('WARNING: Using time-window grouping, even though session-based grouping is recommended for this data set.')


def map_window(window):
    if set(window) == {'Normal'}:
        return 'Normal'
    return 'Anomaly'


def do_sample(source, train_ratio):
    header = True
    sequences_extracted = {}
    tw_groups = {} # Only used for time-window based grouping
    tw_labels = {} # Only used for time-window based grouping
    labels = {}
    with open(source + '/parsed.csv') as extracted, open(source + '/' + source.split('_')[0] + '_train', 'w+') as train, open(source + '/' + source.split('_')[0] + '_test_normal', 'w+') as test_norm, open(source + '/' + source.split('_')[0] + '_test_abnormal', 'w+') as test_abnormal:
        if grouping_method == 'sliding_window':
            # store data in df
            df = pd.read_csv(extracted, sep=';')

            #get eventlables per window
            eventlabel_windows = np.lib.stride_tricks.as_strided(df.eventlabel,
                                                                 ((len(df) - window_size) // step_size + 1, window_size),
                                                                 (df.id.values.strides[0] * step_size,
                                                                  df.id.values.strides[0]))

            #get sequences per window and store in df
            df_sequences_moving_window = pd.DataFrame(np.lib.stride_tricks.as_strided(df.event_type,
                                                                                      ((len(df) - window_size) // step_size + 1, window_size),
                                                                                      (df.id.values.strides[0] * step_size,df.id.values.strides[0])))

            #get normal/anomaly label per window and store to df
            df_sequences_moving_window['label'] = np.array(list(map(map_window, eventlabel_windows)))

            #set id per window
            df_sequences_moving_window.insert(loc=0, column='id', value=df_sequences_moving_window.index + 1)

            # get normal & abnormal sequences
            df_normal = df_sequences_moving_window[df_sequences_moving_window['label'] == 'Normal'].drop(['label'],axis=1)
            df_anormal = df_sequences_moving_window[df_sequences_moving_window['label'] == 'Anomaly'].drop(['label'],axis=1)

            sequences_extracted = {'Normal': df_normal.set_index('id').T.to_dict('list'),
                                   'Anomaly': df_anormal.set_index('id').T.to_dict('list')}
        else:
            cnt = 0
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
                        print(str(cnt) + ' lines processed, ' + str(num_normal) + ' normal and ' + str(num_seq_anom) + ' anomalous sequences found so far')
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
            print('Sampled ' + str(len(sequences_extracted['Normal'])) + ' normal and ' + str(num_sampled_anom) + ' anomalous sequences')
        num_train_logs = math.ceil(train_ratio * len(sequences_extracted['Normal']))
        print('Randomly selecting ' + str(num_train_logs) + ' sequences from ' + str(len(sequences_extracted['Normal'])) + ' normal sequences for training')
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
