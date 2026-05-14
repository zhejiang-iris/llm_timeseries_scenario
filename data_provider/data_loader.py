import glob
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        # self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.raw_dates = pd.to_datetime(df_stamp['date'].values)


    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def window_dates(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        s_end = s_begin + self.seq_len
        return self.raw_dates[s_begin], self.raw_dates[s_end - 1]


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', percent=100,
                 seasonal_patterns=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.raw_dates = pd.to_datetime(df_stamp['date'].values)

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None, use_aux_data=False, return_source_id=False):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        self.use_aux_data = use_aux_data
        self.return_source_id = return_source_id

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.sources[0]['enc_in']
        self.tot_len = self.sources[0]['tot_len']

    def __read_data__(self):
        df_raw = self._read_frame(self.data_path)
        main_source, fine_tune_range = self._build_source(
            df_raw,
            self.data_path,
            use_main_split=True
        )
        self.sources = [main_source]
        self.scaler = main_source['scaler']
        self.data_x = main_source['data_x']
        self.data_y = main_source['data_y']
        self.data_stamp = main_source['data_stamp']
        self.raw_dates = main_source['raw_dates']

        if self.use_aux_data and self.set_type == 0:
            self.sources.extend(self._build_aux_sources(fine_tune_range))
            self.cumulative_lengths = np.cumsum([source['length'] for source in self.sources])
        else:
            self.cumulative_lengths = np.array([main_source['length']])

    def _read_frame(self, data_path):
        return pd.read_csv(os.path.join(self.root_path, data_path))

    def _normalize_frame(self, df_raw):
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        df_raw = df_raw.copy()
        if 'date' not in df_raw.columns:
            if 'Day' in df_raw.columns:
                df_raw = df_raw.rename(columns={'Day': 'date'})
            else:
                df_raw = df_raw.rename(columns={df_raw.columns[0]: 'date'})

        return df_raw

    def _select_target(self, df_raw):
        if self.target not in df_raw.columns:
            numeric_cols = [col for col in df_raw.columns if col != 'date']
            return numeric_cols[-1]
        return self.target

    def _build_source(self, df_raw, data_path, use_main_split, date_range=None):
        scaler = StandardScaler()
        df_raw = self._normalize_frame(df_raw)
        target = self._select_target(df_raw)

        cols = list(df_raw.columns)
        cols.remove(target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [target]]
        df_raw['date'] = pd.to_datetime(df_raw['date'])

        if use_main_split:
            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            num_vali = len(df_raw) - num_train - num_test
            border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            if self.set_type == 0:
                border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

            train_slice = slice(border1s[0], border2s[0])
            selected_frame = df_raw.iloc[border1:border2].copy()
        else:
            start_date, end_date = date_range
            df_raw = df_raw[(df_raw['date'] >= start_date) & (df_raw['date'] <= end_date)].copy()
            if len(df_raw) < self.seq_len + self.pred_len:
                return None, None

            train_slice = slice(0, len(df_raw))
            selected_frame = df_raw.copy()

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[target]]

        if self.scale:
            train_data = df_data.iloc[train_slice]
            scaler.fit(train_data.values)
            data = scaler.transform(df_data.values)
        else:
            data = df_data.values

        if use_main_split:
            source_data = data[border1:border2]
        else:
            source_data = data

        df_stamp = selected_frame[['date']].copy()
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        raw_dates = pd.to_datetime(selected_frame['date'].values)
        source = {
            'data_x': source_data.astype(np.float32),
            'data_y': source_data.astype(np.float32),
            'data_stamp': data_stamp.astype(np.float32),
            'raw_dates': raw_dates,
            'source_id': os.path.splitext(os.path.basename(data_path))[0],
            'scaler': scaler,
        }
        source['enc_in'] = source['data_x'].shape[-1]
        source['tot_len'] = len(source['data_x']) - self.seq_len - self.pred_len + 1
        source['length'] = source['tot_len'] * source['enc_in']
        return source, (raw_dates[0], raw_dates[-1])

    def _build_aux_sources(self, fine_tune_range):
        aux_sources = []
        main_path = os.path.abspath(os.path.join(self.root_path, self.data_path))
        patterns = ['*.csv', '*.txt', '*.tsv']
        aux_paths = []
        for pattern in patterns:
            aux_paths.extend(glob.glob(os.path.join(self.root_path, pattern)))

        for aux_path in sorted(set(aux_paths)):
            if os.path.abspath(aux_path) == main_path:
                continue

            aux_data_path = os.path.relpath(aux_path, self.root_path)
            try:
                aux_source, _ = self._build_source(
                    self._read_frame(aux_data_path),
                    aux_data_path,
                    use_main_split=False,
                    date_range=fine_tune_range
                )
            except Exception as exc:
                warnings.warn(f'Skipping auxiliary dataset {aux_data_path}: {exc}')
                continue

            if aux_source is not None and aux_source['length'] > 0:
                aux_sources.append(aux_source)

        return aux_sources

    def __getitem__(self, index):
        source = self.sources[0]
        if len(self.sources) > 1:
            source_idx = int(np.searchsorted(self.cumulative_lengths, index, side='right'))
            prev_len = 0 if source_idx == 0 else self.cumulative_lengths[source_idx - 1]
            index = index - prev_len
            source = self.sources[source_idx]

        return self._get_source_item(source, index)

    def _get_source_item(self, source, index):
        tot_len = source['tot_len']
        feat_id = index // tot_len
        s_begin = index % tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = source['data_x'][s_begin:s_end, feat_id:feat_id + 1]
        seq_y = source['data_y'][r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = source['data_stamp'][s_begin:s_end]
        seq_y_mark = source['data_stamp'][r_begin:r_end]

        if self.return_source_id:
            return seq_x, seq_y, seq_x_mark, seq_y_mark, source['source_id']

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return int(self.cumulative_lengths[-1])

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def window_dates(self, index):
        source = self.sources[0]
        if len(self.sources) > 1:
            source_idx = int(np.searchsorted(self.cumulative_lengths, index, side='right'))
            prev_len = 0 if source_idx == 0 else self.cumulative_lengths[source_idx - 1]
            index = index - prev_len
            source = self.sources[source_idx]

        s_begin = index % source['tot_len']
        s_end = s_begin + self.seq_len
        return source['raw_dates'][s_begin], source['raw_dates'][s_end - 1]


class Dataset_M4(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask
