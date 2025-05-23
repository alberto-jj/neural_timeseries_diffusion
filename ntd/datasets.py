import os

import numpy as np
import torch
import xarray as xr
from scipy import io
from torch.utils.data.dataset import Dataset
import mne
import glob

from ntd.utils.utils import standardize_array

### Datasets

# Macaque ECoG session times
# Session : 1
# time: 12.28 [s] : AwakeEyesOpened-Start
# time: 908.91 [s] : AwakeEyesOpened-End
# time: 1007.11 [s] : AwakeEyesClosed-Start
# time: 1927.43 [s] : AwakeEyesClosed-End

# Session : 2
# time: 85.98 [s] : AnestheticInjection
# time: 553.55 [s] : Anesthetized-Start
# time: 1098.10 [s] : Anesthetized-End
# time: 2389.56 [s] : RecoveryEyesClosed-Start
# time: 3590.76 [s] : RecoveryEyesClosed-End

# Session : 3
# time: 38.51 [s] : RecoveryEyesOpened-Start
# time: 1277.81 [s] : RecoveryEyesOpened-End


def load_ecog_data(session_path, channels, signal_length, start_time, end_time):
    """
    Load ECoG data for a given session.

    Args:
        session_path: Path to session folder
        channels: List of channels to load and arange
        signal_length: Length of time windows to split into
        start_time: Time in seconds to start loading data from
        end_time: Time in seconds to stop loading data from

    Returns:
        Array of shape (num_time_windows, num_channels, signal_length)
    """

    assert start_time < end_time
    time_file = os.path.join(session_path, "ECoGTime.mat")
    time = io.loadmat(time_file)
    time = time["ECoGTime"].squeeze()
    traces = []
    for chan in channels:
        ecog_file = os.path.join(session_path, f"ECoG_ch{chan}.mat")
        data = io.loadmat(ecog_file)
        traces += [data[f"ECoGData_ch{chan}"]]
    ecog = np.concatenate(traces, axis=0)
    time_window = np.where((time > start_time) & (time < end_time))[0]
    ecog_time_window = ecog[:, time_window]

    splitter = (
        np.arange(1, (ecog_time_window.shape[1] // signal_length) + 1, dtype=int)
        * signal_length
    )
    splitted = np.split(ecog, splitter, axis=1)
    return np.array(splitted[:-1])


class FIFDataLoader(Dataset):
    """
    Dataset loader for EEG data from .fif files.

    Processes data into tensors and optionally includes subject IDs and class labels.
    Assumes filenames follow a convention like CLASSLABEL_SUBJECTID_*.fif for label extraction.
    """

    def __init__(
        self,
        file_path,
        n_epochs,
        condition_on_subject_id=False,
        condition_on_class_label=False,
    ):
        """
        Initializes the FIFDataLoader.

        Args:
            file_path (str): Path to a directory of .fif files or a single .fif file.
            n_epochs (int): Number of epochs to use from each subject/file.
                            The data is expected to have 20 epochs, each 5s long (1000 time points at 200Hz).
            condition_on_subject_id (bool): If True, extract and store subject IDs.
            condition_on_class_label (bool): If True, extract and store class labels.
        """
        self.file_path = file_path
        self.n_epochs_per_subject = n_epochs # n_epochs to use from each subject's file
        self.condition_on_subject_id = condition_on_subject_id
        self.condition_on_class_label = condition_on_class_label

        all_epoch_data = []
        all_subject_ids_str = []
        all_class_labels_str = []
        
        self.label_to_int_id = {}
        self.next_class_int_id = 0
        self.subject_str_to_int_id = {}
        self.next_subject_int_id = 0

        # Expected data parameters
        self.expected_channels = 19
        self.expected_time_points = 1000 # 5 seconds * 200 Hz

        if os.path.isdir(self.file_path):
            # Sort files for deterministic order
            file_paths = sorted(glob.glob(os.path.join(self.file_path, "*.fif")))
        elif os.path.isfile(self.file_path) and self.file_path.endswith(".fif"):
            file_paths = [self.file_path]
        else:
            raise ValueError(
                "file_path must be a valid .fif file or a directory containing .fif files."
            )

        if not file_paths:
            raise ValueError(f"No .fif files found in {self.file_path}.")

        for fp in file_paths:
            try:
                # verbose='WARNING' to reduce MNE's console output
                epochs = mne.read_epochs(fp, preload=True, verbose="WARNING")
            except Exception as e:
                print(f"Warning: Could not read or process {fp}. Skipping. Error: {e}")
                continue

            if len(epochs) < self.n_epochs_per_subject:
                print(
                    f"Warning: Not enough epochs in {fp} (found {len(epochs)}, required {self.n_epochs_per_subject}). Skipping this file."
                )
                continue

            # Data shape: (n_epochs, n_channels, n_times)
            # MNE's get_data() returns data in Volts.
            # Ensure data is float64 for mne processing, will be converted to float32 for torch later.
            # Select the specified n_epochs_per_subject and up to expected_time_points
            epoch_data = epochs.get_data(picks="eeg").astype(np.float64)[
                : self.n_epochs_per_subject, :, : self.expected_time_points
            ]
            
            # Validate data shape
            if epoch_data.shape[1] != self.expected_channels:
                print(
                    f"Warning: Data in {fp} has {epoch_data.shape[1]} channels, expected {self.expected_channels}. Skipping this file."
                )
                continue
            if epoch_data.shape[2] != self.expected_time_points:
                print(
                    f"Warning: Data in {fp} has {epoch_data.shape[2]} time points, expected {self.expected_time_points}. Skipping this file."
                )
                continue
            
            all_epoch_data.append(epoch_data)

            filename = os.path.basename(fp)
            parts = filename.split("_") # e.g., CLASSLABEL_SUBJECTID_*.fif

            # Extract class label
            if self.condition_on_class_label:
                if len(parts) >= 2: # Expect at least CLASSLABEL_SUBJECTID
                    class_label_str = parts[0]
                    if class_label_str not in self.label_to_int_id:
                        self.label_to_int_id[class_label_str] = self.next_class_int_id
                        self.next_class_int_id += 1
                    all_class_labels_str.extend([class_label_str] * epoch_data.shape[0])
                else:
                    default_class_label = "UnknownClass"
                    print(
                        f"Warning: Filename {filename} does not follow CLASSLABEL_SUBJECTID_*.fif convention for class label. Assigning '{default_class_label}'."
                    )
                    if default_class_label not in self.label_to_int_id:
                        self.label_to_int_id[default_class_label] = self.next_class_int_id
                        self.next_class_int_id +=1
                    all_class_labels_str.extend([default_class_label] * epoch_data.shape[0])

            # Extract subject ID
            if self.condition_on_subject_id:
                if len(parts) >= 2: # Expect at least CLASSLABEL_SUBJECTID
                    subject_id_str = parts[1]
                    if subject_id_str not in self.subject_str_to_int_id:
                        self.subject_str_to_int_id[subject_id_str] = self.next_subject_int_id
                        self.next_subject_int_id += 1
                    all_subject_ids_str.extend([subject_id_str] * epoch_data.shape[0])
                else: # Fallback if filename doesn't have enough parts
                    default_subject_id = filename.split(".")[0] # Use filename without extension as ID
                    print(
                        f"Warning: Filename {filename} does not follow CLASSLABEL_SUBJECTID_*.fif convention for subject ID. Using '{default_subject_id}' as ID."
                    )
                    if default_subject_id not in self.subject_str_to_int_id:
                        self.subject_str_to_int_id[default_subject_id] = self.next_subject_int_id
                        self.next_subject_int_id += 1
                    all_subject_ids_str.extend([default_subject_id] * epoch_data.shape[0])

        if not all_epoch_data:
            raise ValueError(
                "No valid data loaded. All files were skipped or resulted in errors. Check warnings for details."
            )

        # Concatenate data from all files
        self.data_array_np = np.concatenate(all_epoch_data, axis=0)
        # Standardize data across epochs and time dimensions (mean 0, std 1)
        self.data_array_np = standardize_array(self.data_array_np, ax=(0, 2)) # ax=(0,2) standardizes each channel independently over epochs and time
        self.data = torch.from_numpy(self.data_array_np).float() # Convert to float32 tensor

        if self.condition_on_subject_id:
            self.subject_ids = torch.tensor(
                [self.subject_str_to_int_id[sid_str] for sid_str in all_subject_ids_str], dtype=torch.long
            )
        else:
            self.subject_ids = None # Keep as None if not conditioning

        if self.condition_on_class_label:
            self.class_labels = torch.tensor(
                [self.label_to_int_id[cls_str] for cls_str in all_class_labels_str], dtype=torch.long
            )
        else:
            self.class_labels = None # Keep as None if not conditioning

    def __len__(self):
        """Returns the total number of epochs accumulated from all processed files."""
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns a dictionary containing the signal and optional subject ID and class label for a given epoch index.

        Args:
            index (int): Index of the epoch.

        Returns:
            dict: A dictionary with 'signal', and optionally 'subject_id' and 'class_label' tensors.
        """
        item = {"signal": self.data[index]}
        if self.condition_on_subject_id and self.subject_ids is not None:
            item["subject_id"] = self.subject_ids[index]
        if self.condition_on_class_label and self.class_labels is not None:
            item["class_label"] = self.class_labels[index]
        return item


class TychoConditionalDataset(Dataset):
    """
    Dataset of for the macaque ECoG data.

    Conditional version: Awake and anesthetized data is used.
    Provides class information.
    """

    def __init__(
        self,
        signal_length=1000,
        channels=[1, 2, 3, 4],
        session_one_start=1007.11,
        session_one_end=1007.11 + 544.55,
        session_two_start=553.55,
        session_two_end=1098.10,
        with_class_cond=False,
        filepath=None,
    ):
        super().__init__()
        self.signal_length = signal_length
        self.with_class_cond = with_class_cond

        if channels == "all":
            channels = [i for i in range(1, 129)]

        ### Session 1 is awake
        filepath_sess1 = os.path.join(filepath, "Session1")
        self.array_after_anesthetized = load_ecog_data(
            filepath_sess1, channels, signal_length, session_one_start, session_one_end
        )

        ### Session 2 contains anesthetized
        filepath_sess2 = os.path.join(filepath, "Session2")
        self.array_anesthetized = load_ecog_data(
            filepath_sess2, channels, signal_length, session_two_start, session_two_end
        )

        self.cond_vector = np.concatenate(
            (
                np.ones(len(self.array_anesthetized)),
                np.zeros(len(self.array_after_anesthetized)),
            )
        )
        temp_array = np.concatenate(
            [self.array_anesthetized, self.array_after_anesthetized], axis=0
        )
        self.array, self.arr_mean, self.arr_std = standardize_array(
            temp_array, ax=(0, 2), return_mean_std=True
        )

    def __getitem__(self, index):
        return_dict = {}
        return_dict["signal"] = torch.from_numpy(np.float32(self.array[index]))
        class_mem = self.cond_vector[index]
        return_dict["label"] = torch.tensor([np.float32(class_mem)])
        cond = self.get_cond(class_mem=class_mem)
        if cond is not None:
            return_dict["cond"] = cond
        return return_dict

    def get_cond(self, class_mem=0.0):
        if not self.with_class_cond:
            return None
        to_cat = []
        if self.with_class_cond:
            assert class_mem in [0.0, 1.0]
            to_cat.append(class_mem * torch.ones(1, self.signal_length))
        return torch.cat(to_cat, dim=0)

    def __len__(self):
        return len(self.array)


class TychoUnconditionalDataset(Dataset):
    """
    Dataset of for the macaque ECoG data.

    Unconditional version: Only awake data is used.
    """

    def __init__(
        self,
        signal_length,
        channels=[1, 2, 3, 4],
        session_one_start=0.0,
        session_one_end=10000.0,
        filepath=None,
    ):
        super().__init__()
        self.signal_length = signal_length

        if channels == "all":
            channels = [i for i in range(1, 129)]
        self.num_channels = len(channels)

        filepath_sess1 = os.path.join(filepath, "Session1")
        self.array = standardize_array(
            load_ecog_data(
                filepath_sess1,
                channels,
                signal_length,
                session_one_start,
                session_one_end,
            ),
            ax=(0, 2),
        )

    def __getitem__(self, index):
        return_dict = {}
        return_dict["signal"] = torch.from_numpy(np.float32(self.array[index]))
        cond = self.get_cond()
        if cond is not None:
            return_dict["cond"] = cond
        return return_dict

    def get_cond(self):
        return None

    def __len__(self):
        return len(self.array)


class NER_BCI(Dataset):
    """
    Dataset of for the Inria EEG BCI data.

    Supports positional embeddings.
    """

    def __init__(
        self,
        patient_id,
        filepath=None,
    ):
        super().__init__()

        self.signal_length = 260
        self.num_channels = 56

        temp_array = np.load(os.path.join(filepath, f"{patient_id}_data.npy"))
        self.data_array = standardize_array(temp_array, ax=(0, 2))

    def __getitem__(self, index):
        return_dict = {}
        return_dict["signal"] = torch.from_numpy(np.float32(self.data_array[index]))
        cond = self.get_cond()
        if cond is not None:
            return_dict["cond"] = cond
        return return_dict

    def get_cond(self):
        return None

    def __len__(self):
        return len(self.data_array)


class CRCNS_LFP(Dataset):
    """
    Dataset of for the CRCNS rat LFP data.
    """

    def __init__(
        self,
        sessions=["Session1_PETHphaseLocking", "Session3_SleepAfterRun"],
        lfp_types=["HC_LFP", "PFC_LFP", "THAL_LFP"],
        signal_length=2 * 600,
        filepath=None,
    ):
        super().__init__()
        self.signal_length = signal_length
        self.num_signal_channel = len(lfp_types)

        session_arrays = []
        for sess in sessions:
            lfp_arr_ls = []
            for lfp in lfp_types:
                mat_res = io.loadmat(
                    os.path.join(
                        filepath,
                        f"{sess}/PrimaryData/LFPs",
                        f"{lfp}_600Hz.mat",
                    )
                )
                sig = mat_res[lfp]
                splitter = (
                    np.arange(1, (sig.shape[1] // signal_length) + 1, dtype=int)
                    * signal_length
                )
                splitted = np.split(sig, splitter, axis=1)
                lfp_arr_ls.append(np.array(splitted[:-1]))
            session_arrays.append(
                standardize_array(np.concatenate(lfp_arr_ls, axis=1), ax=(0, 2))
            )

        self.array = np.concatenate(session_arrays, axis=0)

    def __getitem__(self, index):
        return_dict = {}
        return_dict["signal"] = torch.from_numpy(np.float32(self.array[index]))
        cond = self.get_cond()
        if cond is not None:
            return_dict["cond"] = cond
        return return_dict

    def get_cond(self):
        return None

    def __len__(self):
        return len(self.array)


class AJILE_LFP(Dataset):
    """
    AJILE dataset.

    Provides class information and supports conditional and masked training.
    """

    def __init__(
        self,
        signal_length=1001,
        data_array=None,
        label_array=None,
        mask_array=None,
        train_mean=None,
        train_std=None,
        conditional_training=False,
        masked_training=True,
    ):
        super().__init__()
        self.conditional_training = conditional_training
        self.masked_training = masked_training

        _num_time_windows, self.num_channels, self.signal_length = data_array.shape
        assert self.signal_length == signal_length
        self.data_array = data_array
        self.label_array = label_array
        self.mask_array = mask_array
        self.train_mean = train_mean
        self.train_std = train_std

    def __getitem__(self, index):
        return_dict = {}
        return_dict["signal"] = torch.from_numpy(np.float32(self.data_array[index]))
        return_dict["label"] = torch.tensor([np.float32(self.label_array[index])])
        cond = self.get_cond(index=index)
        if cond is not None and self.masked_training:
            return_dict["mask"] = torch.from_numpy(np.float32(self.mask_array[index]))
            cond[self.num_channels :] *= return_dict["mask"]
            cond[: self.num_channels] *= return_dict["mask"]
            return_dict["cond"] = cond
        if cond is not None:
            return_dict["cond"] = cond
        if self.masked_training:
            return_dict["mask"] = torch.from_numpy(np.float32(self.mask_array[index]))
        return return_dict

    def get_cond(self, index=0):
        cond = None
        if self.conditional_training:
            condition_mask = torch.zeros(self.num_channels, self.signal_length)
            num_cond_channels = np.random.randint(self.num_channels + 1)
            cond_channel = list(
                np.random.choice(
                    self.num_channels,
                    size=num_cond_channels,
                    replace=False,
                )
            )
            # Cond channels are indicated by 1.0
            condition_mask[cond_channel, :] = 1.0
            condition_signal = (
                torch.from_numpy(np.float32(self.data_array[index])) * condition_mask
            )
            cond = torch.cat((condition_mask, condition_signal), dim=0)
        return cond

    def get_train_mean_and_std(self):
        return torch.from_numpy(np.float32(self.train_mean)), torch.from_numpy(
            np.float32(self.train_std)
        )

    def __len__(self):
        return len(self.data_array)


def load_ajile_data(
    pat_id,
    lower_q=25,
    upper_q=75,
    iqr_factor=5.0,
    time_before_event=2.0,  # in seconds
    time_after_event=2.0,  # in seconds
    filepath=None,
):
    """
    Adapted from https://github.com/BruntonUWBio/HTNet_generalized_decoding
    Load data from preprocessed AJILE dataset and peform standardization and outlier detection.
    Outlier detection is based on the range (max - min) of each channel. Channels with a range beyond
    the iqr_factor are marked as outliers and are ignored during standardization.

    Args:
        pat_id: AJILE Patient ID ("001" - "012")
        lower_q: Lower percentile for outlier detection
        upper_q: Upper percentile for outlier detection
        iqr_factor: Inter quantile factor for outlier detection
        time_before_event: Time before event in seconds
        time_after_event: Time after event in seconds

    Returns:
        X_train: Training data, array of shape (num_time_windows, num_channels, signal_length)
        y_train: Training labels
        mask_train: Outlier mask for training data
        X_test: Test data
        y_test: Test labels
        mask_test: Outlier mask for test data
        fin_mean: Mean used for standardization
        fin_std: Standard deviation used for standardization
    """

    # Gather each subjects data, and concatenate all days
    ep_data_in = xr.open_dataset(os.path.join(filepath, f"{pat_id}_ecog_data.nc"))
    ep_times = np.asarray(ep_data_in.time)
    time_inds = np.nonzero(
        np.logical_and(ep_times >= -time_before_event, ep_times <= time_after_event)
    )[0]
    n_ecog_chans = len(ep_data_in.channels) - 1

    days_all_in = np.asarray(ep_data_in.events)
    days_train = np.unique(days_all_in)[:-1]
    day_test_curr = np.unique(days_all_in)[-1]
    days_test_inds = np.nonzero(days_all_in == day_test_curr)[0]

    # Compute indices of days_train in xarray dataset
    days_train_inds = []
    for day_tmp in list(days_train):
        days_train_inds.extend(np.nonzero(days_all_in == day_tmp)[0])

    # Extract data and labels
    X_train = (
        ep_data_in[
            dict(
                events=days_train_inds,
                channels=slice(0, n_ecog_chans),
                time=time_inds,
            )
        ]
        .to_array()
        .values.squeeze()
    )
    y_train = -1.0 + (
        ep_data_in[
            dict(events=days_train_inds, channels=ep_data_in.channels[-1], time=0)
        ]
        .to_array()
        .values.squeeze()
    )
    X_test = (
        ep_data_in[
            dict(
                events=days_test_inds,
                channels=slice(0, n_ecog_chans),
                time=time_inds,
            )
        ]
        .to_array()
        .values.squeeze()
    )
    y_test = -1.0 + (
        ep_data_in[
            dict(events=days_test_inds, channels=ep_data_in.channels[-1], time=0)
        ]
        .to_array()
        .values.squeeze()
    )

    _num_train_samples, num_channel, signal_length = X_train.shape
    _num_test_samples, test_num_channel, test_signal_length = X_train.shape
    assert num_channel == test_num_channel
    assert signal_length == test_signal_length

    # Remove outliers
    assert lower_q < upper_q
    assert iqr_factor >= 0.0

    train_ranges = np.ptp(X_train, axis=2)  # B, C
    test_ranges = np.ptp(X_test, axis=2)  # B, C

    lower = np.percentile(train_ranges, lower_q, axis=0, keepdims=True)  # (B), C
    upper = np.percentile(train_ranges, upper_q, axis=0, keepdims=True)
    iqr = upper - lower

    train_mask = (
        (train_ranges == 0.0)
        | (train_ranges < lower - (iqr_factor * iqr))
        | (train_ranges > upper + (iqr_factor * iqr))
    )
    train_mask = np.repeat(train_mask[:, :, np.newaxis], signal_length, axis=2)

    test_mask = (
        (test_ranges == 0.0)
        | (test_ranges < lower - (iqr_factor * iqr))
        | (test_ranges > upper + (iqr_factor * iqr))
    )
    test_mask = np.repeat(test_mask[:, :, np.newaxis], signal_length, axis=2)

    # Values with True are masked and ignored
    masked_train_array = np.ma.array(X_train, mask=train_mask)
    fin_mean = np.ma.mean(masked_train_array, axis=(0, 2), keepdims=True)
    fin_std = np.ma.std(masked_train_array, axis=(0, 2), keepdims=True)

    # Same standardization for both train and test
    X_train = (X_train - fin_mean) / fin_std
    X_test = (X_test - fin_mean) / fin_std

    return (
        X_train,
        y_train,
        1.0 - train_mask.astype(float),
        X_test,
        y_test,
        1.0 - test_mask.astype(float),
        fin_mean,
        fin_std,
    )
