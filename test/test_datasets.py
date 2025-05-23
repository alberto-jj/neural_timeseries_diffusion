import unittest
import torch
import numpy as np
import mne
import os
import shutil
import tempfile
from ntd.datasets import FIFDataLoader

class TestFIFDataLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.n_epochs_per_file = 20
        self.n_channels = 19
        self.n_times = 1000
        self.sfreq = 200

        # --- Create dummy data and save to .fif files ---
        self.file_paths = []
        self.event_id = {'event': 1}
        self.events = np.array([[i, 0, 1] for i in range(self.n_epochs_per_file)])

        # File 1: CLASS1_SUBJ1_eeg.fif
        data1 = np.random.rand(self.n_epochs_per_file, self.n_channels, self.n_times).astype(np.float64)
        epochs1 = mne.EpochsArray(data1, mne.create_info(self.n_channels, self.sfreq, ch_types='eeg'),
                                  events=self.events, event_id=self.event_id, verbose=False)
        fp1 = os.path.join(self.temp_dir, "CLASS1_SUBJ1_eeg.fif")
        epochs1.save(fp1, overwrite=True, verbose=False)
        self.file_paths.append(fp1)

        # File 2: CLASS1_SUBJ2_eeg.fif
        data2 = np.random.rand(self.n_epochs_per_file, self.n_channels, self.n_times).astype(np.float64)
        epochs2 = mne.EpochsArray(data2, mne.create_info(self.n_channels, self.sfreq, ch_types='eeg'),
                                  events=self.events, event_id=self.event_id, verbose=False)
        fp2 = os.path.join(self.temp_dir, "CLASS1_SUBJ2_eeg.fif")
        epochs2.save(fp2, overwrite=True, verbose=False)
        self.file_paths.append(fp2)

        # File 3: CLASS2_SUBJ3_eeg.fif
        data3 = np.random.rand(self.n_epochs_per_file, self.n_channels, self.n_times).astype(np.float64)
        epochs3 = mne.EpochsArray(data3, mne.create_info(self.n_channels, self.sfreq, ch_types='eeg'),
                                  events=self.events, event_id=self.event_id, verbose=False)
        fp3 = os.path.join(self.temp_dir, "CLASS2_SUBJ3_eeg.fif")
        epochs3.save(fp3, overwrite=True, verbose=False)
        self.file_paths.append(fp3)

        # File 4: Insufficient epochs (LESS_SUBJ4_eeg.fif)
        data_less = np.random.rand(self.n_epochs_per_file // 2, self.n_channels, self.n_times).astype(np.float64)
        epochs_less = mne.EpochsArray(data_less, mne.create_info(self.n_channels, self.sfreq, ch_types='eeg'),
                                      events=self.events[:self.n_epochs_per_file // 2], event_id=self.event_id, verbose=False)
        self.fp_less_epochs = os.path.join(self.temp_dir, "LESS_SUBJ4_eeg.fif")
        epochs_less.save(self.fp_less_epochs, overwrite=True, verbose=False)

        # File 5: Mismatched channels (BADCHANNELS_SUBJ5_eeg.fif)
        data_bad_channels = np.random.rand(self.n_epochs_per_file, self.n_channels - 1, self.n_times).astype(np.float64)
        epochs_bad_channels = mne.EpochsArray(data_bad_channels, mne.create_info(self.n_channels - 1, self.sfreq, ch_types='eeg'),
                                             events=self.events, event_id=self.event_id, verbose=False)
        self.fp_bad_channels = os.path.join(self.temp_dir, "BADCHANNELS_SUBJ5_eeg.fif")
        epochs_bad_channels.save(self.fp_bad_channels, overwrite=True, verbose=False)

        # File 6: Mismatched times (BADTIMES_SUBJ6_eeg.fif)
        data_bad_times = np.random.rand(self.n_epochs_per_file, self.n_channels, self.n_times // 2).astype(np.float64)
        epochs_bad_times = mne.EpochsArray(data_bad_times, mne.create_info(self.n_channels, self.sfreq, ch_types='eeg'),
                                          events=self.events, event_id=self.event_id, verbose=False)
        self.fp_bad_times = os.path.join(self.temp_dir, "BADTIMES_SUBJ6_eeg.fif")
        epochs_bad_times.save(self.fp_bad_times, overwrite=True, verbose=False)
        
        # File 7: Invalid FIF file (empty)
        self.fp_invalid_fif = os.path.join(self.temp_dir, "INVALID_SUBJ7_eeg.fif")
        with open(self.fp_invalid_fif, 'w') as f:
            f.write("This is not a fif file.")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_load_single_file_unconditional(self):
        loader = FIFDataLoader(file_path=self.file_paths[0], n_epochs=self.n_epochs_per_file)
        self.assertEqual(len(loader), self.n_epochs_per_file)
        self.assertEqual(loader.data.shape, (self.n_epochs_per_file, self.n_channels, self.n_times))
        self.assertIsInstance(loader.data, torch.FloatTensor)
        item = loader[0]
        self.assertIn("signal", item)
        self.assertNotIn("subject_id", item)
        self.assertNotIn("class_label", item)

    def test_load_directory_unconditional(self):
        # Uses first 3 files created in setUp
        loader = FIFDataLoader(file_path=self.temp_dir, n_epochs=self.n_epochs_per_file)
        self.assertEqual(len(loader), self.n_epochs_per_file * 3) # Assuming 3 valid files are loaded
        self.assertEqual(loader.data.shape, (self.n_epochs_per_file * 3, self.n_channels, self.n_times))
        self.assertIsInstance(loader.data, torch.FloatTensor)

    def test_load_conditional_subject_id(self):
        loader = FIFDataLoader(file_path=self.temp_dir, n_epochs=self.n_epochs_per_file, condition_on_subject_id=True)
        self.assertEqual(len(loader), self.n_epochs_per_file * 3) # SUBJ1, SUBJ2, SUBJ3
        self.assertIsNotNone(loader.subject_ids)
        self.assertIsInstance(loader.subject_ids, torch.LongTensor)
        self.assertEqual(loader.subject_ids.shape[0], self.n_epochs_per_file * 3)
        
        # Check if IDs are assigned correctly based on sorted file loading
        # Expected IDs: SUBJ1 -> 0, SUBJ2 -> 1, SUBJ3 -> 2
        # The actual subject strings are SUBJ1, SUBJ2, SUBJ3 (from file_paths[0], file_paths[1], file_paths[2])
        # Their int IDs will depend on the order they are processed by glob and then sorted.
        # setUp creates them as CLASS1_SUBJ1, CLASS1_SUBJ2, CLASS2_SUBJ3. Sorted they are.
        # Then BADCHANNELS_SUBJ5, BADTIMES_SUBJ6, INVALID_SUBJ7, LESS_SUBJ4
        # Valid files loaded are CLASS1_SUBJ1, CLASS1_SUBJ2, CLASS2_SUBJ3
        # Subject string to int ID map: {'SUBJ1':0, 'SUBJ2':1, 'SUBJ3':2}
        
        expected_subj_ids_flat = []
        for i in range(3): # SUBJ1, SUBJ2, SUBJ3
             expected_subj_ids_flat.extend([i] * self.n_epochs_per_file)
        
        self.assertTrue(torch.equal(loader.subject_ids, torch.tensor(expected_subj_ids_flat, dtype=torch.long)))
        item = loader[0]
        self.assertIn("subject_id", item)

    def test_load_conditional_class_label(self):
        loader = FIFDataLoader(file_path=self.temp_dir, n_epochs=self.n_epochs_per_file, condition_on_class_label=True)
        self.assertEqual(len(loader), self.n_epochs_per_file * 3)
        self.assertIsNotNone(loader.class_labels)
        self.assertIsInstance(loader.class_labels, torch.LongTensor)
        self.assertEqual(loader.class_labels.shape[0], self.n_epochs_per_file * 3)

        # Expected labels: CLASS1 -> 0, CLASS2 -> 1
        # Files: CLASS1_SUBJ1, CLASS1_SUBJ2, CLASS2_SUBJ3
        # Class string to int ID map: {'CLASS1':0, 'CLASS2':1}
        expected_class_ids_flat = []
        expected_class_ids_flat.extend([0] * self.n_epochs_per_file) # CLASS1_SUBJ1
        expected_class_ids_flat.extend([0] * self.n_epochs_per_file) # CLASS1_SUBJ2
        expected_class_ids_flat.extend([1] * self.n_epochs_per_file) # CLASS2_SUBJ3
        self.assertTrue(torch.equal(loader.class_labels, torch.tensor(expected_class_ids_flat, dtype=torch.long)))
        item = loader[0]
        self.assertIn("class_label", item)

    def test_insufficient_epochs(self):
        # This test expects that the file with fewer epochs is skipped,
        # and if other valid files exist, they are loaded.
        # If only this file was provided, it should raise ValueError.
        
        # Scenario 1: Only the insufficient file is provided
        with self.assertRaisesRegex(ValueError, "No valid data loaded."):
            FIFDataLoader(file_path=self.fp_less_epochs, n_epochs=self.n_epochs_per_file)

        # Scenario 2: Insufficient file along with valid files.
        # The dataloader should skip the insufficient file and load the others.
        # Create a new temp dir for this specific test to isolate files.
        specific_test_dir = tempfile.mkdtemp()
        shutil.copy(self.file_paths[0], specific_test_dir) # Copy a valid file
        shutil.copy(self.fp_less_epochs, specific_test_dir) # Copy the insufficient epoch file
        
        try:
            # Expect a warning print, but unittest can't easily check stdout without more setup.
            # We check that only the valid file's epochs are loaded.
            loader = FIFDataLoader(file_path=specific_test_dir, n_epochs=self.n_epochs_per_file)
            self.assertEqual(len(loader), self.n_epochs_per_file) # Only one valid file loaded
        finally:
            shutil.rmtree(specific_test_dir)


    def test_invalid_file_path(self):
        with self.assertRaisesRegex(ValueError, "No .fif files found in /non/existent/path."):
            FIFDataLoader(file_path="/non/existent/path", n_epochs=self.n_epochs_per_file)

    def test_invalid_fif_file(self):
        # Expect the dataloader to skip this file (due to MNE read error) and raise ValueError if no other files are loaded.
        with self.assertRaisesRegex(ValueError, "No valid data loaded."):
             FIFDataLoader(file_path=self.fp_invalid_fif, n_epochs=self.n_epochs_per_file)
        
        # Test with a valid file and an invalid one - should load the valid one
        specific_test_dir_invalid = tempfile.mkdtemp()
        shutil.copy(self.file_paths[0], specific_test_dir_invalid)
        shutil.copy(self.fp_invalid_fif, specific_test_dir_invalid)
        try:
            loader = FIFDataLoader(file_path=specific_test_dir_invalid, n_epochs=self.n_epochs_per_file)
            self.assertEqual(len(loader), self.n_epochs_per_file) # Only the valid file
        finally:
            shutil.rmtree(specific_test_dir_invalid)


    def test_mismatched_channels(self):
        # Expect the dataloader to skip this file and raise ValueError if no other files are loaded.
        with self.assertRaisesRegex(ValueError, "No valid data loaded."):
            FIFDataLoader(file_path=self.fp_bad_channels, n_epochs=self.n_epochs_per_file)

        # Test with a valid file and a mismatched channel one - should load the valid one
        specific_test_dir_ch = tempfile.mkdtemp()
        shutil.copy(self.file_paths[0], specific_test_dir_ch)
        shutil.copy(self.fp_bad_channels, specific_test_dir_ch)
        try:
            loader = FIFDataLoader(file_path=specific_test_dir_ch, n_epochs=self.n_epochs_per_file)
            self.assertEqual(len(loader), self.n_epochs_per_file) # Only the valid file
        finally:
            shutil.rmtree(specific_test_dir_ch)


    def test_mismatched_times(self):
        # Expect the dataloader to skip this file and raise ValueError if no other files are loaded.
        with self.assertRaisesRegex(ValueError, "No valid data loaded."):
            FIFDataLoader(file_path=self.fp_bad_times, n_epochs=self.n_epochs_per_file)

        # Test with a valid file and a mismatched time one - should load the valid one
        specific_test_dir_tm = tempfile.mkdtemp()
        shutil.copy(self.file_paths[0], specific_test_dir_tm)
        shutil.copy(self.fp_bad_times, specific_test_dir_tm)
        try:
            loader = FIFDataLoader(file_path=specific_test_dir_tm, n_epochs=self.n_epochs_per_file)
            self.assertEqual(len(loader), self.n_epochs_per_file) # Only the valid file
        finally:
            shutil.rmtree(specific_test_dir_tm)


if __name__ == '__main__':
    unittest.main()
