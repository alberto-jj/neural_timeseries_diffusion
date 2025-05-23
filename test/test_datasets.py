import unittest
import torch
import numpy as np
import mne
import os
import shutil
import tempfile
from ntd.datasets import FIFDataLoader

class TestFIFDataLoader(unittest.TestCase):
    def _create_fif_file(self, dir_path, filename_prefix, sfreq, n_channels, n_times, n_epochs, event_id_val=1):
        """Helper function to create a dummy FIF file."""
        data = np.random.rand(n_epochs, n_channels, n_times).astype(np.float64) * 10e-6
        ch_names = [f'EEG {i+1:02}' for i in range(n_channels)]
        ch_types = ['eeg'] * n_channels
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        info.set_montage('standard_1020', on_missing='ignore')
        
        events = np.array([[i, 0, event_id_val] for i in range(n_epochs)])
        event_id_dict = {f'event{event_id_val}': event_id_val}
        
        epochs_obj = mne.EpochsArray(data, info, events=events, event_id=event_id_dict, tmin=0, verbose=False)
        
        file_path = os.path.join(dir_path, f"{filename_prefix}_eeg.fif")
        epochs_obj.save(file_path, overwrite=True, verbose=False)
        return file_path

    def setUp(self):
        self.temp_dir_base = tempfile.mkdtemp() # Base for all test temp dirs

        # Standard parameters for most tests
        self.std_sfreq = 200.0
        self.std_n_channels = 19
        self.std_n_times = 1000
        self.std_n_epochs = 20 # Max epochs in standard files

        # --- Create a general directory with consistent files for basic tests ---
        self.general_data_dir = os.path.join(self.temp_dir_base, "general_data")
        os.makedirs(self.general_data_dir, exist_ok=True)

        self.fp1 = self._create_fif_file(self.general_data_dir, "CONSISTENT_SUBJ1", 
                                          self.std_sfreq, self.std_n_channels, self.std_n_times, self.std_n_epochs)
        self.fp2 = self._create_fif_file(self.general_data_dir, "CONSISTENT_SUBJ2", 
                                          self.std_sfreq, self.std_n_channels, self.std_n_times, self.std_n_epochs)
        # File with fewer epochs than std_n_epochs, but otherwise consistent
        self.fp_less_epochs_std_params = self._create_fif_file(self.general_data_dir, "LESSEPOCHS_SUBJ3",
                                                              self.std_sfreq, self.std_n_channels, self.std_n_times, 10)

        # --- Files for invalid/corrupted tests ---
        self.invalid_files_dir = os.path.join(self.temp_dir_base, "invalid_files")
        os.makedirs(self.invalid_files_dir, exist_ok=True)
        self.fp_invalid_fif = os.path.join(self.invalid_files_dir, "INVALID_SUBJ7_eeg.fif")
        with open(self.fp_invalid_fif, 'w') as f:
            f.write("This is not a fif file.")

        # --- Directories for inconsistency tests ---
        # Files are named such that the default sorted glob order is S1, S2, S3
        self.inconsistent_sfreq_dir = os.path.join(self.temp_dir_base, "inconsistent_sfreq")
        os.makedirs(self.inconsistent_sfreq_dir, exist_ok=True)
        self.s1_sfreq200 = self._create_fif_file(self.inconsistent_sfreq_dir, "FILE_S1_SFREQ200", 200.0, self.std_n_channels, self.std_n_times, 5)
        self.s2_sfreq250 = self._create_fif_file(self.inconsistent_sfreq_dir, "FILE_S2_SFREQ250", 250.0, self.std_n_channels, self.std_n_times, 5) # Inconsistent
        self.s3_sfreq200 = self._create_fif_file(self.inconsistent_sfreq_dir, "FILE_S3_SFREQ200", 200.0, self.std_n_channels, self.std_n_times, 5)

        self.inconsistent_channels_dir = os.path.join(self.temp_dir_base, "inconsistent_channels")
        os.makedirs(self.inconsistent_channels_dir, exist_ok=True)
        self.s1_chan19 = self._create_fif_file(self.inconsistent_channels_dir, "FILE_S1_CHAN19", self.std_sfreq, 19, self.std_n_times, 5)
        self.s2_chan10 = self._create_fif_file(self.inconsistent_channels_dir, "FILE_S2_CHAN10", self.std_sfreq, 10, self.std_n_times, 5) # Inconsistent
        self.s3_chan19 = self._create_fif_file(self.inconsistent_channels_dir, "FILE_S3_CHAN19", self.std_sfreq, 19, self.std_n_times, 5)

        self.inconsistent_times_dir = os.path.join(self.temp_dir_base, "inconsistent_times")
        os.makedirs(self.inconsistent_times_dir, exist_ok=True)
        self.s1_times1000 = self._create_fif_file(self.inconsistent_times_dir, "FILE_S1_TIMES1000", self.std_sfreq, self.std_n_channels, 1000, 5)
        self.s2_times500 = self._create_fif_file(self.inconsistent_times_dir, "FILE_S2_TIMES500", self.std_sfreq, self.std_n_channels, 500, 5) # Inconsistent
        self.s3_times1000 = self._create_fif_file(self.inconsistent_times_dir, "FILE_S3_TIMES1000", self.std_sfreq, self.std_n_channels, 1000, 5)
        
        # --- Directory for adopted parameters test ---
        self.adopted_params_dir = os.path.join(self.temp_dir_base, "adopted_params")
        os.makedirs(self.adopted_params_dir, exist_ok=True)
        self.alt_sfreq = 100.0
        self.alt_n_channels = 10
        self.alt_n_times = 500
        self.alt_n_epochs = 15
        self._create_fif_file(self.adopted_params_dir, "ALT_PARAM_S1", self.alt_sfreq, self.alt_n_channels, self.alt_n_times, self.alt_n_epochs)
        self._create_fif_file(self.adopted_params_dir, "ALT_PARAM_S2", self.alt_sfreq, self.alt_n_channels, self.alt_n_times, self.alt_n_epochs)


    def tearDown(self):
        shutil.rmtree(self.temp_dir_base)

    def test_load_single_file_unconditional_all_epochs(self):
        # n_epochs = None (load all available: std_n_epochs = 20)
        loader = FIFDataLoader(file_path=self.fp1, n_epochs=None)
        self.assertEqual(len(loader), self.std_n_epochs)
        self.assertEqual(loader.data.shape, (self.std_n_epochs, self.std_n_channels, self.std_n_times))
        self.assertIsInstance(loader.data, torch.FloatTensor)
        self.assertEqual(loader.sfreq, self.std_sfreq)
        self.assertEqual(loader.n_channels, self.std_n_channels)
        self.assertEqual(loader.n_times, self.std_n_times)
        item = loader[0]
        self.assertIn("signal", item)
        self.assertNotIn("subject_id", item)
        self.assertNotIn("class_label", item)

    def test_load_single_file_n_epochs_less_than_available(self):
        # Request 10 epochs, file has 20 (self.fp1)
        requested_epochs = 10
        loader = FIFDataLoader(file_path=self.fp1, n_epochs=requested_epochs)
        self.assertEqual(len(loader), requested_epochs)
        self.assertEqual(loader.data.shape, (requested_epochs, self.std_n_channels, self.std_n_times))

    def test_load_single_file_n_epochs_more_than_available(self):
        # Request 30 epochs, file has 20 (self.fp1)
        # The file fp_less_epochs_std_params has 10 epochs.
        loader = FIFDataLoader(file_path=self.fp_less_epochs_std_params, n_epochs=30)
        self.assertEqual(len(loader), 10) # Loads all available (10)
        self.assertEqual(loader.data.shape, (10, self.std_n_channels, self.std_n_times))


    def test_load_single_file_n_epochs_zero(self):
        # n_epochs = 0 (load all available: std_n_epochs = 20 from fp1)
        loader = FIFDataLoader(file_path=self.fp1, n_epochs=0)
        self.assertEqual(len(loader), self.std_n_epochs)
        self.assertEqual(loader.data.shape, (self.std_n_epochs, self.std_n_channels, self.std_n_times))

    def test_load_directory_unconditional_n_epochs_none(self):
        # Uses general_data_dir: fp1 (20 epochs), fp2 (20 epochs), fp_less_epochs_std_params (10 epochs)
        # Total = 20 + 20 + 10 = 50
        loader = FIFDataLoader(file_path=self.general_data_dir, n_epochs=None)
        self.assertEqual(len(loader), self.std_n_epochs * 2 + 10) 
        self.assertEqual(loader.data.shape, (self.std_n_epochs * 2 + 10, self.std_n_channels, self.std_n_times))
        self.assertIsInstance(loader.data, torch.FloatTensor)
        self.assertEqual(loader.sfreq, self.std_sfreq)
        self.assertEqual(loader.n_channels, self.std_n_channels)
        self.assertEqual(loader.n_times, self.std_n_times)

    def test_load_directory_n_epochs_fixed(self):
        # Uses general_data_dir. Request n_epochs = 15.
        # fp1 (20 epochs) -> loads 15
        # fp2 (20 epochs) -> loads 15
        # fp_less_epochs_std_params (10 epochs) -> loads 10
        # Total = 15 + 15 + 10 = 40
        requested_epochs = 15
        loader = FIFDataLoader(file_path=self.general_data_dir, n_epochs=requested_epochs)
        self.assertEqual(len(loader), requested_epochs * 2 + 10)
        self.assertEqual(loader.data.shape, (requested_epochs * 2 + 10, self.std_n_channels, self.std_n_times))

    def test_load_conditional_subject_id(self):
        # Uses general_data_dir. n_epochs=None. Files: CONSISTENT_SUBJ1, CONSISTENT_SUBJ2, LESSEPOCHS_SUBJ3
        # Expected total epochs = 20+20+10 = 50
        loader = FIFDataLoader(file_path=self.general_data_dir, n_epochs=None, condition_on_subject_id=True)
        self.assertEqual(len(loader), self.std_n_epochs * 2 + 10)
        self.assertIsNotNone(loader.subject_ids)
        self.assertIsInstance(loader.subject_ids, torch.LongTensor)
        self.assertEqual(loader.subject_ids.shape[0], self.std_n_epochs * 2 + 10)
        
        # Files are CONSISTENT_SUBJ1, CONSISTENT_SUBJ2, LESSEPOCHS_SUBJ3 (sorted order)
        # Expected map: {'SUBJ1':0, 'SUBJ2':1, 'SUBJ3':2}
        self.assertDictEqual(loader.subject_str_to_int_id, {'SUBJ1':0, 'SUBJ2':1, 'SUBJ3':2})
        
        expected_ids = []
        expected_ids.extend([0] * self.std_n_epochs) # SUBJ1
        expected_ids.extend([1] * self.std_n_epochs) # SUBJ2
        expected_ids.extend([2] * 10)               # SUBJ3 (10 epochs)
        self.assertTrue(torch.equal(loader.subject_ids, torch.tensor(expected_ids, dtype=torch.long)))
        item = loader[0]
        self.assertIn("subject_id", item)

    def test_load_conditional_class_label(self):
        # Uses general_data_dir. n_epochs=None. Files: CONSISTENT_SUBJ1, CONSISTENT_SUBJ2, LESSEPOCHS_SUBJ3
        # Filename prefixes are CONSISTENT, CONSISTENT, LESSEPOCHS
        # Expected total epochs = 20+20+10 = 50
        loader = FIFDataLoader(file_path=self.general_data_dir, n_epochs=None, condition_on_class_label=True)
        self.assertEqual(len(loader), self.std_n_epochs * 2 + 10)
        self.assertIsNotNone(loader.class_labels)
        self.assertIsInstance(loader.class_labels, torch.LongTensor)
        
        # Expected map: {'CONSISTENT':0, 'LESSEPOCHS':1}
        self.assertDictEqual(loader.label_to_int_id, {'CONSISTENT':0, 'LESSEPOCHS':1})
        expected_ids = []
        expected_ids.extend([0] * self.std_n_epochs) # CONSISTENT_SUBJ1
        expected_ids.extend([0] * self.std_n_epochs) # CONSISTENT_SUBJ2
        expected_ids.extend([1] * 10)               # LESSEPOCHS_SUBJ3
        self.assertTrue(torch.equal(loader.class_labels, torch.tensor(expected_ids, dtype=torch.long)))
        item = loader[0]
        self.assertIn("class_label", item)

    def test_invalid_file_path(self):
        with self.assertRaisesRegex(ValueError, "No .fif files found in /non/existent/path."):
            FIFDataLoader(file_path="/non/existent/path", n_epochs=self.std_n_epochs)

    def test_invalid_fif_file_alone(self):
        # Only the invalid text file is provided
        with self.assertRaisesRegex(ValueError, "No valid EEG data loaded."):
             FIFDataLoader(file_path=self.fp_invalid_fif, n_epochs=self.std_n_epochs)
        
    def test_invalid_fif_file_with_valid(self):
        # Mix of valid and the invalid text file
        mixed_dir = os.path.join(self.temp_dir_base, "mixed_invalid")
        os.makedirs(mixed_dir, exist_ok=True)
        shutil.copy(self.fp1, mixed_dir) # Valid file
        shutil.copy(self.fp_invalid_fif, mixed_dir) # Invalid file

        loader = FIFDataLoader(file_path=mixed_dir, n_epochs=None)
        self.assertEqual(len(loader), self.std_n_epochs) # Should load only the valid file
        self.assertEqual(loader.sfreq, self.std_sfreq) # Params from valid file

    def test_no_fif_files_in_directory(self):
        empty_dir = os.path.join(self.temp_dir_base, "empty_dir")
        os.makedirs(empty_dir, exist_ok=True)
        with self.assertRaisesRegex(ValueError, f"No .fif files found in {empty_dir}."): # Regex uses f-string
            FIFDataLoader(file_path=empty_dir)

    def test_adopted_parameters_all_consistent_alternative(self):
        # All files in adopted_params_dir have alternative consistent parameters
        loader = FIFDataLoader(file_path=self.adopted_params_dir, n_epochs=None)
        # Two files, each with self.alt_n_epochs (15)
        self.assertEqual(len(loader), self.alt_n_epochs * 2)
        self.assertEqual(loader.sfreq, self.alt_sfreq)
        self.assertEqual(loader.n_channels, self.alt_n_channels)
        self.assertEqual(loader.n_times, self.alt_n_times)
        self.assertEqual(loader.data.shape, (self.alt_n_epochs * 2, self.alt_n_channels, self.alt_n_times))

    def test_inconsistent_sfreq(self):
        # Files: S1_SFREQ200 (5 epochs), S2_SFREQ250 (5 epochs, skipped), S3_SFREQ200 (5 epochs)
        # First file S1_SFREQ200 sets sfreq=200.0
        loader = FIFDataLoader(file_path=self.inconsistent_sfreq_dir, n_epochs=None)
        self.assertEqual(loader.sfreq, 200.0)
        self.assertEqual(len(loader), 5 + 5) # S1 and S3 loaded
        self.assertEqual(loader.data.shape, (10, self.std_n_channels, self.std_n_times))

    def test_inconsistent_channels(self):
        # Files: S1_CHAN19 (5 epochs), S2_CHAN10 (5 epochs, skipped), S3_CHAN19 (5 epochs)
        # First file S1_CHAN19 sets n_channels=19
        loader = FIFDataLoader(file_path=self.inconsistent_channels_dir, n_epochs=None)
        self.assertEqual(loader.n_channels, 19)
        self.assertEqual(len(loader), 5 + 5) # S1 and S3 loaded
        self.assertEqual(loader.data.shape, (10, 19, self.std_n_times))

    def test_inconsistent_times(self):
        # Files: S1_TIMES1000 (5 epochs), S2_TIMES500 (5 epochs, skipped), S3_TIMES1000 (5 epochs)
        # First file S1_TIMES1000 sets n_times=1000
        loader = FIFDataLoader(file_path=self.inconsistent_times_dir, n_epochs=None)
        self.assertEqual(loader.n_times, 1000)
        self.assertEqual(len(loader), 5 + 5) # S1 and S3 loaded
        self.assertEqual(loader.data.shape, (10, self.std_n_channels, 1000))
        
    def test_inconsistent_first_file_skipped_then_consistent_adopted(self):
        # Create a dir with: 1. invalid_fif, 2. adopted_param_S1, 3. adopted_param_S2
        test_dir = os.path.join(self.temp_dir_base, "invalid_first_then_consistent")
        os.makedirs(test_dir, exist_ok=True)
        
        # Create an invalid file that will be sorted first
        fp_invalid_first = os.path.join(test_dir, "AAAA_INVALID_eeg.fif")
        with open(fp_invalid_first, 'w') as f:
            f.write("This is not a fif file.")

        self._create_fif_file(test_dir, "BBBB_ALT_PARAM_S1", self.alt_sfreq, self.alt_n_channels, self.alt_n_times, self.alt_n_epochs)
        self._create_fif_file(test_dir, "CCCC_ALT_PARAM_S2", self.alt_sfreq, self.alt_n_channels, self.alt_n_times, self.alt_n_epochs)

        loader = FIFDataLoader(file_path=test_dir, n_epochs=None)
        self.assertEqual(len(loader), self.alt_n_epochs * 2)
        self.assertEqual(loader.sfreq, self.alt_sfreq)
        self.assertEqual(loader.n_channels, self.alt_n_channels)
        self.assertEqual(loader.n_times, self.alt_n_times)

    def test_all_files_inconsistent_loads_first_only(self):
        # File A: sfreq=200. File B: sfreq=250. File C: sfreq=300.
        # Only File A should be loaded.
        test_dir = os.path.join(self.temp_dir_base, "all_inconsistent")
        os.makedirs(test_dir, exist_ok=True)
        self._create_fif_file(test_dir, "FILE_A_SF200", 200.0, self.std_n_channels, self.std_n_times, 5)
        self._create_fif_file(test_dir, "FILE_B_SF250", 250.0, self.std_n_channels, self.std_n_times, 5)
        self._create_fif_file(test_dir, "FILE_C_SF300", 300.0, self.std_n_channels, self.std_n_times, 5)

        loader = FIFDataLoader(file_path=test_dir, n_epochs=None)
        self.assertEqual(loader.sfreq, 200.0)
        self.assertEqual(len(loader), 5) # Only the first file's epochs
        self.assertEqual(loader.data.shape, (5, self.std_n_channels, self.std_n_times))

    def test_no_valid_data_loaded_error_all_skipped(self):
        # All files are readable by MNE but inconsistent with the first one.
        # First file (sfreq=200) is processed. Second file (sfreq=250) is skipped.
        # This test is slightly different: create only two files, first valid, second inconsistent.
        # The previous `test_all_files_inconsistent_loads_first_only` covers the "loads first only"
        # This test ensures that if the *only other files* are inconsistent, it's not an error, but rather just loads what's consistent.
        # A "No valid EEG data loaded" would occur if the FIRST file is bad, and subsequent ones are also bad OR no other files.
        # The case of "all files are unreadable" is tested by test_invalid_fif_file_alone
        # The case of "no .fif files found" is tested by test_no_fif_files_in_directory
        
        # This scenario: first readable file sets params, all other readable files are inconsistent.
        # Should load only the first file, not raise "No valid data loaded".
        test_dir_one_valid_many_inconsistent = os.path.join(self.temp_dir_base, "one_valid_many_inconsistent")
        os.makedirs(test_dir_one_valid_many_inconsistent, exist_ok=True)
        
        # First file is valid and sets the parameters
        self._create_fif_file(test_dir_one_valid_many_inconsistent, "VALID_SF200", 200.0, self.std_n_channels, self.std_n_times, 7)
        # Subsequent files are inconsistent
        self._create_fif_file(test_dir_one_valid_many_inconsistent, "INCONSISTENT_SF250", 250.0, self.std_n_channels, self.std_n_times, 5)
        self._create_fif_file(test_dir_one_valid_many_inconsistent, "INCONSISTENT_CHAN10", 200.0, 10, self.std_n_times, 6)

        loader = FIFDataLoader(file_path=test_dir_one_valid_many_inconsistent, n_epochs=None)
        self.assertEqual(len(loader), 7) # Only data from the first file
        self.assertEqual(loader.sfreq, 200.0)
        self.assertEqual(loader.n_channels, self.std_n_channels)
        self.assertEqual(loader.n_times, self.std_n_times)


if __name__ == '__main__':
    unittest.main()
