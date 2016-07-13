
import numpy as np
import scipy.stats as stats
import sklearn.preprocessing as preprocessing
import scipy.signal as signal
import pywt

from mfcc_feature.mfcc import mfcc

_floatX = np.float32
_intX = np.int32


class EegSignalFeature(object):

    def __init__(self):
        pass

    def fft(self, data):
        """
        Apply Fast Fourier Transform to the last axis
        np.fft.rfft: Compute the one-dimensional discrete Fourier Transform for real input.
        This function computes the one-dimensional n-point discrete Fourier Transform (DFT) of a real-valued array by
        means of an efficient algorithm called the Fast Fourier Transform (FFT).
        """
        return np.fft.rfft(data, axis=data.ndim - 1)

    def slice(self, data, start, end=None):
        """
        Take a slice of data on the last axis
        """
        s = [slice(None), ] * data.ndim
        s[-1] = slice(start, end)
        return data[s]

    def magnitude(self, data):
        """
        Take magnitude of complex data
        """
        return np.absolute(data)

    def magnitude_and_phase(self, data):
        """
        Take the magnitudes and phases of complex data and append them together.
        np.angle: Return the angle of the complex argument
        input data is preprocessed by fft and have shape [m, n], so return shape will be [m, 2n]
        """
        magnitudes = np.absolute(data)
        phases = np.angle(data)
        return np.concatenate((magnitudes, phases), axis=1)

    def lpf(self, data, frequency):
        """
        Low-pass filter using FIR window
        """
        # nyq - the Nyquist rate
        nyq = frequency * 0.5
        cutoff = min(frequency, nyq - 1)
        h = signal.firwin(numtaps=101, cutoff=cutoff, nyq=nyq)

        for i in range(len(data)):
            data_point = data[i]
            for j in range(len(data_point)):
                data_point[j] = signal.lfilter(h, 1.0, data_point[j])

        return data

    def mfcc(self, data):
        """
        Mel-frequency cepstrum coefficients
        """
        all_ceps = list()
        for channel in data:
            ceps, mspec, spec = mfcc(channel)
            all_ceps.append(ceps.ravel())
        all_ceps = np.asarray(all_ceps, dtype=_floatX)
        return all_ceps

    def log(self, data):
        """
        Apply LogE
        """
        indices = np.where(data <= 0)
        data[indices] = np.max(data)
        data[indices] = (np.min(data) * 0.1)
        return np.log(data)

    def log2(self, data):
        """
        Apply Log2
        """
        indices = np.where(data <= 0)
        data[indices] = np.max(data)
        data[indices] = (np.min(data) * 0.1)
        return np.log2(data)

    def log10(self, data):
        """
        Apply Log10
        """
        indices = np.where(data <= 0)
        data[indices] = np.max(data)
        data[indices] = (np.min(data) * 0.1)
        return np.log10(data)

    def stats(self, data):
        """
        Subtract the mean, then take (min, max, standard_deviation) for each channel.
        """

        shape = data.shape
        out = np.empty((shape[0], 3))
        channel_mean = np.mean(data, axis=1).reshape([shape[0], 1])
        data = data - np.tile(channel_mean, [1, shape[1]])
        out[:, 0] = np.std(data, axis=1)
        out[:, 1] = np.min(data, axis=1)
        out[:, 2] = np.max(data, axis=1)

        return out

    def moment_per_channel(self, data, n):
        """
        Calculate the Nth moment per channel.
        A moment is a specific quantitative measure of the shape of a set of points. It is often used to calculate
        coefficients of skewness and kurtosis due to its close relationship with them.
        """
        return stats.moment(data, moment=n, axis=data.ndim - 1)

    def unit_scale(self, data):
        """
        Scale across the last axis.
        with data being [ch, features], this is scaling each channel
        """
        return preprocessing.scale(data, axis=data.ndim - 1)

    def unit_scale_feature(self, data):
        """
        scale acorss the first (feature) axis.
        with data being [ch, features], this is scaling each feature
        """
        return preprocessing.scale(data, axis=0)

    def correlation_matrix(self, data):
        """
        Calculate correlation coefficients matrix across all EEG channels.
        np.corrcoef: Return Pearson product-moment correlation coefficients.
        """
        return np.corrcoef(data)

    def eigenvalues(self, data):
        """
        Take eigenvalues of a matrix, and sort them by magnitude in order to
        make them useful as features (as they have no inherent order).
        np.linalg.eig: Compute the eigenvalues and right eigenvectors of a square array
        """
        w, v = np.linalg.eig(data)
        w = np.absolute(w)
        w.sort()
        return w

    def upper_right_triangle(self, data):
        """
        Take the upper right triangle of a matrix
        """
        # make sure data is a square matrix
        assert data.ndim == 2 and data.shape[0] == data.shape[1]
        accum = []
        for i in range(data.shape[0]):
            for j in range(i + 1, data.shape[1]):
                accum.append(data[i, j])

        return np.asarray(accum)

    def resample(self, data, sampling_rate):
        """
        Resample time-series data.
        """
        if data.shape[-1] > sampling_rate:
            return signal.resample(data, sampling_rate, axis=data.ndim - 1)
        return data

    def resample_hanning(self, data, sampling_rate):
        """
        Resample time-series data using a Hanning window
        """
        out = signal.resample(data, sampling_rate, axis=data.ndim - 1, window=signal.hann(M=data.shape[data.ndim-1]))
        return out

    # never test or used
    def daub_wavelet_stats(self, data, n):
        """
       Daubechies wavelet coefficients. For each block of co-efficients take (mean, std, min, max)
       """
        shape = data.shape
        out = np.empty((shape[0], 4 * (n * 2 + 1)), dtype=_floatX)

        def set_stats(outi, x, offset):
            outi[offset * 4] = np.mean(x)
            outi[offset * 4 + 1] = np.std(x)
            outi[offset * 4 + 2] = np.min(x)
            outi[offset * 4 + 3] = np.max(x)

        for i in range(len(data)):
            outi = out[i]
            new_data = pywt.wavedec(data[i], 'db%d' % n, level=n * 2)
            for i, x in enumerate(new_data):
                set_stats(outi, x, i)

        return out

    def overlapping_fft_deltas(self, data, n_parts, parts_per_window, start, end):
        """
        Calculate overlapping FFT windows. The time window will be split up into num_parts,
        and parts_per_window determines how many parts form an FFT segment.

        e.g. num_parts=4 and parts_per_windows=2 indicates 3 segments
        parts = [0, 1, 2, 3]
        segment0 = parts[0:1]
        segment1 = parts[1:2]
        segment2 = parts[2:3]

        Then the features used are (segment2-segment1, segment1-segment0)
        this function is operating on data preprocessed by fft, which the last axis is the frequencies.

        NOTE: Experimental, not sure if this works properly.
        """

        axis = data.ndim - 1
        parts = np.split(data, n_parts, axis=axis)

        # if slice end is 208, we want 208hz
        partial_size = 1.0 / n_parts * parts_per_window
        # if slice end is 208, and partial_size is 0.5, then end should be 104
        partial_end = int(end * partial_size)

        partials = []
        for i in range(n_parts - parts_per_window + 1):
            combined_parts = parts[i:i + parts_per_window]
            if parts_per_window > 1:
                d = np.concatenate(combined_parts, axis=axis)
            else:
                d = combined_parts
            d = self.slice(d, start, partial_end)
            d = self.fft(d)
            d = self.magnitude(d)
            d = self.log10(d)
            partials.append(d)

        diffs = []
        for i in range(1, len(partials)):
            diffs.append(partials[i] - partials[i - 1])

        return np.concatenate(diffs, axis=axis)

    def fft_with_overlapping_fft_deltas(self, data, n_parts, parts_per_window, start, end):
        """
        As above but appends the whole FFT to the overlapping data.

        see overlapping_fft_deltas for more information.

        NOTE: Experimental, not sure if this works properly.
        """
        axis = data.ndim - 1
        full_fft = self.fft(data)
        full_fft = self.magnitude(full_fft)
        full_fft = self.log10(full_fft)

        parts = np.split(data, n_parts, axis=axis)

        # if slice end is 208, we want 208hz
        partial_size = 1.0 / n_parts * parts_per_window
        # if slice end is 208, and partial_size is 0.5, then end should be 104
        partial_end = int(end * partial_size)

        partials = []
        for i in range(n_parts - parts_per_window + 1):
            d = np.concatenate(parts[i:i + parts_per_window], axis=axis)
            d = self.slice(d, start, partial_end)
            d = self.fft(d)
            d = self.magnitude(d)
            d = self.log10(d)
            partials.append(d)

        out = [full_fft]
        for i in range(1, len(partials)):
            out.append(partials[i] - partials[i - 1])

        return np.concatenate(out, axis=axis)

    def frequency_correlation(self, data, start_hz, end_hz, option, resample_size=None, max_hz=None, with_fft=False, with_corr=True, with_eigen=True):
        """
        Correlation in the frequency domain. First take FFT with (start, end) slice options,
        then calculate correlation co-efficients on the FFT output, followed by calculating
        eigenvalues on the correlation co-efficients matrix.

        The output features are (fft, upper_right_diagonal(correlation_coefficients), eigenvalues)

        Features can be selected/omitted using the constructor arguments.
        """
        # resample data
        if max_hz is not None and data.shape[1] > max_hz:
            data = self.resample(data, max_hz)

        fft_data = self.fft(data)
        fft_data = self.slice(fft_data, start_hz, end_hz)
        fft_data = self.magnitude(fft_data)
        # resample after fft
        if resample_size is not None and fft_data.shape[1] > resample_size:
            fft_data = self.resample(fft_data, resample_size)

        fft_data = self.log10(fft_data)

        # resample after fft
        # if resample_size is not None and fft_data.shape[1] > resample_size:
        #     fft_data = self.resample(fft_data, resample_size)

        out = []
        if with_fft:
            out.append(fft_data.ravel())
        else:
            pass

        temp_data = None
        if with_corr or with_eigen:
            temp_data = np.copy(fft_data)
            if option == 'usf':
                temp_data = self.unit_scale_feature(temp_data)
            elif option == 'us':
                temp_data = self.unit_scale(temp_data)
            else:
                pass
            temp_data = self.correlation_matrix(temp_data)

        if with_corr:
            out.append(self.upper_right_triangle(temp_data))
        else:
            pass

        if with_eigen:
            out.append(self.eigenvalues(temp_data))
        else:
            pass

        return np.concatenate(out, axis=0)

    def time_correlation(self, data, option, max_hz=None, with_corr=True, with_eigen=True):
        """
        Correlation in the time domain. First downsample the data, then calculate correlation co-efficients
        followed by calculating eigenvalues on the correlation co-efficients matrix.

        The output features are (upper_right_diagonal(correlation_coefficients), eigenvalues)

        Features can be selected/omitted using the constructor arguments.
        """
        # so that correlation matrix calculation doesn't crash
        for ch in data:
            if np.alltrue(ch == 0.0):
                ch[-1] += 0.00001
        # resample data
        if max_hz is not None and data.shape[1] > max_hz:
            data = self.resample(data, max_hz)

        if option == 'usf':
            temp_data = self.unit_scale_feature(data)
        elif option == 'us':
            temp_data = self.unit_scale(data)
        else:
            temp_data = data

        temp_data = self.correlation_matrix(temp_data)

        out = []
        if with_corr:
            out.append(self.upper_right_triangle(temp_data))
        else:
            pass

        if with_eigen:
            out.append(self.eigenvalues(temp_data))
        else:
            pass

        for d in out:
            assert d.ndim == 1

        return np.concatenate(out, axis=0)

    def time_frequency_correlation(self, data, start_hz, end_hz, option, resample_size=None, max_hz=None):
        """
        Combines time and frequency correlation, taking both correlation coefficients and eigenvalues.
        """

        time_correlation_feature = self.time_correlation(data, option, max_hz)
        frequency_correlation_feature = self.frequency_correlation(data, start_hz, end_hz, option, resample_size, max_hz)
        assert time_correlation_feature.ndim == frequency_correlation_feature.ndim
        return np.concatenate((time_correlation_feature, frequency_correlation_feature), axis=time_correlation_feature.ndim - 1)

    def fft_with_time_frequency_correlation(self, data, start_hz, end_hz, option, resample_size, max_hz=None):
        """
        Combines FFT with time and frequency correlation, taking both correlation coefficients and eigenvalues.
        """
        time_correlation_feature = self.time_correlation(data, option, max_hz)
        frequency_correlation_feature = self.frequency_correlation(data, start_hz, end_hz, option, resample_size, max_hz, with_fft=True)
        assert time_correlation_feature.ndim == frequency_correlation_feature.ndim
        return np.concatenate((time_correlation_feature, frequency_correlation_feature), axis=time_correlation_feature.ndim-1)

