import numpy as np
from astropy.convolution import convolve
try:
    from astropy.version import version_info as astropy_version_info
except:
    from astropy import version
    astropy_version_info = (version.major, version.minor, version.bugfix)

THREADED_VERSION = (3,1)

max_exponents = {1: 15, 2: 7, 3: 5}
max_exponent = max(max_exponents.values())
lengths = []
for n in range(max_exponent):
    lengths.append(2**n)
n_dims = (1,2,3)
nan_interpolate = (True, False)

class ConvolveBenchmarks:

    params = [n_dims, lengths, nan_interpolate]

    def setup(self, n_dim, length, nan_interpolate):

        if length > max_exponents[n_dim]:
            # Tell ASV to skip this test instance
            raise NotImplementedError

        self.array = np.random.random([length]*n_dim)
        self.kernel = np.random.random([length-1]*n_dim)
        self.params = [self.array, self.kernel]

        self.kargs = {'nan_treatment':'interpolate'}
        if astropy_version_info >= THREADED_VERSION:
            self.kargs.add('n_threads', 1)

        if nan_interpolate:
            # Make a pixel a NaN to force NaN interpolation
            zeroth_pix = tuple([0]*n_dim)
            self.array[zeroth_pix] = np.nan

    def time_convolve(self):
        convolve(*self.params, **self.kargs)

    def time_convolve_boundary_none(self):
        convolve(*self.params, boundary=None, **self.kargs)

    def time_convolve_boundary_fill(self):
        convolve(*self.params, boundary='fill', **self.kargs)

    def time_convolve_boundary_extend(self):
        convolve(*self.params, boundary='extend', **self.kargs)

    def time_convolve_boundary_wrap(self):
        convolve(*self.params, boundary='wrap', **self.kargs)

class ConvolveThreadedBenchmarks(ConvolveBenchmarks):

    def setup(self):
        super().setup()
        if astropy_version_info >= THREADED_VERSION:
            self.args.add('n_threads', 0)
        else:
            # Tell ASV to skip this test instance
            raise NotImplementedError
