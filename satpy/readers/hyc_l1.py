#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2022-2023 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.

"""Reader for the PRISMA L1 HDF5 data."""

import logging
import os
import shutil
import tempfile
import zipfile
from datetime import datetime
from glob import glob

import xarray as xr
from pyresample.geometry import SwathDefinition

from satpy.readers.hdf5_utils import HDF5FileHandler

logger = logging.getLogger(__name__)


class HYCL1FileHandler(HDF5FileHandler):
    """File handler for PRISMA HYC L1 HDF5 files."""

    def __init__(self, filename, filename_info, filetype_info):
        """Prepare the class for dataset reading."""
        self.input_filename = filename
        if os.fspath(filename).endswith('zip'):
            # unzip L1B file
            self.filename = self._unzip_file(filename)
        else:
            self.filename = filename

        super().__init__(self.filename, filename_info, filetype_info)

        # load bands info for dims
        self._load_bands()
        self._meta = None
        self.area = None

    def _unzip_file(self, filename):
        """Unzip L1 file."""
        self.root_dir = tempfile.mkdtemp()

        with zipfile.ZipFile(filename, "r") as zf:
            logger.debug(f'Unzip {filename} to {self.root_dir}')
            zf.extractall(self.root_dir)

        return glob(os.path.join(self.root_dir, '*.he5'))[0]

    def __del__(self):
        """Delete the object."""
        if self.root_dir:
            shutil.rmtree(self.root_dir)

    @property
    def start_time(self):
        """Time for first observation."""
        return datetime.strptime(self['/attr/Product_StartTime'], "%Y-%m-%dT%H:%M:%S.%f")

    @property
    def end_time(self):
        """Time for final observation."""
        return datetime.strptime(self['/attr/Product_StopTime'], "%Y-%m-%dT%H:%M:%S.%f")

    @property
    def sensor(self):
        """Get sensor name."""
        # hard code here
        return 'hyc'

    @property
    def platform_name(self):
        """Get platform name."""
        return 'PRISMA'

    def _get_metadata(self, data):
        """Derive metadata."""
        if self._meta is None:
            self._meta = {'sza': self['/attr/Sun_zenith_angle'],
                          'saa': self['/attr/Sun_azimuth_angle'],
                          'filename': self.input_filename,
                          'sensor': self.sensor,
                          'platform_name': self.platform_name,
                          }

        return self._meta

    def _load_bands(self):
        # read wavelength which is the dim for other variables
        self.bands_vnir = xr.DataArray(self['/attr/List_Cw_Vnir'], dims='bands_vnir').rename('bands_vnir')
        self.bands_swir = xr.DataArray(self['/attr/List_Cw_Swir'], dims='bands_swir').rename('bands_swir')
        self.bands_vnir.attrs['units'] = 'nm'
        self.bands_swir.attrs['units'] = 'nm'

        # read fwhm which is the anciilary variable of bands
        self.fwhm_vnir = xr.DataArray(self['/attr/List_Fwhm_Vnir'], dims='bands_vnir').rename('fwhm_vnir')
        self.fwhm_swir = xr.DataArray(self['/attr/List_Fwhm_Swir'], dims='bands_swir').rename('fwhm_swir')
        self.fwhm_vnir.attrs['units'] = 'nm'
        self.fwhm_swir.attrs['units'] = 'nm'
        self.fwhm_vnir.attrs['standard_name'] = 'full width at half maximum'
        self.fwhm_swir.attrs['standard_name'] = 'full width at half maximum'

        # assign fwhm as bands coords
        self.bands_vnir.coords['fwhm_vnir'] = self.fwhm_vnir
        self.bands_swir.coords['fwhm_swir'] = self.fwhm_swir

    def calibrate(self, data, ds_info):
        """Calibrate data."""
        # get calibration method and detector name
        calibration = ds_info['calibration']

        if calibration == 'counts':
            # original DN values
            calibrated_data = data
        elif calibration == 'radiance':
            if ds_info['name'] == 'vnir':
                calibrated_data = data / self['/attr/ScaleFactor_Vnir'] - self['/attr/Offset_Vnir']
            elif ds_info['name'] == 'swir':
                calibrated_data = data / self['/attr/ScaleFactor_Swir'] - self['/attr/Offset_Swir']
        # elif calibration == 'reflectance':
        else:
            raise ValueError("Unknown calibration %s for dataset %s" % (calibration, self.name))

        # add units
        calibrated_data.attrs['units'] = ds_info['units']

        return calibrated_data

    def get_dataset(self, dataset_id, ds_info):
        """Load data variable and metadata."""
        file_key = ds_info.get('file_key', ds_info['name'])
        data = self[file_key]

        if data.ndim == 3:
            # nHypAcrossPixel, nBands, nHypAlongPixel
            band_dim = f"bands_{ds_info['name']}"
            data = data.rename({data.dims[0]: 'x', data.dims[1]: band_dim, data.dims[2]: 'y'})
            data = data.transpose(band_dim, 'y', 'x')
        elif data.ndim == 2:
            # nHypAcrossPixel, nHypAlongPixel
            data = data.rename({data.dims[0]: 'x', data.dims[1]: 'y'})
            data = data.transpose('y', 'x')

        if dataset_id['name'] in ['vnir', 'swir']:
            data = self.calibrate(data, ds_info)

        # add area
        self.get_area()
        data.attrs['area'] = self.area

        # add metadata and bands coords
        data.attrs.update(self._get_metadata(data))
        band_dims = [s for s in data.dims if 'bands' in s]
        if len(band_dims) > 0:
            data.coords[band_dims[0]] = getattr(self, band_dims[0])

        return data

    def get_area(self):
        """Get the VNIR lonlats as the area.

        Coregistered cubes correspond to the Radimetric Cubes
            where also spatial coregistration of SWIR and PAN with respect to VNIR channel is performed.
        """
        if self.area is None:
            lons = self['HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Longitude_VNIR']
            lats = self['HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Latitude_VNIR']

            # set to correct order
            lons = lons.rename({lons.dims[0]: 'x', lons.dims[1]: 'y'})
            lons = lons.transpose('y', 'x')
            lats = lats.rename({lats.dims[0]: 'x', lats.dims[1]: 'y'})
            lats = lats.transpose('y', 'x')

            self.area = SwathDefinition(lons, lats)
            self.area.name = '_'.join([self.sensor, str(self.start_time)])
