#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016-2017.

# Author(s):

#
#   David Hoese <david.hoese@ssec.wisc.edu>
#

# This file is part of satpy.

# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.

"""Helpers for reading netcdf-based files.

"""
import warnings
import netCDF4
import logging
import xarray as xr

from satpy import CHUNK_SIZE
from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.utils import np2str

LOG = logging.getLogger(__name__)


class NetCDF4FileHandler(BaseFileHandler):

    """Small class for inspecting a NetCDF4 file and retrieving its metadata/header data.

    File information can be accessed using bracket notation. Variables are
    accessed by using:

        wrapper["var_name"]

    Or:

        wrapper["group/subgroup/var_name"]

    Attributes can be accessed by using a variables `.attrs` dictionary:

        wrapper["group/subgroup/var_name"].attrs["units"]

    Or if needed in a YAML configuration file as a string:

        wrapper["group/subgroup/var_name/attr/units"]

    And for global attributes in a YAML configuration file as a string:

        wrapper["/attr/platform_short_name"]

    """

    def __init__(self, filename, filename_info, filetype_info,
                 mask_and_scale=False, decode_cf=True, autoclose=False,
                 engine='netcdf4', chunks=None):
        super(NetCDF4FileHandler, self).__init__(
            filename, filename_info, filetype_info)

        try:
            file_handle = xr.open_dataset(self.filename,
                                          decode_cf=decode_cf,
                                          mask_and_scale=mask_and_scale,
                                          autoclose=autoclose,
                                          engine=engine,
                                          chunks=chunks or CHUNK_SIZE)
            self.file_handle = file_handle
        except IOError:
            LOG.exception(
                'Failed reading file %s. Possibly corrupted file', self.filename)
            raise

    def __getitem__(self, key):
        if '/attr/' in key:
            var_name, attr_name = key.split('/attr/')
            if not var_name:
                return self.file_handle.attrs[attr_name]
            else:
                return self.file_handle[var_name].attrs[attr_name]
        elif key.endswith('/shape'):
            warnings.warn("Deprecated use of 'var_name/shape' in file handler.")
            return self.file_handle[key[:-6]].shape
        elif key.endswith('/dtype'):
            warnings.warn("Deprecated use of 'var_name/dtype' in file handler.")
            return self.file_handle
        else:
            return self.file_handle[key]

    def __contains__(self, item):
        return item in self.file_handle

    def get(self, item, default=None):
        return self.file_handle.get(item, default)
