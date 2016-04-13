import re
import os
from glob import glob

import numpy as np

from astropy import units as u
from astropy.table import QTable

### ELVIS simulation loaders


def read_elvis_z0(fn):
    tab = QTable.read(fn, format='ascii.commented_header', data_start=0, header_start=1)

    col_name_re = re.compile(r'(.*?)(\(.*\))?$')
    for col in tab.columns.values():
        match = col_name_re.match(col.name)
        if match:
            nm, unit = match.groups()
            if nm != col.name:
                col.name = nm
            if unit is not None:
                col.unit = u.Unit(unit[1:-1])  # 1:-1 to get rid of the parenthesis

    return tab


def load_elvii_z0(data_dir=os.path.abspath('elvis_data/PairedCatalogs/'), isolated=False, inclhires=False):
    tables = {}

    fntoload = [fn for fn in glob(os.path.join(data_dir, '*.txt')) if 'README' not in fn]
    for fn in fntoload:
        simname = os.path.split(fn)[-1][:-4]
        if simname.startswith('i'):
            if not isolated:
                continue
        else:
            if isolated:
                continue
        if not inclhires and 'HiRes' in fn:
            continue
        print('Loading', fn)
        tables[simname] = read_elvis_z0(fn)

        annotate_table_z0(tables[simname])
    return tables

def annotate_table_z0(tab):
    for i in (0, 1):
        idi = tab['ID'][i]
        tab['sat_of_{}'.format(i)] = tab['UpID'] == idi

        dx = tab['X'] - tab['X'][i]
        dy = tab['Y'] - tab['Y'][i]
        dz = tab['Z'] - tab['Z'][i]
        tab['host{}_dist'.format(i)] = (dx**2 + dy**2 + dz**2)**0.5

    tab['sat_of_either'] = tab['sat_of_0']|tab['sat_of_1']
