import re
import os
from glob import glob

import numpy as np
from scipy import optimize

from astropy import units as u
from astropy.table import QTable
from astropy.coordinates import (SkyCoord, ICRS, SphericalRepresentation,
                                 CartesianRepresentation,
                                 UnitSphericalRepresentation, Distance, Angle)

from astropy.coordinates.angles import rotation_matrix

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



galactic_center = SkyCoord(0*u.deg, 0*u.deg, frame='galactic')

def add_oriented_radecs(elvis_tab, hostidx=0, targetidx=1,
                        target_coord=SkyCoord(0*u.deg, 0*u.deg),
                        earth_distance=8.5*u.kpc, earth_vrot=220*u.km/u.s,
                        roll_angle=0*u.deg):
    """
    Computes a spherical coordinate system centered on the `hostidx` halo,
    re-oriented so that `targetidx` is at the `target_coord` coordinate
    location.

    Note that this adds columns 'host<n>_*' to
    `elvis_tab`, and will *overwrite* them if  they already exist.
    """
    if hasattr(target_coord, 'spherical'):
        target_lat = target_coord.spherical.lat
        target_lon = target_coord.spherical.lon
    else:
        target_lat = target_coord.lat
        target_lon = target_coord.lon

    def offset_repr(rep, vector, newrep=None):
        if newrep is None:
            newrep = rep.__class__
        newxyz = rep.to_cartesian().xyz + vector.reshape(3, 1)
        return CartesianRepresentation(newxyz).represent_as(newrep)

    def rotate_repr(rep, matrix, newrep=None):
        if newrep is None:
            newrep = rep.__class__
        newxyz = np.dot(matrix.view(np.ndarray), rep.to_cartesian().xyz)
        return CartesianRepresentation(newxyz).represent_as(newrep)

    rep = CartesianRepresentation(elvis_tab['X'], elvis_tab['Y'], elvis_tab['Z'])
    # first we offset the catalog to have its origin at host
    rep = offset_repr(rep, -rep.xyz[:, hostidx])

    # now rotate so that host1 is along the z-axis, and apply the arbitrary roll angle
    usph = rep.represent_as(UnitSphericalRepresentation)
    M1 = rotation_matrix(usph.lon[targetidx], 'z')
    M2 = rotation_matrix(90*u.deg-usph.lat[targetidx], 'y')
    M3 = rotation_matrix(roll_angle, 'z')
    Mfirst = M3*M2*M1
    rep = rotate_repr(rep, Mfirst)

    # now determine the location of the earth in this system
    # need diagram to explain this, but it uses SSA formula
    theta = target_coord.separation(galactic_center)  # target to GC angle
    D = rep.z[targetidx]  # distance to the target host
    R = earth_distance
    # srho = (R/D) * np.sin(theta)
    # sdelta_p = (srho * np.cos(theta) + (1 - srho**2)**0.5)
    # sdelta_m = (srho * np.cos(theta) - (1 - srho**2)**0.5)
    d1, d2 = R * np.cos(theta), (D**2 - (R * np.sin(theta))**2)**0.5
    dp, dm = d1 + d2, d1 - d2
    sdelta = (dp/D) * np.sin(theta)

    x = R * sdelta
    z = R * (1-sdelta**2)**0.5
    earth_location = u.Quantity([x, 0*u.kpc, z])

    # now offset to put earth at the origin
    rep = offset_repr(rep, -earth_location)
    sph = rep.represent_as(SphericalRepresentation)

    # rotate to put the target at its correct spot
    # first sent the target host to 0,0
    M1 = rotation_matrix(sph[targetidx].lon, 'z')
    M2 = rotation_matrix(-sph[targetidx].lat, 'y')
    # now rotate from origin to target lat,lon
    M3 = rotation_matrix(target_lat, 'y')
    M4 = rotation_matrix(-target_lon, 'z')
    Mmiddle = M4*M3*M2*M1
    rep = rotate_repr(rep, Mmiddle)

    # now one more rotation about the target to stick the GC in the right place
    def tomin(ang, inrep=rep[hostidx], axis=rep[targetidx].xyz, target=galactic_center.icrs):
        newr = rotate_repr(inrep, rotation_matrix(ang[0]*u.deg, axis))
        return ICRS(newr).separation(target).radian
    rot_angle = optimize.minimize(tomin, np.array(0).ravel(), method='Nelder-Mead')['x'][0]
    Mlast = rotation_matrix(rot_angle*u.deg, rep[targetidx].xyz)
    rep = rotate_repr(rep, Mlast)

    sph = rep.represent_as(SphericalRepresentation)
    elvis_tab['host{}_lat'.format(hostidx)] = sph.lat.to(u.deg)
    elvis_tab['host{}_lon'.format(hostidx)] = sph.lon.to(u.deg)
    elvis_tab['host{}_dist'.format(hostidx)] = sph.distance

    # now compute  velocities
    # host galactocentric
    dvxg = u.Quantity((elvis_tab['Vx'])-elvis_tab['Vx'][hostidx])
    dvyg = u.Quantity((elvis_tab['Vy'])-elvis_tab['Vy'][hostidx])
    dvzg = u.Quantity((elvis_tab['Vz'])-elvis_tab['Vz'][hostidx])

    earth_location_in_xyz = np.dot(Mfirst.T, earth_location)
    dxg = elvis_tab['X'] - elvis_tab['X'][0] - earth_location_in_xyz[0]
    dyg = elvis_tab['Y'] - elvis_tab['Y'][0] - earth_location_in_xyz[0]
    dzg = elvis_tab['Z'] - elvis_tab['Z'][0] - earth_location_in_xyz[0]
    vrg = (dvxg*dxg + dvyg*dyg + dvzg*dzg) * (dxg**2+dyg**2+dzg**2)**-0.5
    elvis_tab['host{}_galvr'.format(hostidx)] = vrg.to(u.km/u.s)

    # "vLSR-like"
    # first figure out the rotation direction
    earth_rotdir = SkyCoord(90*u.deg, 0*u.deg, frame='galactic').icrs

    #now apply the component from that everywhere
    offset_angle = earth_rotdir.separation(ICRS(sph))
    vrlsr = vrg - earth_vrot*np.cos(offset_angle)

    elvis_tab['host{}_vrlsr'.format(hostidx)] = vrlsr.to(u.km/u.s)
