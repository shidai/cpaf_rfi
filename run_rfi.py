import math
import numpy as np
import numpy.ma as ma
from numpy import linalg as la

import matplotlib.pyplot as plt
import argparse
#import glob

#import pyfits
from astropy.io import fits
#import psrfits_srch as srch
from psrfits_archive import fits_srch
from rfi_covariance import acm

###############
parser = argparse.ArgumentParser(description='Read PSRFITS format search mode data')
parser.add_argument('-f',  '--input_file',  metavar='Input file name',  nargs='+', required=True, help='Input file name')
#parser.add_argument('-o',  '--output_file', metavar='Output file name', nargs='+', required=True, help='Output file name')
#parser.add_argument('-sub',  '--subband_range', metavar='Subint ragne', nargs='+', default = [0, 0], type = int, help='Subint range')
#parser.add_argument('-npart',  '--nparts', metavar='Number of chunks in time', default = 50, type = int, help='Npart')
parser.add_argument('-step',  '--subint_step', metavar='Step of subint', default = 20, type = int, help='Subint step')
parser.add_argument('-freq',  '--freq_range', metavar='Freq range (MHz)', nargs='+', default = [0, 0], type = int, help='Frequency range (MHz)')
parser.add_argument('-sig',  '--sigma', metavar='Threshold', default = 3, type = float, help='Masking threshold')
parser.add_argument('-nsub',  '--num_subint', metavar='Total number of subint', default = 4250, type = int, help='Total number of subint in the search mode file')

args = parser.parse_args()
#sub_start = int(args.subband_range[0])
#sub_end = int(args.subband_range[1])
#npart = int(args.nparts)
step = int(args.subint_step)
freq_start = int(args.freq_range[0])
freq_end = int(args.freq_range[1])
sigma = float(args.sigma)
#dm = float(args.chn_dm[0])
infile = args.input_file
nsub = int(args.num_subint)             # this is the total number of subintegration in the search mode file. This should be the same for all beams

#####################
npart = int(np.ceil(nsub/step))

for i in range(npart):
	sub_start = int(i*step)
	sub_end = int((i+1)*step)
	if sub_end > nsub:
		sub_end = nsub
	print (sub_start, sub_end)

	mb_acm = acm (infile, sub_start, sub_end, freq_start, freq_end, sigma)
	mb_acm.read_data()
	mb_acm.normalise(base=[660,700])
	#mb_acm.plot_data(xr=[600,1000], xtick=50)
	
	mb_acm.cal_acm()
	mb_acm.cal_eigen()
