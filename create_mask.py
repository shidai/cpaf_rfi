import math
import numpy as np
import numpy.ma as ma
from numpy import linalg as la

import matplotlib.pyplot as plt
#from matplotlib import rc
#from matplotlib.patches import Circle

#import subprocess
import argparse
#import glob

#import pyfits
from astropy.io import fits
#import psrfits_srch as srch
from psrfits_archive import fits_srch

##########################


class eig ():
	def __init__ (self, nsub, sub_step, nchan, nbeam, sigma=3.0):
		self.sigma = sigma
		self.nsub = nsub
		self.nchan = nchan
		self.nbeam = nbeam
		self.sub_step = sub_step
		self.npart = int(np.ceil(self.nsub/self.sub_step))

		self.eigval_array = ma.masked_array(np.empty((self.npart, self.nchan, self.nbeam)), mask=np.zeros((self.npart, self.nchan, self.nbeam)))
		self.chan_array = ma.masked_array(np.empty((self.npart, self.nchan, self.nbeam)), mask=np.zeros((self.npart, self.nchan, self.nbeam)))
		#self.eigvec_array = ma.masked_array(np.empty((self.npart, self.nchan, self.nbeam, self.nbeam)), mask=np.zeros((self.npart, self.nchan, self.nbeam, self.nbeam)))
		self.eigvec_array = np.empty((self.npart, self.nchan, self.nbeam, self.nbeam))

		for i in range(self.nchan):
			self.chan_array[:,i,:] = i

		for i in range(self.npart):
			sub_start = int(i*self.sub_step)
			f_eigval = 'eigval_sub{0}.npy'.format(sub_start)
			f_eigvec = 'eigvec_sub{0}.npy'.format(sub_start)

			self.eigval_array[i,:,:] = np.load(f_eigval)
			self.eigvec_array[i,:,:,:] = np.load(f_eigvec)

	def plot_eigval (self, beam=0):
		plt.clf()
		ax = plt.subplot(111)
		ax.imshow(self.eigval_array[:,:,beam], origin='lower', aspect='auto')
		plt.savefig('masking_plot_eigval.png')

	#def generate_filter (self, eig_idx, xlim, xtick):
	def generate_filter (self, eig_idx):
		#print (np.ma.count_masked(self.eigval_array))
		self.chan_array.mask = self.chan_array < 200 # hard coded at the moment for MB dspsr data
		self.eigval_array.mask = self.chan_array.mask
		#print (np.ma.count_masked(self.eigval_array))

		###################

		plt.clf()
		ax = plt.subplot(111)
		ax.imshow(self.eigval_array[:,:,eig_idx], origin='lower', aspect='auto')
		plt.savefig('masking_plot_eigval_masked.png')

		#print (self.eigval.shape, ma.mean(self.eigval), ma.std(self.eigval))
		#for i in range(self.nbeam):
		#	print (self.eigval[:,i].shape, ma.mean(self.eigval[:,i]), ma.std(self.eigval[:,i]))

		# treating each eig val independently; even after normalisation, there is a gradient in eig val
		for idx in range(eig_idx):
			nchan_zap0 = np.ma.count_masked(self.eigval_array)
			#nchan_zap0 = len(self.dat_freq[self.eigval.mask[:,idx]])
			#nchan_zap0 = ma.count_masked(self.eigval) 
			#print (nchan_zap0)

			ave_eigval = ma.mean(self.eigval_array[:,:,idx])
			std_eigval = ma.std(self.eigval_array[:,:,idx])
			#print (ave_eigval, std_eigval, ma.mean(self.eigval[:,1]), ma.std(self.eigval[:,1]))
			#print (default_mask.shape, self.eigval[:,0].shape)
		
			self.eigval_array[:,:,idx] = ma.masked_greater(self.eigval_array[:,:,idx], ave_eigval+self.sigma*std_eigval)
			self.eigval_array[:,:,idx] = ma.masked_less(self.eigval_array[:,:,idx], ave_eigval-self.sigma*std_eigval)
			#self.eigval.mask = ma.fabs(self.eigval - ave_eigval) > self.sigma*std_eigval
			#print (self.dat_freq[self.eigval.mask[:,0]], np.arange(self.nchan)[self.eigval.mask[:,0]])
			#print (len(self.dat_freq[self.eigval.mask[:,eig_idx]]), ma.std(self.eigval[:,eig_idx]))

			nchan_zap = np.ma.count_masked(self.eigval_array)
			#nchan_zap = len(self.dat_freq[self.eigval.mask[:,idx]])
			#print (nchan_zap0, nchan_zap)

			count = 0
			while nchan_zap != nchan_zap0:
			#while count < 5:
				#print (count)
				nchan_zap0 = nchan_zap
				ave_eigval = ma.mean(self.eigval_array[:,:,idx])
				std_eigval = ma.std(self.eigval_array[:,:,idx])
				#print (count, ave_eigval, std_eigval)
				self.eigval_array[:,:,idx] = ma.masked_greater(self.eigval_array[:,:,idx], ave_eigval+self.sigma*std_eigval)
				self.eigval_array[:,:,idx] = ma.masked_less(self.eigval_array[:,:,idx], ave_eigval-self.sigma*std_eigval)
				#self.eigval.mask = ma.fabs(self.eigval - ave_eigval) > self.sigma*std_eigval
				nchan_zap = np.ma.count_masked(self.eigval_array)
				#nchan_zap = len(self.dat_freq[self.eigval.mask[:,idx]])
				#print (len(self.dat_freq[self.eigval.mask[:,eig_idx]]), ma.std(self.eigval[:,eig_idx]))
				#print (count, ave_eigval, std_eigval, nchan_zap)
				#print (np.arange(self.nchan)[self.eigval.mask[:,idx]])
				print(nchan_zap)
				count += 1

	def generate_mask (self, base, eig_idx, eigvec_sig):
		# generate mask for each beam
		# self.eigval_array shape (self.npart, self.nchan, self.nbeam)
		# self.eigvec_array shape (self.npart, self.nchan, self.nbeam, self.nbeam)

		#self.rfi_mask = np.empty((self.nchan, self.nbeam))
		self.rfi_mask = np.ones((self.npart, self.nchan, self.nbeam))
		eig_mask = self.eigval_array.mask[:,:,eig_idx]

		init_mask = np.moveaxis(np.array(self.nbeam*[eig_mask]), 0, -1)  # make an array with shape (self.npart, self.nchan, self.nbeam)
		#ave_eigvec = np.mean(self.eigvec_array[:, :, :, eig_idx])
		#std_eigvec = np.std(self.eigvec_array[:, :, :, eig_idx])
		#print (ave_eigvec, std_eigvec)
		#ave_eigvec = np.mean(self.eigvec_array[:, :, :, eig_idx][init_mask])
		#std_eigvec = np.std(self.eigvec_array[:, :, :, eig_idx][init_mask])
		#print (ave_eigvec, std_eigvec)
		ave_eigvec = np.mean(self.eigvec_array[:, base[0]:base[1], :, eig_idx])
		std_eigvec = np.std(self.eigvec_array[:, base[0]:base[1], :, eig_idx])
		print (ave_eigvec, std_eigvec)
		for j in range(self.npart):
			for i in range(self.nchan):
				if i <=200: 
					self.rfi_mask[j,i,:] = 0
				elif self.eigval_array.mask[j,i,eig_idx] == 1:
					#self.rfi_mask[i, :] = 0
					#beam_mask = np.fabs(self.eigvec[i, :, 0] - ave_eigvec) > 0.01*std_eigvec
					beam_mask = np.fabs(self.eigvec_array[j, i, :, eig_idx]) > eigvec_sig*std_eigvec
					self.rfi_mask[j, i, beam_mask] = 0
				#else:
				#	self.rfi_mask[i,:] = 1

		self.rfi_mask = np.array(self.rfi_mask, dtype=bool)
		print (np.array_equal(self.rfi_mask[:,:,0], self.rfi_mask[:,:,-1]))

	def plot_result (self, beam, xr, xtick):
		#mb = fits_srch (filenames)
		#mb.read_data()
		#for i in range(self.nbeam):
		#	print ('{0} nsub:{1} nsblk:{2} nbits:{3} nchan:{4}'.format(mb.filenames[i], mb.nsub, mb.nsblk, mb.nbits, mb.nchan))
		print (self.eigval_array[:,:,beam].shape)

		plt.clf()
		'''
		plt.figure(figsize=(12,12))
		ax = plt.subplot(111)
		ax.set_title('Masked eig value', fontsize=16)
		ax.set_xlim(200,self.nchan)
		ax.set_xlabel('Channel number', fontsize=16)
		ax.set_ylabel('Time', fontsize=16)
		ax.imshow(self.eigval_array[:,:,beam[0]], origin='lower', aspect='auto')
		#ax.imshow(self.eigval_array.mask[:,:,beam[0]], origin='lower', aspect='auto')
		'''
		
		plt.figure(figsize=(12,12))
		plt.subplots_adjust(hspace=0.05, wspace=0.05)
		i = 0
		for beam_id in beam:
			plot_idx = i+1
			ax = plt.subplot(2, 2, plot_idx)
			ax.set_title('beam{0}'.format(beam_id+1))
			ax.set_ylabel('Time', fontsize=16)
			
			ax.set_xticks([])
			ax.set_yticks([])
			if plot_idx == 3 or plot_idx == 4:
				ax.set_xticks(np.arange(0, self.nchan, xtick))
				ax.set_xlabel('Channel number', fontsize=16)

			ax.set_xlim(xr[0], xr[1])
			ax.imshow(self.rfi_mask[:,:,beam_id], origin='lower', aspect='auto')
			i += 1

		plt.savefig('masking_plot_eigval_masked.png')

		#################

##########################

if __name__ == "__main__":
	######################
	#rc('text', usetex=True)
	# read in parameters
	
	parser = argparse.ArgumentParser(description='Read PSRFITS format search mode data')
	#parser.add_argument('-o',  '--output_file', metavar='Output file name', nargs='+', required=True, help='Output file name')
	parser.add_argument('-nsub',  '--nsubint', metavar='Total num of subint', default = 100, type = int, help='Total number of subint')
	parser.add_argument('-nchn',  '--nchan', metavar='Total num of channel', default = 1024, type = int, help='Total number of channel')
	parser.add_argument('-nbeam',  '--nbeam', metavar='Total num of beam', default = 13, type = int, help='Total number of beam')
	parser.add_argument('-step',  '--sub_step', metavar='Step in subintegration', default = 20, type = int, help='Step in subintegration')
	parser.add_argument('-sig',  '--sigma', metavar='Threshold', default = 3, type = float, help='Masking threshold')
	#parser.add_argument('-f',  '--input_file',  metavar='Input file name',  nargs='+', default = [], help='Input file name')
	
	args = parser.parse_args()
	sigma = int(args.sigma)
	nbeam = int(args.nbeam)
	nchan = int(args.nchan)
	nsub =  int(args.nsubint)
	sub_step = int(args.sub_step)
	#filenames = args.input_file
	
	eig = eig (sub_step=sub_step, nsub=nsub, nchan=nchan, nbeam=nbeam, sigma=sigma)
	#eig.plot_eigval(beam=0)
	eig.generate_filter(eig_idx=1)
	eig.generate_mask(base=[600,650], eig_idx=0, eigvec_sig=0.5)
	eig.plot_result(beam=[0,1,2,12], xr=[200, 1024], xtick=200)
	
