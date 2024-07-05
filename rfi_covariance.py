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


class acm (fits_srch):
	def __init__ (self, filenames, sub0=0, sub1=0, freq0=0, freq1=0, sigma=5.0):
		self.sub0 = sub0
		print ('Initialising a multi-beam object\n')
		super().__init__ (filenames, sub0, sub1, freq0, freq1)
		self.sigma = sigma

		#self.read_data()
		for i in range(self.nbeam):
			print ('{0} nsub:{1} nsblk:{2} nbits:{3} nchan:{4}'.format(self.filenames[i], self.nsub, self.nsblk, self.nbits, self.nchan))

	def normalise (self, base):
		'''
		std = np.std(self.nbarray, axis=1)
		mean = np.mean(self.nbarray, axis=1)
		print (np.amax(std), np.amin(std))
		print (np.amax(mean), np.amin(mean))
	
		for i in range(self.nbeam):
			for j in range(self.nchan):
				if std[i,j] != 0:
					#self.nbarray[i,:,j] = (self.nbarray[i,:,j] - mean[i,j])/std[i,j]
					self.nbarray[i,:,j] = self.nbarray[i,:,j]/std[i,j]
		'''
		for i in range(self.nbeam):
			std = np.std(self.nbarray[i,:,base[0]:base[1]])
			mean = np.mean(self.nbarray[i,:,base[0]:base[1]])
			self.nbarray[i,:,:] = (self.nbarray[i,:,:] - mean)/std

	def cal_acm (self):
		self.acm = np.empty((self.nchan, self.nbeam, self.nbeam))
		'''
		#print (self.nbarray.shape)
		for i in range(self.nchan):
			self.acm[i] = np.cov(self.nbarray[:,:,i])
		#print (self.acm.shape)
		'''
		for i in range(self.nchan):
			for j in range(self.nbeam):
				for k in range(self.nbeam):
					self.acm[i,j,k] = np.correlate(self.nbarray[j,:,i], self.nbarray[k,:,i])    # use correlation instead of covariance
	
	def sim_acm (self):
		print ("Simulating....\n")
		#self.acm = np.random.randn(self.nchan, self.nbeam, self.nbeam)
		x1, x2, x3 = self.nbarray.shape
		#self.nbarray = np.random.randn(x1, x2, x3)   # assuming the same noise level in each beam
		for i in range(self.nbeam):
			A = 3*np.random.randn(1)
			B = 0.1*np.random.randn(1)
			self.nbarray[i, :, :] = np.random.randn(x2, x3)*A + B  # assuming the same noise level in each beam

		# adding some RFI
		rfi = np.random.randn(x2,6)*10
		#for i in range(self.nbeam):
		for i in [0,4,12]:
			#A = 2*np.random.randn(1)+10
			#rfi = np.random.randn(x2,6)*A
			#self.nbarray[i, :, 900:905] += i
			self.nbarray[i, :, 904:910] += rfi
			#self.nbarray[i, :, 100:105] += i*0.1

	def plot_acm (self, chan):
		plt.clf()
		tmp = np.array(self.acm[chan], copy=True)
		for i in range(self.nbeam):
			tmp[i,i] = np.nan

		ax = plt.subplot(111)
		#ax.imshow(self.acm[chan], origin='lower', aspect='auto')
		ax.imshow(tmp, origin='lower', aspect='auto')
		plt.savefig('acm.png')
		#plt.show()

	#def plot_data (self, beams=[0,1,2]):
	def plot_data (self, xr, xtick):
		plt.clf()
		'''
		nplot = len(beams)
		fig, ax = plt.subplots(2,nplot,figsize=(16,16), gridspec_kw={'height_ratios': [1, 4]})
		plt.subplots_adjust(wspace=0.01, hspace=0.01)

		for i in range(nplot):
			ax[0,i].set_xticks([])
			ax[1,i].set_yticks([])
			ax[0,i].set_yticks([])
			ax[0,i].set_title('Beam{0}'.format(beams[i]))

			ax[1,i].set_xticks(np.arange(0,self.nchan+1,256))

			ax[0,i].plot(np.arange(self.nchan), np.mean(self.nbarray[i,:,:], axis=0))
			ax[1,i].imshow(self.nbarray[i,:,:], origin='lower', aspect='auto')

			# some rough masking
			spec = np.mean(self.nbarray[i,:,:], axis=0)
			std = np.std(spec[200:])   # hard coded fro MB bpsr data
			mean = np.mean(spec[200:])   # hard coded fro MB bpsr data
			mask = (np.fabs(spec-mean) > 5*std) & (np.arange(self.nchan)>200)
			mask_chn = np.arange(self.nchan)[mask]
			mask_freq = self.dat_freq[mask]
			print (i, mask_chn, mask_freq)

		ax[1,0].set_ylabel('Sample')
		ax[1,0].set_yticks(np.arange(0, self.nsamp+1, 256))
		plt.savefig('inspect_data.png')
		'''
		plt.figure(figsize=(12,12))
		plt.subplots_adjust(hspace=0.05, wspace=0.05)
		i = 0
		for i in np.arange(self.nbeam):
			plot_idx = i+1
			ax = plt.subplot(4, 4, plot_idx)
			ax.set_title('beam{0}'.format(plot_idx))
			
			if plot_idx == 13:
				ax.set_xticks(np.arange(0, self.nchan, xtick))
				#ax.set_xticks(np.arange(0, self.nchan, 50))
			else:
				ax.set_xticks([])
				ax.set_yticks([])

			ax.set_xlim(xr[0], xr[1])
			#ax.set_xlim(700, 1024)
			#ax.set_ylim(2800, 3300)
			lc = np.sum(self.nbarray[i], axis=1)
			#print (np.argmax(lc))
			ax.imshow(self.nbarray[i], origin='lower', aspect='auto', interpolation='none')
			i += 1
		plt.savefig('rfi_data.png')
	
	def cal_eigen (self):
		#eigval, eigvec = la.eig(self.acm[chan])
		#print (eigval)
		#print (eigvec)
		
		self.eigval = ma.masked_array(np.empty((self.nchan, self.nbeam)), mask=np.zeros((self.nchan, self.nbeam)))
		self.freq_array = ma.masked_array(np.empty((self.nchan, self.nbeam)), mask=np.zeros((self.nchan, self.nbeam)))
		self.chan_array = ma.masked_array(np.empty((self.nchan, self.nbeam)), mask=np.zeros((self.nchan, self.nbeam)))
		#self.eigval = np.empty((self.nchan, self.nbeam))
		#self.eigval_mask = np.empty((self.nchan, self.nbeam))
		self.eigvec = np.empty((self.nchan, self.nbeam, self.nbeam))

		for i in range(self.nchan):
			u, s, vh = la.svd(self.acm[i])
			#self.eigval[i] = s
			# normalise eig val
			self.eigval[i] = np.power(s, 2)/np.sum(np.power(s, 2))
			# normalise eig vec
			#self.eigvec[i] = vh
			#self.eigvec[i] = vh/la.norm(vh)
			self.eigvec[i] = u
			#self.eigvec[i] = u/la.norm(u)
			#print (i, len(s))
			#print (s)
			#print (u)
			self.freq_array[i,:] = self.dat_freq[i]
			self.chan_array[i,:] = i

		plt.clf()
		ax = plt.subplot(211)
		ax.set_ylim(200,1000)
		im = ax.imshow(self.eigval, origin='lower', aspect='auto')
		#im = ax.imshow(self.eigval[self.freq_mask,:], origin='lower', aspect='auto')
		cbar = plt.colorbar(im, ax=ax, extend='both')
		cbar.minorticks_on()

		ax = plt.subplot(212)
		ax.hist(np.ndarray.flatten(self.eigval), bins=30)

		plt.savefig('eigval.png')
		#plt.show()

		np.save('eigval_sub%d'%self.sub0, ma.getdata(self.eigval))
		np.save('eigvec_sub%d'%self.sub0, ma.getdata(self.eigvec))

	def generate_filter (self, eig_idx, xlim, xtick):
		self.chan_array.mask = self.chan_array < 200 # hard coded at the moment for MB dspsr data
		self.freq_array.mask = self.chan_array.mask
		self.eigval.mask = self.chan_array.mask
		#print (self.eigval.shape, ma.mean(self.eigval), ma.std(self.eigval))
		#for i in range(self.nbeam):
		#	print (self.eigval[:,i].shape, ma.mean(self.eigval[:,i]), ma.std(self.eigval[:,i]))

		# treating each eig val independently; even after normalisation, there is a gradient in eig val
		for idx in range(eig_idx):
			nchan_zap0 = len(self.dat_freq[self.eigval.mask[:,idx]])
			#nchan_zap0 = ma.count_masked(self.eigval) 
			#print (nchan_zap0)

			ave_eigval = ma.mean(self.eigval[:,idx])
			std_eigval = ma.std(self.eigval[:,idx])
			#print (ave_eigval, std_eigval, ma.mean(self.eigval[:,1]), ma.std(self.eigval[:,1]))
			#print (default_mask.shape, self.eigval[:,0].shape)
		
			self.eigval[:,idx] = ma.masked_greater(self.eigval[:,idx], ave_eigval+self.sigma*std_eigval)
			self.eigval[:,idx] = ma.masked_less(self.eigval[:,idx], ave_eigval-self.sigma*std_eigval)
			#self.eigval.mask = ma.fabs(self.eigval - ave_eigval) > self.sigma*std_eigval
			#print (self.dat_freq[self.eigval.mask[:,0]], np.arange(self.nchan)[self.eigval.mask[:,0]])
			#print (len(self.dat_freq[self.eigval.mask[:,eig_idx]]), ma.std(self.eigval[:,eig_idx]))

			nchan_zap = len(self.dat_freq[self.eigval.mask[:,idx]])
			#print (nchan_zap)
			count = 0
			while nchan_zap != nchan_zap0:
			#while count < 5:
				#print (count)
				nchan_zap0 = nchan_zap
				ave_eigval = ma.mean(self.eigval[:,idx])
				std_eigval = ma.std(self.eigval[:,idx])
				#print (count, ave_eigval, std_eigval)
				self.eigval[:,idx] = ma.masked_greater(self.eigval[:,idx], ave_eigval+self.sigma*std_eigval)
				self.eigval[:,idx] = ma.masked_less(self.eigval[:,idx], ave_eigval-self.sigma*std_eigval)
				#self.eigval.mask = ma.fabs(self.eigval - ave_eigval) > self.sigma*std_eigval
				nchan_zap = len(self.dat_freq[self.eigval.mask[:,idx]])
				#print (len(self.dat_freq[self.eigval.mask[:,eig_idx]]), ma.std(self.eigval[:,eig_idx]))
				#print (count, ave_eigval, std_eigval, nchan_zap)
				#print (np.arange(self.nchan)[self.eigval.mask[:,idx]])
				count += 1

		# combine all the masked channel together
		self.freq_mask = np.any(self.eigval.mask, axis=1)
		print (np.arange(self.nchan)[self.freq_mask])

		#print (self.eigvec[:,:,0].shape)
		plt.clf()
		plt.figure(figsize=(9,9))
		ax = plt.subplot(211)
		#ax = plt.subplot(221)
		ax.set_title('Spectrum')
		#ax.set_ylim(0.5, 6)
		ax.set_xticks(np.arange(0,self.nchan,xtick))
		#ax.set_xlim(200, 1024)
		ax.set_xlim(xlim[0], xlim[1])
		#ax.imshow(self.eigvec[:,:,0], origin='lower', aspect='auto', interpolation='none')

		spec = np.mean(self.nbarray[:,:,:], axis=1)
		for i in range(self.nbeam):
		#for i in [0,1,2,3,4,5,6,7,8,9,10,11]:
			#spec = np.mean(self.nbarray[i,:,:], axis=0)
			ax.scatter(np.arange(self.nchan)[self.freq_mask], spec[i, self.freq_mask]+i, label='beam{0}'.format(i), alpha=1, marker='+', s=60)
			ax.plot(np.arange(self.nchan)[200:], spec[i, 200:]+i, color='k')
		ax.legend(loc='upper right', fontsize=6)

		#ax.imshow(spec, origin='lower', aspect='auto', interpolation='none')

		ax.grid()
		#plt.show()
		###################
		ax = plt.subplot(212)
		#ax = plt.subplot(223)
		ax.set_title('Mask')
		ax.set_xticks(np.arange(0,self.nchan,xtick))
		#ax.set_xlim(200, 1024)
		ax.set_xlim(xlim[0], xlim[1])
		
		mask = np.empty((self.nbeam, self.nchan))
		for i in range(self.nbeam):
			mask[i,:] = self.freq_mask
			
		#print (spec.shape, mask.shape)
		spec = ma.masked_array(spec, mask=mask)
		for i in range(self.nbeam):
			ax.plot(np.arange(self.nchan)[200:], spec[i, 200:]+i, color='k')

		#ax.imshow(spec, origin='lower', aspect='auto', interpolation='none')

		ax.grid()
		plt.savefig('zapping_spectrum.png')

		###################
		plt.clf()
		plt.figure(figsize=(9,9))
		ax = plt.subplot(111)
		ax.set_title('Eig value')
		#ax.set_xlim(200, 1024)
		ax.set_xticks(np.arange(0,self.nchan,xtick))
		ax.set_xlim(xlim[0], xlim[1])
		for i in range(self.nbeam):
			plot_eigval = ma.getdata(self.eigval)[:,i]
			ax.scatter(np.arange(self.nchan)[self.eigval.mask[:,i]], plot_eigval[self.eigval.mask[:,i]]+0.05*i, alpha=1, marker='+', s=60)
			#ax.plot(np.arange(self.nchan)[200:], self.eigval[200:,i]+0.05*i, color='k')
			ax.plot(np.arange(self.nchan)[200:], plot_eigval[200:]+0.05*i, color='k')
		
		ax.grid()
		plt.savefig('zapping_eig.png')

	def generate_mask (self, chn, base):
		# generate mask for each beam
		# self.eigval shape (self.nchan, self.nbeam)
		# self.eigvec shape (self.nchan, self.nbeam, self.nbeam)

		#self.rfi_mask = np.empty((self.nchan, self.nbeam))
		self.rfi_mask = np.ones((self.nchan, self.nbeam))
		eig_mask = self.eigval.mask[:,0]
		init_mask = np.swapaxes(np.array(self.nbeam*[eig_mask]), 0, 1)
		print (init_mask.shape)
		ave_eigvec = np.mean(self.eigvec[:, :, 0])
		std_eigvec = np.std(self.eigvec[:, :, 0])
		print (ave_eigvec, std_eigvec)
		ave_eigvec = np.mean(self.eigvec[:, :, 0][init_mask])
		std_eigvec = np.std(self.eigvec[:, :, 0][init_mask])
		print (ave_eigvec, std_eigvec)
		ave_eigvec = np.mean(self.eigvec[660:700, :, 0])
		std_eigvec = np.std(self.eigvec[660:700, :, 0])
		print (ave_eigvec, std_eigvec)
		for i in range(self.nchan):
			if i <=200: 
				self.rfi_mask[i,:] = 0
			elif self.eigval.mask[i,0] == 1:
				#self.rfi_mask[i, :] = 0
				#beam_mask = np.fabs(self.eigvec[i, :, 0] - ave_eigvec) > 0.01*std_eigvec
				beam_mask = np.fabs(self.eigvec[i, :, 0]) > 0.1*std_eigvec
				self.rfi_mask[i, beam_mask] = 0
			#else:
			#	self.rfi_mask[i,:] = 1

		self.rfi_mask = np.array(self.rfi_mask, dtype=bool)
		#### making diagnostic plot #####
		'''
		#chn = 907
		#chn = 864
		spec = np.mean(self.nbarray[:,:,:], axis=1)
		#print(spec[:, chn], np.argmax(spec[:, chn]))
		#print(self.eigval[chn, :], ma.getdata(self.eigval[chn,:]), self.eigval.mask[chn,:])
		#print(self.eigvec[chn, :, 0])

		plt.clf()
		plt.figure(figsize=(9,9))
		ax = plt.subplot(111)
		#ax.scatter(np.arange(self.nbeam), spec[:, chn], color='grey')
		#ax.plot(np.arange(self.nbeam), spec[:, chn], color='grey')
		ax.scatter(np.arange(self.nbeam), spec[:, chn] - np.mean(spec[:, base[0]:base[1]], axis=1), color='grey')
		ax.plot(np.arange(self.nbeam), spec[:, chn] - np.mean(spec[:, base[0]:base[1]], axis=1), color='grey')

		ax.plot(np.arange(self.nbeam), self.eigvec[chn, :, 0], color='red')
		#ax.plot(np.arange(self.nbeam), self.eigvec[chn, :, 1]-0.05, color='blue')
		#ax.plot(np.arange(self.nbeam), self.eigvec[chn, :, 2]-0.1, color='green')
		#ax.plot(np.arange(self.nbeam), self.eigvec[chn, :, 11]-0.15, color='k')

		#added_vec = np.sum(self.eigvec[chn, :, :], axis=0)
		#added_vec = np.sum(self.eigvec[chn, self.eigval.mask[chn, :], :], axis=0)
		#added_vec = np.empty(self.nbeam)
		#for i in range(self.nbeam):
		#	#added_vec[i] = np.sum(self.eigvec[chn, i, :])
		#	added_vec[i] = np.sum(self.eigvec[chn, i, self.eigval.mask[chn, ])
		#ax.plot(np.arange(self.nbeam), added_vec, color='pink')

		ax.hlines(0, self.nbeam, 0, ls='--')
		plt.savefig('test.png')
		'''
	def plot_spec (self, xr, xtick):
		plt.clf()
		plt.figure(figsize=(12,12))
		plt.subplots_adjust(hspace=0.05, wspace=0.05)
		i = 0
		for i in np.arange(self.nbeam):
			plot_idx = i+1
			ax = plt.subplot(4, 4, plot_idx)
			ax.set_title('beam{0}'.format(plot_idx))
			
			if plot_idx == 13:
				ax.set_xticks(np.arange(0, self.nchan, xtick))
				#ax.set_xticks(np.arange(0, self.nchan, 50))
			else:
				ax.set_xticks([])
				ax.set_yticks([])

			ax.set_xlim(xr[0], xr[1])
			spec = np.sum(self.nbarray[i], axis=0)
			ax.plot(np.arange(self.nchan), spec, color='k')

			ax.scatter(np.arange(self.nchan)[self.rfi_mask[:,i]], spec[self.rfi_mask[:,i]], color='r', alpha=0.5)
			i += 1
		plt.savefig('spectrum.png')
	

##########################

if __name__ == "__main__":
	######################
	#rc('text', usetex=True)
	# read in parameters
	
	parser = argparse.ArgumentParser(description='Read PSRFITS format search mode data')
	parser.add_argument('-f',  '--input_file',  metavar='Input file name',  nargs='+', required=True, help='Input file name')
	#parser.add_argument('-o',  '--output_file', metavar='Output file name', nargs='+', required=True, help='Output file name')
	parser.add_argument('-sub',  '--subband_range', metavar='Subint ragne', nargs='+', default = [0, 0], type = int, help='Subint range')
	parser.add_argument('-freq',  '--freq_range', metavar='Freq range (MHz)', nargs='+', default = [0, 0], type = int, help='Frequency range (MHz)')
	parser.add_argument('-sig',  '--sigma', metavar='Threshold', default = 10, type = float, help='Masking threshold')
	
	args = parser.parse_args()
	sub_start = int(args.subband_range[0])
	sub_end = int(args.subband_range[1])
	freq_start = int(args.freq_range[0])
	freq_end = int(args.freq_range[1])
	sigma = float(args.sigma)
	#dm = float(args.chn_dm[0])
	infile = args.input_file
	
	#read_data (infile[0], sub_start, sub_end)
	mb_acm = acm (infile, sub_start, sub_end, freq_start, freq_end, sigma)
	mb_acm.read_data()
	mb_acm.normalise(base=[660,700])
	mb_acm.plot_data(xr=[600,1000], xtick=50)
	#mb_acm.sim_acm() # instead of reading in data, simulate
	#print (mb_acm.acm.shape)
	
	#mb_acm.plot_bandpass()
	
	mb_acm.cal_acm()
	#mb_acm.plot_acm(903)
	#print (np.mean(mb_acm.acm[1]))
	#print (np.mean(mb_acm.acm[903]))
	mb_acm.cal_eigen()

	mb_acm.generate_filter(eig_idx=1,xlim=[850,930],xtick=10)
	mb_acm.generate_mask(chn=865, base=[660,700])
	#mb_acm.plot_spec (xr=[330,350], xtick=10)
	mb_acm.plot_spec (xr=[800,1000], xtick=50)

