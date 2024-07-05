import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sys
import glob
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import argparse

class archive ():
	def __init__ (self, fits_name):
		hdulist = fits.open(fits_name)  # open the file
		#hdulist.info() # print info
		#hdulist['SUBINT'].header # print header of 'SUBINT'
		self.mjd = float(hdulist[0].header['STT_IMJD']) + float(hdulist[0].header['STT_SMJD'])/86400
		self.bw = float(hdulist[0].header['OBSBW'])
		self.freq = float(hdulist[0].header['OBSFREQ'])
		self.stt_imjd = float(hdulist[0].header['STT_IMJD'])
		self.stt_smjd = float(hdulist[0].header['STT_SMJD'])
		self.stt_offs = float(hdulist[0].header['STT_OFFS'])
		self.obs_mode = hdulist[0].header['OBS_MODE']

		tsamp = hdulist['SUBINT'].header['TBIN']
		if tsamp != '*':
			self.tsamp = float(hdulist['SUBINT'].header['TBIN'])

		self.nbin = int(hdulist['SUBINT'].header['NBIN'])
		self.npol = int(hdulist['SUBINT'].header['NPOL'])
		self.nchn = int(hdulist['SUBINT'].header['NCHAN'])
		self.nsub = int(hdulist['SUBINT'].header['NAXIS2'])

		self.tbdata = hdulist['SUBINT'].data # open the table, put the data into tbdata
		self.tsub = self.tbdata['TSUBINT']
		self.length = self.tsub*self.nsub
		
		#self.profile_raw = np.rot90(np.squeeze(self.tbdata['DATA']))
		#self.dat_offs = np.squeeze(self.tbdata['DAT_OFFS']).reshape((self.nchn, self.npol))
		#self.dat_scl = np.squeeze(self.tbdata['DAT_SCL']).reshape((self.nchn, self.npol))
		self.profile_raw = self.tbdata['DATA']
		self.dat_offs = self.tbdata['DAT_OFFS'].reshape((self.npol, self.nchn))
		self.dat_scl = self.tbdata['DAT_SCL'].reshape((self.npol, self.nchn))
		self.dat_wts = self.tbdata['DAT_WTS'].reshape((self.nsub,self.nchn))
		self.dat_freq = self.tbdata['DAT_FREQ']		

		#self.dat_wts[self.dat_wts==0.] = 'nan'
		hdulist.close()

	def get_profile (self, scl=True, wts=True):
		self.profile = np.zeros((self.nsub, self.npol, self.nchn, self.nbin))
		if scl == True:
			for i in range(self.nsub):
				for j in range(self.npol):
					for k in range(self.nchn):
						if wts == True:
							self.profile[i,j,k,:] += self.dat_wts[i,k]*(self.dat_scl[j,k]*self.profile_raw[i,j,k,:] + self.dat_offs[j,k])
						else:
							self.profile[i,j,k,:] += self.dat_scl[j,k]*self.profile_raw[i,j,k,:] + self.dat_offs[j,k]

	def remove_baseline (self, sub, delta=0.4):
		'''
		delta: off pulse fraction
		'''
		I = self.profile[sub,0,:,:]
		#I = np.mean(I, axis=0)
		I = np.nanmean(I, axis=0)
		num = int(self.nbin*delta)
		m = self.nbin
		
		std = []
		for i in range(m):
			if (i+num) < m:
				temp = I[i:(i+num)]
				#std.append(np.std(temp))
				std.append(np.nanstd(temp))
			else:
				temp = np.concatenate((I[i:m],I[0:(i+num-m)]))
				#std.append(np.std(temp))
				std.append(np.nanstd(temp))
		
		std = np.array(std)
		#print std
		index = np.argmin(std)
		
		baseline = np.nanmean(self.profile[sub,:,:,index:(index+num)], axis=-1)
		#baseline = np.mean(self.profile[sub,:,:,index:(index+num)], axis=-1)
		#print (baseline[0,:])
		
		for i in range(self.npol):
			for j in range(self.nchn):
				self.profile[sub,i,j,:] = self.profile[sub,i,j,:] - baseline[i,j]

	def centering (self, sub):
		I = self.profile[sub,0,:,:]
		I = np.nanmean(I, axis=0)
		#I = np.mean(I, axis=0)
		m = self.nbin
		num = 5  # number of phase bins to average
		
		mean = []
		for i in range(m):
			if (i+num) < m:
				temp = I[i:(i+num)]
				mean.append(np.nanmean(temp))
				#mean.append(np.mean(temp))
			else:
				temp = np.concatenate((I[i:m],I[0:(i+num-m)]))
				#mean.append(np.mean(temp))
				mean.append(np.nanmean(temp))
		
		index = np.argmax(mean)
		shift = int(m/2 - index)
		for i in range(self.npol):
			for j in range(self.nchn):
				self.profile[sub,i,j,:] = np.roll(self.profile[sub,i,j,:], shift, axis=-1)

	def fscrunch (self, n):
		self.profile = self.profile.reshape(self.nsub, self.npol, int(self.nchn/n), n, self.nbin)
		#self.profile = np.mean(self.profile, axis=-2)
		self.profile = np.nanmean(self.profile, axis=-2)
		
		self.dat_freq = self.dat_freq.reshape(self.nsub, int(self.nchn/n), n)
		#self.dat_freq = np.mean(self.dat_freq, axis=-1)
		self.dat_freq = np.nanmean(self.dat_freq, axis=-1)

		self.nchn = int(self.nchn/n)
		
	def bscrunch (self, n):
		r = self.nbin % n
		if (self.nbin % n) == 0:
			self.profile = self.profile.reshape(self.nsub, self.npol, self.nchn, int(self.nbin/n), n)
			#self.profile = np.mean(self.profile, axis=-1)
			self.profile = np.nanmean(self.profile, axis=-1)
		
			self.nbin = int(self.nbin/n)
		else:
			self.profile = self.profile[:,:,:,:(self.nbin-r)].reshape(self.nsub, self.npol, self.nchn, int(self.nbin/n), n)
			#self.profile = np.mean(self.profile, axis=-1)
			self.profile = np.nanmean(self.profile, axis=-1)
		
			self.nbin = int(self.nbin/n)

	def cal_pa (self, sub, delta=0.4):
		self.linear = np.sqrt(np.power(self.profile[sub, 1, :, :], 2.) + np.power(self.profile[sub, 2, :, :], 2.))
		## determine the off pulse phase
		I = self.profile[sub,0,:,:]
		#I = np.mean(I, axis=0)
		I = np.nanmean(I, axis=0)
		num = int(self.nbin*delta)
		m = self.nbin
		
		std = []
		for i in range(m):
			if (i+num) < m:
				temp = I[i:(i+num)]
				#std.append(np.std(temp))
				std.append(np.nanstd(temp))
			else:
				temp = np.concatenate((I[i:m],I[0:(i+num-m)]))
				#std.append(np.std(temp))
				std.append(np.nanstd(temp))
		
		std = np.array(std)
		#print std
		index = np.argmin(std)
		#baseline = np.mean(self.profile[sub,:,:,index:(index+num)], axis=-1)
		baseline = np.nanmean(self.profile[sub,:,:,index:(index+num)], axis=-1)

		# calculate Q and U rms
		#self.sigma = np.std(self.profile[sub,:,:,index:(index+num)], axis=-1)
		self.sigma = np.nanstd(self.profile[sub,:,:,index:(index+num)], axis=-1)

		# correcting Linear pol bias
		for i in range(self.nchn):
			mask = self.linear[i,:] > self.sigma[0, i]*1.57
			self.linear[i,mask] = np.sqrt((self.linear[i,mask]/self.sigma[0,i])**2. - 1.)*self.sigma[0,i]

			mask = self.linear[i,:] <= self.sigma[0, i]*1.57
			self.linear[i,mask] = 0.
			#for j in range(nbin):
			#	if linear[i,j] > I_sig[i]*1.57:
			#		linear[i,j] = np.sqrt((linear[i,j]/I_sig[i])**2. - 1.)*I_sig[i]
			#	else:
			#		linear[i,j] = 0.
		#l_baseline = np.mean(self.linear[:,index:(index+m)], axis=-1)
		l_baseline = np.nanmean(self.linear[:,index:(index+m)], axis=-1)
		for i in range(self.nchn):
			self.linear[i,:] = self.linear[i,:] - l_baseline[i]
		#self.l_sig = np.std(self.linear[:,index:(index+num)], axis=-1)
		self.l_sig = np.nanstd(self.linear[:,index:(index+num)], axis=-1)
		#print("Linear RMS %f"%np.mean(self.l_sig))
		print("Linear RMS %f"%np.nanmean(self.l_sig))

		# calculating PA
		self.pa = np.empty((self.nchn, self.nbin))
		self.pa_err = np.empty((self.nchn, self.nbin))
		self.phase_pa = np.empty((self.nchn, self.nbin))

		self.pa[:] = np.nan
		self.pa_err[:] = np.nan
		self.phase_pa[:] = np.nan
		for chn in range(self.nchn):
		    for j in range(self.nbin):
		        if np.fabs(self.linear[chn,j]/self.l_sig[chn]) >= 5:
		            U = self.profile[sub,2,chn,j]
		            Q = self.profile[sub,1,chn,j]
		            sig_U = self.sigma[2,chn]
		            sig_Q = self.sigma[1,chn]

		            self.pa[chn,j] = (np.arctan2(U, Q)/2.)*180/np.pi
		            self.phase_pa[chn,j] = j
		    
		            part1 = Q/(Q*Q+U*U)
		            part2 = -U/(Q*Q+U*U)
		            self.pa_err[chn,j] = 0.5*(180/np.pi)*np.sqrt(part1*part1*sig_U*sig_U+part2*part2*sig_Q*sig_Q)
		
		#print(pa.shape, err.shape)
		#mask = err/pa <= 0.1
		#return pa, err, phase_bin

class fits_srch ():
	def __init__ (self, filenames, sub0=0, sub1=0, freq0=0, freq1=0):
		self.filenames = filenames
		self.sub0 = sub0
		self.sub1 = sub1
		
		self.nbeam = len(filenames)
		
		# here I assume that all beams use the same observational setup
		hdulist = fits.open(self.filenames[0])
		self.obsbw = hdulist['PRIMARY'].header['OBSBW']
		#self.ibeam = hdulist['PRIMARY'].header['IBEAM']
		self.nsub =  int(hdulist['SUBINT'].header['NAXIS2'])
		self.nbits = int(hdulist['SUBINT'].header['NBITS'])
		self.nchan = int(hdulist['SUBINT'].header['NCHAN'])
		self.nsblk = int(hdulist['SUBINT'].header['NSBLK'])
		self.npol =  int(hdulist['SUBINT'].header['NPOL'])
		self.nstot = int(hdulist['SUBINT'].header['NSTOT'])

		tbdata = hdulist['SUBINT'].data
		self.dat_freq = tbdata['DAT_FREQ'][0]
		#self.dat_scls = tbdata['DAT_SCL']
		#self.dat_offs = tbdata['DAT_OFFS']

		hdulist.close()
		
		#self.nbarray = np.empty((self.nbeam, self.nsblk*int(sub1-sub0), self.nchan))
		#self.freq_mask = self.dat_freq < 1520
		if freq0 == 0 and freq1 == 0:
			self.freq_mask = self.dat_freq > freq0
		else:
			self.freq_mask = (self.dat_freq > freq0) & (self.dat_freq < freq1)

		if self.sub0 == 0 and self.sub1 == 0:
			self.nsamp = self.nsub*self.nsblk
			self.nbarray = np.empty((self.nbeam, self.nsblk*self.nsub, self.nchan))
		else:
			self.nsamp = (self.sub1-self.sub0)*self.nsblk
			self.nbarray = np.empty((self.nbeam, self.nsblk*int(self.sub1-self.sub0), self.nchan))
	
	def read_data (self):
		for i in range(self.nbeam):
			hdulist = fits.open(self.filenames[i])
			tbdata = hdulist['SUBINT'].data
			hdulist.close()
			print ('Read in data...')
			print ('{0} nsub:{1} nsblk:{2} nbits:{3} nchan:{4}'.format(self.filenames[i], self.nsub, self.nsblk, self.nbits, self.nchan))

			if self.sub0 == 0 and self.sub1 == 0:
				data = np.squeeze(tbdata['DATA'])
				data = np.reshape(data, (self.nsub*self.nsblk, int(self.nchan/(8/self.nbits))))
			else:
				data = np.squeeze(tbdata['DATA'][self.sub0:self.sub1, :, :, :])
				data = np.reshape(data, ((self.sub1-self.sub0)*self.nsblk, int(self.nchan/(8/self.nbits))))
			
			#############################################################
				
			##### unpack data #####
			#data_unpack = unpack_data (data, nbits, nchan, nsamp)
			temp = np.reshape(np.unpackbits(data, axis=-1), (self.nsamp, self.nchan, self.nbits))
				
			self.nbarray[i] = np.squeeze(np.packbits(np.insert(temp, [0,0,0,0,0,0], 0, axis=-1), axis=-1))
			#data_unpack = np.moveaxis(data_unpack, 0, -1)
			print ('Shape of the unpacked data: {0}\n'.format(self.nbarray[i].shape))

	def plot_bandpass (self, beam=np.nan):
		plt.clf()
		ax1 = plt.subplot(111)
		ax1.set_xlabel('Channel')
		ax1.set_xlim(0, self.nchan)
		ax1.set_xticks(np.arange(0,self.nchan,200))

		ax2 = ax1.twiny()
		ax2.set_xlabel('Frequency (MHz)')
		ax2.set_xlim(0, self.nchan)
		ax2.set_xticks(np.arange(0,self.nchan,200))
		ax2.set_xticklabels(np.round(self.dat_freq[0] - np.arange(0,self.nchan,200)*self.obsbw/self.nchan,1))

		if self.npol == 1:
			if np.isnan(beam):
				bp = np.mean(self.nbarray, axis=1)
			else:
				bp = np.mean(self.nbarray[beam], axis=0)
		else:
			if np.isnan(beam):
				bp = np.mean(self.nbarray[:,:,0,:], axis=1)
			else:
				bp = np.mean(self.nbarray[beam,:,0,:], axis=0)

		if np.isnan(beam):
			bp_name = 'allbeams_bandpass.png'
			for i in range(self.nbeam):
				#ax1.plot(self.dat_freq, bp[i]+i, label='beam{0}'.format(i))
				ax1.plot(np.arange(self.nchan), bp[i]+i, label='beam{0}'.format(i))
		else:
			#ax1.plot(self.dat_freq, bp)
			ax1.plot(np.arange(self.nchan), bp)
			bp_name = self.filenames[beam] + '.bandpass.png'

		#ax.legend(loc='upper right')
		plt.savefig(bp_name)
	
######################
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Read PSRFITS format search mode data')
	parser.add_argument('-f',  '--input_file',  metavar='Input file name', nargs='+',  required=True, help='Input file name')
	parser.add_argument('-sub0',  '--subband_start', metavar='Starting subband', required=True, help='Starting subband')
	parser.add_argument('-sub1',  '--subband_end', metavar='Ending subband',  required=True, help='Ending subband')

	args = parser.parse_args()
	sub_start = int(args.subband_start[0])
	sub_end = int(args.subband_end[0])
	#dm = float(args.chn_dm[0])
	infile = args.input_file

	srch = fits_srch (infile, sub_start, sub_end)
	srch.read_data()

	srch.plot_bandpass(0)
