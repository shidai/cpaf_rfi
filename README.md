# cpaf_rfi
RFI codes for the cryoPAF (or multibeam system in general)

Examples:

python rfi_covariance.py -f *.sf -sub 0 20 -freq 1182 1520
python run_rfi.py -f *.sf -step 20 -freq 1182 1520
python create_mask.py -nchn 1024 -nsub 4250 -nbeam 13 -step 20 -sig 3
