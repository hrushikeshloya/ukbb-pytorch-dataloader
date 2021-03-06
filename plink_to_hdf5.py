## Convert .Bed to .HDF5 file  (saving mean and std genotype seperately)
import pandas as pd
import os
from pysnptools.snpreader import SnpData
from pysnptools.snpreader import Pheno, Bed
import h5py
import numpy as np
from tqdm import tqdm
import argparse

def main(args):
    genome_path = args.genome_path
    phenotype_path = args.phenotype_path
    phenotype = pd.read_csv(phenotype_path)
    snp_on_disk = Bed(genome_path, count_A1=True)
    iid = pd.DataFrame(snp_on_disk.iid, columns=['FID','IID'])
    phenotype['IID'] = phenotype['IID'].astype(str)
    phenotype['FID'] = phenotype['FID'].astype(str)
    iid_merged = pd.merge(iid, phenotype, on=['IID','FID'])
    x = snp_on_disk[snp_on_disk.iid_to_index(iid_merged[['FID','IID']].values),:].read( dtype='int8' ,  _require_float32_64=False ).val
    mean_genotype = np.mean(x, axis=0)
    std_genotype = np.zeros(x.shape[1])  ## Low memory exact std dev    
    for i in tqdm(range(0, x.shape[1], 500)):
        std_genotype[i:min(i+500, x.shape[1])] = np.std(x[:,i:min(i+500, x.shape[1])], axis=0)
    y = iid_merged[args.phen_col].values
    mean_phenotype = y.mean()
    std_phenotype = y.std()
    h = h5py.File(args.out + '.hdf5', 'w')
    dset1 = h.create_dataset('data', (x.shape[0], x.shape[1]), chunks=(500, x.shape[1]), dtype= 'int8')
    for i in tqdm(range(0, x.shape[0], 500)):
        dset1[i:min(i+500, x.shape[0]),:] = x[i:min(i+500, x.shape[0])]
    dset2 = h.create_dataset('label', data=(y - mean_phenotype)/std_phenotype)
    dset3 = h.create_dataset('mean_genotype', data=mean_genotype)
    dset4 = h.create_dataset('std_genotype', data=std_genotype)
    h.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--genome_path', help= 'Genotype file in PLINK format')
    parser.add_argument('--phenotype_path', help= 'Covariate Adjusted Phenotype file in csv format')
    parser.add_argument('--phen_col', help='Column name corresponding to the phenotype in the phenotype file')
    parser.add_argument('--out', help='Output destination')
    parser.add_argument('--chunk_size', help='Chunk size of the HDF5 file', type = int, default = 500)
    args = parser.parse_args()
    main(args)