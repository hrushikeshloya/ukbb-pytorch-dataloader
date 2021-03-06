## genome_path = location of the HDF5 generated in step 1
## batch_size = chunk_size of the HDF5 files
def load_data(genome_path, i, batch_size):
    h5_file = h5py.File(genome_path , 'r')
    x = h5_file['data']
    y = h5_file['label']
    xsub = x[i:min(i+batch_size, x.shape[0]),:]
    ysub = y[i:min(i+batch_size, x.shape[0])]
    return torch.from_numpy(xsub).float(), torch.from_numpy(ysub).float()



