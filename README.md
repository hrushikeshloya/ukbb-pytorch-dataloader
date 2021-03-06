# ukbb-pytorch-dataloader
PyTorch Dataloader for UK Biobank genotype data

An efficient implemention of using SNP marker data availble in the UK Biobank for machine learning in PyTorch

Step 1: Convert the PLINK files to HDF5 format

Step 2: Use a fast HDF5 Dataloader

_ _ _

# How to use ?

```
python plink_to_hdf5.py --help
```

_ _ _

** Note: Its important to keep the batch size equal to the chunk size of HDF5 files for peak performance
