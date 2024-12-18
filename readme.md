# VISTA Uncovers Missing Gene Expression and Spatial-induced Information for Spatial Transcriptomic Data Analysis

# Install

To install VISTA, please install [scvi](https://docs.scvi-tools.org/en/stable/tutorials/index.html), [scanpy](https://scanpy-tutorials.readthedocs.io/en/latest/index.html) and [pyg](https://pytorch-geometric.readthedocs.io/en/latest/index.html) in ahead, using:

```
pip install scvi-tools==0.20.3 
pip install scanpy==1.9.3
pip install torch_geometric 
```

or

```
conda install scvi-tools -c conda-forge
conda install scanpy
conda install pyg -c pyg
```

You also need to install [FAISS](https://github.com/facebookresearch/faiss):

```
conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl
```

Then you can install vista based on :

```
git clone https://github.com/HelloWorldLTY/VISTA.git
cd VISTA
pip install .
```

We provide a yml file for related packages, feel free to use and check.


# Key functions

The general process of running VISTA contains three steps: pre-processing, model construction and training.

## Pre-processing:

Please ensure your reference scRNA-seq dataset and spatial data for imputation are all count-based data. And also please ensure the cell type label infromation of these two datasets is stored in the key **scClassify** of the anndata object.

```
import scanpy as sc
adata_sc = sc.read("path_scrnaseq")
adata_st = sc.read("path_spatial")
# should print out the cell type labels.
print(adata_st.obs.scClassify.unique()) 
# should print out the cell type labels.
print(adata_sc.obs.scClassify.unique()) 
```

Then you can filter genes with lower correlation by:

```
info_gene = calcualte_pse_correlation(seq_data, spatial_data, 'scClassify')
adata_st = adata_st[:,info_gene]
```

Now you also need to store the cell id in the key **names** and spatial information in the key **spatial** of the spatial data:
```
adata_st.obs['names'] = adata_st.obs_names
# should print out the cell id
print(adata_st.obs['names']) 
# should print out the information of spatial location, as a n x 2 matrix.
print(adata_st.obsm['spatial'])
```

## Model construction:

Now you can set up the data that VISTA needs:

```
from vista import GIMVI_GCN
GIMVI_GCN.setup_anndata(adata_st, batch_key="batch", obs_names = 'names')
GIMVI_GCN.setup_anndata(adata_sc)
```

And then you can set up the model:
```
model = GIMVI_GCN(adata_sc, adata_st)
```

## Training:

Now you can start training and save the imputation results.

```
# train for 200 epochs
model.train(200)

fish_imputation_norm = model.get_imputed_values(normalized=True)[0]
fish_imputation_raw = model.get_imputed_values(normalized=False)[0]
fish_imputation_theta = model.get_imputed_theta(normalized=False)[0]

spatial_data_imputed = sc.AnnData(fish_imputation_raw, obs = adata_st.obs, var = adata_sc.var)

spatial_data_imputed.obsm['imputed'] = fish_imputation_norm
spatial_data_imputed.obsm['imputed_raw'] = fish_imputation_raw
spatial_data_imputed.obsm['imputed_raw_theta'] =  fish_imputation_theta

spatial_data_imputed.write_h5ad("osmfish_imputation.h5ad")
```

More information for API and hyper-parameter settings can be found in the folder vista and the example demo_imputation.ipynb file.

# Evaluation

The codes for evaluation can be found in the folder metrics.

For Tangram, we implemented a mini-batch version based on this [idea](https://github.com/broadinstitute/Tangram/issues/100). The mini-batch version can be found in this [link](https://github.com/HelloWorldLTY/Tangram_v2.git). For the rest of the methods, please refer their project website for installation and usage: [gimVI](https://docs.scvi-tools.org/en/0.20.3/tutorials/notebooks/gimvi_tutorial.html), [TransImp](https://transpa.readthedocs.io/en/latest/install.html), and [SpaGE](https://github.com/tabdelaal/SpaGE). 

We also need the following packages to run the metrics for evaluation:

[scib](https://github.com/theislab/scib), [SpatialDE](https://github.com/Teichlab/SpatialDE), and [squidpy](https://github.com/scverse/squidpy).


# Applications

The codes for downstream applications can be found in the folder notebooks. To runn all the analysis for applications, you need to install the following packages:

[scVelo](https://scvelo.readthedocs.io/en/stable/), [VeloVI](https://velovi.readthedocs.io/en/latest/index.html), and [SIMVI](https://github.com/KlugerLab/SIMVI).


# Acknowledgement

We refer the codes from the following packages to implement VISTA. Many thanks to these great developers:

[gimVI](https://github.com/scverse/scvi-tools/tree/main/scvi/external/gimvi), [H2GCN](https://github.com/GitEventhandler/H2GCN-PyTorch/blob/master/model.py), [SIMVI](https://github.com/KlugerLab/SIMVI), and [SpatialBenchmarking](https://github.com/QuKunLab/SpatialBenchmarking).

If you have any questions about this project, please contact tianyu.liu@yale.edu.

# Citation

```
@article{liu2024vista,
  title={VISTA Uncovers Missing Gene Expression and Spatial-induced Information for Spatial Transcriptomic Data Analysis},
  author={Liu, Tianyu and Lin, Yingxin and Luo, Xiao and Sun, Yizhou and Zhao, Hongyu},
  journal={bioRxiv},
  pages={2024--08},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```
