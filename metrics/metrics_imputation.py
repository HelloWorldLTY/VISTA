import scanpy as sc
import numpy as np
import pandas as pd
from anndata import AnnData
import scipy
import os
sc.logging.print_header()
print(f"squidpy=={sq.__version__}")

def cal_ssim(im1,im2,M):
    
    """
        calculate the SSIM value between two arrays.
        Detail usages can be found in PredictGenes.ipynb
    
    
    Parameters
        -------
        im1: array1, shape dimension = 2
        im2: array2, shape dimension = 2
        M: the max value in [im1, im2]
        
    """
    
    assert len(im1.shape) == 2 and len(im2.shape) == 2
    assert im1.shape == im2.shape
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, M
    C1 = (k1*L) ** 2
    C2 = (k2*L) ** 2
    C3 = C2/2
    l12 = (2*mu1*mu2 + C1)/(mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2*sigma1*sigma2 + C2)/(sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3)/(sigma1*sigma2 + C3)
    ssim = l12 * c12 * s12
    
    return ssim

def scale_max(df):
    
    
    """
        Divided by maximum value to scale the data between [0,1].
        Please note that these datafrmae are scaled data by column.
        Detail usages can be found in PredictGenes.ipynb
        
        
        Parameters
        -------
        df: dataframe, each col is a feature.

    """
    
    result = pd.DataFrame()
    for label, content in df.items():
        content = content/content.max()
        result = pd.concat([result, content],axis=1)
    return result



def scale_z_score(df):
    
    """
        scale the data by Z-score to conform the data to the standard normal distribution, that is, the mean value is 0, the standard deviation is 1, and the conversion function is 0.
        Please note that these datafrmae are scaled data by column.
        Detail usages can be found in PredictGenes.ipynb
        
        
        Parameters
        -------
        df: dataframe, each col is a feature.
        
        """
    
    result = pd.DataFrame()
    for label, content in df.items():
        content = stats.zscore(content)
        content = pd.DataFrame(content,columns=[label])
        result = pd.concat([result, content],axis=1)
    return result




def scale_plus(df):
    
    
    """
        Divided by the sum of the data to scale the data between (0,1), and the sum of data is 1.
        Please note that these datafrmae are scaled data by column.
        Detail usages can be found in PredictGenes.ipynb
        
        
        Parameters
        -------
        df: dataframe, each col is a feature.
        
    """
    
    result = pd.DataFrame()
    for label, content in df.items():
        content = content/content.sum()
        result = pd.concat([result,content],axis=1)
    return result

class count:
    ###This was used for calculating the accuracy of each integration methods in predicting genes.
    
    def __init__(self, adata1, adata2, tool, outdir, metric):
        
        """
            Parameters
            -------
            raw_count_path: str (eg. Insitu_count.txt)
            spatial transcriptomics count data file with Tab-delimited as reference, spots X genes, each col is a gene. Please note that the file has no index).
            
            impute_count_path: str (eg. result_gimVI.csv)
            predicted result file with comma-delimited. spots X genes, each row is a spot, each col is a gene.
            
            tool: str
            choose tools you want to use. ['SpaGE','gimVI','novoSpaRc','SpaOTsc','Seurat','LIGER','Tangram_image','Tangram_seq']
            
            outdir: str
            predicted result directory
            
            metric:list
            choose metrics you want to use. ['SSIM','PCC','RMSE','JS']
            
        """
        
        self.raw_count = adata1.to_df()
        self.impute_count = adata2.to_df()
        self.impute_count = self.impute_count.fillna(1e-20)
        self.tool = tool
        self.outdir = outdir
        self.metric = metric
        
    def ssim(self, raw, impute, scale = 'scale_max'):
        
        ###This was used for calculating the SSIM value between two arrays.
        
        if scale == 'scale_max':
            raw = scale_max(raw)
            impute = scale_max(impute)
        else:
            print ('Please note you do not scale data by max')
        if raw.shape[1] == impute.shape[1]:
            result = pd.DataFrame()
            for label in raw.columns:
                raw_col =  raw.loc[:,label]
                impute_col = impute.loc[:,label]
                
                M = [raw_col.max(),impute_col.max()][raw_col.max()>impute_col.max()]
                raw_col_2 = np.array(raw_col)
                raw_col_2 = raw_col_2.reshape(raw_col_2.shape[0],1)
                
                impute_col_2 = np.array(impute_col)
                impute_col_2 = impute_col_2.reshape(impute_col_2.shape[0],1)
                
                ssim = cal_ssim(raw_col_2,impute_col_2,M)
                
                ssim_df = pd.DataFrame(ssim, index=["SSIM"],columns=[label])
                result = pd.concat([result, ssim_df],axis=1)
            return result
        else:
            print("columns error")
            
    def spearmanr(self, raw, impute, scale = None):
        
        ###This was used for calculating the Pearson value between two arrays.
        
        if raw.shape[1] == impute.shape[1]:
            result = pd.DataFrame()
            for label in raw.columns:
                raw_col =  raw.loc[:,label]
                impute_col = impute.loc[:,label]
                pearsonr, _ = scipy.stats.spearmanr(raw_col,impute_col)
                pearson_df = pd.DataFrame(pearsonr, index=["Spearman"],columns=[label])
                result = pd.concat([result, pearson_df],axis=1)
            return result
        
    def JS(self, raw, impute, scale = 'scale_plus'):
        
        ###This was used for calculating the JS value between two arrays.
        
        if scale == 'scale_plus':
            raw = scale_plus(raw)
            impute = scale_plus(impute)
        else:
            print ('Please note you do not scale data by plus')    
        if raw.shape[1] == impute.shape[1]:
            result = pd.DataFrame()
            for label in raw.columns:
                raw_col =  raw.loc[:,label]
                impute_col = impute.loc[:,label]
                
                M = (raw_col + impute_col)/2
                KL = 0.5*scipy.stats.entropy(raw_col,M)+0.5*scipy.stats.entropy(impute_col,M)
                KL_df = pd.DataFrame(KL, index=["JS"],columns=[label])
                
                
                result = pd.concat([result, KL_df],axis=1)
            return result
        
    def RMSE(self, raw, impute, scale = 'zscore'):
        
        ###This was used for calculating the RMSE value between two arrays.
        
        if scale == 'zscore':
            raw = scale_z_score(raw)
            impute = scale_z_score(impute)
        else:
            print ('Please note you do not scale data by zscore')
        if raw.shape[1] == impute.shape[1]:
            result = pd.DataFrame()
            for label in raw.columns:
                raw_col =  raw.loc[:,label]
                impute_col = impute.loc[:,label]
                
                RMSE = np.sqrt(((raw_col - impute_col) ** 2).mean())
                RMSE_df = pd.DataFrame(RMSE, index=["RMSE"],columns=[label])
                
                result = pd.concat([result, RMSE_df],axis=1)
            return result
                
        
    def compute_all(self):
        raw = self.raw_count
        impute = self.impute_count
        tool = self.tool
        outdir = self.outdir
        SSIM = self.ssim(raw,impute)
        Pearson = self.spearmanr(raw, impute)
        JS = self.JS(raw, impute)
        RMSE = self.RMSE(raw, impute)
        
        result_all = pd.concat([Pearson, SSIM, RMSE, JS],axis=0)
        
        if not os.path.exists(outdir):
            print ('This is an Error : No impute file folder')
        result_all.T.to_csv(outdir + "metrics_"+tool+".csv",header=1, index=1)
        self.accuracy = result_all
        return result_all
    

###running example###
from scipy import stats
import scipy.stats as st
import scipy

adata_st_true = sc.read("method_impute_result.h5ad")
adata_st_raw = sc.read("original_spatial_data.h5ad")

comp = count(adata_st_raw, adata_st_true, f'save_file_name', './save_folder/' ,'all')
all_result = comp.compute_all()

print(all_result)