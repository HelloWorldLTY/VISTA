import scib_metrics
import scanpy as sc
import squidpy as sq
import numpy as np
sc.logging.print_header()
print(f"squidpy=={sq.__version__}")

import scanpy as sc
import squidpy as sq
import NaiveDE
import SpatialDE

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor="white")

def evaluate_nmi_ari(adata, label = 'scClassify'):
    labels = np.array(list(adata.obs[label]))
    result1 = scib_metrics.nmi_ari_cluster_labels_leiden(adata.obsp['connectivities'], labels = labels, n_jobs = -1)
    result2 = scib_metrics.silhouette_label(adata.obsm['X_pca'], labels = labels, rescale=True, chunk_size=256)
    print(result1)
    print(result2)
    return result1, result2

def calculate_moranI_proportion(adata):
    from collections import Counter
    sq.gr.spatial_neighbors(adata)
    sq.gr.spatial_autocorr(adata, mode="moran", genes=adata.var_names)
    result_dict = dict(Counter(adata.uns["moranI"]['pval_norm_fdr_bh']<0.05))
    if True not in result_dict.keys():
        return 0
    else:
        return result_dict[True]/len(adata.var_names)

def create_spatialde_proportion(adata, n_genes = 1000):
    from collections import Counter
    n = n_genes
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    counts = sc.get.obs_df(adata, keys=list(adata.var_names), use_raw=False)
    total_counts = sc.get.obs_df(adata, keys=["total_counts"])
    norm_expr = NaiveDE.stabilize(counts.T).T
    resid_expr = NaiveDE.regress_out(total_counts, norm_expr.T, "np.log(total_counts)").T
    sample_resid_expr = resid_expr.sample(n=n, axis=1, random_state=1)
    results = SpatialDE.run(adata.obsm["spatial"], sample_resid_expr)
    top10 = results.sort_values("qval")[["g", "l", "qval"]]

    result_dict = dict(Counter(top10['qval']<0.05))
    if True not in result_dict.keys():
        return 0
    else:
        return result_dict[True] / n

def evaluate_cellphonedb(adata, label = "scClassify"):

    res = sq.gr.ligrec(
    adata,
    n_perms=1000,
    cluster_key=label,
    copy=True,
    use_raw=False,
    transmitter_params={"categories": "ligand"},
    receiver_params={"categories": "receptor"},
    )

    return np.mean((res['pvalues'].values<0.05)*1)


###running exmaple###
adata = sc.read("method_impute_result.h5ad")

print(evaluate_nmi_ari(adata))