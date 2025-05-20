
import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
import scipy.sparse

def run_nmf_with_restarts(data, n_components, n_runs, random_state_base=42, max_iter=500):
    """
    Runs NMF multiple times and returns the result with the lowest reconstruction error.

    Parameters:
    -----------
    data : np.ndarray
        Data matrix (cells x genes).
    n_components : int
        Number of components for NMF.
    n_runs : int
        Number of times to run NMF with different random seeds.
    random_state_base : int
        Base for generating random seeds for reproducibility.

    Returns:
    --------
    best_model : NMF model
    best_W : np.ndarray
        Cell loadings matrix (n_cells × n_components).
    best_H : np.ndarray
        Gene loadings matrix (n_components × n_genes).
    lowest_error : float
        The lowest reconstruction error achieved.
    """
    lowest_error = float('inf')
    best_model = None
    best_W = None
    best_H = None

    for i in range(n_runs):
        model = NMF(
            n_components=n_components,
            init='nndsvd',
            random_state=random_state_base + i,
            max_iter=max_iter,
            tol=1e-4
        )
        W = model.fit_transform(data)
        H = model.components_
        
        if model.reconstruction_err_ < lowest_error:
            lowest_error = model.reconstruction_err_
            best_model = model
            best_W = W
            best_H = H
            
    return best_model, best_W, best_H, lowest_error

def get_top_genes(H, gene_names, n_top=100):
    """
    Extracts the top N genes for each NMF component.

    Parameters:
    -----------
    H : np.ndarray
        Gene loadings matrix (n_components × n_genes).
    gene_names : array-like
        List of gene names.
    n_top : int
        Number of top genes to extract per component.

    Returns:
    --------
    top_genes : dict
        Dictionary mapping component index to a list of top gene names.
    """
    top_genes = {}
    for i, component_loadings in enumerate(H):
        top_indices = np.argsort(component_loadings)[::-1][:n_top]
        top_genes[i] = gene_names[top_indices].tolist()
    return top_genes

def jaccard_similarity(set1, set2):
    """
    Calculates Jaccard similarity between two sets.
    """
    intersection = len(set(set1) & set(set2))
    union = len(set(set1) | set(set2))
    return intersection / union if union > 0 else 0

def nmf_workflow(
    adata, 
    sample_column, 
    min_k, 
    max_k, 
    n_nmf_runs, 
    top_n_genes, 
    min_gene_prevalence=0.01,
    overlap_threshold_step1=0.8, 
    overlap_threshold_step2=0.4, 
    redundancy_overlap_min=0.2, 
    redundancy_overlap_max=0.8,
    max_iter=500,
    output_csv_path="NMF_GENES_pipeline.csv"
):
    """
    Runs the NMF pipeline: preprocesses data, runs NMF, and filters programs.

    Parameters:
    -----------
    adata_path : Anndata Object
    sample_column : str
        Column name in adata.obs that identifies samples.
    min_k : int
        Minimum number of NMF components (K).
    max_k : int
        Maximum number of NMF components (K).
    n_nmf_runs : int
        Number of NMF runs for each K to select the best one.
    top_n_genes : int
        Number of top genes to define each NMF program.
    min_gene_prevalence : float
        Minimum fraction of cells in a sample a gene must be present in to be kept.
    overlap_threshold_step1 : float
        Minimum Jaccard overlap for a program to overlap within sample with another k to be kept.
    overlap_threshold_step2 : float
        Minimum Jaccard overlap for cross-sample filtering.
    redundancy_overlap_min : float
        Minimum Jaccard overlap for a program to be redundant.
    redundancy_overlap_max : float
        Maximum Jaccard overlap for a program to be redundant.
    max_iter : int
        Maximum number of iterations for NMF. If there are warnings about convergence, increase this. This will increase runtime.
    output_csv_path : str
        Path to save the final gene programs CSV.

    Returns:
    --------
    final_gene_programs : dict
        Dictionary with final selected gene programs for each sample and K.
    nmf_results : dict
        Dictionary containing all NMF run details (models, W, H, top_genes, errors).
    """
    sample_ids = adata.obs[sample_column].unique()
    nmf_results = {}

    print("Starting NMF pipeline...")

    # Run NMF for each sample
    for sample_id in sample_ids:
        print(f"Processing sample: {sample_id}")
        sample_adata = adata[adata.obs[sample_column] == sample_id].copy()
        
        # Filter genes by prevalence within the sample
        sc.pp.filter_genes(sample_adata, min_cells=min_gene_prevalence * sample_adata.n_obs)


        X_sample = sample_adata.X.toarray() if scipy.sparse.issparse(sample_adata.X) else sample_adata.X
        nmf_results[sample_id] = {}

        for k_val in range(min_k, max_k + 1):
            print(f"  Running NMF with K={k_val} ({n_nmf_runs} runs)")
            model, W, H, error = run_nmf_with_restarts(X_sample, k_val, n_nmf_runs, max_iter=max_iter)
            current_top_genes = get_top_genes(H, sample_adata.var_names, top_n_genes)
            
            nmf_results[sample_id][k_val] = {
                'model': model,
                'W': W,
                'H': H,
                'top_genes': current_top_genes,
                'reconstruction_error': error,
                'gene_names_used': sample_adata.var_names.tolist() # Store gene names used for this NMF
            }
    print("NMF runs completed for all samples and K values.")

    # --- Filtering Steps ---
    # Calculate overlaps within the same sample, different K
    same_sample_overlaps = {}
    for sample_id in nmf_results:
        same_sample_overlaps[sample_id] = {}
        sample_k_values = sorted(nmf_results[sample_id].keys())
        for i in range(len(sample_k_values)):
            for j in range(i + 1, len(sample_k_values)):
                k1, k2 = sample_k_values[i], sample_k_values[j]
                programs_k1 = nmf_results[sample_id][k1]['top_genes']
                programs_k2 = nmf_results[sample_id][k2]['top_genes']
                
                overlaps = {}
                for p1_idx, genes1 in programs_k1.items():
                    for p2_idx, genes2 in programs_k2.items():
                        overlaps[(p1_idx, p2_idx)] = jaccard_similarity(genes1, genes2)
                same_sample_overlaps[sample_id][(k1, k2)] = overlaps
    print("Calculated within-sample overlaps.")

    # Step 1: Keep factors with high overlap within the same sample, different K
    step1_passed_programs = {}
    for sample_id in nmf_results:
        step1_passed_programs[sample_id] = {}
        for k_val in nmf_results[sample_id]:
            step1_passed_programs[sample_id][k_val] = []
            num_components = nmf_results[sample_id][k_val]['H'].shape[0]
            for comp_idx in range(num_components):
                has_high_overlap = False
                for other_k in nmf_results[sample_id]:
                    if other_k == k_val:
                        continue
                    
                    # Determine overlap dict key
                    key_k1, key_k2 = min(k_val, other_k), max(k_val, other_k)
                    if (key_k1, key_k2) not in same_sample_overlaps[sample_id]:
                        continue # Should not happen if calculated correctly
                    
                    overlap_dict = same_sample_overlaps[sample_id][(key_k1, key_k2)]
                    
                    max_o = 0
                    if k_val < other_k: # comp_idx is from k1
                        max_o = max([v for (p1,p2), v in overlap_dict.items() if p1 == comp_idx], default=0)
                    else: # comp_idx is from k2 (other_k < k_val)
                        max_o = max([v for (p1,p2), v in overlap_dict.items() if p2 == comp_idx], default=0)
                        
                    if max_o >= overlap_threshold_step1:
                        has_high_overlap = True
                        break
                if has_high_overlap:
                    step1_passed_programs[sample_id][k_val].append(comp_idx)
    print(f"Step 1 filtering completed (threshold: {overlap_threshold_step1}).")

    # Calculate cross-sample overlaps for programs that passed Step 1
    cross_sample_overlaps = {}
    processed_sample_ids = list(nmf_results.keys()) # Use only samples that had valid matrices
    for i in range(len(processed_sample_ids)):
        for j in range(i + 1, len(processed_sample_ids)):
            s_i, s_j = processed_sample_ids[i], processed_sample_ids[j]
            pair_key = (s_i, s_j)
            cross_sample_overlaps[pair_key] = {}
            
            for k_i in nmf_results[s_i]:
                for k_j in nmf_results[s_j]:
                    programs_i = {idx: genes for idx, genes in nmf_results[s_i][k_i]['top_genes'].items() 
                                  if idx in step1_passed_programs.get(s_i, {}).get(k_i, [])}
                    programs_j = {idx: genes for idx, genes in nmf_results[s_j][k_j]['top_genes'].items()
                                  if idx in step1_passed_programs.get(s_j, {}).get(k_j, [])}
                    
                    if not programs_i or not programs_j: continue

                    overlaps = {}
                    for idx_i, genes_i in programs_i.items():
                        for idx_j, genes_j in programs_j.items():
                            overlaps[(k_i, idx_i, k_j, idx_j)] = jaccard_similarity(genes_i, genes_j)
                    if overlaps:
                        cross_sample_overlaps[pair_key][(k_i, k_j)] = overlaps
    print("Calculated cross-sample overlaps.")

    # Step 2: Keep programs with overlap with another sample
    step2_passed_programs = {}
    for sample_id in processed_sample_ids:
        step2_passed_programs[sample_id] = {}
        for k_val in nmf_results[sample_id]:
            step2_passed_programs[sample_id][k_val] = []
            for comp_idx in step1_passed_programs.get(sample_id, {}).get(k_val, []):
                has_cross_overlap = False
                for other_sample in processed_sample_ids:
                    if other_sample == sample_id: continue
                    
                    s_1, s_2 = min(sample_id, other_sample), max(sample_id, other_sample)
                    pair_key = (s_1, s_2)
                    if pair_key not in cross_sample_overlaps: continue

                    for (k1_cs, k2_cs), overlap_dict_cs in cross_sample_overlaps[pair_key].items():
                        max_o_cs = 0
                        if sample_id == s_1: # current sample_id is the first in pair_key
                            if k1_cs != k_val: continue # Ensure we are looking at current K for sample_id
                            max_o_cs = max([v for (cs_k_i, cs_idx_i, cs_k_j, cs_idx_j), v in overlap_dict_cs.items()
                                            if cs_k_i == k_val and cs_idx_i == comp_idx], default=0)
                        else: # current sample_id is the second in pair_key (s_2)
                            if k2_cs != k_val: continue # Ensure we are looking at current K for sample_id
                            max_o_cs = max([v for (cs_k_i, cs_idx_i, cs_k_j, cs_idx_j), v in overlap_dict_cs.items()
                                            if cs_k_j == k_val and cs_idx_j == comp_idx], default=0)
                        
                        if max_o_cs >= overlap_threshold_step2:
                            has_cross_overlap = True
                            break
                    if has_cross_overlap: break
                if has_cross_overlap:
                    step2_passed_programs[sample_id][k_val].append(comp_idx)
    print(f"Step 2 filtering completed (threshold: {overlap_threshold_step2}).")
    
    # Step 3: Define founder programs (select program with highest cross-sample overlap for each K)
    founder_programs = {}
    for sample_id in processed_sample_ids:
        founder_programs[sample_id] = {}
        for k_val in nmf_results[sample_id]:
            if not step2_passed_programs.get(sample_id, {}).get(k_val, []):
                continue

            max_overlaps_for_k = {} # comp_idx -> max_cross_sample_overlap
            for comp_idx in step2_passed_programs[sample_id][k_val]:
                current_max_o = 0
                for other_sample in processed_sample_ids:
                    if other_sample == sample_id: continue
                    
                    s_1, s_2 = min(sample_id, other_sample), max(sample_id, other_sample)
                    pair_key = (s_1, s_2)
                    if pair_key not in cross_sample_overlaps: continue

                    for (k1_cs, k2_cs), overlap_dict_cs in cross_sample_overlaps[pair_key].items():
                        if sample_id == s_1 and k1_cs == k_val: # current sample_id is s_1
                            comp_o = max([v for (cs_k_i, cs_idx_i, cs_k_j, cs_idx_j), v in overlap_dict_cs.items()
                                          if cs_k_i == k_val and cs_idx_i == comp_idx], default=0)
                            current_max_o = max(current_max_o, comp_o)
                        elif sample_id == s_2 and k2_cs == k_val: # current sample_id is s_2
                            comp_o = max([v for (cs_k_i, cs_idx_i, cs_k_j, cs_idx_j), v in overlap_dict_cs.items()
                                          if cs_k_j == k_val and cs_idx_j == comp_idx], default=0)
                            current_max_o = max(current_max_o, comp_o)
                max_overlaps_for_k[comp_idx] = current_max_o
            
            if max_overlaps_for_k:
                best_program_idx = max(max_overlaps_for_k, key=max_overlaps_for_k.get)
                founder_programs[sample_id][k_val] = best_program_idx
    print("Step 3: Founder program identification completed.")

    # Step 4: Remove redundant programs within the same sample
    final_programs = {}
    for sample_id in processed_sample_ids:
        final_programs[sample_id] = {}
        # Sort K values to process consistently, perhaps not strictly necessary here
        # but good for deterministic behavior if order matters subtly later.
        sorted_k_for_sample = sorted(founder_programs.get(sample_id, {}).keys())

        temp_selected_for_sample = {} # Store initially selected programs for this sample
        for k_val in sorted_k_for_sample:
            if k_val in founder_programs.get(sample_id, {}):
                 temp_selected_for_sample[k_val] = founder_programs[sample_id][k_val]
        
        # Now filter based on redundancy from this temp_selected_for_sample set
        for k_val_outer, comp_idx_outer in temp_selected_for_sample.items():
            should_remove = False
            for k_val_inner, comp_idx_inner in temp_selected_for_sample.items():
                if k_val_outer == k_val_inner:
                    continue

                # Get overlap
                key_k1, key_k2 = min(k_val_outer, k_val_inner), max(k_val_outer, k_val_inner)
                overlap = 0
                if (key_k1, key_k2) in same_sample_overlaps.get(sample_id, {}):
                    overlap_dict = same_sample_overlaps[sample_id][(key_k1, key_k2)]
                    # Adjust comp_idx order based on k order for lookup
                    if k_val_outer < k_val_inner:
                        overlap = overlap_dict.get((comp_idx_outer, comp_idx_inner), 0)
                    else:
                        overlap = overlap_dict.get((comp_idx_inner, comp_idx_outer), 0)
                
                if redundancy_overlap_min <= overlap < redundancy_overlap_max:
                    should_remove = True
                    break 
            
            if not should_remove:
                final_programs[sample_id][k_val_outer] = comp_idx_outer
    print(f"Step 4: Redundancy filtering completed (range: {redundancy_overlap_min}-{redundancy_overlap_max}).")

    # Extract final gene programs
    final_gene_programs_list = []
    for sample_id in final_programs:
        for k_val, comp_idx in final_programs[sample_id].items():
            genes = nmf_results[sample_id][k_val]['top_genes'][comp_idx]
            for gene in genes:
                final_gene_programs_list.append({
                    'sample_id': sample_id,
                    'k': k_val,
                    'program_id': comp_idx,
                    'gene': gene
                })
    
    programs_df = pd.DataFrame(final_gene_programs_list)
    if not programs_df.empty:
        programs_df.to_csv(output_csv_path, index=False)
        print(f"Saved final gene programs to {output_csv_path}")
    else:
        print("No final programs selected after filtering. CSV not saved.")

    return final_programs, nmf_results
