import os
import sys
BASE = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), "..")
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "out")

from metaproteomics import utils
import pandas as pd
from itertools import chain
import shelve
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from collections import Counter
from collections import defaultdict

from metaproteomics.analysis import build_loci ,functional_analysis
from metaproteomics.analysis.DBInfo import DBInfo

f = functional_analysis.Functionizer()

db_info = DBInfo("compil_mgm")
metadata = build_loci.read_metadata(os.path.join(DATA, "metadata.csv"))

samples_keep = {'RTsep_unenr1','RTsep_unenr2','RTsep_unenr3','Ragsep_unenr1','Ragsep_unenr2','Ragsep_unenr3','WTsep_unenr1','WTsep_unenr2','WTsep_unenr3'}
metadata = metadata[metadata.columns[metadata.columns.isin(samples_keep)]]

#%% Parse samples
samples = shelve.open(os.path.join(OUT,"samples.shelve"))
for sample_name, sample_info in tqdm(list(metadata.iteritems())):
    sample = build_loci.Sample(sample_name, sample_info.path, db_info, sample_info)
    samples[sample.sample_name] = sample
#%%
protein_clusters = shelve.open(os.path.join(OUT,"protein_clusters.shelve"))
for name,sample in samples.items():
    protein_clusters[name] = sample.build_protein_clusters()

sample_pep_quant = {sample.sample_name:sample.pep_quant for sample in samples.values()}
grouped_loci = build_loci.group_across_samples(list(chain(*protein_clusters.values())), sample_pep_quant)
for locus in tqdm(grouped_loci):
    locus.annotate()
utils.save(grouped_loci, os.path.join(OUT,"grouped_loci.pkl.gz"), force=True)
#grouped_loci = utils.load(os.path.join(BASE,"grouped_loci.pkl.gz"))
#%% filtering
grouped_loci2 = [x for x in grouped_loci if x.passes_thresh(metadata, min_quant = 10, min_samples = 1, 
                                                            min_samples_per_group = 1, group = "sample type")]
grouped_loci3 = [x for x in grouped_loci2 if x.passes_thresh(metadata, min_quant = 2, min_samples = 2,
                                                             min_samples_per_group = 2, group = "sample type")]
utils.save(grouped_loci3, os.path.join(OUT,"grouped_loci_filt.pkl.gz"), force=True)


#%% Normalize
nf = build_loci.yates_normalization(samples)
for locus in grouped_loci3:
    locus.normalize(nf, field = "norm_quantification")
utils.save(grouped_loci3, os.path.join(OUT,"grouped_loci_filt_norm.pkl.gz"), force=True)

loci = grouped_loci3
#%% write out
df = build_loci.to_df(loci, norm=False)
df = df.iloc[df.index.isin(samples_keep)]
df.T.to_csv(os.path.join(OUT,"df.csv"))
df = build_loci.to_df(loci, norm=True)
df = df.iloc[df.index.isin(samples_keep)]
df.T.to_csv(os.path.join(OUT,"df_norm.csv"))

#%% mark as human/mouse and other
from metaproteomics.analysis import taxonomy
t = taxonomy.Taxonomy(host="wl-cmadmin.scripps.edu")
# human or mouse taxids and ancestors (up to the phylum chordata (7711)):
chordata = set(t.taxid_to_taxonomy(7711)['lineage'])
human = set(t.taxid_to_taxonomy(9606)['lineage']) - chordata
mouse = set(t.taxid_to_taxonomy(10090)['lineage']) - chordata
human_mouse = human | mouse
for locus in loci:
    locus.human_mouse = True if locus.lca in human_mouse else False
    
#%% Build a locus -> metadata table

import re
for locus in loci:
    names = [x['d'] for x in locus.prot_info]
    gn=set(chain(*[re.findall("GN=([^ ]*)",name) for name in names]))
    locus.gn = ','.join(gn) if gn else ''
    locus.gn1 = list(gn)[0] if len(gn)==1 else ''
locus_df = pd.DataFrame({locus.cluster_id: {'name': locus.name, 'human_mouse': locus.human_mouse, 'lca':locus.lca, 'gn':locus.gn, 'gn1':locus.gn1} for locus in loci}).T
locus_df.to_csv(os.path.join(OUT,"locus_df.csv"))
