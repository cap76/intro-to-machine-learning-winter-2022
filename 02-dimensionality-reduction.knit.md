# Dimensionality reduction {#dimensionality-reduction}

In machine learning, dimensionality reduction refers broadly to any modelling approach that reduces the number of variables in a dataset to a few highly informative or representative ones. This is necessitated by the fact that large datasets with many variables are inherently difficult for humans to develop a clear intuition for. Dimensionality reduction is therefore an integral step in the analysis of large, complex (biological) datasets, allowing exploratory analyses and more intuitive visualisation that may aid interpretability, as well as forming a chain in the link of more complex analyses.

<div class="figure" style="text-align: center">
<img src="images/swiss_roll_manifold_sculpting.png" alt="Example of a dimensionality reduction. Here we have a two-dimensional dataset embeded in a three-dimensional space (swiss roll dataset)." width="55%" />
<p class="caption">(\#fig:dimreduc)Example of a dimensionality reduction. Here we have a two-dimensional dataset embeded in a three-dimensional space (swiss roll dataset).</p>
</div>

In biological applications, systems-level measurements are typically used to decipher complex mechanisms. These include measurements of gene expression from collections of microarrays or RNA-sequencing experiments that provide quantitative measurments for tens-of-thousands of genes. Studies like these, based on bulk measurements (that is pooled material), provide observations for many variables (in this case many genes) but with relatively few samples e.g., few time points or conditions. The imbalance between the number of variables and the number of observations is referred to as large *p*, small *n*, and makes statistical analysis difficult. Dimensionality reduction techniques therefore prove to be a useful first step in any analysis, identifying potential structure that exists in the dataset or highlighting which (combinations of) variables are the most informative.

The increasing prevalence of single cell RNA-sequencing (scRNA-seq) means the scale of datasets has shifted away from large *p*, small *n*, towards providing measurements of many variables but with a corresponding large number of observations (large *n*) albeit from potentially heterogeneous populations. scRNA-sequencing was largely driven by the need to investigate the transcrptomes of cells that were limited in quantity, such as embryonic cells, with early applications in mouse blastomeres. As of 2017, scRNA-seq experiments routinely generate datasets with tens to hundreds-of-thousands of cells. Indeed, in 2016, the [10x Genomics million cell experiment](https://community.10xgenomics.com/t5/10x-Blog/Our-1-3-million-single-cell-dataset-is-ready-to-download/ba-p/276) provided sequencing for over 1.3 million cells taken from the cortex, hippocampus and ventricular zone of embryonic mice, and large international consortiums, such as the [Human Cell Atlas](https://www.humancellatlas.org) aim to create a comprehensive maps of all cell types in the human body. A key goal when dealing with datasets of this magnitude is the identification of subpopulations of cells that may have gone undetected in bulk experiments; another, perhaps more ambitious task, aims to take advantage of any heterogeneity within the population in order to identify a temporal or mechanistic progression of developmental processes or disease.

Of course, whilst dimensionality reduction allows humans to inspect the dataset manually, particularly when the data can be represented in two or three dimensions, we should keep in mind that humans are exceptionally good at identifying patterns in two or three dimensional data, even when no real structure exists. It is therefore useful to employ other statistical approaches to search for patterns in the reduced dimensional space. In this sense, dimensionality reduction forms an integral component in the analysis of complex datasets that will typically be combined a variety of machine learning techniques, such as classification, regression, and clustering.

<div class="figure" style="text-align: center">
<img src="images/GB1.jpg" alt="Humans are exceptionally good at identifying patterns in two and three-dimensional spaces - sometimes too good. To illustrate this, note the Great Britain shapped cloud in the image (presumably drifting away from an EU shaped cloud, not shown). More whimsical shaped clouds can also be seen if you have a spare afternoon.  Golcar Matt/Weatherwatchers [BBC News](http://www.bbc.co.uk/news/uk-england-leeds-40287817)" width="35%" />
<p class="caption">(\#fig:humanpattern)Humans are exceptionally good at identifying patterns in two and three-dimensional spaces - sometimes too good. To illustrate this, note the Great Britain shapped cloud in the image (presumably drifting away from an EU shaped cloud, not shown). More whimsical shaped clouds can also be seen if you have a spare afternoon.  Golcar Matt/Weatherwatchers [BBC News](http://www.bbc.co.uk/news/uk-england-leeds-40287817)</p>
</div>

In this chapter we will explore two forms of dimensionality reduction: principle component analysis ([PCA](#linear-dimensionality-reduction)) and t-distributed stochastic neighbour embedding ([tSNE](#nonlinear-dimensionality-reduction)), highlighting the advantages and potential pitfalls of each method. As an illustrative example, we will use these approaches to analyse single cell RNA-sequencing data of early human development. Finally, we will illustrate the use of dimensionality redution on an image dataset.

## Linear Dimensionality Reduction {#linear-dimensionality-reduction}

The most widely used form of dimensionality reduction is principle component analysis (PCA), which was introduced by Pearson in the early 1900's, and independently rediscovered by Hotelling. PCA has a long history of use in biological and ecological applications, with early use in population studies, and later for the analysis of gene expression data.

PCA is not a dimensionality reduction technique *per se*, but an alternative way of representing the data that more naturally captures the variance in the system. Specifically, it finds a new co-ordinate system, so that the new "x-axis" (which is called the first principle component; PC1) is aligned along the direction of greatest variance, with an orthogonal "y-axis" aligned along the direction with second greatest variance (the second principle component; PC2), and so forth. At this stage there has been no inherent reduction in the dimensionality of the system, we have simply rotated the data around.

To illustrate PCA we will use a dataset from GEO (GSE5325). This dataset contains gene expression profiles for $105$ breast tumour samples measured using Swegene Human 27K RAP UniGene188 arrays. Within the population of cells, the original analysis focused on the expression of *GATA3* and *XBP1*, whose expression was known to correlate with estrogen receptor status [^](Breast cancer cells may be estrogen receptor positive, ER$^+$, or negative, ER$^-$, indicating capacity to respond to estrogen signalling, which has impliations for treatment), representing a two dimensional system. A pre-processed dataset containing the expression levels for *GATA3* and *XBP1*, and ER status, can be loaded into R using the code, below:



















































