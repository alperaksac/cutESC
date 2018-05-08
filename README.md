# cutESC
CutESC: Cut-Edge for Spatial Clustering based on Proximity Graph

The CutESC algorithm removes edges when a dynamically calculated cut-edge value for the edge's endpoints is below a threshold. The dynamic cut-edge value is calculated by using statistical features and spatial distribution of data based on its neighborhood. Also, the algorithm works without any prior information and preliminary parameter settings while automatically discovering clusters with non-uniform densities, arbitrary shapes, and outliers. But there is an option which allows users to set two parameters to adapt clustering for particular problems easily.

<img src="results/compound.png" alt="compund" style="width: 400px;"/>
<img src="results/t8.8k.png" alt="t8.8k" style="width: 400px;"/>

## Dependencies

* Coded in Python 3.x.
* Using [Anaconda](https://www.continuum.io/downloads) is recommended.
* See [`requirements.txt`](requirements.txt) for a full list of requirements. Most of these are easy to get through pip, e.g.:
```bash
$ pip install -r requirements.txt
```

If you use this code, please cite the following [paper]():

```
@InProceedings{RubioICPR2016,
   author = {Alper Aksac and Tansel Ozyer and Reda Alhajj},
   title = {CutESC: Cut-Edge for Spatial Clustering based on Proximity Graph},
   year = 2018
}
```