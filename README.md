# cs-pca

Principal Component Analysis implemented in C#

# Install

```bash
Install-Package cs-pca
```

# Usage

The sample codes below shows how to use the library to reduce the number of dimension or reconstruct the original data from the reduced data:

```cs 
List<double[]> source = GetNormalizedData();
List<double[]> Z; // PCA output 
double variance_retained;
K = 5; // dimension of the Z (note that Z will have K+1 dimensions where the first dimension will be ignored)
PCA.PCADimReducer.CompressData(source, K, out Z, out variance_retained);

// To reconstruct some compressed data point from Z 
List<double[]> compressed_data_point = GetCompressedDataPoints(); // K+1 dimension data points 
List<double[]> uncompressed_data_point = ReconstructData(compressed_data_point, Z);
```