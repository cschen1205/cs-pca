using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MathNet.Numerics.LinearAlgebra.Generic;
using MathNet.Numerics.LinearAlgebra.Double;

namespace PCA
{
    public class PCADimReducer
    {
        /// <summary>
        /// Compress M input data points from original dimension N to new dimension K
        /// </summary>
        /// <param name="Xinput">M input data points, each having dimension N</param>
        /// <param name="K">The new dimension of the data</param>
        /// <param name="U_reduce">The MxK matrix, where M is the set of the data set (i.e. M = Xinput.Count), K is the new feature dimension count </param>
        /// <param name="variance_retained"></param>
        /// <returns></returns>
        public List<double[]> CompressData(List<double[]> Xinput, int K, out Matrix<double> U_reduce, out double variance_retained)
        {
            int dimension = Xinput[0].Length;
            int m = Xinput.Count;

            int dimension2=dimension-1;
            Matrix<double> X = new DenseMatrix(m, dimension2, 0);
            for (int i = 0; i < m; ++i)
            {
                double[] rec = Xinput[i];
                for (int d = 1; d < dimension; ++d)
                {
                    X[i, d - 1] = rec[d];
                }
            }
            Matrix<double> X_transpose = X.Transpose();
            Matrix<double> Sigma = X_transpose.Multiply(X).Multiply(1.0 / m);

            var svd=Sigma.Svd(true);
            Matrix<double> U=svd.U();
            Vector<double> S = svd.S();

            U_reduce = new DenseMatrix(m, K);
            for (int i = 0; i < m; ++i)
            {
                for (int d = 0; d < K; ++d)
                {
                    U_reduce[i, d] = U[i, d];
                }
            }


            double Skk=0;
            double Smm=S.Sum();
            for(int i=0; i < K; ++i)
            {
                Skk+=S[i];
            }
            variance_retained=Skk/Smm;

            List<double[]> Zoutput = new List<double[]>();
            for (int i = 0; i < m; ++i)
            {
                double[] rec_x = Xinput[i];
                double[] rec_z = new double[K+1];
                
                for (int d = 0; d < K; ++d)
                {
                    double z=0;
                    for(int d2=0; d2 < dimension2; ++d2)
                    {
                        z+=U_reduce[d2, d] * rec_x[d2+1]; //index must start at 1
                    }
                    rec_z[d + 1] = z;
                }
                Zoutput.Add(rec_z);
            }
            return Zoutput;
        }

        /// <summary>
        /// Reconstruct the K-dimension input Zinput to its original N-dimension form using the compressed matrix U_reduce (obtained from CompressData method)
        /// </summary>
        /// <param name="Zinput">The K-dimension input data point</param>
        /// <param name="U_reduce">The M x K matrix obtained from CompressData method</param>
        /// <returns>The reconstructed N-dimension output data point</returns>
        public List<double[]> ReconstructData(List<double[]> Zinput, Matrix<double> U_reduce)
        {
            int m=Zinput.Count;
            int K=Zinput[0].Length-1;
            Matrix<double> Z = new DenseMatrix(m, K);
            for (int i = 0; i < m; ++i)
            {
                double[] rec_z= Zinput[i];
                for(int d=0; d < K; ++d)
                {
                    Z[i, d] = rec_z[d + 1];
                }
            }

            Matrix<double> X = Z.Multiply(U_reduce.Transpose());

            int N=X.ColumnCount;
            List<double[]> Xoutput = new List<double[]>();
            for (int i = 0; i < m; ++i)
            {
                double[] rec_x = new double[N+1];
                for (int d = 0; d < N; ++d)
                {
                    rec_x[d+1] = X[i, d]; //index must start at 1!
                }
                Xoutput.Add(rec_x);
            }

            return Xoutput;
        }

        public List<double[]> CompressData(List<double[]> Xinput, out Matrix<double> U_reduce, out int K, double variance_retained_threshold)
        {
            int dimension = Xinput[0].Length;
            int m = Xinput.Count;

            int N = dimension - 1;
            Matrix<double> X = new DenseMatrix(m, N, 0);
            for (int i = 0; i < m; ++i)
            {
                double[] rec = Xinput[i];
                for (int d = 1; d < dimension; ++d)
                {
                    X[i, d - 1] = rec[d];
                }
            }
            Matrix<double> X_transpose = X.Transpose();
            Matrix<double> Sigma = X_transpose.Multiply(X).Multiply(1.0 / m);

            var svd = Sigma.Svd(true);
            Matrix<double> U = svd.U();
            Vector<double> S = svd.S();

            
            double Skk = 0;
            double Smm = S.Sum();
            K = 0;
            for (int i = 0; i < m; ++i)
            {
                Skk += S[i];
                double variance_retained = Skk / Smm;
                if (variance_retained >= variance_retained_threshold)
                {
                    K = i;
                    break;
                }
            }

            U_reduce = new SparseMatrix(m, K);
            for (int i = 0; i < m; ++i)
            {
                for (int d = 0; d < K; ++d)
                {
                    U_reduce[i, d] = U[i, d];
                }
            }

            List<double[]> Zoutput = new List<double[]>();
            for (int i = 0; i < m; ++i)
            {
                double[] rec_x = Xinput[i];
                double[] rec_z = new double[K+1];

                for (int d = 0; d < K; ++d)
                {
                    double z = 0;
                    for (int d2 = 0; d2 < N; ++d2)
                    {
                        z += U_reduce[d2, d] * rec_x[d2 + 1]; 
                    }
                    rec_z[d + 1] = z;
                }
                Zoutput.Add(rec_z);
            }
            return Zoutput;
        }
    }
}
