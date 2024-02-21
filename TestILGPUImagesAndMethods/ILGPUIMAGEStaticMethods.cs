using ILGPU.Runtime;
using ILGPU;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TestILGPUImagesAndMethods
{
    static public class ILGPUIMAGEStaticMethods
    {
        #region floatMethos
        public static void ReadRoiKernel(Index2D index, ArrayView2D<float, Stride2D.DenseY> image, ArrayView2D<float, Stride2D.DenseY> outputRoi, PointF roiOrigin)
        {
            
            outputRoi[index] = At(image, index.X + roiOrigin.X, index.Y + roiOrigin.Y);
        }
        public static float At(ArrayView2D<float, Stride2D.DenseY> image, float x, float y)
        {
            int ix = (int)x; int iy = (int)y;
            float dx = x - ix; float dy = y - iy;

            return
                (ILGPUImageGenericStaticMethods<float>.At(image, ix, iy) * (1.0f - dx) + ILGPUImageGenericStaticMethods<float>.At(image, ix + 1, iy) * dx) * (1.0f - dy) +
                (ILGPUImageGenericStaticMethods<float>.At(image, ix, iy + 1) * (1.0f - dx) + ILGPUImageGenericStaticMethods<float>.At(image, ix + 1, iy + 1) * dx) * dy
                ;
        }
        #endregion
        #region byteMethos
        public static void ReadRoiKernel(Index2D index, ArrayView2D<byte, Stride2D.DenseY> image, ArrayView2D<byte, Stride2D.DenseY> outputRoi, PointF roiOrigin)
        {
            outputRoi[index] = At(image, index.X + roiOrigin.X, index.Y + roiOrigin.Y);
        }
        public static byte At(ArrayView2D<byte, Stride2D.DenseY> image, float x, float y)
        {
            int ix = (int)x; int iy = (int)y;
            float dx = x - ix; float dy = y - iy;

            return (byte)(
                (ILGPUImageGenericStaticMethods<byte>.At(image, ix, iy) * (1.0f - dx) + ILGPUImageGenericStaticMethods<byte>.At(image, ix + 1, iy) * dx) * (1.0f - dy) +
                (ILGPUImageGenericStaticMethods<byte>.At(image, ix, iy + 1) * (1.0f - dx) + ILGPUImageGenericStaticMethods<byte>.At(image, ix + 1, iy + 1) * dx) * dy
                );
        }
        /// <summary>
        /// simply histogram assumes that the image has values between 0 and nBin-1
        /// </summary>
        /// <param name="index"></param>
        /// <param name="image"></param>
        /// <param name="histogram"></param>
        /// <param name="nBin"></param>
        /// <param name="totLen">roiw*roih somma dell'istogramma per la rinormalizzazione, metteere 1 per istogramma non normalizzato</param>
        public static void HistogramSimplyAtRoi(Index2D index, ArrayView2D<byte, Stride2D.DenseY> image, ArrayView1D<int, Stride1D.Dense> histogram, byte nBin, PointF roiOrigin)
        {
            byte tmpVal = At(image, index.X + roiOrigin.X, index.Y + roiOrigin.Y);
            if (tmpVal < 0) tmpVal = 0;
            if (tmpVal >= nBin) tmpVal = (byte)(nBin - 1);
            Atomic.Add(ref histogram[tmpVal], 1);
            
        }
        /// <summary>
        /// simply histogram assumes that the image has values between 0 and nBin-1
        /// </summary>
        /// <param name="index"></param>
        /// <param name="image"></param>
        /// <param name="histogram"></param>
        /// <param name="nBin"></param>
        /// <param name="totLen">roiw*roih somma dell'istogramma per la rinormalizzazione, metteere 1 per istogramma non normalizzato</param>
        public static void HistogramSimplyAtRoi(Index2D index, ArrayView2D<byte, Stride2D.DenseY> image, ArrayView1D<int, Stride1D.Dense> histogram, byte nBin, Point roiOrigin)
        {
            byte tmpVal = At(image, index.X + roiOrigin.X, index.Y + roiOrigin.Y);
            //histogram[tmpVal] += 1; 
            Atomic.Add(ref histogram[tmpVal], 1);
        }
        #endregion
    }
}
