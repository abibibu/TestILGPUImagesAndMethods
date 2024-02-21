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
    public static class ILGPUImageGenericStaticMethods<T> where T : unmanaged
    {
        public static void ReadRoiKernel(Index2D index, ArrayView2D<T, Stride2D.DenseY> image, ArrayView2D<T, Stride2D.DenseY> outputRoi, Point roiOrigin)
        {
            outputRoi[index] = At(image, index.X + roiOrigin.X, index.Y + roiOrigin.Y);
        }
        public static T At(ArrayView2D<T, Stride2D.DenseY> image, int x, int y)
        {
            int w = image.IntExtent.X;
            int h = image.IntExtent.Y;
            if (x < 0) x = 0; if (x >= w) x = w - 1;
            if (y < 0) y = 0; if (y >= h) y = h - 1;
            return image[x, y];
        }
    }
}
