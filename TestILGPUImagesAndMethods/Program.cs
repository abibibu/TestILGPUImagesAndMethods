using ILGPU.Runtime;
using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TestILGPUImagesAndMethods
{
    internal class Program
    {
        static void Main(string[] args)
        {
            //prova test a
            SampleStruct sp = new SampleStruct(0.5, 1000);
            double[] x = new double[100];
            for (int i = 0; i < x.Length; i++)
            {
                x[i] = i * 123.456;
            }
            double[] result = new double[x.Length];
            for (int i = 0; i < x.Length; i++)
            {
                result[i] = sp.Evaluate(x[i]);
            }




            var ctx = Context.CreateDefault();
            var acc = ctx.GetPreferredDevice(false).CreateAccelerator(ctx);
            SampleStructILGPU spGPU = new SampleStructILGPU(sp,acc);
            var kernel = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, SampleStructILGPU>(SampleStructILGPU.EvaluateKernel);
            Index1D idx = new Index1D(x.Length);
            var xg = acc.Allocate1D<double>(idx);
            var yg = acc.Allocate1D<double>(idx);
            xg.CopyFromCPU(x);
            kernel(idx, xg, yg, spGPU);
            acc.Synchronize();
            var result1 = yg.GetAsArray1D();

            //this does not complie because of the arrays (I think...)
            //var kernel2 = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, SampleStruct>(SampleStructILGPU.EvaluateKernel2);
            //kernel2(idx, xg, yg, sp);
            //acc.Synchronize();
            //var result2 = yg.GetAsArray1D();

        }
    }
    struct SampleStruct
    {
        public double[] prm;
        public SampleStruct(double seed, int len)
        {
            prm = new double[len];
            for (int i = 0; i < len; i++)
                prm[i] = seed + i;
        }
        public double Evaluate(double x)
        {
            double y = 0;
            for (int i = 0; i < prm.Length; i++)
            {
                y += x * prm[i];
            }
            return y;
        }
    }

    struct SampleStructILGPU
    {
        public ArrayView1D<double, Stride1D.Dense> prm;
        public SampleStructILGPU(SampleStruct smp, Accelerator dv)
        {
            prm = dv.Allocate1D<double>(smp.prm.Length);
            prm.CopyFromCPU(smp.prm);
        }
        //code duplication! how to avoid this?
        public double Evaluate(double x)
        {
            double y = 0;
            for (int i = 0; i < prm.Length; i++)
            {
                y += x * prm[i];
            }
            return y;
        }
        static public void EvaluateKernel(Index1D idx, ArrayView1D<double, Stride1D.Dense> x, ArrayView1D<double, Stride1D.Dense> y, SampleStructILGPU smp)
        {
            y[idx] = smp.Evaluate(x[idx]);
        }
        static public void EvaluateKernel2(Index1D idx, ArrayView1D<double, Stride1D.Dense> x, ArrayView1D<double, Stride1D.Dense> y, SampleStruct smp)
        {
            y[idx] = smp.Evaluate(x[idx]);
        }

    }
}
