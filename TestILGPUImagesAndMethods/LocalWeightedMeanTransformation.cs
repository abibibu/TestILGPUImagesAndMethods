using ILGPU;
using ILGPU.Runtime;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace TestILGPUImagesAndMethods
{
    public struct LocalWeightedMeanTransformation
    {
        //articolo: "Y:\05 Michele\Articoli\registrazione DE projection\Goshtasby - Image registration by local approximation methods.pdf"
        //al capitolo Local Weighted Mean
        //miglirabile utilizzando jagged array di punti per velocizzare la ricerca
        readonly public Vector2[] fix;//(x_i, y_i) nell'articolo
        //Vector2[] mov;//X_i nell'articolo
        readonly public PolynomialInterpolator2ndOrder[] polynomials;//P_i nell'articolo
        readonly public float[] lastNearestNeighbourDistance;//R_n nell'articolo
        /// <summary>
        /// lwm trasformation: calcola la trasformazione che manda i punti mov ai punti fix, ovvero fix = trasform(mov)
        /// articolo: "Y:\05 Michele\Articoli\registrazione DE projection\Goshtasby - Image registration by local approximation methods.pdf"
        /// al capitolo Local Weighted Mean
        /// </summary>
        /// <param name="fix_xy">punti prima della trasformazione NB: i la distribuzioni di punti deve essere abbastanza equispaziata, il numero minimo di punti é 12</param>
        /// <param name="mov_uv">punti dopo la trasformazione </param>
        public LocalWeightedMeanTransformation(Vector2[] fix_xy, Vector2[] mov_uv, int numberOfClosestPoints = 12)
        {
            if (fix_xy == null || mov_uv == null) throw new ArgumentNullException();
            if (fix_xy.Length != mov_uv.Length) throw new Exception("fix and mov lenghts must be equal");
            if (fix_xy.Length < numberOfClosestPoints) throw new Exception("points lenghts must be at least equal to selected numberOfClosestPoints");
            this.fix = fix_xy;
            //this.mov = mov_uv;
            int pointsLenght = fix_xy.Length;

            //creo i polinomi
            //per ogni punto cerco i  numberOfClosestPoints nearest neighbour
            int[] idx = new int[pointsLenght];
            double[] radiuses = new double[pointsLenght];
            polynomials = new PolynomialInterpolator2ndOrder[pointsLenght];
            lastNearestNeighbourDistance = new float[pointsLenght];
            for (int i = 0; i < pointsLenght; i++)
            {
                for (int j = 0; j < pointsLenght; j++)
                {
                    //inizializzo l'array di indici
                    idx[j] = j;
                    //calcolo i raggi
                    radiuses[j] = Vector2.DistanceSquared(fix_xy[j], fix_xy[i]);
                }
                //sorting in base alla distanza portandomi dietro gli indici
                Array.Sort(radiuses, idx);
                var selectedfix_xy = new Vector2[numberOfClosestPoints];
                var selectedmov_uv = new Vector2[numberOfClosestPoints];
                for (int j = 0; j < numberOfClosestPoints; j++)
                {
                    selectedfix_xy[j] = fix_xy[idx[j]];
                    selectedmov_uv[j] = mov_uv[idx[j]];
                }

                polynomials[i] = new PolynomialInterpolator2ndOrder(selectedfix_xy, selectedmov_uv);
                lastNearestNeighbourDistance[i] = (float)Math.Sqrt(radiuses[numberOfClosestPoints - 1]);
            }
        }
        public LocalWeightedMeanTransformation(PointF[] fix_xy, PointF[] mov_uv, int numberOfClosestPoints = 12) :
            this(fix_xy.Select(t => new Vector2(t.X, t.Y)).ToArray(), mov_uv.Select(t => new Vector2(t.X, t.Y)).ToArray(), numberOfClosestPoints)
        {

        }


        public Vector2 TransformInverse(Vector2 pFixXy)
        {
            float x = 0, y = 0, WeightsSum = 0;
            //per il momento é forza bruta, si puó ottimizzare in qualche modo credo...
            for (int i = fix.Length - 1; i >= 0; i--)
            {
                float Dx = (pFixXy.X - fix[i].X);
                float Dy = (pFixXy.Y - fix[i].Y);
                float R = (float)Math.Sqrt(Dx * Dx + Dy * Dy) / lastNearestNeighbourDistance[i];
                float W = R <= 1 ? 1 - 3 * R * R + 2 * R * R * R : 0;
                if (W > 0)
                {
                    Vector2 polyVal = polynomials[i].Evaluate(pFixXy);
                    x += W * polyVal.X;
                    y += W * polyVal.Y;
                    WeightsSum += W;
                }
            }
            //questo se capita crea delle discontinuitá, bisogna assicurarsi che tutti i punti siano nel dominio ed altrimenti inventarseli
            if (WeightsSum == 0)
            {
                //x = pFixXy.X;
                //y = pFixXy.Y;
                //throw new Exception("Point too far from control points");
                return pFixXy;

            }
            x /= WeightsSum;
            y /= WeightsSum;

            return new Vector2((float)(x), (float)(y));
        }
        static public void TestMatlab()
        {
            //stesso test di matlab
            Vector2[] fix = new Vector2[]
            {
                new Vector2(10,8),
                new Vector2(12,2),
                new Vector2(17,6),
                new Vector2(14,10),
                new Vector2(7,20),
                new Vector2(10,4)
            };
            double[] a = new double[] { 1, 2, 3, 4, 5, 6 };
            double[] b = new double[] { 2.3, 3, 4, 5, 6, 7.5 };

            Vector2[] mov = fix.Select((x) => new Vector2(
                (float)(a[0] + a[1] * x.X + a[2] * x.Y + a[3] * x.X * x.Y + a[4] * x.X * x.X + a[5] * x.Y * x.Y),
                (float)(b[0] + b[1] * x.X + b[2] * x.Y + b[3] * x.X * x.Y + b[4] * x.X * x.X + b[5] * x.Y * x.Y)
                )).ToArray();

            var tform = new LocalWeightedMeanTransformation(fix, mov);

            Vector2[] movComputed = fix.Select(x => tform.TransformInverse(x)).ToArray();

        }
        static public void TesImageRototraslation()
        {
            //M_Generic_classLib.LogClass.init();
            //array di punti equispaziati a cui applico una rototraslazione
            //int step = 128;
            int step = 128;

            int xmin = 0;
            int xmax = 512;
            int ymin = 0;
            int ymax = 512;
            List<Vector2> fix = new List<Vector2>();
            for (int y = ymin; y <= ymax + step; y += step)
            {
                for (int x = xmin; x <= xmax + step; x += step)
                {
                    fix.Add(new Vector2(x, y));
                }
            }

            Matrix3x2 M = Matrix3x2.Multiply(Matrix3x2.CreateRotation(0.3f), Matrix3x2.CreateTranslation(20, -13));

            Vector2[] mov = fix.Select((x) => Vector2.Transform(x, M)).ToArray();

            var tform = new LocalWeightedMeanTransformation(fix.ToArray(), mov);

            Vector2[] movComputed = fix.Select(x => tform.TransformInverse(x)).ToArray();

            double error = mov.Zip(movComputed, (f, m) => Vector2.Distance(f, m)).Sum() / (double)mov.Length;

            //creo un secondo array con le coordinate di un immagine

            Vector2[] fix2 = new Vector2[(xmax - xmin) * (ymax - ymin)];
            Stopwatch st = new Stopwatch();
            st.Start();
            Parallel.For(ymin, ymax, y =>
            //for (int y = ymin; y < ymax; y++)
            {
                for (int x = xmin; x < xmax; x++)
                {
                    fix2[(xmax - xmin) * y + x] = tform.TransformInverse(new Vector2(x, y));
                }
            }
            );
            st.Stop();
            Console.WriteLine(st.Elapsed);//196 punti di controllo per un immagine da 512*521 20ms
            double error2 = 0;

            for (int y = ymin; y < ymax; y++)
            {
                for (int x = xmin; x < xmax; x++)
                {
                    error2 += Vector2.Distance(fix2[(xmax - xmin) * y + x], Vector2.Transform(new Vector2(x, y), M));
                }
            }
            error2 /= (double)((xmax - xmin) * (ymax - ymin));
            Console.WriteLine("N: " + fix.Count() + " Error: " + error2);
            //M_Generic_classLib.LogClass.Info("N: " + fix.Count() + " Error: " + error2 + " Time: " + st.Elapsed);

        }



    }

    public struct PolynomialInterpolator2ndOrder
    {
        private readonly QuadraticParam paramX;
        private readonly QuadraticParam paramY;

        internal PolynomialInterpolator2ndOrder(Vector2[] fix_xy, Vector2[] mov_uv)
        {
            var X = DenseMatrix.Create(fix_xy.Length, 6, 0);
            var u = DenseVector.Create(fix_xy.Length, 0);
            var v = DenseVector.Create(fix_xy.Length, 0);
            Vector2 tmp;
            for (int i = fix_xy.Length - 1; i >= 0; i--)
            {
                tmp = fix_xy[i];
                X[i, 0] = 1; X[i, 1] = tmp.X; X[i, 2] = tmp.Y; X[i, 3] = tmp.X * tmp.Y; X[i, 4] = tmp.X * tmp.X; X[i, 5] = tmp.Y * tmp.Y;
                u[i] = mov_uv[i].X;
                v[i] = mov_uv[i].Y;
            }
            var t = X.QR();

            var paramX = t.Solve(u).ToArray();
            var paramY = t.Solve(v).ToArray();
            bool undefinitedPoly = false;
            foreach (var h in paramX)
            {
                if (!h.IsFinite())
                {
                    undefinitedPoly = true;
                }
            }
            //se i parametri sono indefiniti creo un polinomio che lascia la x dov'era prima
            if (undefinitedPoly) { paramX = new double[] { 0, 1, 0, 0, 0, 0 }; }
            foreach (var h in paramY)
            {
                if (!h.IsFinite())
                {
                    undefinitedPoly = true;
                }
            }

            //se i parametri sono indefiniti creo un polinomio che lascia la y dov'era prima
            if (undefinitedPoly) { paramX = new double[] { 0, 0, 1, 0, 0, 0 }; }
            this.paramX = new QuadraticParam(paramX);
            this.paramY = new QuadraticParam(paramY);

            //test
            //var mov_uv_calc = new Vector2[mov_uv.Length];
            //for (int i = fix_xy.Length - 1; i >= 0; i--)
            //{
            //    mov_uv_calc[i] = Evaluate(fix_xy[i]);
            //}
        }
        public Vector2 Evaluate(Vector2 x)
        {
            float xnew = paramX.a + paramX.b * x.X + paramX.c * x.Y + paramX.d * x.X * x.Y + paramX.e * x.X * x.X + paramX.f * x.Y * x.Y;
            float ynew = paramY.a + paramY.b * x.X + paramY.c * x.Y + paramY.d * x.X * x.Y + paramY.e * x.X * x.X + paramY.f * x.Y * x.Y;
            return new Vector2(xnew, ynew);
        }

        struct QuadraticParam
        {
            public float a, b, c, d, e, f;
            public QuadraticParam(double[] param)
            {
                a = (float)param[0];
                b = (float)param[1];
                c = (float)param[2];
                d = (float)param[3];
                e = (float)param[4];
                f = (float)param[5];
            }
        }
    }
    public struct LocalWeightedMeanTransformationILGPU
    {
        ArrayView1D<Vector2, Stride1D.Dense> fix;
        ArrayView1D<PolynomialInterpolator2ndOrder, Stride1D.Dense> polynomials;
        ArrayView1D<float, Stride1D.Dense> lastNearestNeighbourDistance;
        Index1D idx;
        public LocalWeightedMeanTransformationILGPU(LocalWeightedMeanTransformation lw, Accelerator acc)
        {
            idx = new Index1D(lw.fix.Length);
            fix = acc.Allocate1D<Vector2>(lw.fix);
            polynomials = acc.Allocate1D<PolynomialInterpolator2ndOrder>(lw.polynomials);
            lastNearestNeighbourDistance = acc.Allocate1D<float>(lw.lastNearestNeighbourDistance);
        }
        public Vector2 TransformInverse(PointF pFixXy)
        {
            return TransformInverse(new Vector2(pFixXy.X, pFixXy.Y));
        }

        public Vector2 TransformInverse(Vector2 pFixXy)
        {
            float x = 0, y = 0, WeightsSum = 0;
            float Dx, Dy, R, W;
            Vector2 polyVal;
            //per il momento é forza bruta, si puó ottimizzare in qualche modo credo...
            for (int i = (int)fix.Length - 1; i >= 0; i--)
            {
                Dx = (pFixXy.X - fix[i].X);
                Dy = (pFixXy.Y - fix[i].Y);
                R = (float)(Math.Sqrt(Dx * Dx + Dy * Dy)) / lastNearestNeighbourDistance[i];
                W = R <= 1 ? 1 - 3 * R * R + 2 * R * R * R : 0;
                if (W > 0)
                {
                    polyVal = polynomials[i].Evaluate(pFixXy);
                    x += W * polyVal.X;
                    y += W * polyVal.Y;
                    WeightsSum += W;
                }
            }
            //questo se capita crea delle discontinuitá, bisogna assicurarsi che tutti i punti siano nel dominio ed altrimenti inventarseli
            if (WeightsSum == 0)
            {
                //x = pFixXy.X;
                //y = pFixXy.Y;
                //throw new Exception("Point too far from control points");
                return pFixXy;

            }
            x /= WeightsSum;
            y /= WeightsSum;

            return new Vector2(x,y);
        }
    }
}
