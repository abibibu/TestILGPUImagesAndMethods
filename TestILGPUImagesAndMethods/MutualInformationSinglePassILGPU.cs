using ILGPU.Runtime;
using ILGPU;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;
using System.Numerics;
using System.Diagnostics;
using System.Drawing.Printing;
using Emgu.CV.Structure;
using static System.Net.Mime.MediaTypeNames;
using System.Dynamic;

namespace TestILGPUImagesAndMethods
{
    public class MutualInformationSinglePassILGPU
    {
        internal Accelerator accelerator;
        internal MemoryBuffer2D<float, Stride2D.DenseY> image;
        internal MemoryBuffer2D<float, Stride2D.DenseY> imageTransformed;
        internal Index2D imageIndex;
        internal LocalWeightedMeanTransformationReader LWMTReader;
        internal Size imSize;
        internal KeyPointsILGPU_struct findKeyPointsILGPU;
        internal KeyPointsILGPU_noStruct findKeyPointsILGPUNoStruct;
        internal byte[] imageArr, imageTransformedArr;
        bool useStruct;
        public static Accelerator MakeAccellerator()
        {
            // Initialize ILGPU.
            //Context context = Context.Create(builder => builder.AllAccelerators().
            //                                            EnableAlgorithms().Math(MathMode.Fast32BitOnly).
            //                                            Inlining(InliningMode.Aggressive).
            //                                            AutoAssertions().
            //                                            Optimize(OptimizationLevel.O1));
            Context context = Context.Create(builder => builder.AllAccelerators().EnableAlgorithms()
                                                        );

            return context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);
        }
        public MutualInformationSinglePassILGPU(Size imSize, Accelerator accelerator, bool useStruct)
        {
            this.useStruct = useStruct;
            this.imSize = imSize;
            this.accelerator = accelerator;
            imageIndex = new Index2D(imSize.Width, imSize.Height);
            image = accelerator.Allocate2DDenseY<float>(imageIndex);
            imageTransformed = accelerator.Allocate2DDenseY<float>(imageIndex);
            imageArr = new byte[image.LengthInBytes];
            imageTransformedArr = new byte[imageTransformed.LengthInBytes];
            accelerator.Synchronize();
            //I know this is super ugly but is only for understanding purpose...
            if(useStruct)
                findKeyPointsILGPU = new KeyPointsILGPU_struct(imSize, accelerator);
            else
                findKeyPointsILGPUNoStruct = new KeyPointsILGPU_noStruct(imSize, accelerator);
        }
        public Image<Gray, float> RegisterStraight(Mat fix, Mat mov)
        {
            CreateTransformation(fix, mov);
            return ApplyTransformation(mov);
        }
        public void CreateTransformation(Mat fix, Mat mov)
        {
            PointF[] pFix = new PointF[1];
            PointF[] pMov = new PointF[1];
            if (useStruct)
            {
                findKeyPointsILGPU.LoadImages(fix, mov, accelerator);
                ( pFix,  pMov) = findKeyPointsILGPU.FindKeyPoints();
            }
            else
            {
                findKeyPointsILGPUNoStruct.LoadImages(fix, mov, accelerator);
                ( pFix,  pMov) = findKeyPointsILGPUNoStruct.FindKeyPoints();
            }
            LWMTReader = new LocalWeightedMeanTransformationReader(pFix, pMov, accelerator);

        }
        public Image<Gray, float> ApplyTransformation(Mat image)
        {
            //lento, un sacco di allocazioni in ram...
            if (LWMTReader == null)
                return new Image<Gray, float>(10, 10);
            this.image.CopyFromCPU((float[,])(image.GetData(true)));
            LWMTReader.ApplyInverse(imageIndex, this.image.View, imageTransformed.View);
            accelerator.Synchronize();
            var bytes = imageTransformed.GetRawData().Array;
            var calc = imageTransformed.GetAsArray2D<float>();
            var outimg = new Image<Gray, float>(imSize);
            outimg.Bytes = bytes;
            return outimg;

        }
        public void ApplyTransformation(Mat image, ref Mat imout)
        {
            //molto veloce yeye
            if (LWMTReader == null)
                return;
            image.CopyTo<byte>(imageArr);
            this.image.AsRawArrayView().CopyFromCPU(imageArr);
            LWMTReader.ApplyInverse(imageIndex, this.image.View, imageTransformed.View);
            accelerator.Synchronize();
            imageTransformed.AsRawArrayView().CopyToCPU(imageTransformedArr);
            imout.SetTo<byte>(imageTransformedArr);
        }
    }
    public class KeyPointsILGPU_struct
    {

        public struct KeyPointsCalculator
        {
            internal ArrayView1D<PointF, Stride1D.Dense> keyPointFixGpu;
            internal ArrayView1D<PointF, Stride1D.Dense> keyPointMovGpu;

            internal ArrayView1D<float, Stride1D.Dense> RoiFixedEntropy;
            internal ArrayView1D<float, Stride1D.Dense> RoiMutualInformation;
            internal ArrayView2D<float, Stride2D.DenseY> MutualInformation;
            /// <summary>
            /// one hisotgram with nHistoBin for each roi
            /// </summary>
            internal ArrayView2D<short, Stride2D.DenseX> roiHistogramsFix;
            internal ArrayView2D<short, Stride2D.DenseX> roiHistogramsMov;
            /// <summary>
            /// one joint hisotgram with nHistoBin*nHistoBin for each roi
            /// </summary>
            internal ArrayView2D<short, Stride2D.DenseX> roiJointHistograms;
            /// <summary>
            /// one hisotgram with nHistoBin for each test point of the image
            /// </summary>
            internal ArrayView2D<short, Stride2D.DenseX> histograms;
            /// <summary>
            /// one joint hisotgram with nHistoBin*nHistoBin for each test point of the image
            /// </summary>
            internal ArrayView2D<short, Stride2D.DenseX> jointHistograms;

            Size imageSize, roiSize;
            public int nBin, nRoiPerWidth, nRoiPerHeight, margin, totalNumberOfRois, nStepForRoiSide, nStepPerPixel, maxDistanceFromCenter;
            float stepWidth;
            public const int nHistoBin = 64;
            public KeyPointsCalculator(Size size, Accelerator acc, int nBin = 1, int nRoiPerWidth = 8, int margin = 6)
            {


                imageSize = size;
                this.nBin = nBin;
                this.nRoiPerWidth = nRoiPerWidth;
                this.margin = margin;

                //l'idea é quella di creare un memory buffer 2d:
                //dato il numero di roi per lato: nRoiX*nRoiY = nTotRoi
                //per ogni roi creo un quadrato di lato 2*maxDist*4 con un valore di mutual information per ogni quarto di pixel attorno al centro della roi e scrivo tutta sta roba in un immagine
                //che a questo punto avré dimensioni (nRoiX*8*maxDist, nRoiy*8*maxDist)
                //il punto di minimo per ogni roi sará il minimo nel quadrato con le MI della roi

                //1-rebin and create set of rois
                //voglio ottenere 4x4 roi di lato circa 60 con margine di circa 6 pixel
                int binw = imageSize.Width / nBin;
                int binh = imageSize.Height / nBin;
                int roiSide = (binw - margin / nBin) / nRoiPerWidth;

                //le roi devono essere equispaziate per cui in base all'altezza e ai margini calcolo il numero delle roi in Y
                //2*margin + nRoiPerHeight*roiSide = fixedImage.Height
                nRoiPerHeight = (binh - nBin / margin) / roiSide;
                totalNumberOfRois = nRoiPerWidth * nRoiPerHeight;

                //creo gli array dove scriveró gli istogrammi
                roiHistogramsFix = acc.Allocate2DDenseX<short>(new Index2D(nHistoBin, totalNumberOfRois));
                roiHistogramsMov = acc.Allocate2DDenseX<short>(new Index2D(nHistoBin, totalNumberOfRois));
                roiJointHistograms = acc.Allocate2DDenseX<short>(new Index2D(nHistoBin * nHistoBin, totalNumberOfRois));

                //qui va un controllo sulle size se non ho esattamente 512
                maxDistanceFromCenter = 7;
                nStepPerPixel = 1;
                stepWidth = 1.0f / nStepPerPixel;
                //il numero di passi per lato di ogni roi: voglio fare step da 1/4 di pixel quindi 4*maxDist*2 (l'ultimo 2 é che voglio maxDist sia a destra che a sinistra
                nStepForRoiSide = nStepPerPixel * maxDistanceFromCenter * 2;
                //creo il memory buffer con le mutual information come spiegato sopra
                MutualInformation = acc.Allocate2DDenseY<float>(new Index2D(nRoiPerWidth * nStepForRoiSide, nRoiPerHeight * nStepForRoiSide));
                //creo il memeory buffer per gli istogrammi
                histograms = acc.Allocate2DDenseX<short>(new Index2D(nHistoBin, totalNumberOfRois * nStepForRoiSide * nStepForRoiSide));
                jointHistograms = acc.Allocate2DDenseX<short>(new Index2D(nHistoBin * nHistoBin, totalNumberOfRois * nStepForRoiSide * nStepForRoiSide));//1.6 giga bytes, che sia troppo?


                //inizializzo le coppie di punti
                var keyPointFix = new PointF[totalNumberOfRois];
                RoiFixedEntropy = acc.Allocate1D<float>(totalNumberOfRois);
                RoiMutualInformation = acc.Allocate1D<float>(totalNumberOfRois);
                roiSize = new Size(roiSide, roiSide);

                for (int i = 0; i < totalNumberOfRois; i++)
                {
                    int xpos = i % nRoiPerWidth;
                    int ypos = i / nRoiPerWidth;
                    PointF pfix = new PointF(margin + xpos * roiSide + roiSide * 0.5f, margin + ypos * roiSide + roiSide * 0.5f);
                    //converto in PointF
                    keyPointFix[i] = new PointF(pfix.X, pfix.Y);
                }
                keyPointFixGpu = acc.Allocate1D<PointF>(keyPointFix);
                keyPointMovGpu = acc.Allocate1D<PointF>(keyPointFix.Length);

            }

            public static void KernelValidateRoi(Index1D idx, ArrayView2D<byte, Stride2D.DenseY> fix, ArrayView2D<byte, Stride2D.DenseY> mov, KeyPointsCalculator keyPointsCalculator)
            {
                keyPointsCalculator.ValidateRoi(idx, fix, mov);
            }
            public static void KernelCalcMI(Index2D idx, int idxRoi, ArrayView2D<byte, Stride2D.DenseY> fix, ArrayView2D<byte, Stride2D.DenseY> mov, KeyPointsCalculator keyPointsCalculator)
            {
                keyPointsCalculator.CalcMutualInformation(idx, idxRoi, fix, mov);
            }

            public static void KernelFindPointOfMaxMI(Index1D idx, KeyPointsCalculator keyPointsCalculator)
            {
                keyPointsCalculator.FindPointOfMaxMI(idx);
            }
            void ValidateRoi(Index1D idx, ArrayView2D<byte, Stride2D.DenseY> fix, ArrayView2D<byte, Stride2D.DenseY> mov)
            {
                RoiFixedEntropy[idx] = CalcHentropyAtRoiF(fix, roiHistogramsFix, idx, keyPointFixGpu[idx], roiSize);
                RoiMutualInformation[idx] = RoiFixedEntropy[idx] + CalcHentropyAtRoiF(mov, roiHistogramsMov, idx, keyPointFixGpu[idx], roiSize)
                    - CalcJointHentropytRoiF(fix, mov, roiJointHistograms, idx, keyPointFixGpu[idx], keyPointFixGpu[idx], roiSize);
            }

            void CalcMutualInformation(Index2D idx, int idxRoi, ArrayView2D<byte, Stride2D.DenseY> fix, ArrayView2D<byte, Stride2D.DenseY> mov)
            {
                //trovo il punto nell'immagine fix mov
                PointF centerFix = keyPointFixGpu[idxRoi];
                PointF centerMov = new PointF(centerFix.X + idx.X * stepWidth - maxDistanceFromCenter, centerFix.Y + idx.Y * stepWidth - maxDistanceFromCenter);
                //trovo l'indice della tabellona
                int tabIndexX = idx.X + (idxRoi % nRoiPerWidth) * nStepForRoiSide;
                int tabIndexY = idx.Y + (idxRoi / nRoiPerWidth) * nStepForRoiSide;

                int histogramTableIndex = idx.X + idx.Y * nStepForRoiSide + idxRoi * nStepForRoiSide * nStepForRoiSide;
                //int histogramTableIndex = tabIndexX + tabIndexY* nRoiPerWidth* nStepForRoiSide;

                float mi = RoiFixedEntropy[idxRoi] +
                    CalcHentropyAtRoiF(mov, histograms, histogramTableIndex, centerMov, roiSize) +
                    -CalcJointHentropytRoiF(fix, mov, jointHistograms, histogramTableIndex, centerFix, centerMov, roiSize);

                MutualInformation[tabIndexX, tabIndexY] = mi;
            }

            void FindPointOfMaxMI(Index1D idx)
            {
                //qua metto il valore del punto centrale cosí se sono tutte uguali mi ritorna il punto al centro, anche se non dovrebbe succedere

                int tabIndexX = nStepForRoiSide / 2 + (idx % nRoiPerWidth) * nStepForRoiSide;
                int tabIndexY = nStepForRoiSide / 2 + (idx / nRoiPerWidth) * nStepForRoiSide;
                float tempMax = MutualInformation[tabIndexX, tabIndexY];
                float val;
                PointF centerFix = keyPointFixGpu[idx];
                PointF pMax = centerFix;

                for (int j = nStepForRoiSide - 1; j >= 0; j--)
                    for (int i = nStepForRoiSide - 1; i >= 0; i--)
                    {
                        //trovo l'indice della tabellona
                        tabIndexX = i + (idx % nRoiPerWidth) * nStepForRoiSide;
                        tabIndexY = j + (idx / nRoiPerWidth) * nStepForRoiSide;
                        val = MutualInformation[tabIndexX, tabIndexY];
                        if (val > tempMax)
                        {
                            tempMax = val;
                            pMax = new PointF(centerFix.X + i * stepWidth - maxDistanceFromCenter, centerFix.Y + j * stepWidth - maxDistanceFromCenter);
                        }
                    }
                keyPointMovGpu[idx] = pMax;
            }

            /// <summary>
            /// hentropy -=p*Math.Log(p);
            /// Histogram will have bin width = 1, min = 0, max = nBin-1
            /// Single thread calculation, histogram memory allocation inside the function
            /// </summary>
            /// <param name="image"></param>
            /// <param name="roiCenter"></param>
            /// <param name="roiSize"></param>
            /// <param name="nBin"></param>
            static float CalcHentropyAtRoiF(ArrayView2D<byte, Stride2D.DenseY> image, ArrayView2D<short, Stride2D.DenseX> hist, int histLine, PointF roiCenter, Size roiSize)
            {

                int binIndex = 0;
                float hentropy = 0;
                float p = 0;
                float oneOverSum = 1.0f / (roiSize.Width * roiSize.Height);
                float halfRoiSize = 0.5f * roiSize.Width;
                for (int y = 0; y < roiSize.Height; y++)
                {
                    for (int x = 0; x < roiSize.Width; x++)
                    {
                        binIndex = At(image, roiCenter.X + x - halfRoiSize, roiCenter.Y + y - halfRoiSize);
                        if (binIndex < 0) binIndex = 0;
                        if (binIndex >= nHistoBin) binIndex = nHistoBin - 1;
                        hist[binIndex, histLine]++;
                    }
                }

                short binVal = 0;
                for (int i = 0; i < nHistoBin; i++)
                {
                    binVal = hist[i, histLine];
                    if (binVal > 0)
                    {
                        p = binVal * oneOverSum;
                        hentropy += -p * ILGPU.Algorithms.XMath.Log(p);
                    }
                }

                return hentropy;
            }
            /// <summary>
            /// jointHentropy = entropia ma sul joint histogram delle due immagini
            /// Histogram will have bin width = 1, min = 0, max = nBin-1
            /// Single thread calculation, histogram memory allocation inside the function
            /// </summary>
            /// <param name="fix"></param>
            /// <param name="mov"></param>
            /// <param name="roiCenter"></param>
            /// <param name="roiSize"></param>
            /// <param name="nHistoBin"></param>
            /// <returns></returns>
            static float CalcJointHentropytRoiF(ArrayView2D<byte, Stride2D.DenseY> fix, ArrayView2D<byte, Stride2D.DenseY> mov, ArrayView2D<short, Stride2D.DenseX> hist, int histLine, PointF roiCenterFixImage, PointF roiCenterMovImage, Size roiSize)
            {

                int idxX = 0, idxY = 0;
                float hentropy = 0;
                float p = 0;
                float oneOverSum = 1.0f / (roiSize.Width * roiSize.Height);
                float halfRoiSize = 0.5f * roiSize.Width;

                for (int y = 0; y < roiSize.Height; y++)
                {
                    for (int x = 0; x < roiSize.Width; x++)
                    {
                        //questo non interpola perché la roi fix é ad indici interi, controllare che sia cosí
                        idxY = At(fix, roiCenterFixImage.X + x - halfRoiSize, roiCenterFixImage.Y + y - halfRoiSize);
                        //questo interpola perché la roi mobile é ad indici float
                        idxX = At(mov, roiCenterMovImage.X + x - halfRoiSize, roiCenterMovImage.Y + y - halfRoiSize);
                        if (idxY < 0) idxY = 0;
                        if (idxY >= nHistoBin) idxY = nHistoBin - 1;
                        if (idxX < 0) idxX = 0;
                        if (idxX >= nHistoBin) idxX = nHistoBin - 1;
                        hist[idxY * nHistoBin + idxX, histLine]++;
                    }
                }
                short binVal = 0;
                for (int i = 0; i < nHistoBin * nHistoBin; i++)
                {
                    binVal = hist[i, histLine];
                    if (binVal > 0)
                    {
                        p = binVal * oneOverSum;
                        hentropy += -p * ILGPU.Algorithms.XMath.Log(p);
                    }
                }
                return hentropy;
            }

        }

        internal MemoryBuffer2D<byte, Stride2D.DenseY> fixedImageByte;
        internal MemoryBuffer2D<byte, Stride2D.DenseY> movingImageByte;
        byte[] fixedImageArr, movingImageArr;
        Accelerator accelerator;
        /// <summary>
        /// asd
        /// </summary>
        KeyPointsCalculator keyPointsCalculator;
        Action<Index1D, ArrayView2D<byte, Stride2D.DenseY>, ArrayView2D<byte, Stride2D.DenseY>, KeyPointsCalculator> KernelValidateRoi;

        /// <summary>
        /// KernelCalcMI(Index2D idx, int idxRoi, ArrayView2D<byte, Stride2D.DenseY> fix, ArrayView2D<byte, Stride2D.DenseY> mov, KeyPointsCalculator keyPointsCalculator)
        /// </summary>
        Action<Index2D, int, ArrayView2D<byte, Stride2D.DenseY>, ArrayView2D<byte, Stride2D.DenseY>, KeyPointsCalculator> KernelCalcMutualInformation;

        //public static void KernelFindPointOfMaxMI(Index1D idx, KeyPointsCalculator keyPointsCalculator)
        Action<Index1D, KeyPointsCalculator> KernelFindPointOfMaxMI;

        public KeyPointsILGPU_struct(Size size, Accelerator acc, int nBin = 1, int nRoiPerWidth = 10, int margin = 6)
        {
            Index2D imageIdx = new Index2D(size.Width, size.Height);
            fixedImageByte = acc.Allocate2DDenseY<byte>(imageIdx);
            movingImageByte = acc.Allocate2DDenseY<byte>(imageIdx);
            fixedImageArr = new byte[fixedImageByte.LengthInBytes];
            movingImageArr = new byte[movingImageByte.LengthInBytes];

            keyPointsCalculator = new KeyPointsCalculator(size, acc, nBin, nRoiPerWidth, margin);
            acc.Synchronize();

            KernelValidateRoi = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<byte, Stride2D.DenseY>, ArrayView2D<byte, Stride2D.DenseY>, KeyPointsCalculator>(KeyPointsCalculator.KernelValidateRoi);
            acc.Synchronize();
            KernelCalcMutualInformation = acc.LoadAutoGroupedStreamKernel<Index2D, int, ArrayView2D<byte, Stride2D.DenseY>, ArrayView2D<byte, Stride2D.DenseY>, KeyPointsCalculator>(KeyPointsCalculator.KernelCalcMI);
            acc.Synchronize();
            KernelFindPointOfMaxMI = acc.LoadAutoGroupedStreamKernel<Index1D, KeyPointsCalculator>(KeyPointsCalculator.KernelFindPointOfMaxMI);
            acc.Synchronize();
            accelerator = acc;
        }
        public void LoadImages(Mat fixedImage, Mat movingImage, Accelerator acc)
        {
            //normalizzo le immagini a 64 livelli di grigio, credo che questo sia il modo con meno memoria sprecata
            Mat fix = new Mat();
            Mat mov = new Mat();
            fixedImage.MinMax(out var min, out var max, out var pmin, out var pmax);
            ((fixedImage - min[0]) / (max[0] - min[0]) * 63.0).ConvertTo(fix, Emgu.CV.CvEnum.DepthType.Cv8U);

            movingImage.MinMax(out min, out max, out pmin, out pmax);
            ((movingImage - min[0]) / (max[0] - min[0]) * 63.0).ConvertTo(mov, Emgu.CV.CvEnum.DepthType.Cv8U);
            fix.CopyTo<byte>(fixedImageArr);
            mov.CopyTo<byte>(movingImageArr);
            fixedImageByte.AsRawArrayView().CopyFromCPU(fixedImageArr);
            movingImageByte.AsRawArrayView().CopyFromCPU(movingImageArr);

            accelerator.Synchronize();



#if GPU_DEBUG
            ////test read delle immagini
            //var flt = fixedImageByte.AsContiguous().GetRawData().Array;
            //File.WriteAllBytes("C:\\test\\fixBytes" + fixedImageByte.IntExtent.X + "_" + fixedImageByte.IntExtent.Y + ".raw", flt);
            //flt = movingImageByte.AsContiguous().GetRawData().Array;
            //File.WriteAllBytes("C:\\test\\movBytes" + fixedImageByte.IntExtent.X + "_" + fixedImageByte.IntExtent.Y + ".raw", flt);
#endif
        }
        public (PointF[] pFix, PointF[] pMov) FindKeyPoints()
        {
            //metto a zero gli array degli istogrammi
            keyPointsCalculator.RoiFixedEntropy.AsContiguous().MemSetToZero();
            keyPointsCalculator.RoiMutualInformation.AsContiguous().MemSetToZero();
            keyPointsCalculator.MutualInformation.AsContiguous().MemSetToZero();
            keyPointsCalculator.roiHistogramsFix.AsContiguous().MemSetToZero();
            keyPointsCalculator.roiHistogramsMov.AsContiguous().MemSetToZero();
            keyPointsCalculator.roiJointHistograms.AsContiguous().MemSetToZero();
            keyPointsCalculator.histograms.AsContiguous().MemSetToZero();
            keyPointsCalculator.jointHistograms.AsContiguous().MemSetToZero();
            accelerator.Synchronize();

            //i punti dell'immagine fissa li ho dall'inizio, mi basta scaricarli dalla gpu
            var keyPointFix = keyPointsCalculator.keyPointFixGpu.GetAsArray1D();
            //inizializzo il vettore di punti dell'immagine mov
            var keyPointMov = new PointF[keyPointFix.Length];
            //ora devo trovare i key points usando la mutual information
            //creo un array di booleani dove scriveró quali roi sono valide
            int nRoi = keyPointsCalculator.totalNumberOfRois;
            bool[] validRoi = new bool[nRoi];
            Index1D nRoiIndex = new Index1D(nRoi);
            //KernelValidateRoi(Index1D idx, ArrayView2D<byte, Stride2D.DenseY> fix, ArrayView2D<byte, Stride2D.DenseY> mov, KeyPointsCalculator keyPointsCalculator)
            //lancio un kerne per ogni punto trovato e calcolo per quel punto l'entropia dell'immagine fissa e la mutual information 
            KernelValidateRoi(nRoiIndex, fixedImageByte, movingImageByte, keyPointsCalculator);
            accelerator.Synchronize();



            //leggo i due array, li creo nuovi per ora non sono tanto grandi..
            var fixedHentropy = keyPointsCalculator.RoiFixedEntropy.GetAsArray1D();
            var mutualInfomration = keyPointsCalculator.RoiMutualInformation.GetAsArray1D();
            for (int i = 0; i < nRoi; i++)
            {
                //per ora valuto solo l'entropia
                if (fixedHentropy[i] > 1)//la prima soglia é solo sull'entropia, valutare bene questo parametro insieme alla dimensione
                {
                    if (fixedHentropy[i] + mutualInfomration[i] > 1.8)
                        validRoi[i] = true;
                }
                //test: metto tutto a true
                //validRoi[i] = true;
            }
            //ora se la roi é valida lancio i kernel che andranno a scrivere la MI nella tabellona
            //per ogni roi vengono lanciati nStepForRoiSide*nStepForRoiSide kernel, ognuno dei quli calcolerá la mutual information con il passo specificato in keyPointsCalculator
            Index2D roiIndex = new Index2D(keyPointsCalculator.nStepForRoiSide, keyPointsCalculator.nStepForRoiSide);
            for (int i = 0; i < nRoi; i++)
            {//per ora valuto solo l'entropia

                if (validRoi[i])
                {
                    //KernelCalcMI(Index2D idx, int idxRoi, ArrayView2D<byte, Stride2D.DenseY> fix, ArrayView2D<byte, Stride2D.DenseY> mov, KeyPointsCalculator keyPointsCalculator)
                    KernelCalcMutualInformation(roiIndex, i, fixedImageByte, movingImageByte, keyPointsCalculator);
                }

            }
            accelerator.Synchronize();

            //bon ci siamo, a questo punto lancio i kernel che mi trovano il minimo nel rettangolino
            KernelFindPointOfMaxMI(nRoiIndex, keyPointsCalculator);
            accelerator.Synchronize();

            keyPointMov = keyPointsCalculator.keyPointMovGpu.GetAsArray1D();
            accelerator.Synchronize();

            for (int i = 0; i < nRoi; i++)
            {
                //ora nei punti non validi assegno il punto fix
                if (!validRoi[i])
                {
                    keyPointMov[i] = keyPointFix[i];
                }
            }

#if GPU_DEBUG

            var flt = keyPointsCalculator.roiHistogramsFix.AsContiguous().GetRawData().Array;
            var histRoi = keyPointsCalculator.roiHistogramsFix.GetAsArray2D<short>();
            accelerator.Synchronize();

            File.WriteAllBytes("C:\\test\\roiHistograms_short_" + keyPointsCalculator.roiHistogramsFix.IntExtent.X + "_" + keyPointsCalculator.roiHistogramsFix.IntExtent.Y + ".raw", flt);
            var joint = keyPointsCalculator.roiJointHistograms.GetAsArray2D<short>();
            accelerator.Synchronize();

            Image<Gray, short> jointHistoRoi = new Image<Gray, short>(keyPointsCalculator.nRoiPerWidth* KeyPointsCalculator.nHistoBin, keyPointsCalculator.nRoiPerHeight*KeyPointsCalculator.nHistoBin);
            int jointLen = KeyPointsCalculator.nHistoBin * KeyPointsCalculator.nHistoBin;
            for (int iRoi = 0; iRoi < keyPointsCalculator.totalNumberOfRois; iRoi++)
            {
                int roiX = iRoi % keyPointsCalculator.nRoiPerWidth;
                int roiY = iRoi / keyPointsCalculator.nRoiPerWidth;
                for (int j = 0; j < jointLen; j++)
                {
                    int x = roiX * KeyPointsCalculator.nHistoBin + j % KeyPointsCalculator.nHistoBin;
                    int y = roiY * KeyPointsCalculator.nHistoBin + j / KeyPointsCalculator.nHistoBin;
                    jointHistoRoi.Data[y, x, 0] = joint[j, iRoi];
                }
            }
            File.WriteAllBytes("C:\\test\\roiJointHistograms_short_table_" + jointHistoRoi.Width + "_" + jointHistoRoi.Height + ".raw", jointHistoRoi.Bytes);

            flt = keyPointsCalculator.roiJointHistograms.AsContiguous().GetRawData().Array; accelerator.Synchronize();

            File.WriteAllBytes("C:\\test\\roiJointHistograms_short_linear_" + keyPointsCalculator.roiJointHistograms.IntExtent.X + "_" + keyPointsCalculator.roiJointHistograms.IntExtent.Y + ".raw", flt);
            flt = keyPointsCalculator.MutualInformation.AsContiguous().GetRawData().Array; accelerator.Synchronize();

            File.WriteAllBytes("C:\\test\\MItable_float_" + keyPointsCalculator.MutualInformation.IntExtent.X + "_" + keyPointsCalculator.MutualInformation.IntExtent.Y + ".raw", flt);
            flt = keyPointsCalculator.histograms.AsContiguous().GetRawData().Array; accelerator.Synchronize();

            File.WriteAllBytes("C:\\test\\histograms_short_" + keyPointsCalculator.histograms.IntExtent.X + "_" + keyPointsCalculator.histograms.IntExtent.Y + ".raw", flt);

            //test interpolazione lineare immagine
            var kerRead = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<byte, Stride2D.DenseY>, ArrayView2D<byte, Stride2D.DenseY>, PointF>(ReadImageWithOffset);

            Index2D idxImage = new Index2D(fixedImageByte.IntExtent.X, fixedImageByte.IntExtent.Y);
            var gpuReadImage = accelerator.Allocate2DDenseY<byte>(idxImage);
            accelerator.Synchronize();
            kerRead(idxImage, fixedImageByte, gpuReadImage, new PointF(0.123f, 0.879f));
            accelerator.Synchronize();
            var readimageBytes1 = gpuReadImage.AsContiguous().GetRawData().Array;
            accelerator.Synchronize();
            accelerator.Synchronize();
            kerRead(idxImage, fixedImageByte, gpuReadImage, new PointF(0.123f, 0.879f));
            accelerator.Synchronize();
            var readimageBytes2 = gpuReadImage.AsContiguous().GetRawData().Array;
            accelerator.Synchronize();
            var readFixedImage = fixedImageByte.AsContiguous().GetRawData().Array;
            accelerator.Synchronize();

            File.WriteAllBytes("C:\\test\\readFiximageBytes1_" + idxImage.X + "_" + idxImage.Y + ".raw", readimageBytes1);
            File.WriteAllBytes("C:\\test\\readFiximageBytes2_" + idxImage.X + "_" + idxImage.Y + ".raw", readimageBytes2);
            File.WriteAllBytes("C:\\test\\readFiximageBytes0_" + idxImage.X + "_" + idxImage.Y + ".raw", readFixedImage);

#endif

            //woho dovremmo esserci
            return (keyPointFix, keyPointMov);
        }


        //bilinear interpolation

        public static void ReadImageWithOffset(Index2D i, ArrayView2D<byte, Stride2D.DenseY> imageIn, ArrayView2D<byte, Stride2D.DenseY> imageOut, PointF offset)
        {
            imageOut[i] = At(imageIn, offset.X + i.X, offset.Y + i.Y);
        }

        public static byte At(ArrayView2D<byte, Stride2D.DenseY> image, int x, int y)
        {
            int w = image.IntExtent.X;
            int h = image.IntExtent.Y;
            if (x < 0) x = 0; if (x >= w) x = w - 1;
            if (y < 0) y = 0; if (y >= h) y = h - 1;
            return image[x, y];
        }
        public static float At(ArrayView2D<float, Stride2D.DenseY> image, int x, int y)
        {
            int w = image.IntExtent.X;
            int h = image.IntExtent.Y;
            if (x < 0) x = 0; if (x >= w) x = w - 1;
            if (y < 0) y = 0; if (y >= h) y = h - 1;
            return image[x, y];
        }
        public static byte At(ArrayView2D<byte, Stride2D.DenseY> image, float x, float y)
        {
            int ix = (int)x; int iy = (int)y;
            float dx = x - ix; float dy = y - iy;

            return (byte)(
                (At(image, ix, iy) * (1.0f - dx) + At(image, ix + 1, iy) * dx) * (1.0f - dy) +
                (At(image, ix, iy + 1) * (1.0f - dx) + At(image, ix + 1, iy + 1) * dx) * dy
                );
        }
        public static float At(ArrayView2D<float, Stride2D.DenseY> image, float x, float y)
        {
            int ix = (int)x; int iy = (int)y;
            float dx = x - ix; float dy = y - iy;

            return
                (At(image, ix, iy) * (1.0f - dx) + At(image, ix + 1, iy) * dx) * (1.0f - dy) +
                (At(image, ix, iy + 1) * (1.0f - dx) + At(image, ix + 1, iy + 1) * dx) * dy
                ;
        }

    }
    public class KeyPointsILGPU_noStruct
    {

        public struct KeyPointsCalculator
        {
            internal ArrayView1D<PointF, Stride1D.Dense> keyPointFixGpu;
            internal ArrayView1D<PointF, Stride1D.Dense> keyPointMovGpu;

            internal ArrayView1D<float, Stride1D.Dense> RoiFixedEntropy;
            internal ArrayView1D<float, Stride1D.Dense> RoiMutualInformation;
            internal ArrayView2D<float, Stride2D.DenseY> MutualInformation;
            /// <summary>
            /// one hisotgram with nHistoBin for each roi
            /// </summary>
            internal ArrayView2D<short, Stride2D.DenseX> roiHistogramsFix;
            internal ArrayView2D<short, Stride2D.DenseX> roiHistogramsMov;
            /// <summary>
            /// one joint hisotgram with nHistoBin*nHistoBin for each roi
            /// </summary>
            internal ArrayView2D<short, Stride2D.DenseX> roiJointHistograms;
            /// <summary>
            /// one hisotgram with nHistoBin for each test point of the image
            /// </summary>
            internal ArrayView2D<short, Stride2D.DenseX> histograms;
            /// <summary>
            /// one joint hisotgram with nHistoBin*nHistoBin for each test point of the image
            /// </summary>
            internal ArrayView2D<short, Stride2D.DenseX> jointHistograms;

            public Size imageSize, roiSize;
            public int nBin, nRoiPerWidth, nRoiPerHeight, margin, totalNumberOfRois, nStepForRoiSide, nStepPerPixel, maxDistanceFromCenter;
            public float stepWidth;
            public const int nHistoBin = 64;
            public KeyPointsCalculator(Size size, Accelerator acc, int nBin = 1, int nRoiPerWidth = 8, int margin = 6)
            {


                imageSize = size;
                this.nBin = nBin;
                this.nRoiPerWidth = nRoiPerWidth;
                this.margin = margin;

                //l'idea é quella di creare un memory buffer 2d:
                //dato il numero di roi per lato: nRoiX*nRoiY = nTotRoi
                //per ogni roi creo un quadrato di lato 2*maxDist*4 con un valore di mutual information per ogni quarto di pixel attorno al centro della roi e scrivo tutta sta roba in un immagine
                //che a questo punto avré dimensioni (nRoiX*8*maxDist, nRoiy*8*maxDist)
                //il punto di minimo per ogni roi sará il minimo nel quadrato con le MI della roi

                //1-rebin and create set of rois
                //voglio ottenere 4x4 roi di lato circa 60 con margine di circa 6 pixel
                int binw = imageSize.Width / nBin;
                int binh = imageSize.Height / nBin;
                int roiSide = (binw - margin / nBin) / nRoiPerWidth;

                //le roi devono essere equispaziate per cui in base all'altezza e ai margini calcolo il numero delle roi in Y
                //2*margin + nRoiPerHeight*roiSide = fixedImage.Height
                nRoiPerHeight = (binh - nBin / margin) / roiSide;
                totalNumberOfRois = nRoiPerWidth * nRoiPerHeight;

                //creo gli array dove scriveró gli istogrammi
                roiHistogramsFix = acc.Allocate2DDenseX<short>(new Index2D(nHistoBin, totalNumberOfRois));
                roiHistogramsMov = acc.Allocate2DDenseX<short>(new Index2D(nHistoBin, totalNumberOfRois));
                roiJointHistograms = acc.Allocate2DDenseX<short>(new Index2D(nHistoBin * nHistoBin, totalNumberOfRois));

                //qui va un controllo sulle size se non ho esattamente 512
                maxDistanceFromCenter = 7;
                nStepPerPixel = 1;
                stepWidth = 1.0f / nStepPerPixel;
                //il numero di passi per lato di ogni roi: voglio fare step da 1/4 di pixel quindi 4*maxDist*2 (l'ultimo 2 é che voglio maxDist sia a destra che a sinistra
                nStepForRoiSide = nStepPerPixel * maxDistanceFromCenter * 2;
                //creo il memory buffer con le mutual information come spiegato sopra
                MutualInformation = acc.Allocate2DDenseY<float>(new Index2D(nRoiPerWidth * nStepForRoiSide, nRoiPerHeight * nStepForRoiSide));
                //creo il memeory buffer per gli istogrammi
                histograms = acc.Allocate2DDenseX<short>(new Index2D(nHistoBin, totalNumberOfRois * nStepForRoiSide * nStepForRoiSide));
                jointHistograms = acc.Allocate2DDenseX<short>(new Index2D(nHistoBin * nHistoBin, totalNumberOfRois * nStepForRoiSide * nStepForRoiSide));//1.6 giga bytes, che sia troppo?


                //inizializzo le coppie di punti
                var keyPointFix = new PointF[totalNumberOfRois];
                RoiFixedEntropy = acc.Allocate1D<float>(totalNumberOfRois);
                RoiMutualInformation = acc.Allocate1D<float>(totalNumberOfRois);
                roiSize = new Size(roiSide, roiSide);

                for (int i = 0; i < totalNumberOfRois; i++)
                {
                    int xpos = i % nRoiPerWidth;
                    int ypos = i / nRoiPerWidth;
                    PointF pfix = new PointF(margin + xpos * roiSide + roiSide * 0.5f, margin + ypos * roiSide + roiSide * 0.5f);
                    //converto in PointF
                    keyPointFix[i] = new PointF(pfix.X, pfix.Y);
                }
                keyPointFixGpu = acc.Allocate1D<PointF>(keyPointFix);
                keyPointMovGpu = acc.Allocate1D<PointF>(keyPointFix.Length);

            }

            public static void KernelValidateRoi(
                Index1D idx,
                ArrayView2D<byte, Stride2D.DenseY> fix,
                ArrayView2D<byte, Stride2D.DenseY> mov,
                ArrayView1D<PointF, Stride1D.Dense> keyPointFixGpu,
                ArrayView1D<float, Stride1D.Dense> RoiFixedEntropy,
                ArrayView1D<float, Stride1D.Dense> RoiMutualInformation,
                ArrayView2D<short, Stride2D.DenseX> roiHistogramsFix,
                ArrayView2D<short, Stride2D.DenseX> roiHistogramsMov,
                ArrayView2D<short, Stride2D.DenseX> roiJointHistograms,
                Size roiSize)
            {
                RoiFixedEntropy[idx] = CalcHentropyAtRoiF(fix, roiHistogramsFix, idx, keyPointFixGpu[idx], roiSize);
                RoiMutualInformation[idx] =
                    RoiFixedEntropy[idx]
                    + CalcHentropyAtRoiF(mov, roiHistogramsMov, idx, keyPointFixGpu[idx], roiSize)
                    - CalcJointHentropytRoiF(fix, mov, roiJointHistograms, idx, keyPointFixGpu[idx], keyPointFixGpu[idx], roiSize);
            }
            public static void KernelCalcMI(
                Index2D idx,
                int idxRoi,
                ArrayView2D<byte, Stride2D.DenseY> fix,
                ArrayView2D<byte, Stride2D.DenseY> mov,
                ArrayView1D<PointF, Stride1D.Dense> keyPointFixGpu,
                ArrayView1D<float, Stride1D.Dense> RoiFixedEntropy,
                ArrayView2D<short, Stride2D.DenseX> histograms,
                ArrayView2D<short, Stride2D.DenseX> jointHistograms,
                ArrayView2D<float, Stride2D.DenseY> MutualInformation,
                float stepWidth,
                int maxDistanceFromCenter,
                int nRoiPerWidth,
                int nStepForRoiSide,
                Size roiSize
                )
            {
                //trovo il punto nell'immagine fix mov
                PointF centerFix = keyPointFixGpu[idxRoi];
                PointF centerMov = new PointF(centerFix.X + idx.X * stepWidth - maxDistanceFromCenter, centerFix.Y + idx.Y * stepWidth - maxDistanceFromCenter);
                //trovo l'indice della tabellona
                int tabIndexX = idx.X + (idxRoi % nRoiPerWidth) * nStepForRoiSide;
                int tabIndexY = idx.Y + (idxRoi / nRoiPerWidth) * nStepForRoiSide;

                int histogramTableIndex = idx.X + idx.Y * nStepForRoiSide + idxRoi * nStepForRoiSide * nStepForRoiSide;
                //int histogramTableIndex = tabIndexX + tabIndexY* nRoiPerWidth* nStepForRoiSide;

                float mi = RoiFixedEntropy[idxRoi] +
                    CalcHentropyAtRoiF(mov, histograms, histogramTableIndex, centerMov, roiSize) +
                    -CalcJointHentropytRoiF(fix, mov, jointHistograms, histogramTableIndex, centerFix, centerMov, roiSize);

                MutualInformation[tabIndexX, tabIndexY] = mi;
            }

            public static void KernelFindPointOfMaxMI(
                Index1D idx,
                ArrayView2D<float, Stride2D.DenseY> MutualInformation,
                ArrayView1D<PointF, Stride1D.Dense> keyPointFixGpu,
                ArrayView1D<PointF, Stride1D.Dense> keyPointMovGpu,
                int nStepForRoiSide,
                int nRoiPerWidth,
                float stepWidth,
                int maxDistanceFromCenter)
            {
                //qua metto il valore del punto centrale cosí se sono tutte uguali mi ritorna il punto al centro, anche se non dovrebbe succedere

                int tabIndexX = nStepForRoiSide / 2 + (idx % nRoiPerWidth) * nStepForRoiSide;
                int tabIndexY = nStepForRoiSide / 2 + (idx / nRoiPerWidth) * nStepForRoiSide;
                float tempMax = MutualInformation[tabIndexX, tabIndexY];
                float val;
                PointF centerFix = keyPointFixGpu[idx];
                PointF pMax = centerFix;

                for (int j = nStepForRoiSide - 1; j >= 0; j--)
                    for (int i = nStepForRoiSide - 1; i >= 0; i--)
                    {
                        //trovo l'indice della tabellona
                        tabIndexX = i + (idx % nRoiPerWidth) * nStepForRoiSide;
                        tabIndexY = j + (idx / nRoiPerWidth) * nStepForRoiSide;
                        val = MutualInformation[tabIndexX, tabIndexY];
                        if (val > tempMax)
                        {
                            tempMax = val;
                            pMax = new PointF(centerFix.X + i * stepWidth - maxDistanceFromCenter, centerFix.Y + j * stepWidth - maxDistanceFromCenter);
                        }
                    }
                keyPointMovGpu[idx] = pMax;
            }

            /// <summary>
            /// hentropy -=p*Math.Log(p);
            /// Histogram will have bin width = 1, min = 0, max = nBin-1
            /// Single thread calculation, histogram memory allocation inside the function
            /// </summary>
            /// <param name="image"></param>
            /// <param name="roiCenter"></param>
            /// <param name="roiSize"></param>
            /// <param name="nBin"></param>
            static float CalcHentropyAtRoiF(ArrayView2D<byte, Stride2D.DenseY> image, ArrayView2D<short, Stride2D.DenseX> hist, int histLine, PointF roiCenter, Size roiSize)
            {

                int binIndex = 0;
                float hentropy = 0;
                float p = 0;
                float oneOverSum = 1.0f / (roiSize.Width * roiSize.Height);
                float halfRoiSize = 0.5f * roiSize.Width;
                for (int y = 0; y < roiSize.Height; y++)
                {
                    for (int x = 0; x < roiSize.Width; x++)
                    {
                        binIndex = At(image, roiCenter.X + x - halfRoiSize, roiCenter.Y + y - halfRoiSize);
                        if (binIndex < 0) binIndex = 0;
                        if (binIndex >= nHistoBin) binIndex = nHistoBin - 1;
                        hist[binIndex, histLine]++;
                    }
                }

                short binVal = 0;
                for (int i = 0; i < nHistoBin; i++)
                {
                    binVal = hist[i, histLine];
                    if (binVal > 0)
                    {
                        p = binVal * oneOverSum;
                        hentropy += -p * ILGPU.Algorithms.XMath.Log(p);
                    }
                }

                return hentropy;
            }
            /// <summary>
            /// jointHentropy = entropia ma sul joint histogram delle due immagini
            /// Histogram will have bin width = 1, min = 0, max = nBin-1
            /// Single thread calculation, histogram memory allocation inside the function
            /// </summary>
            /// <param name="fix"></param>
            /// <param name="mov"></param>
            /// <param name="roiCenter"></param>
            /// <param name="roiSize"></param>
            /// <param name="nHistoBin"></param>
            /// <returns></returns>
            static float CalcJointHentropytRoiF(ArrayView2D<byte, Stride2D.DenseY> fix, ArrayView2D<byte, Stride2D.DenseY> mov, ArrayView2D<short, Stride2D.DenseX> hist, int histLine, PointF roiCenterFixImage, PointF roiCenterMovImage, Size roiSize)
            {

                int idxX = 0, idxY = 0;
                float hentropy = 0;
                float p = 0;
                float oneOverSum = 1.0f / (roiSize.Width * roiSize.Height);
                float halfRoiSize = 0.5f * roiSize.Width;

                for (int y = 0; y < roiSize.Height; y++)
                {
                    for (int x = 0; x < roiSize.Width; x++)
                    {
                        //questo non interpola perché la roi fix é ad indici interi, controllare che sia cosí
                        idxY = At(fix, roiCenterFixImage.X + x - halfRoiSize, roiCenterFixImage.Y + y - halfRoiSize);
                        //questo interpola perché la roi mobile é ad indici float
                        idxX = At(mov, roiCenterMovImage.X + x - halfRoiSize, roiCenterMovImage.Y + y - halfRoiSize);
                        if (idxY < 0) idxY = 0;
                        if (idxY >= nHistoBin) idxY = nHistoBin - 1;
                        if (idxX < 0) idxX = 0;
                        if (idxX >= nHistoBin) idxX = nHistoBin - 1;
                        hist[idxY * nHistoBin + idxX, histLine]++;
                    }
                }
                short binVal = 0;
                for (int i = 0; i < nHistoBin * nHistoBin; i++)
                {
                    binVal = hist[i, histLine];
                    if (binVal > 0)
                    {
                        p = binVal * oneOverSum;
                        hentropy += -p * ILGPU.Algorithms.XMath.Log(p);
                    }
                }
                return hentropy;
            }

        }

        internal MemoryBuffer2D<byte, Stride2D.DenseY> fixedImageByte;
        internal MemoryBuffer2D<byte, Stride2D.DenseY> movingImageByte;
        byte[] fixedImageArr, movingImageArr;
        Accelerator accelerator;
        /// <summary>
        /// asd
        /// </summary>
        KeyPointsCalculator keyPointsCalculator;
        Action<Index1D,
                ArrayView2D<byte, Stride2D.DenseY>,
                ArrayView2D<byte, Stride2D.DenseY>,
                ArrayView1D<PointF, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView2D<short, Stride2D.DenseX>,
                ArrayView2D<short, Stride2D.DenseX>,
                ArrayView2D<short, Stride2D.DenseX>,
                Size> KernelValidateRoi;

        /// <summary>
        /// KernelCalcMI(Index2D idx, int idxRoi, ArrayView2D<byte, Stride2D.DenseY> fix, ArrayView2D<byte, Stride2D.DenseY> mov, KeyPointsCalculator keyPointsCalculator)
        /// </summary>
        Action<Index2D,
                int,
                ArrayView2D<byte, Stride2D.DenseY>,
                ArrayView2D<byte, Stride2D.DenseY>,
                ArrayView1D<PointF, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView2D<short, Stride2D.DenseX>,
                ArrayView2D<short, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseY>,
                float,
                int,
                int,
                int,
                Size> KernelCalcMutualInformation;

        //public static void KernelFindPointOfMaxMI(Index1D idx, KeyPointsCalculator keyPointsCalculator)
        Action<Index1D,
                ArrayView2D<float, Stride2D.DenseY>,
                ArrayView1D<PointF, Stride1D.Dense>,
                ArrayView1D<PointF, Stride1D.Dense>,
                int,
                int,
                float,
                int> KernelFindPointOfMaxMI;

        public KeyPointsILGPU_noStruct(Size size, Accelerator acc, int nBin = 1, int nRoiPerWidth = 10, int margin = 6)
        {
            Index2D imageIdx = new Index2D(size.Width, size.Height);
            fixedImageByte = acc.Allocate2DDenseY<byte>(imageIdx);
            movingImageByte = acc.Allocate2DDenseY<byte>(imageIdx);
            fixedImageArr = new byte[fixedImageByte.LengthInBytes];
            movingImageArr = new byte[movingImageByte.LengthInBytes];

            keyPointsCalculator = new KeyPointsCalculator(size, acc, nBin, nRoiPerWidth, margin);
            acc.Synchronize();
            KernelValidateRoi = acc.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView2D<byte, Stride2D.DenseY>,
                ArrayView2D<byte, Stride2D.DenseY>,
                ArrayView1D<PointF, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView2D<short, Stride2D.DenseX>,
                ArrayView2D<short, Stride2D.DenseX>,
                ArrayView2D<short, Stride2D.DenseX>,
                Size>(KeyPointsCalculator.KernelValidateRoi);
            acc.Synchronize();

            KernelCalcMutualInformation = acc.LoadAutoGroupedStreamKernel<Index2D,
                int,
                ArrayView2D<byte, Stride2D.DenseY>,
                ArrayView2D<byte, Stride2D.DenseY>,
                ArrayView1D<PointF, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView2D<short, Stride2D.DenseX>,
                ArrayView2D<short, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseY>,
                float,
                int,
                int,
                int,
                Size>(KeyPointsCalculator.KernelCalcMI);
            acc.Synchronize();

            KernelFindPointOfMaxMI = acc.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView2D<float, Stride2D.DenseY>,
                ArrayView1D<PointF, Stride1D.Dense>,
                ArrayView1D<PointF, Stride1D.Dense>,
                int,
                int,
                float,
                int>(KeyPointsCalculator.KernelFindPointOfMaxMI);
            acc.Synchronize();

            accelerator = acc;
        }
        public void LoadImages(Mat fixedImage, Mat movingImage, Accelerator acc)
        {
            //normalizzo le immagini a 64 livelli di grigio, credo che questo sia il modo con meno memoria sprecata
            Mat fix = new Mat();
            Mat mov = new Mat();
            fixedImage.MinMax(out var min, out var max, out var pmin, out var pmax);
            ((fixedImage - min[0]) / (max[0] - min[0]) * 63.0).ConvertTo(fix, Emgu.CV.CvEnum.DepthType.Cv8U);

            movingImage.MinMax(out min, out max, out pmin, out pmax);
            ((movingImage - min[0]) / (max[0] - min[0]) * 63.0).ConvertTo(mov, Emgu.CV.CvEnum.DepthType.Cv8U);
            fix.CopyTo<byte>(fixedImageArr);
            mov.CopyTo<byte>(movingImageArr);
            fixedImageByte.AsRawArrayView().CopyFromCPU(fixedImageArr);
            movingImageByte.AsRawArrayView().CopyFromCPU(movingImageArr);

            accelerator.Synchronize();



#if GPU_DEBUG
            ////test read delle immagini
            //var flt = fixedImageByte.AsContiguous().GetRawData().Array;
            //File.WriteAllBytes("C:\\test\\fixBytes" + fixedImageByte.IntExtent.X + "_" + fixedImageByte.IntExtent.Y + ".raw", flt);
            //flt = movingImageByte.AsContiguous().GetRawData().Array;
            //File.WriteAllBytes("C:\\test\\movBytes" + fixedImageByte.IntExtent.X + "_" + fixedImageByte.IntExtent.Y + ".raw", flt);
#endif
        }
        public (PointF[] pFix, PointF[] pMov) FindKeyPoints()
        {
            //metto a zero gli array degli istogrammi
            keyPointsCalculator.RoiFixedEntropy.AsContiguous().MemSetToZero();
            keyPointsCalculator.RoiMutualInformation.AsContiguous().MemSetToZero();
            keyPointsCalculator.MutualInformation.AsContiguous().MemSetToZero();
            keyPointsCalculator.roiHistogramsFix.AsContiguous().MemSetToZero();
            keyPointsCalculator.roiHistogramsMov.AsContiguous().MemSetToZero();
            keyPointsCalculator.roiJointHistograms.AsContiguous().MemSetToZero();
            keyPointsCalculator.histograms.AsContiguous().MemSetToZero();
            keyPointsCalculator.jointHistograms.AsContiguous().MemSetToZero();
            accelerator.Synchronize();

            //i punti dell'immagine fissa li ho dall'inizio, mi basta scaricarli dalla gpu
            var keyPointFix = keyPointsCalculator.keyPointFixGpu.GetAsArray1D();
            //inizializzo il vettore di punti dell'immagine mov
            var keyPointMov = new PointF[keyPointFix.Length];
            //ora devo trovare i key points usando la mutual information
            //creo un array di booleani dove scriveró quali roi sono valide
            int nRoi = keyPointsCalculator.totalNumberOfRois;
            bool[] validRoi = new bool[nRoi];
            Index1D nRoiIndex = new Index1D(nRoi);
            //lancio un kerne per ogni punto trovato e calcolo per quel punto l'entropia dell'immagine fissa e la mutual information 
            /*public static void KernelValidateRoi(
                Index1D idx, 
                ArrayView2D<byte, Stride2D.DenseY> fix, 
                ArrayView2D<byte, Stride2D.DenseY> mov,
                ArrayView1D<PointF, Stride1D.Dense> keyPointFixGpu,
                ArrayView1D<float, Stride1D.Dense> RoiFixedEntropy,
                ArrayView1D<float, Stride1D.Dense> RoiMutualInformation,
                ArrayView2D<short, Stride2D.DenseX> roiHistogramsFix, 
                ArrayView2D<short, Stride2D.DenseX> roiHistogramsMov,
                ArrayView2D<short, Stride2D.DenseX> roiJointHistograms,
                Size roiSize)*/
            KernelValidateRoi(nRoiIndex,
                fixedImageByte,
                movingImageByte,
                keyPointsCalculator.keyPointFixGpu,
                keyPointsCalculator.RoiFixedEntropy,
                keyPointsCalculator.RoiMutualInformation,
                keyPointsCalculator.roiHistogramsFix,
                keyPointsCalculator.roiHistogramsMov,
                keyPointsCalculator.roiJointHistograms,
                keyPointsCalculator.roiSize);
            accelerator.Synchronize();



            //leggo i due array, li creo nuovi per ora non sono tanto grandi..
            var fixedHentropy = keyPointsCalculator.RoiFixedEntropy.GetAsArray1D();
            var mutualInfomration = keyPointsCalculator.RoiMutualInformation.GetAsArray1D();
            for (int i = 0; i < nRoi; i++)
            {
                validRoi[i] = false;
                //valuto quale celle sono adatte alla registrazione
                //ATTENZIONE questo parametro funxiona per immagini 978 x 978 
                if (fixedHentropy[i] > 1)//la prima soglia é solo sull'entropia, valutare bene questo parametro insieme alla dimensione
                {
                    if (fixedHentropy[i] + mutualInfomration[i] > 1.8)
                        validRoi[i] = true;
                }


                //test: metto tutto a true
                //validRoi[i] = true;
            }
            //ora se la roi é valida lancio i kernel che andranno a scrivere la MI nella tabellona
            //per ogni roi vengono lanciati nStepForRoiSide*nStepForRoiSide kernel, ognuno dei quli calcolerá la mutual information con il passo specificato in keyPointsCalculator
            Index2D roiIndex = new Index2D(keyPointsCalculator.nStepForRoiSide, keyPointsCalculator.nStepForRoiSide);
            for (int i = 0; i < nRoi; i++)
            {//per ora valuto solo l'entropia

                if (validRoi[i])
                {
                    /*public static void KernelCalcMI(
                Index2D idx, 
                int idxRoi, 
                ArrayView2D<byte, Stride2D.DenseY> fix, 
                ArrayView2D<byte, Stride2D.DenseY> mov,
                ArrayView1D<PointF, Stride1D.Dense> keyPointFixGpu,
                ArrayView1D<float, Stride1D.Dense> RoiFixedEntropy,
                ArrayView2D<short, Stride2D.DenseX> histograms,
                ArrayView2D<short, Stride2D.DenseX> jointHistograms,
                ArrayView2D<float, Stride2D.DenseX> MutualInformation,
                float stepWidth,
                int maxDistanceFromCenter,
                int nRoiPerWidth, 
                int nStepForRoiSide,
                Size roiSize
                    */
                    KernelCalcMutualInformation(new Index2D(keyPointsCalculator.nStepForRoiSide, keyPointsCalculator.nStepForRoiSide),
                        i,
                        fixedImageByte,
                        movingImageByte,
                        keyPointsCalculator.keyPointFixGpu,
                        keyPointsCalculator.RoiFixedEntropy,
                        keyPointsCalculator.histograms,
                        keyPointsCalculator.jointHistograms,
                        keyPointsCalculator.MutualInformation,
                        keyPointsCalculator.stepWidth,
                        keyPointsCalculator.maxDistanceFromCenter,
                        keyPointsCalculator.nRoiPerWidth,
                        keyPointsCalculator.nStepForRoiSide,
                        keyPointsCalculator.roiSize
                        );
                }

            }
            accelerator.Synchronize();

            //bon ci siamo, a questo punto lancio i kernel che mi trovano il minimo nel rettangolino
            /*public static void KernelFindPointOfMaxMI(
                Index1D idx,
                ArrayView2D<float, Stride2D.DenseX> MutualInformation,
                ArrayView1D<PointF, Stride1D.Dense> keyPointFixGpu,
                ArrayView1D<PointF, Stride1D.Dense> keyPointMovGpu,
                int nStepForRoiSide,
                int nRoiPerWidth,
                float stepWidth,
                int maxDistanceFromCenter)*/
            KernelFindPointOfMaxMI(nRoiIndex,
                keyPointsCalculator.MutualInformation,
                keyPointsCalculator.keyPointFixGpu,
                keyPointsCalculator.keyPointMovGpu,
                keyPointsCalculator.nStepForRoiSide,
                keyPointsCalculator.nRoiPerWidth,
                keyPointsCalculator.stepWidth,
                keyPointsCalculator.maxDistanceFromCenter
                );
            accelerator.Synchronize();

            keyPointMov = keyPointsCalculator.keyPointMovGpu.GetAsArray1D();
            accelerator.Synchronize();

            for (int i = 0; i < nRoi; i++)
            {
                //ora nei punti non validi assegno il punto fix
                if (!validRoi[i])
                {
                    keyPointMov[i] = keyPointFix[i];
                }
            }


#if GPU_DEBUG
            var flt3 = keyPointsCalculator.MutualInformation.AsContiguous().GetRawData().Array; accelerator.Synchronize();
            File.WriteAllBytes("C:\\test\\MItable_float_" + keyPointsCalculator.MutualInformation.IntExtent.X + "_" + keyPointsCalculator.MutualInformation.IntExtent.Y + ".raw", flt3);

            var flt = keyPointsCalculator.roiHistogramsFix.AsContiguous().GetRawData().Array;
            var histRoi = keyPointsCalculator.roiHistogramsFix.GetAsArray2D<short>();
            accelerator.Synchronize();

            File.WriteAllBytes("C:\\test\\roiHistograms_short_" + keyPointsCalculator.roiHistogramsFix.IntExtent.X + "_" + keyPointsCalculator.roiHistogramsFix.IntExtent.Y + ".raw", flt);
            var joint = keyPointsCalculator.roiJointHistograms.GetAsArray2D<short>();
            accelerator.Synchronize();

            Image<Gray, short> jointHistoRoi = new Image<Gray, short>(keyPointsCalculator.nRoiPerWidth* KeyPointsCalculator.nHistoBin, keyPointsCalculator.nRoiPerHeight*KeyPointsCalculator.nHistoBin);
            int jointLen = KeyPointsCalculator.nHistoBin * KeyPointsCalculator.nHistoBin;
            for (int iRoi = 0; iRoi < keyPointsCalculator.totalNumberOfRois; iRoi++)
            {
                int roiX = iRoi % keyPointsCalculator.nRoiPerWidth;
                int roiY = iRoi / keyPointsCalculator.nRoiPerWidth;
                for (int j = 0; j < jointLen; j++)
                {
                    int x = roiX * KeyPointsCalculator.nHistoBin + j % KeyPointsCalculator.nHistoBin;
                    int y = roiY * KeyPointsCalculator.nHistoBin + j / KeyPointsCalculator.nHistoBin;
                    jointHistoRoi.Data[y, x, 0] = joint[j, iRoi];
                }
            }
            File.WriteAllBytes("C:\\test\\roiJointHistograms_short_table_" + jointHistoRoi.Width + "_" + jointHistoRoi.Height + ".raw", jointHistoRoi.Bytes);

            flt = keyPointsCalculator.roiJointHistograms.AsContiguous().GetRawData().Array; accelerator.Synchronize();

            File.WriteAllBytes("C:\\test\\roiJointHistograms_short_linear_" + keyPointsCalculator.roiJointHistograms.IntExtent.X + "_" + keyPointsCalculator.roiJointHistograms.IntExtent.Y + ".raw", flt);
            flt = keyPointsCalculator.MutualInformation.AsContiguous().GetRawData().Array; accelerator.Synchronize();

            File.WriteAllBytes("C:\\test\\MItable_float_" + keyPointsCalculator.MutualInformation.IntExtent.X + "_" + keyPointsCalculator.MutualInformation.IntExtent.Y + ".raw", flt);
            flt = keyPointsCalculator.histograms.AsContiguous().GetRawData().Array; accelerator.Synchronize();

            File.WriteAllBytes("C:\\test\\histograms_short_" + keyPointsCalculator.histograms.IntExtent.X + "_" + keyPointsCalculator.histograms.IntExtent.Y + ".raw", flt);

            //test interpolazione lineare immagine
            var kerRead = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<byte, Stride2D.DenseY>, ArrayView2D<byte, Stride2D.DenseY>, PointF>(ReadImageWithOffset);

            Index2D idxImage = new Index2D(fixedImageByte.IntExtent.X, fixedImageByte.IntExtent.Y);
            var gpuReadImage = accelerator.Allocate2DDenseY<byte>(idxImage);
            accelerator.Synchronize();
            kerRead(idxImage, fixedImageByte, gpuReadImage, new PointF(0.123f, 0.879f));
            accelerator.Synchronize();
            var readimageBytes1 = gpuReadImage.AsContiguous().GetRawData().Array;
            accelerator.Synchronize();
            accelerator.Synchronize();
            kerRead(idxImage, fixedImageByte, gpuReadImage, new PointF(0.123f, 0.879f));
            accelerator.Synchronize();
            var readimageBytes2 = gpuReadImage.AsContiguous().GetRawData().Array;
            accelerator.Synchronize();
            var readFixedImage = fixedImageByte.AsContiguous().GetRawData().Array;
            accelerator.Synchronize();

            File.WriteAllBytes("C:\\test\\readFiximageBytes1_" + idxImage.X + "_" + idxImage.Y + ".raw", readimageBytes1);
            File.WriteAllBytes("C:\\test\\readFiximageBytes2_" + idxImage.X + "_" + idxImage.Y + ".raw", readimageBytes2);
            File.WriteAllBytes("C:\\test\\readFiximageBytes0_" + idxImage.X + "_" + idxImage.Y + ".raw", readFixedImage);

            var fixedHentropytess = keyPointsCalculator.RoiFixedEntropy.GetAsArray1D<float>();
            var test_histogram = keyPointsCalculator.histograms.GetAsArray2D<short>();
            accelerator.Synchronize();

#endif

            //woho dovremmo esserci
            return (keyPointFix, keyPointMov);
        }


        //bilinear interpolation

        public static void ReadImageWithOffset(Index2D i, ArrayView2D<byte, Stride2D.DenseY> imageIn, ArrayView2D<byte, Stride2D.DenseY> imageOut, PointF offset)
        {
            imageOut[i] = At(imageIn, offset.X + i.X, offset.Y + i.Y);
        }

        public static byte At(ArrayView2D<byte, Stride2D.DenseY> image, int x, int y)
        {
            int w = image.IntExtent.X;
            int h = image.IntExtent.Y;
            if (x < 0) x = 0; if (x >= w) x = w - 1;
            if (y < 0) y = 0; if (y >= h) y = h - 1;
            return image[x, y];
        }
        public static float At(ArrayView2D<float, Stride2D.DenseY> image, int x, int y)
        {
            int w = image.IntExtent.X;
            int h = image.IntExtent.Y;
            if (x < 0) x = 0; if (x >= w) x = w - 1;
            if (y < 0) y = 0; if (y >= h) y = h - 1;
            return image[x, y];
        }
        public static byte At(ArrayView2D<byte, Stride2D.DenseY> image, float x, float y)
        {
            int ix = (int)x; int iy = (int)y;
            float dx = x - ix; float dy = y - iy;

            return (byte)(
                (At(image, ix, iy) * (1.0f - dx) + At(image, ix + 1, iy) * dx) * (1.0f - dy) +
                (At(image, ix, iy + 1) * (1.0f - dx) + At(image, ix + 1, iy + 1) * dx) * dy
                );
        }
        public static float At(ArrayView2D<float, Stride2D.DenseY> image, float x, float y)
        {
            int ix = (int)x; int iy = (int)y;
            float dx = x - ix; float dy = y - iy;

            return
                (At(image, ix, iy) * (1.0f - dx) + At(image, ix + 1, iy) * dx) * (1.0f - dy) +
                (At(image, ix, iy + 1) * (1.0f - dx) + At(image, ix + 1, iy + 1) * dx) * dy
                ;
        }

    }
    internal class LocalWeightedMeanTransformationReader
    {
        LocalWeightedMeanTransformationILGPU lm;
        Accelerator accelerator;
        Action<Index2D, ArrayView2D<float, Stride2D.DenseY>, ArrayView2D<float, Stride2D.DenseY>, LocalWeightedMeanTransformationILGPU> readRoiKernel;

        public LocalWeightedMeanTransformationReader(PointF[] fixedPoints, PointF[] movingPoints, Accelerator accelerator)
        {
            this.accelerator = accelerator;
            lm = new LocalWeightedMeanTransformationILGPU(new LocalWeightedMeanTransformation(fixedPoints, movingPoints), accelerator);
            readRoiKernel = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseY>, ArrayView2D<float, Stride2D.DenseY>, LocalWeightedMeanTransformationILGPU>(ApplyInverseKernel);
        }

        public void ApplyInverse(Index2D index,
            ArrayView2D<float, Stride2D.DenseY> image,
            ArrayView2D<float, Stride2D.DenseY> mapped)
        {
            readRoiKernel(index, image, mapped, lm);
        }

        static void ApplyInverseKernel(Index2D index,
            ArrayView2D<float, Stride2D.DenseY> image,
            ArrayView2D<float, Stride2D.DenseY> mapped,
            LocalWeightedMeanTransformationILGPU lm)
        {
            var ptRead = lm.TransformInverse(new PointF(index.X, index.Y));
            mapped[index] = KeyPointsILGPU_struct.At(image, ptRead.X, ptRead.Y);
        }
    }
}
