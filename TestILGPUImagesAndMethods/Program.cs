﻿using ILGPU.Runtime;
using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;
using Emgu.CV;
using System.Diagnostics;
using Emgu.CV.Structure;
using System.Windows.Forms;

namespace TestILGPUImagesAndMethods
{
    internal class Program
    {
        [STAThread]

        static void Main(string[] args)
        {
            //image loading
            var fixedUshort = Properties.Resources.HighEnergy978ushort;
            var movingUshort = Properties.Resources.LowEnergy978ushort;

            double I0high = 36670;
            double I0low = 33920;

            Mat fixImgShort = new Mat(978, 978, Emgu.CV.CvEnum.DepthType.Cv16U, 1);
            Mat movImgShort = new Mat(978, 978, Emgu.CV.CvEnum.DepthType.Cv16U, 1);

            fixImgShort.SetTo(fixedUshort);
            movImgShort.SetTo(movingUshort);

            //conversion to projection
            Mat fixImg = new Mat();
            fixImgShort.ConvertTo(fixImg, Emgu.CV.CvEnum.DepthType.Cv32F);
            CvInvoke.Log(fixImg, fixImg);
            fixImg = Math.Log(I0high) - fixImg;
            Mat movImg = new Mat();
            movImgShort.ConvertTo(movImg, Emgu.CV.CvEnum.DepthType.Cv32F);
            CvInvoke.Log(movImg, movImg);
            movImg = Math.Log(I0low) - movImg;
            Mat movImgRegistered = new Mat(978, 978, Emgu.CV.CvEnum.DepthType.Cv32F, 1);


            //show image not registered
            Task t = Task.Run(() => { ShowImage((movImg - fixImg).ToImage<Gray, float>()); });
            







            //creation of the ILGPU class using STRUCT OF ARRAYS for image registration
            Stopwatch st = new Stopwatch();
            st.Reset(); st.Start();
            var mi = new MutualInformationSinglePassILGPU(fixImg.Size, MutualInformationSinglePassILGPU.MakeAccellerator(), true);//approx 16 seconds
            st.Stop(); Console.WriteLine("load registration kernel struct: " + st.ElapsedMilliseconds);

            //first run of registration
            st.Reset(); st.Start();
            mi.CreateTransformation(fixImg, movImg);
            mi.ApplyTransformation(movImg, ref movImgRegistered);
            st.Stop(); Console.WriteLine("first registration run: " + st.ElapsedMilliseconds);

            //20 registration run
            st.Reset(); st.Start();
            for (int i = 0; i < 20; i++)
            {
                mi.CreateTransformation(fixImg, movImg);
                mi.ApplyTransformation(movImg, ref movImgRegistered);
            }
            st.Stop(); Console.WriteLine("20 registration run: " + st.ElapsedMilliseconds);
            st.Stop(); Console.WriteLine("single registration run average: " + st.ElapsedMilliseconds/20);

            //show image registered
            Task t1 = Task.Run(() => { ShowImage((movImgRegistered - fixImg).ToImage<Gray, float>()); });




















            //creation of the ILGPU class using without struct for image registration
            st.Reset(); st.Start();
            var mi2 = new MutualInformationSinglePassILGPU(fixImg.Size, MutualInformationSinglePassILGPU.MakeAccellerator(), false);//approx 15 seconds
            st.Stop(); Console.WriteLine("load registration kernel NO struct: " + st.ElapsedMilliseconds);

            //first run of registration
            st.Reset(); st.Start();
            mi2.CreateTransformation(fixImg, movImg);
            mi2.ApplyTransformation(movImg, ref movImgRegistered);
            st.Stop(); Console.WriteLine("first registration run: " + st.ElapsedMilliseconds);

            //20 registration run
            st.Reset(); st.Start();
            for (int i = 0; i < 20; i++)
            {
                mi2.CreateTransformation(fixImg, movImg);
                mi2.ApplyTransformation(movImg, ref movImgRegistered);
            }
            st.Stop(); Console.WriteLine("20 registration run: " + st.ElapsedMilliseconds);
            st.Stop(); Console.WriteLine("single registration run average: " + st.ElapsedMilliseconds/20);

            //show image registered
            Task t2 = Task.Run(() => { ShowImage((movImgRegistered - fixImg).ToImage<Gray, float>()); });







            Console.ReadKey();


            void ShowImage(Image<Gray, float> im)
            {
                im.MinMax(out var min, out var max, out var pmin, out var pmax);
                var sub0 = (((im - min[0]) / (max[0] - min[0])) * 255.0).Convert<Gray, byte>();
                Emgu.CV.UI.ImageViewer viewer0 = new Emgu.CV.UI.ImageViewer(sub0);
                viewer0.Text = "Not Registered";
                //viewer0.Show();
                Application.Run(viewer0);
            }
        }
    }
}
