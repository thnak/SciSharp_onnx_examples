using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Diagnostics;

namespace ConsoleApp1
{
    class Program
    {
        public static void Main(string[] args)
        {
            // Read paths
            string modelFilePath = "C:\\Users\\thanh\\Documents\\Dotnet\\ConsoleApp1\\ConsoleApp1\\model_softmax.onnx";
            string imageFilePath = "C:\\Users\\thanh\\Documents\\Dotnet\\ConsoleApp1\\ConsoleApp1\\dog.jpeg";

            // Read image
            using Image<Rgb24> image = Image.Load<Rgb24>(imageFilePath);

            // Resize image
            image.Mutate(x =>
            {
                x.Resize(new ResizeOptions
                {
                    Size = new Size(224, 224),
                    Mode = ResizeMode.Crop
                });
            });

            // Preprocess image
            Tensor<float> input = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < accessor.Height; y++)
                {
                    Span<Rgb24> pixelSpan = accessor.GetRowSpan(y);
                    for (int x = 0; x < accessor.Width; x++)
                    {
                        input[0, 0, y, x] = pixelSpan[x].R;
                        input[0, 1, y, x] = pixelSpan[x].G;
                        input[0, 2, y, x] = pixelSpan[x].B;
                                        }
                }
            });

            // Setup inputs
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("images", input)
            };
            
            // Run inference
            SessionOptions sessionOptions = new SessionOptions();
            sessionOptions.AppendExecutionProvider_DML(1);
            sessionOptions.EnableMemoryPattern = true;
            sessionOptions.EnableCpuMemArena = false;
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            sessionOptions.OptimizedModelFilePath = "Optim.onnx";
            sessionOptions.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
            using var session = new InferenceSession(modelFilePath, sessionOptions);
            int[] outputDim = new[] { 1, 1 };
            int[] inputDim = new[] { 1, 1, 1, 1 };
            foreach (var k in session.InputMetadata)
            {
                Console.WriteLine($"{k.Key}");
                inputDim[0] = k.Value.Dimensions[0];
                inputDim[1] = k.Value.Dimensions[1];
                inputDim[2] = k.Value.Dimensions[2];
                inputDim[3] = k.Value.Dimensions[3];
            }

            foreach (var k in session.OutputMetadata)
            {
                outputDim[0] = k.Value.Dimensions[0];
                outputDim[1] = k.Value.Dimensions[1];
            }

            int batchSize = 2;
            if (outputDim[0] == -1)
            {
                outputDim[0] = batchSize;
            }
            else
            {
                batchSize = outputDim[0];
            }
            Console.WriteLine(outputDim[0]);
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = null;
            Stopwatch stopWatch = new Stopwatch();
            stopWatch.Start();
            for (int i = 0; i < 1000; i++)
            {
                 results = session.Run(inputs);
            }
            stopWatch.Stop();
            Console.WriteLine($"somg {stopWatch.ElapsedMilliseconds}ms");
            var valuesRaw = results.ElementAt(0).AsTensor<float>().ToArray();
            var indiesRaw = results.ElementAt(1).AsTensor<long>().ToArray();
            byte[] bytes = new byte[valuesRaw.Length * sizeof(float)];

            
            // BitConverter.GetBytes(valuesRaw, 0, bytes);
            
            

        }
    }
}