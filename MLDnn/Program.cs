using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace MLDnn
{
    /* Necessary packages:
     1) Microsoft.ML
     2) Microsoft.ML.ImageAnalytics.
     3) Microsoft.ML.TensorFlow.
     4) Microsoft.ML.Vision.
     5) SciSharp.TensorFlow.Redist

     This is just a basic project I learned to make based on tutorials to help me learn.

     - Omar Moodie.
     */

    class Program
    {
        static void Main(string[] args)
        {
            // Get ref to images folder.
            var imgFolder = Path.Combine(Environment.CurrentDirectory, "..", "..", "..", "images");

            // Go to images folder, get all files (*) from both folders.
            var files = Directory.GetFiles(imgFolder, "*", SearchOption.AllDirectories);

            var images = files.Select(file => new ImageData
            {
                ImagePath = file,
                Label = Directory.GetParent(file).Name
            });

            var context = new MLContext();

            var imageData = context.Data.LoadFromEnumerable(images);
            var imageDataShuffled = context.Data.ShuffleRows(imageData);

            var testTrainData = context.Data.TrainTestSplit(imageDataShuffled, testFraction: 0.2);

            // Transform the names (individual pic files or pic folders) into keys, numerical keys.
            var validationData = context.Transforms.Conversion.MapValueToKey("LabelKey", "Label", 
                keyOrdinality: Microsoft.ML.Transforms.ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
                .Append(context.Transforms.LoadRawImageBytes("Image", imgFolder, "ImagePath"))
                .Fit(testTrainData.TestSet)
                .Transform(testTrainData.TestSet);

            var imagesPipeline = context.Transforms.Conversion.MapValueToKey("LabelKey", "Label", 
                keyOrdinality: Microsoft.ML.Transforms.ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
                .Append(context.Transforms.LoadRawImageBytes("Image", imgFolder, "ImagePath"));





            var imageDataModel = imagesPipeline.Fit(testTrainData.TrainSet);

            var imageDataView = imageDataModel.Transform(testTrainData.TrainSet);

            var options = new ImageClassificationTrainer.Options()
            {
                Arch = ImageClassificationTrainer.Architecture.ResnetV250,
                Epoch = 100,
                BatchSize = 20,
                LearningRate = 0.01f,
                LabelColumnName = "LabelKey",
                FeatureColumnName = "Image",
                ValidationSet = validationData
            };



            var pipeline = context.MulticlassClassification.Trainers.ImageClassification(options)
                .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel"));


            var model = pipeline.Fit(imageDataView);





            var predictionEngine = context.Model.CreatePredictionEngine<ImageModelInput, ImagePrediction>(model);

            // Go back 3 folders for the test folder.
            var testImagesFolder = Path.Combine(Environment.CurrentDirectory, "..", "..", "..", "test");

            // Get all files in all directories with * and AllDirectories.
            var testFiles = Directory.GetFiles(testImagesFolder, "*", SearchOption.AllDirectories);

            /* The lazy initialization part, use select to get each file and get a new ImageModelInput
             pointer. */
            var testImages = testFiles.Select(file => new ImageModelInput
            {
                ImagePath = file
            });

            Console.WriteLine(Environment.NewLine);

            // When we used select, the best case to go is to use LoadFromEnumerable. It's lazy too.
            var testImagesData = context.Data.LoadFromEnumerable(testImages);

            /*  */
            var testImageDataView = imagesPipeline.Fit(testImagesData).Transform(testImagesData);


            var predictions = model.Transform(testImageDataView);

            // IDataView into strongly typed Enumerable<T>
            var testPredictions = context.Data.CreateEnumerable<ImagePrediction>(predictions, reuseRowObject: false);

            foreach (var prediction in testPredictions)
            {
                var labelIndex = prediction.PredictedLabel;

                Console.WriteLine($"Image: {Path.GetFileName(prediction.ImagePath)}, Predicted Label: {prediction.PredictedLabel}");
            }




            Console.ReadLine();
        }
    }
}
