using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLDnn
{
    // What's coming OUT of the algorithm.
    public class ImagePrediction
    {
        public string ImagePath { get; set; }

        public string Label { get; set; }

        public string PredictedLabel { get; set; }
    }
}
