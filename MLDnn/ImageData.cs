using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLDnn
{
    // The input schema, what's going INTO the algorithm.
    public class ImageData
    {
        [LoadColumn(0)]
        public string ImagePath;

        [LoadColumn(1)]
        public string Label;
    }
}
