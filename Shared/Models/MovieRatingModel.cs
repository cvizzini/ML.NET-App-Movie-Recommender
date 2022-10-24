using Microsoft.ML.Data;

namespace Shared.Models
{
    public class MovieRatingModel
    {
        [LoadColumn(0)]
        public float userId;
        [LoadColumn(1)]
        public float movieId;
        [LoadColumn(2)]
        public float Label;
    }

    public class MovieRatingPredictionModel
    {
        public float Label;
        public float Score;
    }
}
