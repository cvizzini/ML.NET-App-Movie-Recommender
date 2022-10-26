using Microsoft.ML;
using Microsoft.ML.Trainers;
using Shared.Models;

namespace Shared.Trainer
{
    public class MovieModelTrainer
    {
        private MLContext mlContext;
        public MovieModelTrainer(MLContext mlContext)
        {
            this.mlContext = mlContext;
        }

        public (IDataView training, IDataView test) LoadData()
        {
            var trainingDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "recommendation-ratings-train.csv");
            var testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "recommendation-ratings-test.csv");

            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<MovieRatingModel>(trainingDataPath, hasHeader: true, separatorChar: ',');
            IDataView testDataView = mlContext.Data.LoadFromTextFile<MovieRatingModel>(testDataPath, hasHeader: true, separatorChar: ',');

            return (trainingDataView, testDataView);
        }

        public ITransformer BuildAndTrainModel(IDataView trainingDataView)
        {
            IEstimator<ITransformer> estimator = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "userIdEncoded", inputColumnName: "userId")
                .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "movieIdEncoded", inputColumnName: "movieId"));

            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = "userIdEncoded",
                MatrixRowIndexColumnName = "movieIdEncoded",
                LabelColumnName = "Label",
                NumberOfIterations = 20,
                ApproximationRank = 100
            };

            var trainerEstimator = estimator.Append(mlContext.Recommendation().Trainers.MatrixFactorization(options));
            Console.WriteLine("=============== Training the model ===============");
            ITransformer model = trainerEstimator.Fit(trainingDataView);

            return model;
        }

        public void EvaluateModel(IDataView testDataView, ITransformer model)
        {
            Console.WriteLine("=============== Evaluating the model ===============");
            var prediction = model.Transform(testDataView);
            var metrics = mlContext.Regression.Evaluate(prediction, labelColumnName: "Label", scoreColumnName: "Score");
            Console.WriteLine("Root Mean Squared Error : " + metrics.RootMeanSquaredError.ToString());
            Console.WriteLine("RSquared: " + metrics.RSquared.ToString());
        }

        public void UseModelForSinglePrediction(ITransformer model, MovieRatingModel testInput)
        {
            Console.WriteLine("=============== Making a prediction ===============");
            var predictionEngine = mlContext.Model.CreatePredictionEngine<MovieRatingModel, MovieRatingPredictionModel>(model);

            var movieRatingPrediction = predictionEngine.Predict(testInput);

            //Recommend movies with a predicted rating of > 3.5
            if (Math.Round(movieRatingPrediction.Score, 1) > 3.5)
            {
                Console.WriteLine("Movie " + testInput.movieId + " is recommended for user " + testInput.userId);
            }
            else
            {
                Console.WriteLine("Movie " + testInput.movieId + " is not recommended for user " + testInput.userId);
            }
        }

        private record MovieRecommended
        {
            public MovieRecommended(int movieId, float score)
            {
                MovieId = movieId;
                Score = score;
            }

            public int MovieId { get; set; }
            public float Score { get; set; }
        }

        public void UseModelForTop10(ITransformer model, int userId)
        {
            Console.WriteLine($"=============== Making Top 10 predictions for user: {userId} ===============");
            var predictionEngine = mlContext.Model.CreatePredictionEngine<MovieRatingModel, MovieRatingPredictionModel>(model);
            var predictions = new List<MovieRecommended>();
            for (int i = 0; i <= 6947; i++)
            {
                var input = new MovieRatingModel() { userId = userId, movieId = i };

                var movieRatingPrediction = predictionEngine.Predict(input);
                predictions.Add(new MovieRecommended(i, movieRatingPrediction.Score));
            }

            var top10 = predictions.OrderByDescending(x => x.Score).Take(10);

            var count = 1;
            foreach (var predictionModel in top10)
            {
                Console.WriteLine($"{count++}. {predictionModel.MovieId} score {predictionModel.Score}");
            }
        }

        public void SaveModel(DataViewSchema trainingDataViewSchema, ITransformer model)
        {
            var modelPath = Path.Combine(Path.GetTempPath(), "MovieRecommenderModel.zip");

            Console.WriteLine("=============== Saving the model to a file ===============");
            mlContext.Model.Save(model, trainingDataViewSchema, modelPath);
        }

        public ITransformer LoadModel(MLContext mlContext)
        {
            var modelPath = Path.Combine(Path.GetTempPath(), "MovieRecommenderModel.zip");
            DataViewSchema modelSchema;
            Console.WriteLine("=============== Loading the model from a file ===============");
            // Load trained model
            ITransformer trainedModel = mlContext.Model.Load(modelPath, out modelSchema);
            return trainedModel;
        }
    }
}
