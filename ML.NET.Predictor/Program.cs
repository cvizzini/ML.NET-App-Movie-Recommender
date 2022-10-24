using Microsoft.ML;
using Shared.Models;
using Shared.Trainer;

Console.WriteLine("ML.NET Load Training Model - Movie Recommender");

MLContext mlContext = new MLContext();
var movieModelTrainer = new MovieModelTrainer(mlContext);
var model = movieModelTrainer.LoadModel(mlContext);

var testInput = new MovieRatingModel() { userId = 6, movieId = 10 };

movieModelTrainer.UseModelForSinglePrediction(model, testInput);


/*
    Example Output:
    =========================================================
    
        ML.NET Predictor - Movie Recommender
    =============== Making a prediction ===============
    Movie 10 is recommended for user 6

    =========================================================   

 */