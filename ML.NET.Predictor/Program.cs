using Microsoft.ML;
using Shared.Models;
using Shared.Trainer;

Console.WriteLine("ML.NET Load Training Model - Movie Recommender");

MLContext mlContext = new MLContext();
var movieModelTrainer = new MovieModelTrainer(mlContext);
var model = movieModelTrainer.LoadModel(mlContext);

var testInput = new MovieRatingModel() { userId = 6, movieId = 10 };

movieModelTrainer.UseModelForSinglePrediction(model, testInput);

movieModelTrainer.UseModelForTop10(model, 6);



/*
    Example Output:
    =========================================================
    
        ML.NET Predictor - Movie Recommender
    =============== Making a prediction ===============
    Movie 10 is recommended for user 6
    =============== Making Top 10 predictions for user: 6 ===============
    1. 3086 score 5.05236
    2. 5490 score 4.830536
    3. 6818 score 4.783887
    4. 3200 score 4.6664085
    5. 6442 score 4.659237
    6. 3379 score 4.62953
    7. 5915 score 4.613237
    8. 53 score 4.609263
    9. 6201 score 4.599132
    10. 5867 score 4.5860066

    =========================================================   


 */