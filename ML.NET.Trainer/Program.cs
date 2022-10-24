using Microsoft.ML;
using Shared.Models;
using Shared.Trainer;

Console.WriteLine("ML.NET Trainer - Movie Recommender");
MLContext mlContext = new MLContext();
var movieModelTrainer = new MovieModelTrainer(mlContext);

var (training, test) = movieModelTrainer.LoadData();

var model = movieModelTrainer.BuildAndTrainModel(training);

movieModelTrainer.EvaluateModel(training, model);
movieModelTrainer.SaveModel(training.Schema, model);

var testInput = new MovieRatingModel() { userId = 6, movieId = 10 };
movieModelTrainer.UseModelForSinglePrediction(model, testInput);


/*
    Example Output:
    =========================================================

        ML.NET Trainer - Movie Recommender
    =============== Training the model ===============
    iter      tr_rmse          obj
       0       1.5057   2.8336e+05
       1       0.9229   1.4123e+05
       2       0.8677   1.3049e+05
       3       0.8442   1.2629e+05
       4       0.8279   1.2331e+05
       5       0.8145   1.2141e+05
       6       0.8009   1.1937e+05
       7       0.7871   1.1761e+05
       8       0.7727   1.1583e+05
       9       0.7584   1.1420e+05
      10       0.7439   1.1270e+05
      11       0.7297   1.1113e+05
      12       0.7164   1.0987e+05
      13       0.7045   1.0877e+05
      14       0.6925   1.0758e+05
      15       0.6815   1.0670e+05
      16       0.6715   1.0580e+05
      17       0.6621   1.0502e+05
      18       0.6531   1.0427e+05
      19       0.6451   1.0357e+05
    =============== Evaluating the model ===============
    Root Mean Squared Error : 0,6259375628946752
    RSquared: 0,6406839566793838
    =============== Saving the model to a file ===============
    
    =============== Making a prediction ===============
    Movie 10 is recommended for user 6

    =========================================================

    In this output, there are 20 iterations. In each iteration, the measure of error decreases
    and converges closer and closer to 0.
   
    The root of mean squared error (RMS or RMSE) is used to measure the differences between the model 
    predicted values and the test dataset observed values. Technically it's the square root of the 
    of the squares of the errors. The lower it is, the better the model is.
   
    R Squared indicates how well data fits a model. Ranges from 0 to 1. A value of 0 means that the data
    is random or otherwise can't be fit to the model. A value of 1 means that the model exactly matches the data.
    You want your R Squared score to be as close to 1 as possible.

 */