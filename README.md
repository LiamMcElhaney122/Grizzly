# Grizzly ML
Grizzly ML is a C# machine learning library under development. The goal is to have robust framework for creating machine learning models in a compiled language!

It relies on Microsoft dataframes package to load data.

To get started, first load and sanitize your data using C# dataframes!

As of now, the only implemented classifier is RandomForest, but more are planned!

## Prepare and train example
```csharp
    //load your data
    DataFrame df = DataFrame.LoadCsv(@"data_dir", header: true);

    //get rid of unwanted columns
    df=Utils.dropColumns(df, new string[]{ "PassengerId", "Name", "Ticket", "Fare", "Cabin"});

    //split your data into training and testing sets
    (DataFrame, DataFrame) splitDF = Utils.split_train_test(df, .5);
    DataFrame train = splitDF.Item1;
    DataFrame test = splitDF.Item2;

    //specify training x & y column
    DataFrameColumn trs = train["Survived"];
    
    DataFrame yTrain = new DataFrame(new DataFrameColumn[]{trs});
    DataFrame xTrain = Utils.dropColumns(train, new string[] { "Survived" });

    //specify testing x & y column
    DataFrameColumn tes = test["Survived"];
    DataFrame yTest = new DataFrame(new DataFrameColumn[] { tes });
    DataFrame xTest = Utils.dropColumns(test, new string[] { "Survived" });
    
    //Instantiate a new random forrest with 50 trees and max length of 2 
    var rft = new RandomTree(50, 2);

    //Train the tree on the data
    rft.Fit(xTrain, yTrain, xTest, yTest);

    //See the results on the validation test set
    var output = rft.Predict(xTest.Rows);
```
