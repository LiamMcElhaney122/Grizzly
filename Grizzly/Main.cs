using Microsoft.Data.Analysis;
using System;
using System.Collections.Generic;
namespace Grizzly
{
    public static class Utils
    {
        public static DataFrame dropColumns(DataFrame df, String[] columns)
        {
            foreach (string c in columns)
            {
                df.Columns.Remove(c);
            }
            return df;
        }

        public static (DataFrame Train, DataFrame Test) split_train_test(DataFrame df, Double ratio = .25)
        {
            //TODO: Add shuffling for better randomization

            int length = (int)df.Rows.Count;

            int testSize = (int)Math.Floor(length * ratio);
            int trainSize = length - testSize;
            return (df.Tail(trainSize), df.Head(testSize));
        }

        public static String[] selectColumns(int treeLength, int dataLength, DataFrame xTrain, DataFrame yTrain)
        {
            Random rnd = new Random();

            String[] columns = new String[treeLength];

            var cols = xTrain.Columns;

            for (int i = 0; i < treeLength; i++)
            {
                columns[i] = cols[rnd.Next(0, treeLength)].ToString();
            }

            return columns;
        }

    }

    public abstract class Model
    {
        public static LeafNode EvaluateTree(Node node, DataFrameRow row, int rCount = 0)
        {
            rCount++;

            if (node.GetType() == typeof(LeafNode))
            {
                LeafNode o = (LeafNode)node;
                return o;
            }
            else
            {
                NodeBranch NextNode = (NodeBranch)node;

                node = EvaluateTree(NextNode.Compare(row[NextNode.featureIndex]), row, rCount);
            }

            return (LeafNode)node;
        }

        //RandomForst
        public abstract void Fit(DataFrame xTrain, DataFrame yTrain, DataFrame xTest, DataFrame yTest);

        public static Node createTree(int rLength, int treeLength, DataFrame xTrain, DataFrame yTrain, Random r)
        {
            int colCount = 0;
            int rowCount = 0;

            int featureindex = 0;
            int rowIndex = 0;

            DataFrameColumn feature = null;
            DataFrameRow row = null;

            rLength += 1;

            bool bugLoop = true;
            while (bugLoop)
            {

                colCount = xTrain.Columns.Count;
                rowCount = (int)xTrain.Rows.Count;

                featureindex = r.Next(0, colCount);
                rowIndex = r.Next(0, rowCount);

                feature = xTrain.Columns[featureindex];
                row = xTrain.Rows[rowIndex];

                if (featureindex != 2)
                {
                    bugLoop = false;
                }
            }

            if (rLength > treeLength)
            {
                throw new Exception("Reccursive Builder Length exceeded tree length");
            }
            else if (rLength == treeLength)
            {
                return new LeafNode(yTrain[rowIndex, 0]);
            }

            if (feature.GetType() == typeof(Microsoft.Data.Analysis.StringDataFrameColumn))
            {
                return new NodeBranchString(createTree(rLength, treeLength, xTrain, yTrain, r), new LeafNode(yTrain[rowIndex, 0]), xTrain[rowIndex, featureindex].ToString(), featureindex);
            }
            else
            {
                return new NodeBranchFloat(createTree(rLength, treeLength, xTrain, yTrain, r), new LeafNode(yTrain[rowIndex, 0]), (float)xTrain[rowIndex, featureindex], featureindex);
            }
        }

    }

    public class RandomTree : Model
    {

        private int treeCount;
        private int treeDepth;

        NodeBranch[] stumps;

        public RandomTree(int treeCount, int treeDepth)
        {
            this.treeCount = treeCount;
            this.treeDepth = treeDepth;
            this.stumps = new NodeBranch[treeCount];
        }

        public override void Fit(DataFrame xTrain, DataFrame yTrain, DataFrame xTest, DataFrame yTest)
        {
            for (int i = 0; i < treeCount; i++)
            {

                Random r = new Random();

                if (treeDepth > xTrain.Columns.Count)
                {
                    throw new Exception("Tree Depth can not be greater then features. For more complexity, please add more features.");
                }

                if (treeCount > xTrain.Rows.Count)
                {
                    throw new Exception("Tree Count can not be greate then rows. For more trees, please add more training data");
                }

                stumps[i] = (NodeBranch)createTree(0, treeDepth, xTrain, yTrain, r);
            }
        }

        public String[] Predict(DataFrameRowCollection rows)
        {
            String[] output = new String[rows.Count];

            for (int i = 0; i < output.Length; i++)
            {

                var classifiers = new Dictionary<string,
                  int>();

                foreach (var t in stumps)
                {

                    var o = EvaluateTree(t, rows[i]).Output.ToString();

                    if (!classifiers.ContainsKey(o))
                    {

                        classifiers.Add(o, 1);
                    }
                    else
                    {

                        classifiers[o]++;
                    }
                }

                int max = -1;
                string maxO = "";
                foreach (var j in classifiers.Keys)
                {
                    if (classifiers[j] > max)
                    {
                        max = classifiers[j];
                        maxO = j;
                    }

                }
                output[i] = maxO;

            }

            return output;
        }

    }

    public abstract class Node
    {

    }

    public abstract class NodeBranch : Node
    {
        public Node leftNode;
        public Node rightNode;
        public DataFrameColumn feature;
        public int featureIndex;

        public virtual Node Compare(object comparator)
        {
            throw new Exception("Can't call non-implemented class");
        }

    }

    public class NodeBranchString : NodeBranch
    {
        private string comparison;

        public override Node Compare(object comparator)
        {
            if ((string)comparator == comparison)
            {
                return rightNode;
            }
            else
            {
                return leftNode;
            }
        }

        public NodeBranchString(Node lNode, Node rNode, string comparison, int featureIndex)
        {
            leftNode = lNode;
            rightNode = rNode;
            this.featureIndex = featureIndex;
            this.comparison = comparison;
        }
    }

    public class NodeBranchFloat : NodeBranch
    {
        private float comparison;

        public override Node Compare(object comparator)
        {

            if ((float)comparator >= comparison)
            {
                return rightNode;
            }
            else
            {
                return leftNode;
            }
        }

        public NodeBranchFloat(Node lNode, Node rNode, float comparison, int featureIndex)
        {
            leftNode = lNode;
            rightNode = rNode;
            this.featureIndex = featureIndex;
            this.comparison = comparison;
        }
    }

    public class LeafNode : Node
    {
        public Object Output
        {
            get;
        }
        public LeafNode(Object Output)
        {
            this.Output = Output;
        }
    }

}
