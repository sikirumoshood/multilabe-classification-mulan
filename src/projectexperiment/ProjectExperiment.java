package projectexperiment;

import mulan.classifier.lazy.MLkNN;
import mulan.classifier.neural.BPMLL;
import mulan.classifier.transformation.LabelPowerset;
import mulan.classifier.meta.RAkEL;
import weka.classifiers.trees.J48;
import weka.classifiers.bayes.NaiveBayes;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
public class ProjectExperiment {
   
    private static String dataArff; // The formatted data in arff extension
    private static String dataXml;  // Xml file for the relationship between features
   
    public static void main(String[] args) {
       
        //loading the data 

        dataArff = "C:\\Users\\Sikiru Moshood\\Desktop\\DESKTOP\\project\\Datasets\\EmotionsData\\emotions.arff";
        dataXml = "C:\\Users\\Sikiru Moshood\\Desktop\\DESKTOP\\project\\Datasets\\EmotionsData\\emotions.xml";
        
        try{
            
            MultiLabelInstances data = new MultiLabelInstances(dataArff,dataXml);
            RAkEL learner1 = new RAkEL ( new LabelPowerset(new NaiveBayes()));
            MLkNN learner2 = new MLkNN();
            BinaryRelevance learner3 = new BinaryRelevance(new J48());
            BPMLL learner4 = new BPMLL();
            
            System.out.println("\n-----------------------------------------------MLkNN---------------------------------------------------\n");
            Evaluator eval = new Evaluator();
            MultipleEvaluation results;
            
            long startMlknn = System.nanoTime();
            
            results = eval.crossValidate(learner1, data, 10);
            System.out.println(results);
            System.out.println();
            
            long endMlknn = System.nanoTime() - startMlknn;
            double toSeconds = (double)endMlknn/1000000000;
            
            System.out.println("ELAPSED-TIME: "+ toSeconds + " seconds");
            
            System.out.println("\n------------------------------------------------RAkEL---------------------------------------------------\n"); 
            
            
            long startRakel = System.nanoTime();
            results = eval.crossValidate(learner2, data, 10);
            System.out.println(results);
            System.out.println();
            long endRakel = System.nanoTime()- startRakel;
            double toSecondsRakel = (double)endRakel/1000000000;
            
            System.out.println("ELAPSED-TIME: "+ toSecondsRakel + " seconds");
            
            
            System.out.println("\n------------------------------------------------BINARY-RELEVANCE---------------------------------------------------\n"); 
            
            
            long startBinaryRelevance = System.nanoTime();
            
            results = eval.crossValidate(learner3, data, 10);
            System.out.println(results);
            System.out.println();
            long endBinaryRelevance = System.nanoTime() - startBinaryRelevance;
            
            double toSecondsBr = (double)endBinaryRelevance/1000000000;
            
            System.out.println("ELAPSED-TIME: "+ toSecondsBr + " seconds");
            
            System.out.println("\n------------------------------------------------BPMLL---------------------------------------------------\n"); 
            
            long startBpmll = System.nanoTime();
            results = eval.crossValidate(learner4, data, 10);
            System.out.println(results);
            System.out.println();
            long endBpmll = System.nanoTime() - startBpmll;
            
            double toSecondsBpmll = (double)endBpmll/1000000000;
            
            System.out.println("ELAPSED-TIME: "+ toSecondsBpmll + " seconds");
            
            
        }
        catch(Exception e){
            
            System.out.println(e.toString());
        }
        
    }
    
}
