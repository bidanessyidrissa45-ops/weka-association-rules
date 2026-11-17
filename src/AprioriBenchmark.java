import java.io.File;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Locale;

import weka.associations.Apriori;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * TP01 - Bonus, question 2 : temps d'exécution d'Apriori en fonction
 * du seuil de support et du seuil de confiance.
 *
 * Usage : java AprioriBenchmark [fichier.arff]
 *   (par défaut : results/bankModified.arff s'il existe, sinon data/vote.arff)
 *
 * Résultat : results/apriori_benchmark.csv  (support;confiance;temps_ms;nb_regles)
 * puis : python plot.py  pour tracer les graphes.
 */
public class AprioriBenchmark {

    static final double[] SUPPORTS    = {0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65};
    static final double[] CONFIANCES  = {0.50, 0.60, 0.70, 0.80, 0.90};
    static final int REPETITIONS = 5;   // on garde le minimum : le run le moins perturbé (GC, JIT, autres processus)

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.US);
        String chemin = args.length > 0 ? args[0]
                : new File("results/bankModified.arff").exists() ? "results/bankModified.arff"
                : "data/vote.arff";

        Instances data = DataSource.read(chemin);
        System.out.println("Jeu de données : " + chemin + " (" + data.numInstances()
                + " exemples, " + data.numAttributes() + " attributs)");

        // Échauffement de la JVM pour que les premières mesures ne soient pas faussées
        lancer(data, 0.5, 0.9);

        new File("results").mkdirs();
        try (PrintWriter out = new PrintWriter("results/apriori_benchmark.csv")) {
            out.println("support;confiance;temps_ms;nb_regles");
            for (double sup : SUPPORTS) {
                for (double conf : CONFIANCES) {
                    double[] temps = new double[REPETITIONS];
                    int nbRegles = 0;
                    for (int r = 0; r < REPETITIONS; r++) {
                        System.gc(); // libère les règles du run précédent pour ne pas chronométrer le GC
                        long t0 = System.nanoTime();
                        nbRegles = lancer(data, sup, conf);
                        temps[r] = (System.nanoTime() - t0) / 1e6;
                    }
                    Arrays.sort(temps);
                    double tMin = temps[0];
                    out.printf("%.2f;%.2f;%.3f;%d%n", sup, conf, tMin, nbRegles);
                    System.out.printf("support=%.2f  confiance=%.2f  ->  %9.2f ms  %6d règles%n",
                            sup, conf, tMin, nbRegles);
                }
            }
        }
        System.out.println("Résultats écrits dans results/apriori_benchmark.csv");
    }

    /** Lance Apriori avec un support et une confiance fixes, renvoie le nombre de règles trouvées. */
    static int lancer(Instances data, double support, double confiance) throws Exception {
        Apriori apriori = new Apriori();
        // Weka part de minSupport = 1 - delta puis descend de delta jusqu'à lowerBound.
        // Avec delta = 1 - support, le premier (et unique) cycle se fait directement au support voulu.
        // (upperBoundMinSupport reste à 1.0 : sinon Weka supprime les règles de support > upper)
        // lowerBound est recalculé comme 1 - delta : en flottant, 1 - (1 - 0.3) != 0.3, et cet écart
        // suffit à faire faire à Weka un second cycle (temps doublé à certains supports).
        double delta = 1.0 - support;
        apriori.setUpperBoundMinSupport(1.0);
        apriori.setDelta(delta);
        apriori.setLowerBoundMinSupport(1.0 - delta);
        apriori.setMinMetric(confiance);
        apriori.setNumRules(Integer.MAX_VALUE - 1); // on veut toutes les règles, pas seulement les 10 premières
        apriori.buildAssociations(data);
        return apriori.getAssociationRules().getNumRules();
    }
}
