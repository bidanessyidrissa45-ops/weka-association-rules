import java.io.File;
import java.util.Random;

import weka.associations.Apriori;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Remove;

/**
 * TP01 - Bonus, question 1 : automatisation via l'API Weka de ce qui a été fait
 * dans l'interface graphique (Explorer).
 *
 * Les parties suivent la numérotation du compte rendu (.docx) :
 *  3.   securitynet_dataset.arff et weather.nominal.arff : Apriori avec les 4 mesures
 *  4.   bank-full.csv : conversion CSV -> ARFF, suppression de "day", discrétisation,
 *       sauvegarde bank1.arff / bankModified.arff, puis Apriori
 *  6.   iris.arff discrétisé (10 intervalles) : Apriori classe en conclusion (CAR) vs J48
 *  7.   promotions_noel.arff (épicerie) : Apriori aux supports 0.75 / 0.5 / 0.25 / 0.1
 *  8.3  vote.arff (UCI) : exploration par parti + Apriori (normal et CAR)
 *  8.4  iris.arff non discrétisé : J48 en validation croisée 10 plis
 */
public class WekaAssociation {

    static final String DATA = "data/";
    static final String RESULTS = "results/";

    public static void main(String[] args) throws Exception {
        new File(RESULTS).mkdirs();

        partie3_mesures("securitynet_dataset.arff");
        partie3_mesures("weather.nominal.arff");
        partie4_bank();
        partie6_iris();
        partie7_epicerie();
        partie8_vote();
        partie84_irisBrut();
    }

    // ------------------------------------------------------------------ 3
    static void partie3_mesures(String fichier) throws Exception {
        titre("3. " + fichier + " - comparaison des mesures");
        Instances data = DataSource.read(DATA + fichier);

        // Indices de Apriori.TAGS_SELECTION : 0 = Confidence, 1 = Lift, 2 = Leverage, 3 = Conviction
        // Seuils repris de l'interface graphique (cf. figures 6 à 9 du rendu)
        String[] noms = {"Confidence", "Lift", "Leverage", "Conviction"};
        double[] minMetric = {0.6, 1.0, 0.01, 1.1};

        for (int m = 0; m < noms.length; m++) {
            Apriori apriori = new Apriori();
            apriori.setMetricType(new SelectedTag(m, Apriori.TAGS_SELECTION));
            apriori.setMinMetric(minMetric[m]);
            apriori.setNumRules(10);
            apriori.setLowerBoundMinSupport(0.1);

            apriori.buildAssociations(data);
            System.out.println("----- Mesure : " + noms[m] + " (min = " + minMetric[m] + ")");
            System.out.println(apriori);
        }
    }

    // ------------------------------------------------------------------ 4
    static void partie4_bank() throws Exception {
        titre("4. Bank Marketing (bank-full.csv) - prétraitement + Apriori");
        File csv = new File(DATA + "bank-full.csv");
        if (!csv.exists()) {
            System.out.println("data/bank-full.csv introuvable : partie 4 ignorée.");
            System.out.println("Le récupérer sur https://archive.ics.uci.edu/dataset/222/bank+marketing");
            return;
        }

        // 1. Conversion CSV -> Instances (le CSV UCI utilise ';' comme séparateur)
        CSVLoader loader = new CSVLoader();
        loader.setFieldSeparator(";");
        loader.setSource(csv);
        Instances data = loader.getDataSet();
        data.setClassIndex(data.numAttributes() - 1); // y
        System.out.println("Chargé : " + data.numInstances() + " exemples, " + data.numAttributes() + " attributs");

        // Apriori sur les données brutes : refusé car attributs numériques
        try {
            new Apriori().buildAssociations(data);
        } catch (Exception e) {
            System.out.println("Apriori sur données brutes -> " + e.getMessage());
        }

        // 2. Suppression de l'attribut "day"
        data = supprimer(data, "day");

        // 3. Discrétisation de age et balance en 10 intervalles -> bank1.arff
        data = discretiser(data, 10, "age", "balance");
        sauver(data, RESULTS + "bank1.arff");

        // 4. campaign : discrétisé avec le même filtre (10 intervalles), comme dans l'Explorer
        data = discretiser(data, 10, "campaign");

        // 5. Les numériques restants (duration, pdays, previous) sont aussi discrétisés,
        //    sinon Apriori refuse toujours le jeu de données
        data = discretiser(data, 10, "duration", "pdays", "previous");
        sauver(data, RESULTS + "bankModified.arff");

        // 6. Apriori sur bankModified (paramètres par défaut de l'Explorer : 10 règles, confiance 0,9)
        Apriori apriori = new Apriori();
        long t0 = System.nanoTime();
        apriori.buildAssociations(data);
        long t1 = System.nanoTime();
        System.out.println(apriori);
        System.out.printf("Temps Apriori : %.1f ms%n", (t1 - t0) / 1e6);
    }

    // ------------------------------------------------------------------ 6
    static void partie6_iris() throws Exception {
        titre("6. iris.arff discrétisé - Apriori (classe en conclusion) vs J48");
        Instances iris = DataSource.read(DATA + "iris.arff");
        iris.setClassIndex(iris.numAttributes() - 1);

        Discretize disc = new Discretize();
        disc.setBins(10); // 10 intervalles, comme dans l'Explorer
        disc.setInputFormat(iris);
        Instances irisDisc = Filter.useFilter(iris, disc);

        // Apriori en mode "class association rules" : seule la classe apparaît en conclusion
        Apriori apriori = new Apriori();
        apriori.setCar(true);
        apriori.setClassIndex(irisDisc.numAttributes()); // indice 1-based de la classe
        apriori.setNumRules(10);
        apriori.setMinMetric(0.9);
        apriori.buildAssociations(irisDisc);
        System.out.println(apriori);

        // J48 sur les mêmes données discrétisées
        j48(irisDisc);
    }

    // ------------------------------------------------------------------ 7
    static void partie7_epicerie() throws Exception {
        titre("7. promotions_noel.arff (épicerie bio) - Apriori selon le support");
        Instances data = DataSource.read(DATA + "promotions_noel.arff");

        for (double support : new double[] {0.75, 0.5, 0.25, 0.1}) {
            // Toutes les règles au support exact, pour les compter...
            int total = aprioriSupportExact(data, support, Integer.MAX_VALUE - 1).getAssociationRules().getNumRules();
            // ...puis les 10 meilleures, telles que l'Explorer les affiche
            System.out.println("----- Support " + support + " : " + total + " règles (confiance >= 0,9)");
            System.out.println(aprioriSupportExact(data, support, 10));
        }
    }

    /** Apriori en un seul cycle au support demandé (voir AprioriBenchmark pour l'explication). */
    static Apriori aprioriSupportExact(Instances data, double support, int numRules) throws Exception {
        double delta = 1.0 - support;
        Apriori apriori = new Apriori();
        apriori.setUpperBoundMinSupport(1.0);
        apriori.setDelta(delta);
        apriori.setLowerBoundMinSupport(1.0 - delta);
        apriori.setMinMetric(0.9);
        apriori.setNumRules(numRules);
        apriori.buildAssociations(data);
        return apriori;
    }

    // ------------------------------------------------------------------ 8.3
    static void partie8_vote() throws Exception {
        titre("8. vote.arff (UCI Congressional Voting Records) - exploration + Apriori");
        Instances data = DataSource.read(DATA + "vote.arff");
        data.setClassIndex(data.numAttributes() - 1);
        Attribute classe = data.classAttribute();

        // Exploration : pour chaque vote, répartition n / y par parti (équivalent de Preprocess)
        for (int i = 0; i < data.numAttributes() - 1; i++) {
            Attribute att = data.attribute(i);
            int[][] compte = new int[att.numValues()][classe.numValues()];
            for (Instance inst : data) {
                if (!inst.isMissing(i)) compte[(int) inst.value(i)][(int) inst.classValue()]++;
            }
            StringBuilder sb = new StringBuilder(String.format("%-40s manquants=%3d", att.name(),
                    data.attributeStats(i).missingCount));
            for (int v = 0; v < att.numValues(); v++) {
                sb.append(String.format("  %s : %3d dem / %3d rep", att.value(v), compte[v][0], compte[v][1]));
            }
            System.out.println(sb);
        }

        // Apriori avec les paramètres par défaut, puis en mode "classe en conclusion"
        Apriori apriori = new Apriori();
        apriori.buildAssociations(data);
        System.out.println(apriori);

        Apriori car = new Apriori();
        car.setCar(true);
        car.setClassIndex(data.numAttributes()); // indice 1-based de la classe
        car.buildAssociations(data);
        System.out.println(car);
    }

    // ------------------------------------------------------------------ 8.4
    static void partie84_irisBrut() throws Exception {
        titre("8.4 iris.arff non discrétisé (UCI) - J48");
        Instances iris = DataSource.read(DATA + "iris.arff");
        iris.setClassIndex(iris.numAttributes() - 1);
        j48(iris);
    }

    /** Arbre J48 (paramètres par défaut) + validation croisée 10 plis, comme l'onglet Classify. */
    static void j48(Instances data) throws Exception {
        J48 j48 = new J48();
        j48.buildClassifier(data);
        System.out.println(j48);

        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(new J48(), data, 10, new Random(1));
        System.out.println(eval.toSummaryString("=== J48, validation croisée 10 plis ===", false));
        System.out.println(eval.toMatrixString());
    }

    // ------------------------------------------------------------------ utilitaires
    static String indice(Instances data, String... noms) {
        StringBuilder sb = new StringBuilder();
        for (String nom : noms) {
            if (data.attribute(nom) == null) throw new IllegalArgumentException("Attribut inconnu : " + nom);
            if (sb.length() > 0) sb.append(',');
            sb.append(data.attribute(nom).index() + 1); // les filtres Weka sont 1-based
        }
        return sb.toString();
    }

    static Instances supprimer(Instances data, String... noms) throws Exception {
        Remove remove = new Remove();
        remove.setAttributeIndices(indice(data, noms));
        remove.setInputFormat(data);
        return Filter.useFilter(data, remove);
    }

    static Instances discretiser(Instances data, int bins, String... noms) throws Exception {
        Discretize disc = new Discretize();
        disc.setAttributeIndices(indice(data, noms));
        disc.setBins(bins);
        disc.setInputFormat(data);
        return Filter.useFilter(data, disc);
    }

    static void sauver(Instances data, String chemin) throws Exception {
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(chemin));
        saver.writeBatch();
        System.out.println("Sauvegardé : " + chemin);
    }

    static void titre(String t) {
        System.out.println("\n==================== " + t + " ====================\n");
    }
}
