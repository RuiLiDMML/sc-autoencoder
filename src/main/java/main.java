import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.datasets.fetchers.IrisDataFetcher;
import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.deeplearning4j.nn.api.LayerFactory;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class main {
  public static void main(String[] args) {
    IrisDataFetcher fetcher = new IrisDataFetcher();
    fetcher.fetch(150);
    DataSet d = fetcher.next();
    d.normalizeZeroMeanZeroUnitVariance();
    RandomGenerator g = new MersenneTwister(123);

    NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
            .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
            .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
            .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
            .learningRate(1e-1f)
            .nIn(d.numInputs())
            .rng(g)
            .nOut(3).build();

    LayerFactory layerFactory = LayerFactories.getFactory(RBM.class);
    RBM r = layerFactory.create(conf);
    r.fit(d.getFeatureMatrix());
  }
}
