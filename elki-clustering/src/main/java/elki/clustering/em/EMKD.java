/*
 * This file is part of ELKI:
 * Environment for Developing KDD-Applications Supported by Index-Structures
 * 
 * Copyright (C) 2020
 * ELKI Development Team
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
package elki.clustering.em;


import static elki.math.linearalgebra.VMath.times;
import static elki.math.linearalgebra.VMath.minus;
import static elki.math.linearalgebra.VMath.plus;
import static elki.math.linearalgebra.VMath.argmax;
import static elki.math.linearalgebra.VMath.timesTranspose;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import elki.clustering.ClusteringAlgorithm;
import elki.clustering.kmeans.KMeans;
import elki.data.Cluster;
import elki.data.Clustering;
import elki.data.DoubleVector;
import elki.data.NumberVector;
import elki.data.model.MeanModel;
import elki.data.type.SimpleTypeInformation;
import elki.data.type.TypeInformation;
import elki.data.type.TypeUtil;
import elki.database.datastore.DataStoreFactory;
import elki.database.datastore.DataStoreUtil;
import elki.database.datastore.WritableDataStore;
import elki.database.ids.*;
import elki.database.relation.MaterializedRelation;
import elki.database.relation.Relation;
import elki.logging.Logging;
import elki.result.Metadata;
import elki.utilities.optionhandling.OptionID;
import elki.utilities.optionhandling.Parameterizer;
import elki.utilities.optionhandling.constraints.CommonConstraints;
import elki.utilities.optionhandling.parameterization.Parameterization;
import elki.utilities.optionhandling.parameters.DoubleParameter;
import elki.utilities.optionhandling.parameters.IntParameter;
import elki.utilities.optionhandling.parameters.ObjectParameter;

import net.jafama.FastMath;
import sun.rmi.runtime.Log;

public class EMKD<M extends MeanModel> implements ClusteringAlgorithm<Clustering<M>> {
  //
  /**
   * Factory for producing the initial cluster model.
   */
  private EMClusterModelFactory<NumberVector, M> mfactory;

  private static final Logging LOG = Logging.getLogger(EMKD.class);
  
  private static final double MIN_LOGLIKELIHOOD = -100000;

  /**
   * Retain soft assignments.
   */
  private boolean soft;

  /**
   * Delta parameter
   */
  private double delta;

  /**
   * Soft assignment result type.
   */
  public static final SimpleTypeInformation<double[]> SOFT_TYPE = new SimpleTypeInformation<>(double[].class);
  
  private int k = 3;

  // currently not used
  private int miniter;

  private int maxiter;
  
  protected ArrayModifiableDBIDs sorted;


  public EMKD(int k, double delta, EMClusterModelFactory<NumberVector, M> mfactory, int miniter, int maxiter, boolean soft) {
    this.k = k;
    this.delta = delta;
    this.mfactory = mfactory;
    this.miniter = miniter;
    this.maxiter = maxiter;
    this.soft = soft;
    }
  
  /**
   * I took the run method from EM.java and I am rewriting it to work on KD trees.
   * @param relation
   * @return
   */
  public Clustering<M> run(Relation<? extends NumberVector> relation) {
    
    double[] test = {2.,1.};
    double[][] test2 = timesTranspose(test, test);
    for(int i = 0; i < test2.length; i++) {
      LOG.verbose(Arrays.toString(test2[i]));
    }
    if(relation.size() == 0) {
      throw new IllegalArgumentException("database empty: must contain elements");
    }
    //build kd-tree
    sorted = DBIDUtil.newArray(relation.getDBIDs());
    double[] dimwidth = analyseDimWidth(relation);
    mrkdNode tree = new mrkdNode(relation, sorted.iter(), 0, sorted.size(), dimwidth);
    
    LOG.verbose("root:");
    double[][] rootcov = tree.cov;
    LOG.verbose("center:");
    LOG.verbose(Arrays.toString(tree.center));
    LOG.verbose("covariance:");
    for(int i = 0; i < rootcov.length; i++) {
      LOG.verbose(Arrays.toString(rootcov[i]));
    }
    
    // initial models
    ArrayList<? extends EMClusterModel<NumberVector, M>> models = new ArrayList<EMClusterModel<NumberVector,M>>(mfactory.buildInitialModels(relation, k));
    WritableDataStore<double[]> probClusterIGivenX = DataStoreUtil.makeStorage(relation.getDBIDs(), DataStoreFactory.HINT_HOT | DataStoreFactory.HINT_SORTED, double[].class);
    double loglikelihood = assignProbabilitiesToInstances(relation, models, probClusterIGivenX);
    //DoubleStatistic likestat = new DoubleStatistic(this.getClass().getName() + ".loglikelihood");
    //LOG.statistics(likestat.setDouble(loglikelihood));

    // iteration unless no change
    int it = 0;
    //double bestloglikelihood = loglikelihood; // For detecting instabilities.
    for(; it < maxiter || maxiter < 0; it++) {
      
      // recalculate probabilities
      ClusterData[] newstats = makeStats(tree, relation.size(), models);
      
      updateClusters(newstats, models, relation.size());
      // here i need to finish the makeStats and then apply them
      
      //LOG.statistics(likestat.setDouble(loglikelihood));
      //if(loglikelihood - bestloglikelihood > delta) {
        //lastimprovement = it;
        //bestloglikelihood = loglikelihood;
      //}
      //if(it >= miniter && (Math.abs(loglikelihood - oldloglikelihood) <= delta || lastimprovement < it >> 1)) {
      //  break;
      //}
    }
    //LOG.statistics(new LongStatistic(KEY + ".iterations", it));

    // fill result with clusters and models
    List<ModifiableDBIDs> hardClusters = new ArrayList<>(k);
    for(int i = 0; i < k; i++) {
      hardClusters.add(DBIDUtil.newArray());
    }

    loglikelihood = assignProbabilitiesToInstances(relation, models, probClusterIGivenX);
    
    // provide a hard clustering
    // add each point to cluster of max density
    for(DBIDIter iditer = relation.iterDBIDs(); iditer.valid(); iditer.advance()) {
      hardClusters.get(argmax(probClusterIGivenX.get(iditer))).add(iditer);
    }
    Clustering<M> result = new Clustering<>();
    Metadata.of(result).setLongName("EM Clustering");
    // provide models within the result
    for(int i = 0; i < k; i++) {
      result.addToplevelCluster(new Cluster<>(hardClusters.get(i), models.get(i).finalizeCluster()));
    }
    if(soft) {
      Metadata.hierarchyOf(result).addChild(new MaterializedRelation<>("EM Cluster Probabilities", SOFT_TYPE, relation.getDBIDs(), probClusterIGivenX));
    }
    else {
      probClusterIGivenX.destroy();
    }
    return result;
  }

  private void updateClusters(ClusterData[] newstats, ArrayList<? extends EMClusterModel<NumberVector, M>> models, int size) {
    for(int i = 0; i < k; i++) {
      models.get(i).setWeight(FastMath.exp(newstats[i].logApriori_sw - FastMath.log(size)));
      // this might need to change to a set / get method and not a tvar method bcause models might apply changes during set
      // this doesnt affect it currently as there are no changes made so far
      double[] tcenter = times(newstats[i].center_swx, 1./FastMath.exp(newstats[i].logApriori_sw));
      
      LOG.verbose(Arrays.toString(tcenter));
      models.get(i).setCenter(tcenter);
      double[][] tcov = times(newstats[i].cov_swxx,1./FastMath.exp(newstats[i].logApriori_sw))/*minus(, timesTranspose(tcenter, tcenter))*/;
      LOG.verbose("cov");
      for(int j = 0; j < tcov.length;j++) {
        LOG.verbose(Arrays.toString(tcov[j]));
      }
      models.get(i).updateCovariance(tcov);
    }
  }

  public ClusterData[] makeStats(mrkdNode node, int numP, ArrayList<? extends EMClusterModel<NumberVector, M>> models) {
    if(node.isLeaf /*||node.checkStoppingCondition(numP)*/) {
      ClusterData[] res = new ClusterData[k];
      // logarithmic probabilities of clusters in this node
      double[] logProb = new double[k];
      
      for(int i = 0; i < k; i++) {
        logProb[i] = models.get(i).estimateLogDensity(DoubleVector.copy(node.center));
      }
      
      double logDenSum = logSumExp(logProb);
      logProb = minus(logProb,logDenSum);
      
      for(int c = 0; c < logProb.length; c++) {
        double logAPrio = logProb[c] + FastMath.log(node.size);
        double[] center = times(times(node.center, FastMath.exp(logProb[c])),node.size);
        double[][] cov = new double[node.center.length][];
        for(int dim = 0; dim < cov.length; dim++) {//maybe exchange this? it is maybe at the cholsky, but i dont actually think so
          cov[dim] = times(times(node.cov[dim], FastMath.exp(logProb[c])),node.size);
        }
        res[c] = new ClusterData(logAPrio,center,cov);
        // ~~ThisStep is for the approximation part currently left out
        //weightedAppliedThisStep[c] += FastMath.exp(logAPrio);
      }
      //pointsWorkedThisStep += node.size;
      return res;
    }else {
      ClusterData[] lData = makeStats(node.leftChild, numP, models);
      ClusterData[] rData = makeStats(node.rightChild, numP, models);
      for(int c = 0; c < lData.length; c++) {
        lData[c].combine(rData[c]);
      }
      return lData;
    }
  }
/**
 * helper method to retrieve the widths of all data in all dimensions
 * @param relation
 * @return
 */
  private double[] analyseDimWidth(Relation<? extends NumberVector> relation) {
    DBIDIter it = relation.iterDBIDs();
    int d = relation.get(it).getDimensionality();
    double[][] arr = new double[d][2];
    for(int i = 0; i < d; i++) {
      arr[i][0] = Double.MAX_VALUE;
    }
    double[] result = new double[d];
    for(; it.valid(); it.advance()) {
      NumberVector x = relation.get(it);
      for(int i = 0; i < d; i++) {
        double t = x.doubleValue(i);
        arr[i][0] = arr[i][0] < t? arr[i][0]:t;
        arr[i][1] = arr[i][1] > t? arr[i][1]:t;
      }
    }
    for(int i = 0; i < d; i++) {
      result[i] = arr[i][1] - arr[i][0];
    }
    return result;
  }
  

  /**
   * Assigns the current probability values to the instances in the database and
   * compute the expectation value of the current mixture of distributions.
   * <p>
   * Computed as the sum of the logarithms of the prior probability of each
   * instance.
   * 
   * @param relation the database used for assignment to instances
   * @param models Cluster models
   * @param probClusterIGivenX Output storage for cluster probabilities
   * @param <O> Object type
   * @return the expectation value of the current mixture of distributions
   */
  public static <O> double assignProbabilitiesToInstances(Relation<? extends O> relation, List<? extends EMClusterModel<O, ?>> models, WritableDataStore<double[]> probClusterIGivenX) {
    final int k = models.size();
    double emSum = 0.;

    for(DBIDIter iditer = relation.iterDBIDs(); iditer.valid(); iditer.advance()) {
      O vec = relation.get(iditer);
      double[] probs = new double[k];
      for(int i = 0; i < k; i++) {
        double v = models.get(i).estimateLogDensity(vec);
        probs[i] = v > MIN_LOGLIKELIHOOD ? v : MIN_LOGLIKELIHOOD;
      }
      final double logP = logSumExp(probs);
      for(int i = 0; i < k; i++) {
        probs[i] = FastMath.exp(probs[i] - logP);
      }
      probClusterIGivenX.put(iditer, probs);
      emSum += logP;
    }
    return emSum / relation.size();
  }
  
  /**
   * Compute log(sum(exp(x_i)), with attention to numerical issues.
   * 
   * @param x Input
   * @return Result
   */
  private static double logSumExp(double[] x) {
    double max = x[0];
    for(int i = 1; i < x.length; i++) {
      final double v = x[i];
      max = v > max ? v : max;
    }
    final double cutoff = max - 35.350506209; // log_e(2**51)
    double acc = 0.;
    for(int i = 0; i < x.length; i++) {
      final double v = x[i];
      if(v > cutoff) {
        acc += v < max ? FastMath.exp(v - max) : 1.;
      }
    }
    return acc > 1. ? (max + FastMath.log(acc)) : max;
  }
  
  
  // while calculation its in log
  static class ClusterData {
    double logApriori_sw;

    double[] center_swx;

    double[][] cov_swxx;

    public ClusterData(double logApriori, double[] center, double[][] cov) {
      this.logApriori_sw = logApriori;
      this.center_swx = center;
      this.cov_swxx = cov;
    }

    void combine(ClusterData other) {
      this.logApriori_sw = FastMath.log(FastMath.exp(other.logApriori_sw)+FastMath.exp(this.logApriori_sw));
      
      this.center_swx = plus(this.center_swx,other.center_swx);
      
      this.cov_swxx = plus(this.cov_swxx, other.cov_swxx);
    }
  }
  
  class mrkdNode {
    mrkdNode leftChild, rightChild;
    int leftBorder;
    int rightBorder;
    boolean isLeaf = false;

    double[] center;

    int size;

    double[][] cov;

    double[][] hyperboundingbox;

    public mrkdNode(Relation<? extends NumberVector> relation, DBIDArrayIter iter, int left, int right, double[] dimwidth) {
      /**
       * other kdtrees seem to look at [left, right[
       */
      int dim = relation.get(iter).toArray().length;
      
      leftBorder = left;
      rightBorder = right;
      center = new double[dim];
      cov = new double[dim][dim];
      hyperboundingbox = new double[3][dim];
      // size
      size = right - left;
      iter.seek(left);
      hyperboundingbox[0] = relation.get(iter).toArray();
      hyperboundingbox[1] = relation.get(iter).toArray();

      for(int i = 0; i < size; i++) {
        NumberVector vector = relation.get(iter);
        for(int d = 0; d < dim; d++) {
          double value = vector.doubleValue(d);
          // bounding box
          if(value < hyperboundingbox[0][d])
            hyperboundingbox[0][d] = value;
          else if(value > hyperboundingbox[1][d])
            hyperboundingbox[1][d] = value;
          // center
          center[d] += value;
        }
        if(iter.valid())
          iter.advance();
      }

      for(int i = 0; i < dim; i++) {
        center[i] = center[i] / size;
        hyperboundingbox[2][i] = FastMath.abs(hyperboundingbox[1][i] - hyperboundingbox[0][i]);
      }
      iter.seek(left);

      // cov - is this "textbook"?
      // here lies a problem. It seems that this makes it impossible to implement circle and stuff.
      // to know this though, i need to take a second look at the paper and what gets calculated where.
      for(int i = 0; i < size; i++) {
        NumberVector vector = relation.get(iter);
        for(int d1 = 0; d1 < dim; d1++) {
          double value1 = vector.doubleValue(d1);
          for(int d2 = 0; d2 < dim; d2++) {
            double value2 = vector.doubleValue(d2);
            cov[d1][d2] += (value1 - center[d1]) * (value2 - center[d2]);
          }
        }
      }
      for(int d1 = 0; d1 < dim; d1++) {
        for(int d2 = 0; d2 < dim; d2++) {
          cov[d1][d2] = cov[d1][d2] / (double)size;
        }
      }

      final int splitDim = argmax(hyperboundingbox[2]);
      if(hyperboundingbox[2][splitDim] < .1 * dimwidth[splitDim]) {
        isLeaf = true;
//        LOG.verbose("following has " + size + " Points:");
//        for(int i = 0;  i < dim; i++) {
//          LOG.verbose(Arrays.toString(cov[i]));
//        } 
        return;
      }

      double splitpoint = center[splitDim];
      int l = left, r = right - 1;
      while(true) {
        while(l <= r && relation.get(iter.seek(l)).doubleValue(splitDim) <= splitpoint) {
          ++l;
        }
        while(l <= r && relation.get(iter.seek(r)).doubleValue(splitDim) >= splitpoint) {
          --r;
        }
        if(l >= r) {
          break;
        }
        sorted.swap(l++, r--);
      }
      assert relation.get(iter.seek(r)).doubleValue(splitDim) <= splitpoint : relation.get(iter.seek(r)).doubleValue(splitDim) + " not less than " + splitpoint;
      ++r;
      if(r == right) { // Duplicate points!
        isLeaf = true;
        return;
      }
      leftChild = new mrkdNode(relation, iter, left, r,dimwidth);
      rightChild = new mrkdNode(relation, iter, r, right,dimwidth);
    }

    /*
    public boolean checkStoppingCondition(int numP) {
      DBIDArrayIter  it = sorted.iter().seek(leftBorder);
      double[][] mahaDists = new double[k][2];
      // mahaDists[c][0 = min; 1 = max]
      for(int c = 0; c < mahaDists.length; c++) {
        mahaDists[c][0] = Integer.MAX_VALUE;
      }
      
      for(int i = 0; i< size; i++) {
        double[] tdis = currentMahalanobis.get(it);
        for(int c = 0; c < tdis.length; c++) {
          mahaDists[c][0] = mahaDists[c][0] < tdis[c] ? mahaDists[c][0] : tdis[c];
          mahaDists[c][1] = mahaDists[c][1] > tdis[c] ? mahaDists[c][1] : tdis[c];
        }
        if(it.valid())
          it.advance();
      }
      //note that from here mahaDists describes logdensity and is [c][0 = max; 1 = min]
      double maxsum = 0;
      double minsum = 0;
      for(int c = 0; c < mahaDists.length; c++) {
        mahaDists[c][0] =  -.5 * mahaDists[c][0] + classes[c].logNormDet;
        mahaDists[c][1] =  -.5 * mahaDists[c][1] + classes[c].logNormDet;
        maxsum += FastMath.exp(mahaDists[c][0]+ classes[c].data.logApriori_sw) ;
        minsum += FastMath.exp(mahaDists[c][1]+ classes[c].data.logApriori_sw);
      }
      //wmin = amin*p / amin*p + sum_other(amax * p)
      //i guess that the other formular is "similar" it is analog
      // from here on mahaDists describes wmax, wmin
      for(int c = 0; c < mahaDists.length; c++) {
        mahaDists[c][0] =  FastMath.exp(mahaDists[c][0]+ classes[c].data.logApriori_sw) 
            / (minsum-FastMath.exp(mahaDists[c][1]+ classes[c].data.logApriori_sw) + FastMath.exp(mahaDists[c][0]+ classes[c].data.logApriori_sw));
        mahaDists[c][1] =  FastMath.exp(mahaDists[c][1]+ classes[c].data.logApriori_sw) 
            / (maxsum-FastMath.exp(mahaDists[c][0]+ classes[c].data.logApriori_sw) + FastMath.exp(mahaDists[c][1]+ classes[c].data.logApriori_sw));
        // check dis
        double d = mahaDists[c][0]- mahaDists[c][1];
        if(d > weightedAppliedThisStep[c]/pointsWorkedThisStep + (numP - pointsWorkedThisStep)*mahaDists[c][1]*tau) {
          return false;
        }
      }
      return true;
    }
    */
  }
  /**
   * Parameterization class.
   * 
   * @author Erich Schubert
   */
  public static class Par<M extends MeanModel> implements Parameterizer {
    /**
     * Parameter to specify the number of clusters to find, must be an integer
     * greater than 0.
     */
    public static final OptionID K_ID = new OptionID("emkd.k", "The number of clusters to find.");

    /**
     * Parameter to specify the termination criterion for maximization of E(M):
     * E(M) - E(M') &lt; em.delta, must be a double equal to or greater than 0.
     */
    public static final OptionID DELTA_ID = new OptionID("emkd.delta", //
        "The termination criterion for maximization of E(M): E(M) - E(M') < em.delta");

    /**
     * Parameter to specify the EM cluster models to use.
     */
    public static final OptionID INIT_ID = new OptionID("emkd.model", "Model factory.");

    /**
     * Parameter to specify a minimum number of iterations
     */
    public static final OptionID MINITER_ID = new OptionID("emkd.miniter", "Minimum number of iterations.");

    /**
     * Parameter to specify the MAP prior
     */
    public static final OptionID PRIOR_ID = new OptionID("emkd.map.prior", "Regularization factor for MAP estimation.");

    /**
     * Number of clusters.
     */
    protected int k;

    /**
     * Stopping threshold
     */
    protected double delta;

    /**
     * Initialization method
     */
    protected EMClusterModelFactory<NumberVector, M> initializer;

    /**
     * Minimum number of iterations.
     */
    protected int miniter = 1;

    /**
     * Maximum number of iterations.
     */
    protected int maxiter = -1;

    @Override
    public void configure(Parameterization config) {
      new IntParameter(K_ID) //
          .addConstraint(CommonConstraints.GREATER_EQUAL_ONE_INT) //
          .grab(config, x -> k = x);
      new ObjectParameter<EMClusterModelFactory<NumberVector, M>>(INIT_ID, EMClusterModelFactory.class, MultivariateGaussianModelFactory.class) //
          .grab(config, x -> initializer = x);
      new DoubleParameter(DELTA_ID, 1e-7)//
          .addConstraint(CommonConstraints.GREATER_EQUAL_ZERO_DOUBLE) //
          .grab(config, x -> delta = x);
      new IntParameter(MINITER_ID)//
          .addConstraint(CommonConstraints.GREATER_EQUAL_ZERO_INT) //
          .setOptional(true) //
          .grab(config, x -> miniter = x);
      new IntParameter(KMeans.MAXITER_ID)//
          .addConstraint(CommonConstraints.GREATER_EQUAL_ZERO_INT) //
          .setOptional(true) //
          .grab(config, x -> maxiter = x);
    }

    @Override
    public EMKD<M> make() {
      return new EMKD<>(k, delta, initializer, miniter , maxiter, false);
    }
  }
  @Override
  public TypeInformation[] getInputTypeRestriction() {
    return TypeUtil.array(TypeUtil.NUMBER_VECTOR_FIELD);
  }
}
