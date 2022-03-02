/*
 * This file is part of ELKI:
 * Environment for Developing KDD-Applications Supported by Index-Structures
 * 
 * Copyright (C) 2022
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
package elki.clustering.hierarchical;

import elki.database.ids.ArrayDBIDs;
import elki.database.ids.DBIDVar;

/**
 * Merge history representing a hierarchical clustering.
 *
 * @author Erich Schubert
 */
public class ClusterMergeHistory {
  /**
   * The initial DBIDs
   */
  protected ArrayDBIDs ids;

  /**
   * Distance to the parent object.
   */
  protected double[] distances;

  /**
   * Cluster size storage. May be uninitialized!
   */
  protected int[] sizes;

  /**
   * Store merge order (two cluster references per merge).
   */
  protected int[] merges;

  /**
   * Positions for layouting.
   */
  protected int[] positions;

  /**
   * Flag to indicate squared values
   */
  boolean isSquared;

  /**
   * Constructor.
   *
   * @param ids Initial object ids
   * @param merges Merge history 2*(N-1) values
   * @param distances Distances
   * @param sizes Cluster sizes
   * @param isSquared If distances are squared distances
   */
  public ClusterMergeHistory(ArrayDBIDs ids, int[] merges, double[] distances, int[] sizes, boolean isSquared) {
    this.ids = ids;
    this.merges = merges;
    this.distances = distances;
    this.sizes = sizes;
    this.isSquared = isSquared;
    assert distances.length << 1 == merges.length;
    assert sizes == null || distances.length == sizes.length;
  }

  /**
   * Access the i'th singleton via a variable.
   * <p>
   * Note: variables are used here to avoid reallocations.
   * <p>
   * TODO: use an array iterator instead?
   * 
   * @param i Index
   * @param var Variable
   * @return Value
   */
  public DBIDVar assignVar(int i, DBIDVar var) {
    assert i < ids.size() : "Can only assign the first N singleton clusters.";
    return ids.assignVar(i, var);
  }

  /**
   * Get the first partner of merge i.
   *
   * @param i Merge number
   * @return First cluster id to be merged
   */
  public int getMergeA(int i) {
    return merges[i << 1];
  }

  /**
   * Get the second partner of merge i.
   *
   * @param i Merge number
   * @return Second cluster id to be merged
   */
  public int getMergeB(int i) {
    return merges[(i << 1) + 1];
  }

  /**
   * Get merge distance / height
   *
   * @param i Merge index
   * @return Height
   */
  public double getMergeHeight(int i) {
    return distances[i];
  }

  /**
   * Get the size of the cluster merged in step i.
   *
   * @param i Step number
   * @return Cluster size
   */
  public int getSize(int i) {
    return sizes[i];
  }

  /**
   * Number of elements clustered.
   * 
   * @return Size of ids
   */
  public int size() {
    return ids.size();
  }

  /**
   * Number of merges, usually n-1.
   * <p>
   * Note: currently untested to use incomplete results.
   *
   * @return Number of merges
   */
  public int numMerges() {
    return merges.length >> 1;
  }

  /**
   * Indicate whether the stored values are squared values.
   *
   * @return boolean flag
   */
  public boolean isSquared() {
    return isSquared;
  }

  /**
   * Get the object ids in this clustering.
   * 
   * @return object ids
   */
  public ArrayDBIDs getDBIDs() {
    return ids;
  }

  /**
   * Get / compute the positions.
   *
   * @return Dendrogram positions
   */
  public int[] getPositions() {
    if(positions != null) {
      return positions; // Return cached.
    }
    final int n = ids.size();
    positions = new int[n];
    // Temporary positions for merged clusters:
    int[] cpos = new int[sizes.length];
    // Process in reverse merge order
    for(int i = sizes.length - 1; i >= 0; i--) {
      final int a = merges[i << 1], b = merges[(i << 1) + 1];
      final int c = cpos[i], sa = a < n ? 1 : sizes[a - n];
      if(a < n) {
        positions[a] = c;
      }
      else {
        cpos[a - n] = c;
      }
      if(b < n) {
        positions[b] = c + sa;
      }
      else {
        cpos[b - n] = c + sa;
      }
    }
    return positions;
  }
}
