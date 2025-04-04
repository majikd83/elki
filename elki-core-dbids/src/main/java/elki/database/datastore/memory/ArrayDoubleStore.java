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
package elki.database.datastore.memory;

import java.util.Arrays;

import elki.database.datastore.WritableDoubleDataStore;
import elki.database.ids.DBIDEnum;
import elki.database.ids.DBIDRef;

/**
 * A class to answer representation queries using the stored Array.
 *
 * @author Erich Schubert
 * @since 0.5.0
 *
 * @composed - - - elki.database.datastore.DBIDEnum
 */
public class ArrayDoubleStore implements WritableDoubleDataStore {
  /**
   * Data array
   */
  private double[] data;

  /**
   * Default value.
   */
  private double def;

  /**
   * DBID to index map
   */
  private DBIDEnum idmap;

  /**
   * Constructor.
   *
   * @param idmap ID map
   */
  public ArrayDoubleStore(DBIDEnum idmap) {
    this(idmap, Double.NaN);
  }

  /**
   * Constructor.
   *
   * @param idmap ID map
   * @param def Default value
   */
  public ArrayDoubleStore(DBIDEnum idmap, double def) {
    super();
    final int size = idmap.size();
    this.data = new double[size];
    if(def != 0) {
      Arrays.fill(this.data, def);
    }
    this.def = def;
    this.idmap = idmap;
  }

  @Override
  @Deprecated
  public Double get(DBIDRef id) {
    return Double.valueOf(data[idmap.index(id)]);
  }

  @Override
  @Deprecated
  public Double put(DBIDRef id, Double value) {
    final int off = idmap.index(id);
    double ret = data[off];
    data[off] = value.doubleValue();
    return Double.valueOf(ret);
  }

  @Override
  public double doubleValue(DBIDRef id) {
    return data[idmap.index(id)];
  }

  @Override
  public double putDouble(DBIDRef id, double value) {
    final int off = idmap.index(id);
    final double ret = data[off];
    data[off] = value;
    return ret;
  }

  @Override
  public double put(DBIDRef id, double value) {
    final int off = idmap.index(id);
    final double ret = data[off];
    data[off] = value;
    return ret;
  }

  @Override
  public void increment(DBIDRef id, double value) {
    data[idmap.index(id)] += value;
  }

  @Override
  public void clear() {
    Arrays.fill(data, def);
  }

  @Override
  public void destroy() {
    data = null;
    idmap = null;
  }

  @Override
  public void delete(DBIDRef id) {
    throw new UnsupportedOperationException("Can't delete from a static array storage.");
  }
}
