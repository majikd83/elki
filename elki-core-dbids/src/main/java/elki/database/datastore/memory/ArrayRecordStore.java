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

import elki.database.datastore.WritableDataStore;
import elki.database.datastore.WritableRecordStore;
import elki.database.ids.DBIDEnum;
import elki.database.ids.DBIDRef;

/**
 * A class to answer representation queries using the stored Array.
 *
 * @author Erich Schubert
 * @since 0.4.0
 *
 * @composed - - - DBIDEnum
 * @navhas - projectsTo - ArrayRecordStore.StorageAccessor
 */
public class ArrayRecordStore implements WritableRecordStore {
  /**
   * Data array
   */
  private final Object[][] data;

  /**
   * DBID to index map
   */
  private final DBIDEnum idmap;

  /**
   * Constructor with existing data
   *
   * @param data Existing data
   * @param idmap Map for array offsets
   */
  public ArrayRecordStore(Object[][] data, DBIDEnum idmap) {
    super();
    this.data = data;
    this.idmap = idmap;
  }

  @Override
  public <T> WritableDataStore<T> getStorage(int col, Class<? super T> datatype) {
    // TODO: add type checking safety?
    return new StorageAccessor<>(col);
  }

  /**
   * Actual getter
   *
   * @param id Database ID
   * @param index column index
   * @param <T> data type (unchecked cast)
   * @return current value
   */
  @SuppressWarnings("unchecked")
  protected <T> T get(DBIDRef id, int index) {
    return (T) data[idmap.index(id)][index];
  }

  /**
   * Actual setter
   *
   * @param id Database ID
   * @param index column index
   * @param value New value
   * @param <T> data type (unchecked cast)
   * @return old value
   */
  @SuppressWarnings("unchecked")
  protected <T> T set(DBIDRef id, int index, T value) {
    T ret = (T) data[idmap.index(id)][index];
    data[idmap.index(id)][index] = value;
    return ret;
  }

  /**
   * Access a single record in the given data.
   *
   * @author Erich Schubert
   *
   * @param <T> Object data type to access
   */
  protected class StorageAccessor<T> implements WritableDataStore<T> {
    /**
     * Representation index.
     */
    private final int index;

    /**
     * Constructor.
     *
     * @param index In-record index
     */
    protected StorageAccessor(int index) {
      super();
      this.index = index;
    }

    @SuppressWarnings("unchecked")
    @Override
    public T get(DBIDRef id) {
      return (T) ArrayRecordStore.this.get(id, index);
    }

    @Override
    public T put(DBIDRef id, T value) {
      return ArrayRecordStore.this.set(id, index, value);
    }

    @Override
    public void destroy() {
      throw new UnsupportedOperationException("ArrayStore record columns cannot (yet) be destroyed.");
    }

    @Override
    public void delete(DBIDRef id) {
      put(id, null);
    }

    @Override
    public void clear() {
      throw new UnsupportedOperationException("ArrayStore record columns cannot (yet) be cleared.");
    }
  }

  @Override
  public boolean remove(DBIDRef id) {
    throw new UnsupportedOperationException("ArrayStore records cannot be removed.");
  }
}
