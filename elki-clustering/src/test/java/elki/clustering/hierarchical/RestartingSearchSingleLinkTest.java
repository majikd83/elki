/*
 * This file is part of ELKI:
 * Environment for Developing KDD-Applications Supported by Index-Structures
 *
 * Copyright (C) 2024
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

import org.junit.Test;

import elki.Algorithm;
import elki.clustering.AbstractClusterAlgorithmTest;
import elki.clustering.hierarchical.extraction.CutDendrogramByNumberOfClusters;
import elki.data.Clustering;
import elki.database.Database;
import elki.utilities.ELKIBuilder;

/**
 * Perform agglomerative hierarchical clustering, using the index accelerated
 * restarting search single-link (RSSL) algorithm.
 *
 * @author Erich Schubert
 */
public class RestartingSearchSingleLinkTest extends AbstractClusterAlgorithmTest {
  @Test
  public void testSingleLink() {
    Database db = makeSimpleDatabase(UNITTEST + "single-link-effect.ascii", 638);
    Clustering<?> clustering = new ELKIBuilder<>(CutDendrogramByNumberOfClusters.class) //
        .with(CutDendrogramByNumberOfClusters.Par.MINCLUSTERS_ID, 3) //
        .with(Algorithm.Utils.ALGORITHM_ID, RestartingSearchSingleLink.class) //
        .build().autorun(db);
    assertClusterSizes(clustering, new int[] { 9, 200, 429 });
    assertFMeasure(db, clustering, 0.6829722);
  }
}
