use std::ops::Range;

use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_data_structures::graph;
use rustc_data_structures::graph::vec_graph::VecGraph;
use rustc_middle::ty::RegionVid;

use crate::constraints::ConstraintSccIndex;
use crate::region_infer::ConstraintSccs;
use crate::universal_regions::UniversalRegions;

pub(crate) struct ReverseSccGraph {
    graph: VecGraph<ConstraintSccIndex>,
    /// For each SCC, the range of `universal_regions` that use that SCC as
    /// their value.
    scc_regions: FxIndexMap<ConstraintSccIndex, Range<usize>>,
    /// All of the universal regions, in grouped so that `scc_regions` can
    /// index into here.
    universal_regions: Vec<RegionVid>,
}

impl ReverseSccGraph {
    pub(super) fn compute(
        constraint_sccs: &ConstraintSccs,
        universal_regions: &UniversalRegions<'_>,
    ) -> Self {
        let graph = constraint_sccs.reverse();
        let mut paired_scc_regions = universal_regions
            .universal_regions_iter()
            .map(|region| (constraint_sccs.scc(region), region))
            .collect::<Vec<_>>();
        paired_scc_regions.sort();
        let universal_regions = paired_scc_regions.iter().map(|&(_, region)| region).collect();

        let mut scc_regions = FxIndexMap::default();
        let mut start = 0;
        for chunk in paired_scc_regions.chunk_by(|&(scc1, _), &(scc2, _)| scc1 == scc2) {
            let (scc, _) = chunk[0];

            scc_regions.insert(scc, start..start + chunk.len());
            start += chunk.len();
        }
        ReverseSccGraph { graph, scc_regions, universal_regions }
    }

    /// Find all universal regions that are required to outlive the given SCC.
    pub(super) fn upper_bounds(&self, scc0: ConstraintSccIndex) -> impl Iterator<Item = RegionVid> {
        let mut duplicates = FxIndexSet::default();
        graph::depth_first_search(&self.graph, scc0)
            .flat_map(move |scc1| {
                self.scc_regions
                    .get(&scc1)
                    .map_or(&[][..], |range| &self.universal_regions[range.clone()])
            })
            .copied()
            .filter(move |r| duplicates.insert(*r))
    }
}
