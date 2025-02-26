use std::ops::Range;

use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_data_structures::graph;
use rustc_data_structures::graph::vec_graph::VecGraph;
use rustc_middle::ty::RegionVid;

use crate::RegionInferenceContext;
use crate::constraints::ConstraintSccIndex;

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

impl RegionInferenceContext<'_> {
    /// Compute the reverse SCC-based constraint graph (lazily).
    pub(super) fn compute_reverse_scc_graph(&mut self) {
        if self.rev_scc_graph.is_some() {
            return;
        }

        let graph = self.constraint_sccs.reverse();
        let mut paired_scc_regions = self
            .universal_regions()
            .universal_regions_iter()
            .map(|region| (self.constraint_sccs.scc(region), region))
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

        self.rev_scc_graph = Some(ReverseSccGraph { graph, scc_regions, universal_regions });
    }
}
