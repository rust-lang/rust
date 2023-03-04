#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]
use crate::constraints::ConstraintSccIndex;
use crate::RegionInferenceContext;
use itertools::Itertools;
use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_data_structures::graph::vec_graph::VecGraph;
use rustc_data_structures::graph::WithSuccessors;
use rustc_middle::ty::RegionVid;
use std::ops::Range;
use std::rc::Rc;

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
    pub(super) fn upper_bounds<'a>(
        &'a self,
        scc0: ConstraintSccIndex,
    ) -> impl Iterator<Item = RegionVid> + 'a {
        let mut duplicates = FxIndexSet::default();
        self.graph
            .depth_first_search(scc0)
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
    /// Compute and return the reverse SCC-based constraint graph (lazily).
    pub(super) fn reverse_scc_graph(&mut self) -> Rc<ReverseSccGraph> {
        if let Some(g) = &self.rev_scc_graph {
            return g.clone();
        }

        let graph = self.constraint_sccs.reverse();
        let mut paired_scc_regions = self
            .universal_regions
            .universal_regions()
            .map(|region| (self.constraint_sccs.scc(region), region))
            .collect_vec();
        paired_scc_regions.sort();
        let universal_regions = paired_scc_regions.iter().map(|&(_, region)| region).collect();

        let mut scc_regions = FxIndexMap::default();
        let mut start = 0;
        for (scc, group) in &paired_scc_regions.into_iter().group_by(|(scc, _)| *scc) {
            let group_size = group.count();
            scc_regions.insert(scc, start..start + group_size);
            start += group_size;
        }

        let rev_graph = Rc::new(ReverseSccGraph { graph, scc_regions, universal_regions });
        self.rev_scc_graph = Some(rev_graph.clone());
        rev_graph
    }
}
