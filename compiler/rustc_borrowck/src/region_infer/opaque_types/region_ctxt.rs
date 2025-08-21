use std::rc::Rc;

use rustc_data_structures::frozen::Frozen;
use rustc_index::IndexVec;
use rustc_infer::infer::NllRegionVariableOrigin;
use rustc_middle::ty::{RegionVid, UniverseIndex};
use rustc_mir_dataflow::points::DenseLocationMap;

use crate::BorrowckInferCtxt;
use crate::constraints::ConstraintSccIndex;
use crate::handle_placeholders::{SccAnnotations, region_definitions};
use crate::region_infer::reverse_sccs::ReverseSccGraph;
use crate::region_infer::values::RegionValues;
use crate::region_infer::{ConstraintSccs, RegionDefinition, RegionTracker, Representative};
use crate::type_check::MirTypeckRegionConstraints;
use crate::type_check::free_region_relations::UniversalRegionRelations;
use crate::universal_regions::UniversalRegions;

/// A slimmed down version of [crate::region_infer::RegionInferenceContext] used
/// only by opaque type handling.
pub(super) struct RegionCtxt<'a, 'tcx> {
    pub(super) infcx: &'a BorrowckInferCtxt<'tcx>,
    pub(super) definitions: Frozen<IndexVec<RegionVid, RegionDefinition<'tcx>>>,
    pub(super) universal_region_relations: &'a UniversalRegionRelations<'tcx>,
    pub(super) constraint_sccs: ConstraintSccs,
    pub(super) scc_annotations: IndexVec<ConstraintSccIndex, RegionTracker>,
    pub(super) rev_scc_graph: ReverseSccGraph,
    pub(super) scc_values: RegionValues<ConstraintSccIndex>,
}

impl<'a, 'tcx> RegionCtxt<'a, 'tcx> {
    /// Creates a new `RegionCtxt` used to compute defining opaque type uses.
    ///
    /// This does not yet propagate region values. This is instead done lazily
    /// when applying member constraints.
    pub(super) fn new(
        infcx: &'a BorrowckInferCtxt<'tcx>,
        universal_region_relations: &'a Frozen<UniversalRegionRelations<'tcx>>,
        location_map: Rc<DenseLocationMap>,
        constraints: &MirTypeckRegionConstraints<'tcx>,
    ) -> RegionCtxt<'a, 'tcx> {
        let universal_regions = &universal_region_relations.universal_regions;
        let (definitions, _has_placeholders) = region_definitions(infcx, universal_regions);
        let mut scc_annotations = SccAnnotations::init(&definitions);
        let constraint_sccs = ConstraintSccs::new_with_annotation(
            &constraints
                .outlives_constraints
                .graph(definitions.len())
                .region_graph(&constraints.outlives_constraints, universal_regions.fr_static),
            &mut scc_annotations,
        );
        let scc_annotations = scc_annotations.scc_to_annotation;

        // Unlike the `RegionInferenceContext`, we only care about free regions
        // and fully ignore liveness and placeholders.
        let placeholder_indices = Default::default();
        let mut scc_values =
            RegionValues::new(location_map, universal_regions.len(), placeholder_indices);
        for variable in definitions.indices() {
            let scc = constraint_sccs.scc(variable);
            match definitions[variable].origin {
                NllRegionVariableOrigin::FreeRegion => {
                    scc_values.add_element(scc, variable);
                }
                _ => {}
            }
        }

        let rev_scc_graph = ReverseSccGraph::compute(&constraint_sccs, universal_regions);
        RegionCtxt {
            infcx,
            definitions,
            universal_region_relations,
            constraint_sccs,
            scc_annotations,
            rev_scc_graph,
            scc_values,
        }
    }

    pub(super) fn representative(&self, vid: RegionVid) -> Representative {
        let scc = self.constraint_sccs.scc(vid);
        self.scc_annotations[scc].representative
    }

    pub(crate) fn max_placeholder_universe_reached(
        &self,
        scc: ConstraintSccIndex,
    ) -> UniverseIndex {
        self.scc_annotations[scc].max_placeholder_universe_reached()
    }

    pub(super) fn universal_regions(&self) -> &UniversalRegions<'tcx> {
        &self.universal_region_relations.universal_regions
    }

    pub(super) fn eval_equal(&self, r1_vid: RegionVid, r2_vid: RegionVid) -> bool {
        let r1 = self.constraint_sccs.scc(r1_vid);
        let r2 = self.constraint_sccs.scc(r2_vid);

        if r1 == r2 {
            return true;
        }

        let universal_outlives = |sub, sup| {
            self.scc_values.universal_regions_outlived_by(sub).all(|r1| {
                self.scc_values
                    .universal_regions_outlived_by(sup)
                    .any(|r2| self.universal_region_relations.outlives(r2, r1))
            })
        };
        universal_outlives(r1, r2) && universal_outlives(r2, r1)
    }
}
