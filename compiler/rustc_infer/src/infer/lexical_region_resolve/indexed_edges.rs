use rustc_index::IndexVec;
use rustc_type_ir::RegionVid;

use super::Data;
use crate::infer::SubregionOrigin;
use crate::infer::region_constraints::{Constraint, ConstraintKind};

/// Selects either out-edges or in-edges for [`IndexedConstraintEdges::adjacent_edges`].
#[derive(Clone, Copy, Debug)]
pub(super) enum EdgeDirection {
    Out,
    In,
}

/// Type alias for the pairs stored in [`LegacyRegionConstraintData::constraints`],
/// which we are indexing.
type ConstraintPair<'tcx> = (Constraint<'tcx>, SubregionOrigin<'tcx>);

/// An index from region variables to their corresponding constraint edges,
/// used on some error paths.
pub(super) struct IndexedConstraintEdges<'tcx> {
    out_edges: IndexVec<RegionVid, Vec<ConstraintPair<'tcx>>>,
    in_edges: IndexVec<RegionVid, Vec<ConstraintPair<'tcx>>>,
}

impl<'tcx> IndexedConstraintEdges<'tcx> {
    pub(super) fn build_index<D: Data<'tcx>>(num_vars: usize, data: &D) -> Self {
        let mut out_edges = IndexVec::from_fn_n(|_| vec![], num_vars);
        let mut in_edges = IndexVec::from_fn_n(|_| vec![], num_vars);

        for pair @ (c, _) in data
            .constraints()
            .flat_map(|(c, origin)| c.iter_outlives().map(move |c| (c, origin.clone())))
        {
            // Only push a var out-edge for `VarSub...` constraints.
            match c.kind {
                ConstraintKind::VarSubVar | ConstraintKind::VarSubReg => {
                    out_edges[c.sub.as_var()].push(pair);
                }

                ConstraintKind::RegSubVar | ConstraintKind::RegSubReg => {}

                ConstraintKind::VarEqVar | ConstraintKind::VarEqReg | ConstraintKind::RegEqReg => {
                    unreachable!();
                }
            }
        }

        // FIXME: We should merge this loop with the above one eventually.
        // Index in-edges in reverse order, to match what current tests expect.
        // (It's unclear whether this is important or not.)

        for pair @ (c, _) in data
            .constraints()
            .rev()
            .flat_map(|(c, origin)| c.iter_outlives().map(move |c| (c, origin.clone())))
        {
            // Only push a var in-edge for `...SubVar` constraints.
            match c.kind {
                ConstraintKind::VarSubVar | ConstraintKind::RegSubVar => {
                    in_edges[c.sup.as_var()].push(pair);
                }

                ConstraintKind::VarSubReg | ConstraintKind::RegSubReg => {}

                ConstraintKind::VarEqVar | ConstraintKind::VarEqReg | ConstraintKind::RegEqReg => {
                    unreachable!();
                }
            }
        }

        IndexedConstraintEdges { out_edges, in_edges }
    }

    /// Returns either the out-edges or in-edges of the specified region var,
    /// as selected by `dir`.
    pub(super) fn adjacent_edges(
        &self,
        region_vid: RegionVid,
        dir: EdgeDirection,
    ) -> &[ConstraintPair<'tcx>] {
        let edges = match dir {
            EdgeDirection::Out => &self.out_edges,
            EdgeDirection::In => &self.in_edges,
        };
        &edges[region_vid]
    }
}
