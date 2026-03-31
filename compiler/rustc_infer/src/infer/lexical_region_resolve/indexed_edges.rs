use rustc_index::IndexVec;
use rustc_type_ir::RegionVid;

use crate::infer::SubregionOrigin;
use crate::infer::region_constraints::{Constraint, ConstraintKind, RegionConstraintData};

/// Selects either out-edges or in-edges for [`IndexedConstraintEdges::adjacent_edges`].
#[derive(Clone, Copy, Debug)]
pub(super) enum EdgeDirection {
    Out,
    In,
}

/// Type alias for the pairs stored in [`RegionConstraintData::constraints`],
/// which we are indexing.
type ConstraintPair<'tcx> = (Constraint<'tcx>, SubregionOrigin<'tcx>);

/// An index from region variables to their corresponding constraint edges,
/// used on some error paths.
pub(super) struct IndexedConstraintEdges<'data, 'tcx> {
    out_edges: IndexVec<RegionVid, Vec<&'data ConstraintPair<'tcx>>>,
    in_edges: IndexVec<RegionVid, Vec<&'data ConstraintPair<'tcx>>>,
}

impl<'data, 'tcx> IndexedConstraintEdges<'data, 'tcx> {
    pub(super) fn build_index(num_vars: usize, data: &'data RegionConstraintData<'tcx>) -> Self {
        let mut out_edges = IndexVec::from_fn_n(|_| vec![], num_vars);
        let mut in_edges = IndexVec::from_fn_n(|_| vec![], num_vars);

        for pair @ (c, _) in &data.constraints {
            // Only push a var out-edge for `VarSub...` constraints.
            match c.kind {
                ConstraintKind::VarSubVar | ConstraintKind::VarSubReg => {
                    out_edges[c.sub.as_var()].push(pair)
                }
                ConstraintKind::RegSubVar | ConstraintKind::RegSubReg => {}
            }
        }

        // Index in-edges in reverse order, to match what current tests expect.
        // (It's unclear whether this is important or not.)
        for pair @ (c, _) in data.constraints.iter().rev() {
            // Only push a var in-edge for `...SubVar` constraints.
            match c.kind {
                ConstraintKind::VarSubVar | ConstraintKind::RegSubVar => {
                    in_edges[c.sup.as_var()].push(pair)
                }
                ConstraintKind::VarSubReg | ConstraintKind::RegSubReg => {}
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
    ) -> &[&'data ConstraintPair<'tcx>] {
        let edges = match dir {
            EdgeDirection::Out => &self.out_edges,
            EdgeDirection::In => &self.in_edges,
        };
        &edges[region_vid]
    }
}
