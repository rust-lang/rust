//! Polonius analysis and support code:
//! - dedicated constraints
//! - conversion from NLL constraints
//! - debugging utilities
//! - etc.
//!
//! The current implementation models the flow-sensitive borrow-checking concerns as a graph
//! containing both information about regions and information about the control flow.
//!
//! Loan propagation is seen as a reachability problem (with some subtleties) between where the loan
//! is introduced and a given point.
//!
//! Constraints arising from type-checking allow loans to flow from region to region at the same CFG
//! point. Constraints arising from liveness allow loans to flow within from point to point, between
//! live regions at these points.
//!
//! Edges can be bidirectional to encode invariant relationships, and loans can flow "back in time"
//! to traverse these constraints arising earlier in the CFG.
//!
//! When incorporating kills in the traversal, the loans reaching a given point are considered live.
//!
//! After this, the usual NLL process happens. These live loans are fed into a dataflow analysis
//! combining them with the points where loans go out of NLL scope (the frontier where they stop
//! propagating to a live region), to yield the "loans in scope" or "active loans", at a given
//! point.
//!
//! Illegal accesses are still computed by checking whether one of these resulting loans is
//! invalidated.
//!
//! More information on this simple approach can be found in the following links, and in the future
//! in the rustc dev guide:
//! - <https://smallcultfollowing.com/babysteps/blog/2023/09/22/polonius-part-1/>
//! - <https://smallcultfollowing.com/babysteps/blog/2023/09/29/polonius-part-2/>
//!

mod constraints;
mod dump;
pub(crate) mod legacy;
mod liveness_constraints;
mod typeck_constraints;

use std::collections::BTreeMap;

use rustc_index::bit_set::SparseBitMatrix;
use rustc_middle::mir::Body;
use rustc_middle::ty::{RegionVid, TyCtxt};
use rustc_mir_dataflow::points::PointIndex;

pub(crate) use self::constraints::*;
pub(crate) use self::dump::dump_polonius_mir;
use self::liveness_constraints::create_liveness_constraints;
use self::typeck_constraints::convert_typeck_constraints;
use crate::RegionInferenceContext;

/// This struct holds the data needed to create the Polonius localized constraints.
pub(crate) struct PoloniusContext {
    /// The set of regions that are live at a given point in the CFG, used to create localized
    /// outlives constraints between regions that are live at connected points in the CFG.
    live_regions: Option<SparseBitMatrix<PointIndex, RegionVid>>,

    /// The expected edge direction per live region: the kind of directed edge we'll create as
    /// liveness constraints depends on the variance of types with respect to each contained region.
    live_region_variances: BTreeMap<RegionVid, ConstraintDirection>,
}

/// The direction a constraint can flow into. Used to create liveness constraints according to
/// variance.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum ConstraintDirection {
    /// For covariant cases, we add a forward edge `O at P1 -> O at P2`.
    Forward,

    /// For contravariant cases, we add a backward edge `O at P2 -> O at P1`
    Backward,

    /// For invariant cases, we add both the forward and backward edges `O at P1 <-> O at P2`.
    Bidirectional,
}

impl PoloniusContext {
    pub(crate) fn new() -> PoloniusContext {
        Self { live_region_variances: BTreeMap::new(), live_regions: None }
    }

    /// Creates a constraint set for `-Zpolonius=next` by:
    /// - converting NLL typeck constraints to be localized
    /// - encoding liveness constraints
    pub(crate) fn create_localized_constraints<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        regioncx: &RegionInferenceContext<'tcx>,
        body: &Body<'tcx>,
    ) -> LocalizedOutlivesConstraintSet {
        let mut localized_outlives_constraints = LocalizedOutlivesConstraintSet::default();
        convert_typeck_constraints(
            tcx,
            body,
            regioncx.liveness_constraints(),
            regioncx.outlives_constraints(),
            regioncx.universal_regions(),
            &mut localized_outlives_constraints,
        );

        let live_regions = self.live_regions.as_ref().expect(
            "live regions per-point data should have been created at the end of MIR typeck",
        );
        create_liveness_constraints(
            body,
            regioncx.liveness_constraints(),
            live_regions,
            &self.live_region_variances,
            regioncx.universal_regions(),
            &mut localized_outlives_constraints,
        );

        // FIXME: here, we can trace loan reachability in the constraint graph and record this as loan
        // liveness for the next step in the chain, the NLL loan scope and active loans computations.

        localized_outlives_constraints
    }
}
