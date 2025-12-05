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
//!
//! Data flows like this:
//! 1) during MIR typeck, record liveness data needed later: live region variances, as well as the
//!    usual NLL liveness data (just computed on more locals). That's the [PoloniusLivenessContext].
//! 2) once that is done, variance data is transferred, and the NLL region liveness is converted to
//!    the polonius shape. That's the main [PoloniusContext].
//! 3) during region inference, that data and the NLL outlives constraints are used to create the
//!    localized outlives constraints, as described above. That's the [PoloniusDiagnosticsContext].
//! 4) transfer this back to the main borrowck procedure: it handles computing errors and
//!    diagnostics, debugging and MIR dumping concerns.

mod constraints;
mod dump;
pub(crate) mod legacy;
mod liveness_constraints;
mod loan_liveness;
mod typeck_constraints;

use std::collections::BTreeMap;

use rustc_data_structures::fx::FxHashSet;
use rustc_index::bit_set::SparseBitMatrix;
use rustc_index::interval::SparseIntervalMatrix;
use rustc_middle::mir::{Body, Local};
use rustc_middle::ty::{RegionVid, TyCtxt};
use rustc_mir_dataflow::points::PointIndex;

pub(crate) use self::constraints::*;
pub(crate) use self::dump::dump_polonius_mir;
use self::liveness_constraints::create_liveness_constraints;
use self::loan_liveness::compute_loan_liveness;
use self::typeck_constraints::convert_typeck_constraints;
use crate::dataflow::BorrowIndex;
use crate::{BorrowSet, RegionInferenceContext};

pub(crate) type LiveLoans = SparseBitMatrix<PointIndex, BorrowIndex>;

/// This struct holds the liveness data created during MIR typeck, and which will be used later in
/// the process, to compute the polonius localized constraints.
#[derive(Default)]
pub(crate) struct PoloniusLivenessContext {
    /// The expected edge direction per live region: the kind of directed edge we'll create as
    /// liveness constraints depends on the variance of types with respect to each contained region.
    live_region_variances: BTreeMap<RegionVid, ConstraintDirection>,

    /// The regions that outlive free regions are used to distinguish relevant live locals from
    /// boring locals. A boring local is one whose type contains only such regions. Polonius
    /// currently has more boring locals than NLLs so we record the latter to use in errors and
    /// diagnostics, to focus on the locals we consider relevant and match NLL diagnostics.
    pub(crate) boring_nll_locals: FxHashSet<Local>,
}

/// This struct holds the data needed to create the Polonius localized constraints. Its data is
/// transferred and converted from the [PoloniusLivenessContext] at the end of MIR typeck.
pub(crate) struct PoloniusContext {
    /// The liveness data we recorded during MIR typeck.
    liveness_context: PoloniusLivenessContext,

    /// The set of regions that are live at a given point in the CFG, used to create localized
    /// outlives constraints between regions that are live at connected points in the CFG.
    live_regions: SparseBitMatrix<PointIndex, RegionVid>,
}

/// This struct holds the data needed by the borrowck error computation and diagnostics. Its data is
/// computed from the [PoloniusContext] when computing NLL regions.
pub(crate) struct PoloniusDiagnosticsContext {
    /// The localized outlives constraints that were computed in the main analysis.
    localized_outlives_constraints: LocalizedOutlivesConstraintSet,

    /// The liveness data computed during MIR typeck: [PoloniusLivenessContext::boring_nll_locals].
    pub(crate) boring_nll_locals: FxHashSet<Local>,
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
    /// Unlike NLLs, in polonius we traverse the cfg to look for regions live across an edge, so we
    /// need to transpose the "points where each region is live" matrix to a "live regions per point"
    /// matrix.
    // FIXME: avoid this conversion by always storing liveness data in this shape in the rest of
    // borrowck.
    pub(crate) fn create_from_liveness(
        liveness_context: PoloniusLivenessContext,
        num_regions: usize,
        points_per_live_region: &SparseIntervalMatrix<RegionVid, PointIndex>,
    ) -> PoloniusContext {
        let mut live_regions_per_point = SparseBitMatrix::new(num_regions);
        for region in points_per_live_region.rows() {
            for point in points_per_live_region.row(region).unwrap().iter() {
                live_regions_per_point.insert(point, region);
            }
        }

        PoloniusContext { live_regions: live_regions_per_point, liveness_context }
    }

    /// Computes live loans using the set of loans model for `-Zpolonius=next`.
    ///
    /// First, creates a constraint graph combining regions and CFG points, by:
    /// - converting NLL typeck constraints to be localized
    /// - encoding liveness constraints
    ///
    /// Then, this graph is traversed, reachability is recorded as loan liveness, to be used by the
    /// loan scope and active loans computations.
    ///
    /// The constraint data will be used to compute errors and diagnostics.
    pub(crate) fn compute_loan_liveness<'tcx>(
        self,
        tcx: TyCtxt<'tcx>,
        regioncx: &mut RegionInferenceContext<'tcx>,
        body: &Body<'tcx>,
        borrow_set: &BorrowSet<'tcx>,
    ) -> PoloniusDiagnosticsContext {
        let PoloniusLivenessContext { live_region_variances, boring_nll_locals } =
            self.liveness_context;

        let mut localized_outlives_constraints = LocalizedOutlivesConstraintSet::default();
        convert_typeck_constraints(
            tcx,
            body,
            regioncx.liveness_constraints(),
            regioncx.outlives_constraints(),
            regioncx.universal_regions(),
            &mut localized_outlives_constraints,
        );

        create_liveness_constraints(
            body,
            regioncx.liveness_constraints(),
            &self.live_regions,
            &live_region_variances,
            regioncx.universal_regions(),
            &mut localized_outlives_constraints,
        );

        // Now that we have a complete graph, we can compute reachability to trace the liveness of
        // loans for the next step in the chain, the NLL loan scope and active loans computations.
        let live_loans = compute_loan_liveness(
            regioncx.liveness_constraints(),
            regioncx.outlives_constraints(),
            borrow_set,
            &localized_outlives_constraints,
        );
        regioncx.record_live_loans(live_loans);

        PoloniusDiagnosticsContext { localized_outlives_constraints, boring_nll_locals }
    }
}
