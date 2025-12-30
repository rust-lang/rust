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
//!    usual NLL liveness data (just computed on more locals). That's the main [PoloniusContext].
//! 2) during region inference, that data and the NLL outlives constraints are used to create the
//!    localized outlives constraints, as described above. That's the [PoloniusDiagnosticsContext].
//! 3) transfer this back to the main borrowck procedure: it handles computing errors and
//!    diagnostics, debugging and MIR dumping concerns.

mod constraints;
mod dump;
pub(crate) mod legacy;
mod liveness_constraints;

use std::collections::BTreeMap;

use rustc_data_structures::fx::FxHashSet;
use rustc_index::bit_set::SparseBitMatrix;
use rustc_middle::mir::{Body, Local};
use rustc_middle::ty::RegionVid;
use rustc_mir_dataflow::points::PointIndex;

pub(crate) use self::constraints::*;
pub(crate) use self::dump::dump_polonius_mir;
use crate::dataflow::BorrowIndex;
use crate::region_infer::values::LivenessValues;
use crate::{BorrowSet, RegionInferenceContext};

pub(crate) type LiveLoans = SparseBitMatrix<PointIndex, BorrowIndex>;

/// This struct holds the liveness data created during MIR typeck, and which will be used later in
/// the process, to lazily compute the polonius localized constraints.
#[derive(Default)]
pub(crate) struct PoloniusContext {
    /// The expected edge direction per live region: the kind of directed edge we'll create as
    /// liveness constraints depends on the variance of types with respect to each contained region.
    live_region_variances: BTreeMap<RegionVid, ConstraintDirection>,

    /// The regions that outlive free regions are used to distinguish relevant live locals from
    /// boring locals. A boring local is one whose type contains only such regions. Polonius
    /// currently has more boring locals than NLLs so we record the latter to use in errors and
    /// diagnostics, to focus on the locals we consider relevant and match NLL diagnostics.
    pub(crate) boring_nll_locals: FxHashSet<Local>,
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
        regioncx: &mut RegionInferenceContext<'tcx>,
        body: &Body<'tcx>,
        borrow_set: &BorrowSet<'tcx>,
    ) -> PoloniusDiagnosticsContext {
        let PoloniusContext { live_region_variances, boring_nll_locals } = self;

        let localized_outlives_constraints = LocalizedOutlivesConstraintSet::default();

        let liveness = regioncx.liveness_constraints();

        if borrow_set.len() > 0 {
            // From the outlives constraints, liveness, and variances, we can compute reachability
            // on the lazy localized constraint graph to trace the liveness of loans, for the next
            // step in the chain (the NLL loan scope and active loans computations).
            let graph = LocalizedConstraintGraph::new(liveness, regioncx.outlives_constraints());

            let mut live_loans = LiveLoans::new(borrow_set.len());
            let mut visitor = LoanLivenessVisitor { liveness, live_loans: &mut live_loans };
            graph.traverse(
                body,
                liveness,
                &live_region_variances,
                regioncx.universal_regions(),
                borrow_set,
                &mut visitor,
            );
            regioncx.record_live_loans(live_loans);
        }

        PoloniusDiagnosticsContext { localized_outlives_constraints, boring_nll_locals }
    }
}

/// Visitor to record loan liveness when traversing the localized constraint graph.
struct LoanLivenessVisitor<'a> {
    liveness: &'a LivenessValues,
    live_loans: &'a mut LiveLoans,
}

impl LocalizedConstraintGraphVisitor for LoanLivenessVisitor<'_> {
    fn on_node_traversed(&mut self, loan: BorrowIndex, node: LocalizedNode) {
        // Record the loan as being live on entry to this point if it reaches a live region
        // there.
        //
        // This is an approximation of liveness (which is the thing we want), in that we're
        // using a single notion of reachability to represent what used to be _two_ different
        // transitive closures. It didn't seem impactful when coming up with the single-graph
        // and reachability through space (regions) + time (CFG) concepts, but in practice the
        // combination of time-traveling with kills is more impactful than initially
        // anticipated.
        //
        // Kills should prevent a loan from reaching its successor points in the CFG, but not
        // while time-traveling: we're not actually at that CFG point, but looking for
        // predecessor regions that contain the loan. One of the two TCs we had pushed the
        // transitive subset edges to each point instead of having backward edges, and the
        // problem didn't exist before. In the abstract, naive reachability is not enough to
        // model this, we'd need a slightly different solution. For example, maybe with a
        // two-step traversal:
        // - at each point we first traverse the subgraph (and possibly time-travel) looking for
        //   exit nodes while ignoring kills,
        // - and then when we're back at the current point, we continue normally.
        //
        // Another (less annoying) subtlety is that kills and the loan use-map are
        // flow-insensitive. Kills can actually appear in places before a loan is introduced, or
        // at a location that is actually unreachable in the CFG from the introduction point,
        // and these can also be encountered during time-traveling.
        //
        // The simplest change that made sense to "fix" the issues above is taking into account
        // kills that are:
        // - reachable from the introduction point
        // - encountered during forward traversal. Note that this is not transitive like the
        //   two-step traversal described above: only kills encountered on exit via a backward
        //   edge are ignored.
        //
        // This version of the analysis, however, is enough in practice to pass the tests that
        // we care about and NLLs reject, without regressions on crater, and is an actionable
        // subset of the full analysis. It also naturally points to areas of improvement that we
        // wish to explore later, namely handling kills appropriately during traversal, instead
        // of continuing traversal to all the reachable nodes.
        //
        // FIXME: analyze potential unsoundness, possibly in concert with a borrowck
        // implementation in a-mir-formality, fuzzing, or manually crafting counter-examples.
        let location = self.liveness.location_from_point(node.point);
        if self.liveness.is_live_at(node.region, location) {
            self.live_loans.insert(node.point, loan);
        }
    }
}
