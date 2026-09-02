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
mod liveness;
mod liveness_constraints;

use std::collections::BTreeMap;
use std::rc::Rc;

use rustc_data_structures::fx::FxHashSet;
use rustc_index::bit_set::SparseBitMatrix;
use rustc_middle::mir::{Body, Local};
use rustc_middle::ty::{RegionVid, TyCtxt};
use rustc_mir_dataflow::move_paths::MoveData;
use rustc_mir_dataflow::points::{DenseLocationMap, PointIndex};

pub(self) use self::constraints::*;
pub(crate) use self::dump::dump_polonius_mir;
pub(crate) use self::liveness_constraints::record_live_region_variance;
use crate::BorrowSet;
use crate::constraints::OutlivesConstraint;
use crate::dataflow::BorrowIndex;
pub(crate) use crate::polonius::liveness::DeferredLocals;
use crate::region_infer::values::LivenessValues;
use crate::type_check::liveness::{LivenessCalculation, LocalUseMap};
use crate::universal_regions::UniversalRegions;

pub(crate) type LiveLoans = SparseBitMatrix<PointIndex, BorrowIndex>;

/// This struct holds the necessary
///  - liveness data, created during MIR typeck, and which will be used to lazily compute the
///    polonius localized constraints, during NLL region inference as well as MIR dumping,
///  - data needed by the borrowck error computation and diagnostics.
#[derive(Default)]
pub(crate) struct PoloniusContext<'tcx> {
    /// The graph from which we extract the localized outlives constraints.
    graph: Option<LocalizedConstraintGraph>,

    /// The expected edge direction per live region: the kind of directed edge we'll create as
    /// liveness constraints depends on the variance of types with respect to each contained region.
    pub(crate) live_region_variances: BTreeMap<RegionVid, ConstraintDirection>,

    /// The regions that outlive free regions are used to distinguish relevant live locals from
    /// boring locals. A boring local is one whose type contains only such regions. Polonius
    /// currently has more boring locals than NLLs so we record the latter to use in errors and
    /// diagnostics, to focus on the locals we consider relevant and match NLL diagnostics.
    pub(crate) boring_nll_locals: FxHashSet<Local>,

    pub(crate) deferred_locals_for_liveness: DeferredLocals<'tcx>,

    pub(crate) local_use_map: Option<LocalUseMap>,
}

/// The direction a constraint can flow into. Used to create liveness constraints according to
/// variance.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) enum ConstraintDirection {
    /// For covariant cases, we add a forward edge `O at P1 -> O at P2`.
    Forward,

    /// For contravariant cases, we add a backward edge `O at P2 -> O at P1`
    Backward,

    /// For invariant cases, we add both the forward and backward edges `O at P1 <-> O at P2`.
    Bidirectional,
}

impl<'tcx> PoloniusContext<'tcx> {
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
    pub(crate) fn compute_loan_liveness(
        &mut self,
        tcx: TyCtxt<'tcx>,
        liveness: &mut LivenessValues,
        outlives_constraints: impl Iterator<Item = OutlivesConstraint<'tcx>>,
        universal_regions: &UniversalRegions<'tcx>,
        body: &Body<'tcx>,
        move_data: &MoveData<'tcx>,
        location_map: &DenseLocationMap,
        borrow_set: &BorrowSet<'tcx>,
    ) {
        // We don't need to prepare the graph (index NLL constraints, etc.) if we have no loans to
        // trace throughout localized constraints.
        if borrow_set.len() > 0 {
            // From the outlives constraints, liveness, and variances, we can compute reachability
            // on the lazy localized constraint graph to trace the liveness of loans, for the next
            // step in the chain (the NLL loan scope and active loans computations).
            let graph = LocalizedConstraintGraph::new(
                Rc::clone(liveness.location_map()),
                outlives_constraints,
            );

            let local_use_map = self
                .local_use_map
                .as_ref()
                .expect("local use map should be computed before loan liveness");
            let deferred_locals_for_liveness =
                std::mem::take(&mut self.deferred_locals_for_liveness);
            let mut live_loans = LiveLoans::new(borrow_set.len());
            let calc = LivenessCalculation::new(tcx, body, location_map, move_data, &local_use_map);
            let mut traversal = LoanLivenessTraversal {
                liveness,
                live_region_variances: &mut self.live_region_variances,
                live_loans: &mut live_loans,
                universal_regions,
                deferred_locals_for_liveness,
                calc,
            };
            graph.traverse(body, universal_regions, borrow_set, &mut traversal);
            liveness.record_live_loans(live_loans);

            // The graph can be traversed again during MIR dumping, so we store it here.
            self.graph = Some(graph);
        }
    }
}

struct LoanLivenessTraversal<'a, 'tcx> {
    liveness: &'a mut LivenessValues,
    live_region_variances: &'a mut BTreeMap<RegionVid, ConstraintDirection>,
    live_loans: &'a mut LiveLoans,
    universal_regions: &'a UniversalRegions<'tcx>,
    deferred_locals_for_liveness: DeferredLocals<'tcx>,
    calc: LivenessCalculation<'a, 'tcx>,
}

impl LocalizedConstraintGraphTraversal for LoanLivenessTraversal<'_, '_> {
    type Visitor<'a>
        = LoanLivenessVisitor<'a>
    where
        Self: 'a;

    fn mk_visitor(
        &mut self,
        region: RegionVid,
    ) -> (&LivenessValues, &BTreeMap<RegionVid, ConstraintDirection>, Self::Visitor<'_>) {
        if let Some((local, drop_args)) =
            self.deferred_locals_for_liveness.use_deferred_local(region)
        {
            self.calc.compute(local);

            if !self.calc.use_live_at.is_empty() || !self.calc.drop_live_at.is_empty() {
                record_live_region_variance(
                    self.calc.tcx,
                    &mut self.live_region_variances,
                    self.universal_regions,
                    self.calc.body.local_decls[local].ty,
                );
            }
            if !self.calc.use_live_at.is_empty() {
                let local_ty = self.calc.body.local_decls[local].ty;
                self.calc.tcx.for_each_free_region(&local_ty, |live_region| {
                    let region = self.universal_regions.to_region_vid(live_region);
                    self.liveness.add_points(region, &self.calc.use_live_at);
                });
            }
            if !self.calc.drop_live_at.is_empty() {
                for drop_arg in drop_args {
                    self.calc.tcx.for_each_free_region(&drop_arg, |live_region| {
                        let region = self.universal_regions.to_region_vid(live_region);
                        self.liveness.add_points(region, &self.calc.drop_live_at);
                    });
                }
            }
        }

        (
            self.liveness,
            self.live_region_variances,
            LoanLivenessVisitor { liveness: self.liveness, live_loans: self.live_loans },
        )
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
        if self.liveness.is_live_at_point(node.region, node.point) {
            self.live_loans.insert(node.point, loan);
        }
    }
}
