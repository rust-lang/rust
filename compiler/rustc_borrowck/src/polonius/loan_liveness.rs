use std::collections::{BTreeMap, BTreeSet};

use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexSet};
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{
    Body, Local, Location, Place, Rvalue, Statement, StatementKind, Terminator, TerminatorKind,
};
use rustc_middle::ty::{RegionVid, TyCtxt};
use rustc_mir_dataflow::points::PointIndex;

use super::{LiveLoans, LocalizedOutlivesConstraintSet};
use crate::constraints::OutlivesConstraint;
use crate::dataflow::BorrowIndex;
use crate::region_infer::values::LivenessValues;
use crate::type_check::Locations;
use crate::{BorrowSet, PlaceConflictBias, places_conflict};

/// Compute loan reachability, stop at kills, and trace loan liveness throughout the CFG, by
/// traversing the full graph of constraints that combines:
/// - the localized constraints (the physical edges),
/// - with the constraints that hold at all points (the logical edges).
pub(super) fn compute_loan_liveness<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    liveness: &LivenessValues,
    outlives_constraints: impl Iterator<Item = OutlivesConstraint<'tcx>>,
    borrow_set: &BorrowSet<'tcx>,
    localized_outlives_constraints: &LocalizedOutlivesConstraintSet,
) -> LiveLoans {
    let mut live_loans = LiveLoans::new(borrow_set.len());

    // FIXME: it may be preferable for kills to be encoded in the edges themselves, to simplify and
    // likely make traversal (and constraint generation) more efficient. We also display kills on
    // edges when visualizing the constraint graph anyways.
    let kills = collect_kills(body, tcx, borrow_set);

    // Create the full graph with the physical edges we've localized earlier, and the logical edges
    // of constraints that hold at all points.
    let logical_constraints =
        outlives_constraints.filter(|c| matches!(c.locations, Locations::All(_)));
    let graph = LocalizedConstraintGraph::new(&localized_outlives_constraints, logical_constraints);
    let mut visited = FxHashSet::default();
    let mut stack = Vec::new();

    // Compute reachability per loan by traversing each loan's subgraph starting from where it is
    // introduced.
    for (loan_idx, loan) in borrow_set.iter_enumerated() {
        visited.clear();
        stack.clear();

        let start_node = LocalizedNode {
            region: loan.region,
            point: liveness.point_from_location(loan.reserve_location),
        };
        stack.push(start_node);

        while let Some(node) = stack.pop() {
            if !visited.insert(node) {
                continue;
            }

            // Record the loan as being live on entry to this point.
            live_loans.insert(node.point, loan_idx);

            // Here, we have a conundrum. There's currently a weakness in our theory, in that
            // we're using a single notion of reachability to represent what used to be _two_
            // different transitive closures. It didn't seem impactful when coming up with the
            // single-graph and reachability through space (regions) + time (CFG) concepts, but in
            // practice the combination of time-traveling with kills is more impactful than
            // initially anticipated.
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
            // The simplest change that made sense to "fix" the issues above is taking into
            // account kills that are:
            // - reachable from the introduction point
            // - encountered during forward traversal. Note that this is not transitive like the
            //   two-step traversal described above: only kills encountered on exit via a backward
            //   edge are ignored.
            //
            // In our test suite, there are a couple of cases where kills are encountered while
            // time-traveling, however as far as we can tell, always in cases where they would be
            // unreachable. We have reason to believe that this is a property of the single-graph
            // approach (but haven't proved it yet):
            // - reachable kills while time-traveling would also be encountered via regular
            //   traversal
            // - it makes _some_ sense to ignore unreachable kills, but subtleties around dead code
            //   in general need to be better thought through (like they were for NLLs).
            // - ignoring kills is a conservative approximation: the loan is still live and could
            //   cause false positive errors at another place access. Soundness issues in this
            //   domain should look more like the absence of reachability instead.
            //
            // This is enough in practice to pass tests, and therefore is what we have implemented
            // for now.
            //
            // FIXME: all of the above. Analyze potential unsoundness, possibly in concert with a
            // borrowck implementation in a-mir-formality, fuzzing, or manually crafting
            // counter-examples.

            // Continuing traversal will depend on whether the loan is killed at this point, and
            // whether we're time-traveling.
            let current_location = liveness.location_from_point(node.point);
            let is_loan_killed =
                kills.get(&current_location).is_some_and(|kills| kills.contains(&loan_idx));

            for succ in graph.outgoing_edges(node) {
                // If the loan is killed at this point, it is killed _on exit_. But only during
                // forward traversal.
                if is_loan_killed {
                    let destination = liveness.location_from_point(succ.point);
                    if current_location.is_predecessor_of(destination, body) {
                        continue;
                    }
                }
                stack.push(succ);
            }
        }
    }

    live_loans
}

/// The localized constraint graph indexes the physical and logical edges to compute a given node's
/// successors during traversal.
struct LocalizedConstraintGraph {
    /// The actual, physical, edges we have recorded for a given node.
    edges: FxHashMap<LocalizedNode, FxIndexSet<LocalizedNode>>,

    /// The logical edges representing the outlives constraints that hold at all points in the CFG,
    /// which we don't localize to avoid creating a lot of unnecessary edges in the graph. Some CFGs
    /// can be big, and we don't need to create such a physical edge for every point in the CFG.
    logical_edges: FxHashMap<RegionVid, FxIndexSet<RegionVid>>,
}

/// A node in the graph to be traversed, one of the two vertices of a localized outlives constraint.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
struct LocalizedNode {
    region: RegionVid,
    point: PointIndex,
}

impl LocalizedConstraintGraph {
    /// Traverses the constraints and returns the indexed graph of edges per node.
    fn new<'tcx>(
        constraints: &LocalizedOutlivesConstraintSet,
        logical_constraints: impl Iterator<Item = OutlivesConstraint<'tcx>>,
    ) -> Self {
        let mut edges: FxHashMap<_, FxIndexSet<_>> = FxHashMap::default();
        for constraint in &constraints.outlives {
            let source = LocalizedNode { region: constraint.source, point: constraint.from };
            let target = LocalizedNode { region: constraint.target, point: constraint.to };
            edges.entry(source).or_default().insert(target);
        }

        let mut logical_edges: FxHashMap<_, FxIndexSet<_>> = FxHashMap::default();
        for constraint in logical_constraints {
            logical_edges.entry(constraint.sup).or_default().insert(constraint.sub);
        }

        LocalizedConstraintGraph { edges, logical_edges }
    }

    /// Returns the outgoing edges of a given node, not its transitive closure.
    fn outgoing_edges(&self, node: LocalizedNode) -> impl Iterator<Item = LocalizedNode> {
        // The outgoing edges are:
        // - the physical edges present at this node,
        // - the materialized logical edges that exist virtually at all points for this node's
        //   region, localized at this point.
        let physical_edges =
            self.edges.get(&node).into_iter().flat_map(|targets| targets.iter().copied());
        let materialized_edges =
            self.logical_edges.get(&node.region).into_iter().flat_map(move |targets| {
                targets
                    .iter()
                    .copied()
                    .map(move |target| LocalizedNode { point: node.point, region: target })
            });
        physical_edges.chain(materialized_edges)
    }
}

/// Traverses the MIR and collects kills.
fn collect_kills<'tcx>(
    body: &Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    borrow_set: &BorrowSet<'tcx>,
) -> BTreeMap<Location, BTreeSet<BorrowIndex>> {
    let mut collector = KillsCollector { borrow_set, tcx, body, kills: BTreeMap::default() };
    for (block, data) in body.basic_blocks.iter_enumerated() {
        collector.visit_basic_block_data(block, data);
    }
    collector.kills
}

struct KillsCollector<'a, 'tcx> {
    body: &'a Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    borrow_set: &'a BorrowSet<'tcx>,

    /// The set of loans killed at each location.
    kills: BTreeMap<Location, BTreeSet<BorrowIndex>>,
}

// This visitor has a similar structure to the `Borrows` dataflow computation with respect to kills,
// and the datalog polonius fact generation for the `loan_killed_at` relation.
impl<'tcx> KillsCollector<'_, 'tcx> {
    /// Records the borrows on the specified place as `killed`. For example, when assigning to a
    /// local, or on a call's return destination.
    fn record_killed_borrows_for_place(&mut self, place: Place<'tcx>, location: Location) {
        // For the reasons described in graph traversal, we also filter out kills
        // unreachable from the loan's introduction point, as they would stop traversal when
        // e.g. checking for reachability in the subset graph through invariance constraints
        // higher up.
        let filter_unreachable_kills = |loan| {
            let introduction = self.borrow_set[loan].reserve_location;
            let reachable = introduction.is_predecessor_of(location, self.body);
            reachable
        };

        let other_borrows_of_local = self
            .borrow_set
            .local_map
            .get(&place.local)
            .into_iter()
            .flat_map(|bs| bs.iter())
            .copied();

        // If the borrowed place is a local with no projections, all other borrows of this
        // local must conflict. This is purely an optimization so we don't have to call
        // `places_conflict` for every borrow.
        if place.projection.is_empty() {
            if !self.body.local_decls[place.local].is_ref_to_static() {
                self.kills
                    .entry(location)
                    .or_default()
                    .extend(other_borrows_of_local.filter(|&loan| filter_unreachable_kills(loan)));
            }
            return;
        }

        // By passing `PlaceConflictBias::NoOverlap`, we conservatively assume that any given
        // pair of array indices are not equal, so that when `places_conflict` returns true, we
        // will be assured that two places being compared definitely denotes the same sets of
        // locations.
        let definitely_conflicting_borrows = other_borrows_of_local
            .filter(|&i| {
                places_conflict(
                    self.tcx,
                    self.body,
                    self.borrow_set[i].borrowed_place,
                    place,
                    PlaceConflictBias::NoOverlap,
                )
            })
            .filter(|&loan| filter_unreachable_kills(loan));

        self.kills.entry(location).or_default().extend(definitely_conflicting_borrows);
    }

    /// Records the borrows on the specified local as `killed`.
    fn record_killed_borrows_for_local(&mut self, local: Local, location: Location) {
        if let Some(borrow_indices) = self.borrow_set.local_map.get(&local) {
            self.kills.entry(location).or_default().extend(borrow_indices.iter());
        }
    }
}

impl<'tcx> Visitor<'tcx> for KillsCollector<'_, 'tcx> {
    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        // Make sure there are no remaining borrows for locals that have gone out of scope.
        if let StatementKind::StorageDead(local) = statement.kind {
            self.record_killed_borrows_for_local(local, location);
        }

        self.super_statement(statement, location);
    }

    fn visit_assign(&mut self, place: &Place<'tcx>, rvalue: &Rvalue<'tcx>, location: Location) {
        // When we see `X = ...`, then kill borrows of `(*X).foo` and so forth.
        self.record_killed_borrows_for_place(*place, location);
        self.super_assign(place, rvalue, location);
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        // A `Call` terminator's return value can be a local which has borrows, so we need to record
        // those as killed as well.
        if let TerminatorKind::Call { destination, .. } = terminator.kind {
            self.record_killed_borrows_for_place(destination, location);
        }

        self.super_terminator(terminator, location);
    }
}
