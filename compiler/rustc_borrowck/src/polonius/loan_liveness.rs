use std::collections::{BTreeMap, BTreeSet};

use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexSet};
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{
    Body, Local, Location, Place, Rvalue, Statement, StatementKind, Terminator, TerminatorKind,
};
use rustc_middle::ty::{RegionVid, TyCtxt};
use rustc_mir_dataflow::points::PointIndex;

use super::{LiveLoans, LocalizedOutlivesConstraintSet};
use crate::dataflow::BorrowIndex;
use crate::region_infer::values::LivenessValues;
use crate::{BorrowSet, PlaceConflictBias, places_conflict};

/// With the full graph of constraints, we can compute loan reachability, stop at kills, and trace
/// loan liveness throughout the CFG.
pub(super) fn compute_loan_liveness<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    liveness: &LivenessValues,
    borrow_set: &BorrowSet<'tcx>,
    localized_outlives_constraints: &LocalizedOutlivesConstraintSet,
) -> LiveLoans {
    let mut live_loans = LiveLoans::new(borrow_set.len());

    // FIXME: it may be preferable for kills to be encoded in the edges themselves, to simplify and
    // likely make traversal (and constraint generation) more efficient. We also display kills on
    // edges when visualizing the constraint graph anyways.
    let kills = collect_kills(body, tcx, borrow_set);

    let graph = index_constraints(&localized_outlives_constraints);
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

            // Continuing traversal will depend on whether the loan is killed at this point.
            let current_location = liveness.location_from_point(node.point);
            let is_loan_killed =
                kills.get(&current_location).is_some_and(|kills| kills.contains(&loan_idx));

            for succ in outgoing_edges(&graph, node) {
                // If the loan is killed at this point, it is killed _on exit_.
                if is_loan_killed {
                    continue;
                }
                stack.push(succ);
            }
        }
    }

    live_loans
}

/// The localized constraint graph is currently the per-node map of its physical edges. In the
/// future, we'll add logical edges to model constraints that hold at all points in the CFG.
type LocalizedConstraintGraph = FxHashMap<LocalizedNode, FxIndexSet<LocalizedNode>>;

/// A node in the graph to be traversed, one of the two vertices of a localized outlives constraint.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
struct LocalizedNode {
    region: RegionVid,
    point: PointIndex,
}

/// Traverses the constraints and returns the indexable graph of edges per node.
fn index_constraints(constraints: &LocalizedOutlivesConstraintSet) -> LocalizedConstraintGraph {
    let mut edges = LocalizedConstraintGraph::default();
    for constraint in &constraints.outlives {
        let source = LocalizedNode { region: constraint.source, point: constraint.from };
        let target = LocalizedNode { region: constraint.target, point: constraint.to };
        edges.entry(source).or_default().insert(target);
    }

    edges
}

/// Returns the outgoing edges of a given node, not its transitive closure.
fn outgoing_edges(
    graph: &LocalizedConstraintGraph,
    node: LocalizedNode,
) -> impl Iterator<Item = LocalizedNode> + use<'_> {
    graph.get(&node).into_iter().flat_map(|edges| edges.iter().copied())
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
                self.kills.entry(location).or_default().extend(other_borrows_of_local);
            }
            return;
        }

        // By passing `PlaceConflictBias::NoOverlap`, we conservatively assume that any given
        // pair of array indices are not equal, so that when `places_conflict` returns true, we
        // will be assured that two places being compared definitely denotes the same sets of
        // locations.
        let definitely_conflicting_borrows = other_borrows_of_local.filter(|&i| {
            places_conflict(
                self.tcx,
                self.body,
                self.borrow_set[i].borrowed_place,
                place,
                PlaceConflictBias::NoOverlap,
            )
        });

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
