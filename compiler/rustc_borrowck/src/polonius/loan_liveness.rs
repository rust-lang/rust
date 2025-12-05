use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexSet};
use rustc_middle::ty::RegionVid;
use rustc_mir_dataflow::points::PointIndex;

use super::{LiveLoans, LocalizedOutlivesConstraintSet};
use crate::BorrowSet;
use crate::constraints::OutlivesConstraint;
use crate::region_infer::values::LivenessValues;
use crate::type_check::Locations;

/// Compute loan reachability to approximately trace loan liveness throughout the CFG, by
/// traversing the full graph of constraints that combines:
/// - the localized constraints (the physical edges),
/// - with the constraints that hold at all points (the logical edges).
pub(super) fn compute_loan_liveness<'tcx>(
    liveness: &LivenessValues,
    outlives_constraints: impl Iterator<Item = OutlivesConstraint<'tcx>>,
    borrow_set: &BorrowSet<'tcx>,
    localized_outlives_constraints: &LocalizedOutlivesConstraintSet,
) -> LiveLoans {
    let mut live_loans = LiveLoans::new(borrow_set.len());

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
            // The simplest change that made sense to "fix" the issues above is taking into
            // account kills that are:
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

            if liveness.is_live_at(node.region, liveness.location_from_point(node.point)) {
                live_loans.insert(node.point, loan_idx);
            }

            for succ in graph.outgoing_edges(node) {
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
