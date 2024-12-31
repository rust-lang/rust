use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexSet};
use rustc_middle::mir::Body;
use rustc_middle::ty::{RegionVid, TyCtxt};
use rustc_mir_dataflow::points::PointIndex;

use super::{LiveLoans, LocalizedOutlivesConstraintSet};
use crate::BorrowSet;
use crate::region_infer::values::LivenessValues;

/// With the full graph of constraints, we can compute loan reachability, and trace loan liveness
/// throughout the CFG.
pub(super) fn compute_loan_liveness<'tcx>(
    _tcx: TyCtxt<'tcx>,
    _body: &Body<'tcx>,
    liveness: &LivenessValues,
    borrow_set: &BorrowSet<'tcx>,
    localized_outlives_constraints: &LocalizedOutlivesConstraintSet,
) -> LiveLoans {
    let mut live_loans = LiveLoans::new(borrow_set.len());
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

            for succ in outgoing_edges(&graph, node) {
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

/// Index the outlives constraints into a graph of edges per node.
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
