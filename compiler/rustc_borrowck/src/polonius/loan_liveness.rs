use std::collections::BTreeMap;

use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexSet};
use rustc_index::bit_set::SparseBitMatrix;
use rustc_middle::mir::{Body, Location};
use rustc_middle::ty::RegionVid;
use rustc_mir_dataflow::points::PointIndex;

use crate::BorrowSet;
use crate::constraints::OutlivesConstraint;
use crate::polonius::{ConstraintDirection, LiveLoans};
use crate::region_infer::values::LivenessValues;
use crate::type_check::Locations;
use crate::universal_regions::UniversalRegions;

/// Compute loan reachability to approximately trace loan liveness throughout the CFG, by
/// traversing the graph of constraints that lazily combines:
/// - the localized constraints (the physical edges),
/// - with the constraints that hold at all points (the logical edges).
pub(super) fn compute_loan_liveness<'tcx>(
    body: &Body<'tcx>,
    outlives_constraints: impl Iterator<Item = OutlivesConstraint<'tcx>>,
    liveness: &LivenessValues,
    live_regions: &SparseBitMatrix<PointIndex, RegionVid>,
    live_region_variances: &BTreeMap<RegionVid, ConstraintDirection>,
    universal_regions: &UniversalRegions<'tcx>,
    borrow_set: &BorrowSet<'tcx>,
) -> LiveLoans {
    let mut live_loans = LiveLoans::new(borrow_set.len());

    // Create the graph with the physical edges, and the logical edges of constraints that hold at
    // all points.
    let graph = LocalizedConstraintGraph::new(liveness, outlives_constraints);

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

            let location = liveness.location_from_point(node.point);
            if liveness.is_live_at(node.region, location) {
                live_loans.insert(node.point, loan_idx);
            }

            // Then, propagate the loan along the localized constraint graph. The outgoing edges are
            // computed lazily, from:
            // - the various physical edges present at this node,
            // - the materialized logical edges that exist virtually at all points for this node's
            //   region, localized at this point.

            // Universal regions propagate loans along the CFG, i.e. forwards only.
            let is_universal_region = universal_regions.is_universal_region(node.region);

            // The physical edges present at this node are:
            //
            // 1. the typeck edges that flow from region to region *at this point*.
            for &succ in graph.edges.get(&node).into_iter().flatten() {
                let succ = LocalizedNode { region: succ, point: node.point };
                if !visited.contains(&succ) {
                    stack.push(succ);
                }
            }

            // 2a. the liveness edges that flow *forward*, from this node's point to its successors
            // in the CFG.
            if body[location.block].statements.get(location.statement_index).is_some() {
                // Intra-block edges, straight line constraints from each point to its successor
                // within the same block.
                let next_point = node.point + 1;
                if let Some(succ) = compute_forward_successor(
                    node.region,
                    next_point,
                    live_regions,
                    live_region_variances,
                    is_universal_region,
                ) {
                    if !visited.contains(&succ) {
                        stack.push(succ);
                    }
                }
            } else {
                // Inter-block edges, from the block's terminator to each successor block's entry
                // point.
                for successor_block in body[location.block].terminator().successors() {
                    let next_location = Location { block: successor_block, statement_index: 0 };
                    let next_point = liveness.point_from_location(next_location);
                    if let Some(succ) = compute_forward_successor(
                        node.region,
                        next_point,
                        live_regions,
                        live_region_variances,
                        is_universal_region,
                    ) {
                        if !visited.contains(&succ) {
                            stack.push(succ);
                        }
                    }
                }
            }

            // 2b. the liveness edges that flow *backward*, from this node's point to its
            // predecessors in the CFG.
            if !is_universal_region {
                if location.statement_index > 0 {
                    // Backward edges to the predecessor point in the same block.
                    let previous_point = PointIndex::from(node.point.as_usize() - 1);
                    if let Some(succ) = compute_backward_successor(
                        node.region,
                        node.point,
                        previous_point,
                        live_regions,
                        live_region_variances,
                    ) {
                        if !visited.contains(&succ) {
                            stack.push(succ);
                        }
                    }
                } else {
                    // Backward edges from the block entry point to the terminator of the
                    // predecessor blocks.
                    let predecessors = body.basic_blocks.predecessors();
                    for &pred_block in &predecessors[location.block] {
                        let previous_location = Location {
                            block: pred_block,
                            statement_index: body[pred_block].statements.len(),
                        };
                        let previous_point = liveness.point_from_location(previous_location);
                        if let Some(succ) = compute_backward_successor(
                            node.region,
                            node.point,
                            previous_point,
                            live_regions,
                            live_region_variances,
                        ) {
                            if !visited.contains(&succ) {
                                stack.push(succ);
                            }
                        }
                    }
                }
            }

            // And finally, we have the logical edges, materialized at this point.
            for &logical_succ in graph.logical_edges.get(&node.region).into_iter().flatten() {
                let succ = LocalizedNode { region: logical_succ, point: node.point };
                if !visited.contains(&succ) {
                    stack.push(succ);
                }
            }
        }
    }

    live_loans
}

/// Returns the successor for the current region/point node when propagating a loan
/// through forward edges, if applicable, according to liveness and variance.
fn compute_forward_successor(
    region: RegionVid,
    next_point: PointIndex,
    live_regions: &SparseBitMatrix<PointIndex, RegionVid>,
    live_region_variances: &BTreeMap<RegionVid, ConstraintDirection>,
    is_universal_region: bool,
) -> Option<LocalizedNode> {
    // 1. Universal regions are semantically live at all points.
    if is_universal_region {
        let succ = LocalizedNode { region, point: next_point };
        return Some(succ);
    }

    // 2. Otherwise, gather the edges due to explicit region liveness, when applicable.
    if !live_regions.contains(next_point, region) {
        return None;
    }

    // Here, `region` could be live at the current point, and is live at the next point: add a
    // constraint between them, according to variance.

    // Note: there currently are cases related to promoted and const generics, where we don't yet
    // have variance information (possibly about temporary regions created when typeck sanitizes the
    // promoteds). Until that is done, we conservatively fallback to maximizing reachability by
    // adding a bidirectional edge here. This will not limit traversal whatsoever, and thus
    // propagate liveness when needed.
    //
    // FIXME: add the missing variance information and remove this fallback bidirectional edge.
    let direction =
        live_region_variances.get(&region).unwrap_or(&ConstraintDirection::Bidirectional);

    match direction {
        ConstraintDirection::Backward => {
            // Contravariant cases: loans flow in the inverse direction, but we're only interested
            // in forward successors and there are none here.
            None
        }
        ConstraintDirection::Forward | ConstraintDirection::Bidirectional => {
            // 1. For covariant cases: loans flow in the regular direction, from the current point
            // to the next point.
            // 2. For invariant cases, loans can flow in both directions, but here as well, we only
            // want the forward path of the bidirectional edge.
            Some(LocalizedNode { region, point: next_point })
        }
    }
}

/// Returns the successor for the current region/point node when propagating a loan
/// through backward edges, if applicable, according to liveness and variance.
fn compute_backward_successor(
    region: RegionVid,
    current_point: PointIndex,
    previous_point: PointIndex,
    live_regions: &SparseBitMatrix<PointIndex, RegionVid>,
    live_region_variances: &BTreeMap<RegionVid, ConstraintDirection>,
) -> Option<LocalizedNode> {
    // Liveness flows into the regions live at the next point. So, in a backwards view, we'll link
    // the region from the current point, if it's live there, to the previous point.
    if !live_regions.contains(current_point, region) {
        return None;
    }

    // FIXME: add the missing variance information and remove this fallback bidirectional edge. See
    // the same comment in `compute_forward_successor`.
    let direction =
        live_region_variances.get(&region).unwrap_or(&ConstraintDirection::Bidirectional);

    match direction {
        ConstraintDirection::Forward => {
            // Covariant cases: loans flow in the regular direction, but we're only interested in
            // backward successors and there are none here.
            None
        }
        ConstraintDirection::Backward | ConstraintDirection::Bidirectional => {
            // 1. For contravariant cases: loans flow in the inverse direction, from the current
            // point to the previous point.
            // 2. For invariant cases, loans can flow in both directions, but here as well, we only
            // want the backward path of the bidirectional edge.
            Some(LocalizedNode { region, point: previous_point })
        }
    }
}

/// The localized constraint graph indexes the physical and logical edges to lazily compute a given
/// node's successors during traversal.
struct LocalizedConstraintGraph {
    /// The actual, physical, edges we have recorded for a given node. We localize them on-demand
    /// when traversing from the node to the successor region.
    edges: FxHashMap<LocalizedNode, FxIndexSet<RegionVid>>,

    /// The logical edges representing the outlives constraints that hold at all points in the CFG,
    /// which we don't localize to avoid creating a lot of unnecessary edges in the graph. Some CFGs
    /// can be big, and we don't need to create such a physical edge for every point in the CFG.
    logical_edges: FxHashMap<RegionVid, FxIndexSet<RegionVid>>,
}

/// A node in the graph to be traversed, one of the two vertices of a localized outlives constraint.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct LocalizedNode {
    region: RegionVid,
    point: PointIndex,
}

impl LocalizedConstraintGraph {
    /// Traverses the constraints and returns the indexed graph of edges per node.
    fn new<'tcx>(
        liveness: &LivenessValues,
        outlives_constraints: impl Iterator<Item = OutlivesConstraint<'tcx>>,
    ) -> Self {
        let mut edges: FxHashMap<_, FxIndexSet<_>> = FxHashMap::default();
        let mut logical_edges: FxHashMap<_, FxIndexSet<_>> = FxHashMap::default();

        for outlives_constraint in outlives_constraints {
            match outlives_constraint.locations {
                Locations::All(_) => {
                    logical_edges
                        .entry(outlives_constraint.sup)
                        .or_default()
                        .insert(outlives_constraint.sub);
                }

                Locations::Single(location) => {
                    let node = LocalizedNode {
                        region: outlives_constraint.sup,
                        point: liveness.point_from_location(location),
                    };
                    edges.entry(node).or_default().insert(outlives_constraint.sub);
                }
            }
        }

        LocalizedConstraintGraph { edges, logical_edges }
    }
}
