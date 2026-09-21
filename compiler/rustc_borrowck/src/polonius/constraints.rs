use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexSet};
use rustc_index::IndexVec;
use rustc_middle::mir::{Body, Location};
use rustc_middle::ty::RegionVid;
use rustc_mir_dataflow::points::PointIndex;

use crate::BorrowSet;
use crate::constraints::OutlivesConstraint;
use crate::dataflow::BorrowIndex;
use crate::polonius::{ConstraintDirection, LiveRegionVariances};
use crate::region_infer::values::LivenessValues;
use crate::type_check::Locations;
use crate::universal_regions::UniversalRegions;

/// A localized outlives constraint reifies the CFG location where the outlives constraint holds,
/// within the origins themselves as if they were different from point to point: from `a: b`
/// outlives constraints to `a@p: b@p`, where `p` is the point in the CFG.
///
/// This models two sources of constraints:
/// - constraints that traverse the subsets between regions at a given point, `a@p: b@p`. These
///   depend on typeck constraints generated via assignments, calls, etc.
/// - constraints that traverse the CFG via the same region, `a@p: a@q`, where `p` is a predecessor
///   of `q`. These depend on the liveness of the regions at these points, as well as their
///   variance.
///
/// This dual of NLL's [crate::constraints::OutlivesConstraint] therefore encodes the
/// position-dependent outlives constraints used by Polonius, to model the flow-sensitive loan
/// propagation via reachability within a graph of localized constraints.
///
/// That `LocalizedConstraintGraph` can create these edges on-demand during traversal, and we
/// therefore model them as a pair of `LocalizedNode` vertices.
///
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub(super) struct LocalizedNode {
    pub region: RegionVid,
    pub point: PointIndex,
}

/// The localized constraint graph indexes the physical and logical edges to lazily compute a given
/// node's successors during traversal.
pub(super) struct LocalizedConstraintGraph {
    /// The actual, physical, edges we have recorded for a given node. We localize them on-demand
    /// when traversing from the node to the successor region.
    edges: FxHashMap<LocalizedNode, FxIndexSet<RegionVid>>,

    /// The logical edges representing the outlives constraints that hold at all points in the CFG,
    /// which we don't localize to avoid creating a lot of unnecessary edges in the graph. Some CFGs
    /// can be big, and we don't need to create such a physical edge for every point in the CFG.
    logical_edges: IndexVec<RegionVid, FxIndexSet<RegionVid>>,
}

/// The visitor interface when traversing a `LocalizedConstraintGraph`.
pub(super) trait LocalizedConstraintGraphVisitor {
    /// Callback called when traversing a given `loan` encounters a localized `node` it hasn't
    /// visited before.
    fn on_node_traversed(&mut self, _loan: BorrowIndex, _node: LocalizedNode) {}

    /// Callback called when discovering a new `successor` node for the `current_node`.
    fn on_successor_discovered(&mut self, _current_node: LocalizedNode, _successor: LocalizedNode) {
    }
}

impl LocalizedConstraintGraph {
    /// Traverses the constraints and returns the indexed graph of edges per node.
    pub(super) fn new<'tcx>(
        liveness: &LivenessValues,
        outlives_constraints: impl Iterator<Item = OutlivesConstraint<'tcx>>,
    ) -> Self {
        let mut edges: FxHashMap<_, FxIndexSet<_>> = FxHashMap::default();
        let mut logical_edges: IndexVec<_, FxIndexSet<_>> = IndexVec::new();

        for outlives_constraint in outlives_constraints {
            match outlives_constraint.locations {
                Locations::All(_) => {
                    logical_edges
                        .ensure_contains_elem(outlives_constraint.sup, FxIndexSet::default)
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

    /// Traverses the localized constraint graph per-loan, and notifies the `visitor` of discovered
    /// nodes and successors.
    pub(super) fn traverse<'tcx>(
        &self,
        body: &Body<'tcx>,
        liveness: &LivenessValues,
        live_region_variances: &LiveRegionVariances,
        universal_regions: &UniversalRegions<'tcx>,
        borrow_set: &BorrowSet<'tcx>,
        visitor: &mut impl LocalizedConstraintGraphVisitor,
    ) {
        let live_regions = liveness.points();

        let mut visited = FxHashSet::default();
        let mut stack = Vec::new();

        // Compute reachability per loan by traversing each loan's subgraph starting from where it
        // is introduced.
        for (loan_idx, loan) in borrow_set.iter_enumerated() {
            visited.clear();
            stack.clear();

            let start_node = LocalizedNode {
                region: loan.region,
                point: liveness.point_from_location(loan.reserve_location),
            };
            visited.insert(start_node);
            stack.push(start_node);

            while let Some(node) = stack.pop() {
                // We've reached a node we haven't visited before.
                let location = liveness.location_from_point(node.point);
                visitor.on_node_traversed(loan_idx, node);

                // When we find a _new_ successor, we'd like to
                // - visit it eventually,
                // - and let the generic visitor know about it.
                let mut successor_found = |succ| {
                    if visited.insert(succ) {
                        stack.push(succ);
                        visitor.on_successor_discovered(node, succ);
                    }
                };

                // Then, we propagate the loan along the localized constraint graph. The outgoing
                // edges are computed lazily, from:
                // - the various physical edges present at this node,
                // - the materialized logical edges that exist virtually at all points for this
                //   node's region, localized at this point.

                // Universal regions propagate loans along the CFG, i.e. forwards only.
                let is_universal_region = universal_regions.is_universal_region(node.region);

                // Note: there currently are cases related to promoted and const generics, where we don't yet
                // have variance information (possibly about temporary regions created when typeck sanitizes the
                // promoteds). Until that is done, we conservatively fallback to maximizing reachability by
                // adding a bidirectional edge here. This will not limit traversal whatsoever, and thus
                // propagate liveness when needed.
                //
                // FIXME: add the missing variance information and remove this fallback bidirectional edge.
                let liveness_direction = if is_universal_region {
                    ConstraintDirection::Forward
                } else {
                    live_region_variances
                        .get(node.region)
                        .copied()
                        .flatten()
                        .unwrap_or(ConstraintDirection::Bidirectional)
                };

                // The physical edges present at this node are:
                //
                // 1. the typeck edges that flow from region to region *at this point*.
                for &succ in self.edges.get(&node).into_flat_iter() {
                    let succ = LocalizedNode { region: succ, point: node.point };
                    successor_found(succ);
                }

                // 2a. the liveness edges that flow *forward*, from this node's point to its
                // successors in the CFG.
                //
                // - for covariant cases: loans flow in the regular direction, from the current point
                // to the next point.
                // - for invariant cases, loans can flow in both directions, but here we're only
                // interested in the forward path of the bidirectional edge.
                //
                // We still need to check liveness for each next point though.
                if matches!(
                    liveness_direction,
                    ConstraintDirection::Forward | ConstraintDirection::Bidirectional
                ) {
                    if body[location.block].statements.get(location.statement_index).is_some() {
                        // Intra-block edges, straight line constraints from each point to its successor
                        // within the same block.
                        let next_point = node.point + 1;
                        if live_regions.contains(node.region, next_point) {
                            successor_found(LocalizedNode {
                                region: node.region,
                                point: next_point,
                            });
                        }
                    } else {
                        // Inter-block edges, from the block's terminator to each successor block's
                        // entry point.
                        for successor_block in body[location.block].terminator().successors() {
                            let next_location =
                                Location { block: successor_block, statement_index: 0 };
                            let next_point = liveness.point_from_location(next_location);
                            if live_regions.contains(node.region, next_point) {
                                successor_found(LocalizedNode {
                                    region: node.region,
                                    point: next_point,
                                });
                            }
                        }
                    }
                }

                // 2b. the liveness edges that flow *backward*, from this node's point to its
                // predecessors in the CFG.
                //
                // - for contravariant cases: loans flow in the inverse direction, from the current
                // point to the previous point.
                // - for invariant cases, loans can flow in both directions, but here we only
                // want the backward path of the bidirectional edge.
                //
                // Liveness flows into the regions live at the next point. So, in a backwards view, we'll link
                // the region from the current point, if it's live there, to the previous point.
                if matches!(
                    liveness_direction,
                    ConstraintDirection::Backward | ConstraintDirection::Bidirectional
                ) && live_regions.contains(node.region, node.point)
                {
                    if location.statement_index > 0 {
                        // Backward edges to the predecessor point in the same block.
                        let previous_point = PointIndex::from(node.point.as_usize() - 1);
                        successor_found(LocalizedNode {
                            region: node.region,
                            point: previous_point,
                        });
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
                            successor_found(LocalizedNode {
                                region: node.region,
                                point: previous_point,
                            });
                        }
                    }
                }

                // And finally, we have the logical edges, materialized at this point.
                let logical_succs = self.logical_edges.get(node.region);
                for &logical_succ in logical_succs.into_flat_iter() {
                    let succ = LocalizedNode { region: logical_succ, point: node.point };
                    successor_found(succ);
                }
            }
        }
    }
}
