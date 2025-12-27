use std::collections::BTreeMap;

use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexSet};
use rustc_index::bit_set::SparseBitMatrix;
use rustc_middle::mir::{Body, Location};
use rustc_middle::ty::{RegionVid, TyCtxt};
// use rustc_mir_dataflow::fmt::DebugWithAdapter;
use rustc_mir_dataflow::points::PointIndex;

use super::{LiveLoans, LocalizedOutlivesConstraintSet};
use crate::constraints::OutlivesConstraint;
use crate::polonius::ConstraintDirection;
// use crate::polonius::LocalizedOutlivesConstraint;
use crate::region_infer::values::LivenessValues;
use crate::type_check::Locations;
use crate::universal_regions::UniversalRegions;
use crate::{BorrowSet, RegionInferenceContext};

/// Compute loan reachability to approximately trace loan liveness throughout the CFG, by
/// traversing the full graph of constraints that combines:
/// - the localized constraints (the physical edges),
/// - with the constraints that hold at all points (the logical edges).
pub(super) fn compute_loan_liveness<'tcx>(
    _tcx: TyCtxt<'tcx>,
    liveness: &LivenessValues,
    // outlives_constraints: impl Iterator<Item = OutlivesConstraint<'tcx>>,
    regioncx: &RegionInferenceContext<'tcx>,
    borrow_set: &BorrowSet<'tcx>,
    localized_outlives_constraints: &LocalizedOutlivesConstraintSet,
    body: &Body<'tcx>,
    universal_regions: &UniversalRegions<'tcx>,
    live_regions: &SparseBitMatrix<PointIndex, RegionVid>,
    live_region_variances: &BTreeMap<RegionVid, crate::polonius::ConstraintDirection>,
) -> LiveLoans {
    // let _timer = std::time::Instant::now();

    let mut live_loans = LiveLoans::new(borrow_set.len());

    // let mut live_loans_orig = LiveLoans::new(borrow_set.len());

    // We want to traverse a (region, point) node to its successors computed by typeck.
    let mut outlives_per_node: FxHashMap<LocalizedNode, Vec<OutlivesConstraint<'tcx>>> =
        FxHashMap::default();
    for outlives_constraint in regioncx.outlives_constraints() {
        match outlives_constraint.locations {
            Locations::All(_) => {
                // We don't turn constraints holding at all points into physical edges at every
                // point in the graph. They are encoded into *traversal* instead: a given node's
                // successors will combine these logical edges with the regular, physical, localized
                // edges.
                continue;
            }

            Locations::Single(location) => {
                let node = LocalizedNode {
                    region: outlives_constraint.sup,
                    point: liveness.point_from_location(location),
                };
                outlives_per_node.entry(node).or_default().push(outlives_constraint);
            }
        }
    }

    // eprintln!(
    //     "compute_loan_liveness - 1ndexing constraints took: {} ns, {:?}, outlives constraints: {}",
    //     _timer.elapsed().as_nanos(),
    //     body.span,
    //     regioncx.outlives_constraints().count(),
    // );

    // let _timer = std::time::Instant::now();

    let outlives_constraints = regioncx.outlives_constraints();

    // Create the full graph with the physical edges we've localized earlier, and the logical edges
    // of constraints that hold at all points.
    let logical_constraints =
        outlives_constraints.filter(|c| matches!(c.locations, Locations::All(_)));
    let graph = LocalizedConstraintGraph::new(&localized_outlives_constraints, logical_constraints);

    // eprintln!(
    //     "compute_loan_liveness - 2ndexing constraints took: {} ns, {:?}, logical constraints: {}, localized constraints: {}",
    //     _timer.elapsed().as_nanos(),
    //     body.span,
    //     graph.logical_edges.len(),
    //     localized_outlives_constraints.outlives.len(),
    // );

    // let _timer = std::time::Instant::now();

    // let mut visited = FxHashSet::default();
    // let mut stack = Vec::new();

    // // Compute reachability per loan by traversing each loan's subgraph starting from where it is
    // // introduced.
    // for (loan_idx, loan) in borrow_set.iter_enumerated() {
    //     visited.clear();
    //     stack.clear();

    //     let start_node = LocalizedNode {
    //         region: loan.region,
    //         point: liveness.point_from_location(loan.reserve_location),
    //     };
    //     stack.push(start_node);

    //     while let Some(node) = stack.pop() {
    //         if !visited.insert(node) {
    //             continue;
    //         }

    //         // Record the loan as being live on entry to this point if it reaches a live region
    //         // there.
    //         //
    //         // This is an approximation of liveness (which is the thing we want), in that we're
    //         // using a single notion of reachability to represent what used to be _two_ different
    //         // transitive closures. It didn't seem impactful when coming up with the single-graph
    //         // and reachability through space (regions) + time (CFG) concepts, but in practice the
    //         // combination of time-traveling with kills is more impactful than initially
    //         // anticipated.
    //         //
    //         // Kills should prevent a loan from reaching its successor points in the CFG, but not
    //         // while time-traveling: we're not actually at that CFG point, but looking for
    //         // predecessor regions that contain the loan. One of the two TCs we had pushed the
    //         // transitive subset edges to each point instead of having backward edges, and the
    //         // problem didn't exist before. In the abstract, naive reachability is not enough to
    //         // model this, we'd need a slightly different solution. For example, maybe with a
    //         // two-step traversal:
    //         // - at each point we first traverse the subgraph (and possibly time-travel) looking for
    //         //   exit nodes while ignoring kills,
    //         // - and then when we're back at the current point, we continue normally.
    //         //
    //         // Another (less annoying) subtlety is that kills and the loan use-map are
    //         // flow-insensitive. Kills can actually appear in places before a loan is introduced, or
    //         // at a location that is actually unreachable in the CFG from the introduction point,
    //         // and these can also be encountered during time-traveling.
    //         //
    //         // The simplest change that made sense to "fix" the issues above is taking into
    //         // account kills that are:
    //         // - reachable from the introduction point
    //         // - encountered during forward traversal. Note that this is not transitive like the
    //         //   two-step traversal described above: only kills encountered on exit via a backward
    //         //   edge are ignored.
    //         //
    //         // This version of the analysis, however, is enough in practice to pass the tests that
    //         // we care about and NLLs reject, without regressions on crater, and is an actionable
    //         // subset of the full analysis. It also naturally points to areas of improvement that we
    //         // wish to explore later, namely handling kills appropriately during traversal, instead
    //         // of continuing traversal to all the reachable nodes.
    //         //
    //         // FIXME: analyze potential unsoundness, possibly in concert with a borrowck
    //         // implementation in a-mir-formality, fuzzing, or manually crafting counter-examples.

    //         if liveness.is_live_at(node.region, liveness.location_from_point(node.point)) {
    //             // live_loans.insert(node.point, loan_idx);
    //             live_loans_orig.insert(node.point, loan_idx);
    //         }

    //         let _location = liveness.location_from_point(node.point);
    //         for succ in graph.outgoing_edges(node) {
    //             // eprintln!(
    //             //     "B - region {:?} at {:?} has successor region {:?} at {:?}",
    //             //     node.region,
    //             //     _location,
    //             //     succ.region,
    //             //     liveness.location_from_point(succ.point),
    //             // );

    //             // if node.region.as_u32() == 17 && succ.region.as_u32() == 5 {
    //             //     eprintln!("B2, node: {node:?}, succ: {succ:?}");
    //             //     // eprintln!("outlives: {:?}", outlives_per_point.get(&node.point));
    //             //     for c in &localized_outlives_constraints.outlives {
    //             //         if c.source == node.region && c.target == succ.region {
    //             //             eprintln!("localized outlives constraint: {c:?}");
    //             //         }
    //             //     }
    //             // }

    //             stack.push(succ);
    //         }
    //     }
    // }

    // eprintln!(
    //     "compute_loan_liveness - loan propagat1on took:     {} ns, {:?}, logical constraints: {}, localized constraints: {}",
    //     _timer.elapsed().as_nanos(),
    //     body.span,
    //     graph.logical_edges.len(),
    //     localized_outlives_constraints.outlives.len(),
    // );

    // let _timer = std::time::Instant::now();

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

            // The outgoing edges are:
            // - the physical edges present at this node,
            // - the materialized logical edges that exist virtually at all points for this node's
            //   region, localized at this point.

            let location = liveness.location_from_point(node.point);

            // let mut successors = Vec::new();

            // The physical edges present at this node are:
            //
            // 1. the typeck edges that flow from region to region *at this point*
            for outlives_constraint in outlives_per_node.get(&node).into_iter().flatten() {
                let succ = LocalizedNode { region: outlives_constraint.sub, point: node.point };
                stack.push(succ);
                // successors.push(succ);
            }

            // 2a. the liveness edges that flow *forward*, from this node's point to its successors
            // in the CFG.
            // FIXME: this shouldn't need to take a successors vec, and only return an optional
            // successor.
            fn propagate_loans_forward(
                region: RegionVid,
                next_point: PointIndex,
                live_regions: &SparseBitMatrix<PointIndex, RegionVid>,
                live_region_variances: &BTreeMap<RegionVid, ConstraintDirection>,
                universal_regions: &UniversalRegions<'_>,
                successors: &mut Vec<LocalizedNode>,
            ) {
                // 1. Universal regions are semantically live at all points.
                if universal_regions.is_universal_region(region) {
                    let succ = LocalizedNode { region, point: next_point };
                    successors.push(succ);

                    // FIXME: we shouldn't need to continue the function and should return here;
                    // universal regions are most likely not even marked live at the next point
                }

                // 2. Gather the edges due to explicit region liveness.
                let Some(next_live_regions) = live_regions.row(next_point) else {
                    // There are no constraints to add: there are no live regions at the next point.
                    return;
                };

                // `region` could be live at the current point, and is live at the next
                // point: add a constraint between them, according to variance.
                if !next_live_regions.contains(region) {
                    return;
                }

                // Note: there currently are cases related to promoted and const generics,
                // where we don't yet have variance information (possibly about temporary
                // regions created when typeck sanitizes the promoteds). Until that is done,
                // we conservatively fallback to maximizing reachability by adding a
                // bidirectional edge here. This will not limit traversal whatsoever, and
                // thus propagate liveness when needed.
                //
                // FIXME: add the missing variance information and remove this fallback
                // bidirectional edge.
                let direction = live_region_variances
                    .get(&region)
                    .unwrap_or(&ConstraintDirection::Bidirectional);

                match direction {
                    ConstraintDirection::Forward => {
                        // Covariant cases: loans flow in the regular direction, from the
                        // current point to the next point.
                        let succ = LocalizedNode { region, point: next_point };
                        successors.push(succ);
                    }
                    ConstraintDirection::Backward => {
                        // Contravariant cases: loans flow in the inverse direction, but
                        // we're only interested in forward successors and there
                        // are none here.
                    }
                    ConstraintDirection::Bidirectional => {
                        // For invariant cases, loans can flow in both directions, but
                        // here as well, we only want the forward path of the
                        // bidirectional edge.
                        let succ = LocalizedNode { region, point: next_point };
                        successors.push(succ);
                    }
                }
            }

            if body[location.block].statements.get(location.statement_index).is_some() {
                // Intra-block edges, straight line constraints from each point to its successor
                // within the same block.
                let next_location = Location {
                    block: location.block,
                    statement_index: location.statement_index + 1,
                };
                let next_point = liveness.point_from_location(next_location);
                // FIXME: the above should not be needed, the next point in a block should be
                // current_point_idx + 1, only an unknown block needs some translation from the
                // locationmap
                propagate_loans_forward(
                    node.region,
                    next_point,
                    live_regions,
                    live_region_variances,
                    universal_regions,
                    // &mut successors,
                    &mut stack,
                );
            } else {
                // Inter-block edges, from the block's terminator to each successor block's entry
                // point.
                for successor_block in body[location.block].terminator().successors() {
                    let next_location = Location { block: successor_block, statement_index: 0 };
                    let next_point = liveness.point_from_location(next_location);
                    propagate_loans_forward(
                        node.region,
                        next_point,
                        live_regions,
                        live_region_variances,
                        universal_regions,
                        // &mut successors,
                        &mut stack,
                    );
                }
            }

            // 2b. the liveness edges that flow *backward*, from this node's point to its predecessors
            // in the CFG.
            fn propagate_loans_backward(
                region: RegionVid,
                current_point: PointIndex,
                previous_point: PointIndex,
                live_regions: &SparseBitMatrix<PointIndex, RegionVid>,
                live_region_variances: &BTreeMap<RegionVid, ConstraintDirection>,
                successors: &mut Vec<LocalizedNode>,
            ) {
                // Since liveness flows into the regions live at the next point, we compute the
                // regions live at the current point, and gather the backward edges flowing to the
                // predecessor point.
                let Some(current_live_regions) = live_regions.row(current_point) else {
                    // There are no constraints to add: there are no live regions at the current point.
                    return;
                };

                // `region` could be live at the previous point, and is live at the current
                // point: add a constraint between them, according to variance.
                if !current_live_regions.contains(region) {
                    return;
                }

                // Note: there currently are cases related to promoted and const generics,
                // where we don't yet have variance information (possibly about temporary
                // regions created when typeck sanitizes the promoteds). Until that is done,
                // we conservatively fallback to maximizing reachability by adding a
                // bidirectional edge here. This will not limit traversal whatsoever, and
                // thus propagate liveness when needed.
                //
                // FIXME: add the missing variance information and remove this fallback
                // bidirectional edge.
                let direction = live_region_variances
                    .get(&region)
                    .unwrap_or(&ConstraintDirection::Bidirectional);

                match direction {
                    ConstraintDirection::Forward => {
                        // Covariant cases: loans flow in the regular direction, but we're only
                        // interested in backward successors and there are none here.
                    }
                    ConstraintDirection::Backward => {
                        // Contravariant cases: loans flow in the inverse direction, from the next
                        // point to the current point.
                        let succ = LocalizedNode { region, point: previous_point };
                        successors.push(succ);
                    }
                    ConstraintDirection::Bidirectional => {
                        // For invariant cases, loans can flow in both directions, but
                        // here as well, we only want the backward path of the
                        // bidirectional edge.
                        let succ = LocalizedNode { region, point: previous_point };
                        successors.push(succ);
                    }
                }
            }

            if location.statement_index > 0 {
                // Intra-block edges. We want the predecessor point in the same block.
                let previous_location = Location {
                    block: location.block,
                    statement_index: location.statement_index - 1,
                };
                // FIXME: here as well, the previous point within a block is just the index - 1, we don't
                // need to go through a Location.
                let previous_point = liveness.point_from_location(previous_location);
                propagate_loans_backward(
                    node.region,
                    node.point,
                    previous_point,
                    live_regions,
                    live_region_variances,
                    // &mut successors,
                    &mut stack,
                );
            } else {
                // Block entry point, we want the inter-block edges. The terminator of the predecessor block.
                for &pred in &body.basic_blocks.predecessors()[location.block] {
                    let previous_location =
                        Location { block: pred, statement_index: body[pred].statements.len() };
                    let previous_point = liveness.point_from_location(previous_location);
                    propagate_loans_backward(
                        node.region,
                        node.point,
                        previous_point,
                        live_regions,
                        live_region_variances,
                        // &mut successors,
                        &mut stack,
                    );
                }
            }

            // 3. The logical edges, materialized at this point.
            for logical_succ in graph.logical_edges.get(&node.region).into_iter().flatten() {
                let succ = LocalizedNode { region: *logical_succ, point: node.point };
                // successors.push(succ);
                stack.push(succ);
            }

            // ---

            // let mut valid_edges: Vec<_> = graph.outgoing_edges(node).collect();
            // let mut invalid_edges: Vec<_> = successors.iter().copied().collect();

            // valid_edges.sort();
            // invalid_edges.sort();
            // valid_edges.dedup();
            // invalid_edges.dedup();

            // fn prepare(it: impl Iterator<Item = LocalizedNode>) -> Vec<LocalizedNode> {
            //     let mut edges: Vec<_> = it.collect();
            //     edges.sort();
            //     edges.dedup();
            //     edges
            // }

            // let _physical_edges = prepare(
            //     graph.edges.get(&node).into_iter().flat_map(|targets| targets.iter().copied()),
            // );
            // let print = |edges: &Vec<LocalizedNode>| {
            //     edges
            //         .iter()
            //         .map(|e| format!(
            //             "{}@{:?}",
            //             e.region.as_u32(),
            //             liveness.location_from_point(e.point),
            //         ))
            //         .collect::<Vec<_>>()
            // };

            // assert_eq!(
            //     valid_edges,
            //     invalid_edges,
            //     // "edges differed for region {:?} at {:?}, valid: {:?} (physical: {:?}), invalid: {:?}",
            //     "edges differed for region {:?} at {:?}, valid: {:?}, invalid: {:?}",
            //     node.region,
            //     location,
            //     print(&valid_edges),
            //     // print(&_physical_edges),
            //     print(&invalid_edges),
            // );

            // for succ in successors {
            //     stack.push(succ);
            // }
        }
    }

    // for (point, point_orig) in live_loans.rows().zip(live_loans_orig.rows()) {
    //     assert_eq!(point, point_orig, "points differ");
    //     let loans_new = live_loans.row(point);
    //     let loans_orig = live_loans_orig.row(point);
    //     assert_eq!(
    //         loans_orig,
    //         loans_new,
    //         "live loans differ at {:?}",
    //         liveness.location_from_point(point),
    //     );
    // }

    // eprintln!(
    //     "compute_loan_liveness - loan propagat2on took:     {} ns, {:?}, logical constraints: {}, localized constraints: {}",
    //     _timer.elapsed().as_nanos(),
    //     body.span,
    //     graph.logical_edges.len(),
    //     localized_outlives_constraints.outlives.len(),
    // );

    live_loans
}

/// The localized constraint graph indexes the physical and logical edges to compute a given node's
/// successors during traversal.
struct LocalizedConstraintGraph {
    /// The actual, physical, edges we have recorded for a given node.
    // edges: FxHashMap<LocalizedNode, FxIndexSet<LocalizedNode>>,

    /// The logical edges representing the outlives constraints that hold at all points in the CFG,
    /// which we don't localize to avoid creating a lot of unnecessary edges in the graph. Some CFGs
    /// can be big, and we don't need to create such a physical edge for every point in the CFG.
    logical_edges: FxHashMap<RegionVid, FxIndexSet<RegionVid>>,
}

/// A node in the graph to be traversed, one of the two vertices of a localized outlives constraint.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Ord, PartialOrd)]
struct LocalizedNode {
    region: RegionVid,
    point: PointIndex,
}

impl LocalizedConstraintGraph {
    /// Traverses the constraints and returns the indexed graph of edges per node.
    fn new<'tcx>(
        _constraints: &LocalizedOutlivesConstraintSet,
        logical_constraints: impl Iterator<Item = OutlivesConstraint<'tcx>>,
    ) -> Self {
        // let edges: FxHashMap<_, FxIndexSet<_>> = FxHashMap::default();
        // for constraint in &constraints.outlives {
        //     let source = LocalizedNode { region: constraint.source, point: constraint.from };
        //     let target = LocalizedNode { region: constraint.target, point: constraint.to };
        //     edges.entry(source).or_default().insert(target);
        // }

        let mut logical_edges: FxHashMap<_, FxIndexSet<_>> = FxHashMap::default();
        for constraint in logical_constraints {
            logical_edges.entry(constraint.sup).or_default().insert(constraint.sub);
        }

        LocalizedConstraintGraph { logical_edges }
    }

    // /// Returns the outgoing edges of a given node, not its transitive closure.
    // fn outgoing_edges(&self, node: LocalizedNode) -> impl Iterator<Item = LocalizedNode> {
    //     // The outgoing edges are:
    //     // - the physical edges present at this node,
    //     // - the materialized logical edges that exist virtually at all points for this node's
    //     //   region, localized at this point.
    //     let physical_edges =
    //         self.edges.get(&node).into_iter().flat_map(|targets| targets.iter().copied());
    //     let materialized_edges =
    //         self.logical_edges.get(&node.region).into_iter().flat_map(move |targets| {
    //             targets
    //                 .iter()
    //                 .copied()
    //                 .map(move |target| LocalizedNode { point: node.point, region: target })
    //         });
    //     physical_edges.chain(materialized_edges)
    // }
}
