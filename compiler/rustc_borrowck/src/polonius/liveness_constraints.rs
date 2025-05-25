use std::collections::BTreeMap;

use rustc_index::bit_set::SparseBitMatrix;
use rustc_middle::mir::{Body, Location};
use rustc_middle::ty::relate::{self, Relate, RelateResult, TypeRelation};
use rustc_middle::ty::{self, RegionVid, Ty, TyCtxt, TypeVisitable};
use rustc_mir_dataflow::points::PointIndex;

use super::{
    ConstraintDirection, LocalizedOutlivesConstraint, LocalizedOutlivesConstraintSet,
    PoloniusLivenessContext,
};
use crate::region_infer::values::LivenessValues;
use crate::universal_regions::UniversalRegions;

/// Propagate loans throughout the CFG: for each statement in the MIR, create localized outlives
/// constraints for loans that are propagated to the next statements.
pub(super) fn create_liveness_constraints<'tcx>(
    body: &Body<'tcx>,
    liveness: &LivenessValues,
    live_regions: &SparseBitMatrix<PointIndex, RegionVid>,
    live_region_variances: &BTreeMap<RegionVid, ConstraintDirection>,
    universal_regions: &UniversalRegions<'tcx>,
    localized_outlives_constraints: &mut LocalizedOutlivesConstraintSet,
) {
    for (block, bb) in body.basic_blocks.iter_enumerated() {
        let statement_count = bb.statements.len();
        for statement_index in 0..=statement_count {
            let current_location = Location { block, statement_index };
            let current_point = liveness.point_from_location(current_location);

            if statement_index < statement_count {
                // Intra-block edges, straight line constraints from each point to its successor
                // within the same block.
                let next_location = Location { block, statement_index: statement_index + 1 };
                let next_point = liveness.point_from_location(next_location);
                propagate_loans_between_points(
                    current_point,
                    next_point,
                    live_regions,
                    live_region_variances,
                    universal_regions,
                    localized_outlives_constraints,
                );
            } else {
                // Inter-block edges, from the block's terminator to each successor block's entry
                // point.
                for successor_block in bb.terminator().successors() {
                    let next_location = Location { block: successor_block, statement_index: 0 };
                    let next_point = liveness.point_from_location(next_location);
                    propagate_loans_between_points(
                        current_point,
                        next_point,
                        live_regions,
                        live_region_variances,
                        universal_regions,
                        localized_outlives_constraints,
                    );
                }
            }
        }
    }
}

/// Propagate loans within a region between two points in the CFG, if that region is live at both
/// the source and target points.
fn propagate_loans_between_points(
    current_point: PointIndex,
    next_point: PointIndex,
    live_regions: &SparseBitMatrix<PointIndex, RegionVid>,
    live_region_variances: &BTreeMap<RegionVid, ConstraintDirection>,
    universal_regions: &UniversalRegions<'_>,
    localized_outlives_constraints: &mut LocalizedOutlivesConstraintSet,
) {
    // Universal regions are semantically live at all points.
    // Note: we always have universal regions but they're not always (or often) involved in the
    // subset graph. For now, we emit all their edges unconditionally, but some of these subgraphs
    // will be disconnected from the rest of the graph and thus, unnecessary.
    //
    // FIXME: only emit the edges of universal regions that existential regions can reach.
    for region in universal_regions.universal_regions_iter() {
        localized_outlives_constraints.push(LocalizedOutlivesConstraint {
            source: region,
            from: current_point,
            target: region,
            to: next_point,
        });
    }

    let Some(current_live_regions) = live_regions.row(current_point) else {
        // There are no constraints to add: there are no live regions at the current point.
        return;
    };
    let Some(next_live_regions) = live_regions.row(next_point) else {
        // There are no constraints to add: there are no live regions at the next point.
        return;
    };

    for region in next_live_regions.iter() {
        if !current_live_regions.contains(region) {
            continue;
        }

        // `region` is indeed live at both points, add a constraint between them, according to
        // variance.
        if let Some(&direction) = live_region_variances.get(&region) {
            add_liveness_constraint(
                region,
                current_point,
                next_point,
                direction,
                localized_outlives_constraints,
            );
        } else {
            // Note: there currently are cases related to promoted and const generics, where we
            // don't yet have variance information (possibly about temporary regions created when
            // typeck sanitizes the promoteds). Until that is done, we conservatively fallback to
            // maximizing reachability by adding a bidirectional edge here. This will not limit
            // traversal whatsoever, and thus propagate liveness when needed.
            //
            // FIXME: add the missing variance information and remove this fallback bidirectional
            // edge.
            let fallback = ConstraintDirection::Bidirectional;
            add_liveness_constraint(
                region,
                current_point,
                next_point,
                fallback,
                localized_outlives_constraints,
            );
        }
    }
}

/// Adds `LocalizedOutlivesConstraint`s between two connected points, according to the given edge
/// direction.
fn add_liveness_constraint(
    region: RegionVid,
    current_point: PointIndex,
    next_point: PointIndex,
    direction: ConstraintDirection,
    localized_outlives_constraints: &mut LocalizedOutlivesConstraintSet,
) {
    match direction {
        ConstraintDirection::Forward => {
            // Covariant cases: loans flow in the regular direction, from the current point to the
            // next point.
            localized_outlives_constraints.push(LocalizedOutlivesConstraint {
                source: region,
                from: current_point,
                target: region,
                to: next_point,
            });
        }
        ConstraintDirection::Backward => {
            // Contravariant cases: loans flow in the inverse direction, from the next point to the
            // current point.
            localized_outlives_constraints.push(LocalizedOutlivesConstraint {
                source: region,
                from: next_point,
                target: region,
                to: current_point,
            });
        }
        ConstraintDirection::Bidirectional => {
            // For invariant cases, loans can flow in both directions: we add both edges.
            localized_outlives_constraints.push(LocalizedOutlivesConstraint {
                source: region,
                from: current_point,
                target: region,
                to: next_point,
            });
            localized_outlives_constraints.push(LocalizedOutlivesConstraint {
                source: region,
                from: next_point,
                target: region,
                to: current_point,
            });
        }
    }
}
