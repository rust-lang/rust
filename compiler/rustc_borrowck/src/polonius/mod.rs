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
pub(crate) use constraints::*;
mod dump;
pub(crate) use dump::dump_polonius_mir;
pub(crate) mod legacy;

use rustc_middle::mir::{Body, Location};
use rustc_mir_dataflow::points::PointIndex;

use crate::RegionInferenceContext;
use crate::constraints::OutlivesConstraint;
use crate::region_infer::values::LivenessValues;
use crate::type_check::Locations;
use crate::universal_regions::UniversalRegions;

/// Creates a constraint set for `-Zpolonius=next` by:
/// - converting NLL typeck constraints to be localized
/// - encoding liveness constraints
pub(crate) fn create_localized_constraints<'tcx>(
    regioncx: &mut RegionInferenceContext<'tcx>,
    body: &Body<'tcx>,
) -> LocalizedOutlivesConstraintSet {
    let mut localized_outlives_constraints = LocalizedOutlivesConstraintSet::default();
    convert_typeck_constraints(
        body,
        regioncx.liveness_constraints(),
        regioncx.outlives_constraints(),
        &mut localized_outlives_constraints,
    );
    create_liveness_constraints(
        body,
        regioncx.liveness_constraints(),
        regioncx.universal_regions(),
        &mut localized_outlives_constraints,
    );

    // FIXME: here, we can trace loan reachability in the constraint graph and record this as loan
    // liveness for the next step in the chain, the NLL loan scope and active loans computations.

    localized_outlives_constraints
}

/// Propagate loans throughout the subset graph at a given point (with some subtleties around the
/// location where effects start to be visible).
fn convert_typeck_constraints<'tcx>(
    body: &Body<'tcx>,
    liveness: &LivenessValues,
    outlives_constraints: impl Iterator<Item = OutlivesConstraint<'tcx>>,
    localized_outlives_constraints: &mut LocalizedOutlivesConstraintSet,
) {
    for outlives_constraint in outlives_constraints {
        match outlives_constraint.locations {
            Locations::All(_) => {
                // For now, turn logical constraints holding at all points into physical edges at
                // every point in the graph.
                // FIXME: encode this into *traversal* instead.
                for (block, bb) in body.basic_blocks.iter_enumerated() {
                    let statement_count = bb.statements.len();
                    for statement_index in 0..=statement_count {
                        let current_location = Location { block, statement_index };
                        let current_point = liveness.point_from_location(current_location);

                        localized_outlives_constraints.push(LocalizedOutlivesConstraint {
                            source: outlives_constraint.sup,
                            from: current_point,
                            target: outlives_constraint.sub,
                            to: current_point,
                        });
                    }
                }
            }

            _ => {}
        }
    }
}

/// Propagate loans throughout the CFG: for each statement in the MIR, create localized outlives
/// constraints for loans that are propagated to the next statements.
pub(crate) fn create_liveness_constraints<'tcx>(
    body: &Body<'tcx>,
    liveness: &LivenessValues,
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
                    liveness,
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
                        liveness,
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
    _liveness: &LivenessValues,
    universal_regions: &UniversalRegions<'_>,
    localized_outlives_constraints: &mut LocalizedOutlivesConstraintSet,
) {
    // Universal regions are semantically live at all points.
    // Note: we always have universal regions but they're not always (or often) involved in the
    // subset graph. For now, we emit all their edges unconditionally, but some of these subgraphs
    // will be disconnected from the rest of the graph and thus, unnecessary.
    // FIXME: only emit the edges of universal regions that existential regions can reach.
    for region in universal_regions.universal_regions_iter() {
        localized_outlives_constraints.push(LocalizedOutlivesConstraint {
            source: region,
            from: current_point,
            target: region,
            to: next_point,
        });
    }
}
