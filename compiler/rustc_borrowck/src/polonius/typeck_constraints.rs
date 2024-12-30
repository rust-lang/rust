use rustc_middle::mir::{Body, Location};

use super::{LocalizedOutlivesConstraint, LocalizedOutlivesConstraintSet};
use crate::constraints::OutlivesConstraint;
use crate::region_infer::values::LivenessValues;
use crate::type_check::Locations;

/// Propagate loans throughout the subset graph at a given point (with some subtleties around the
/// location where effects start to be visible).
pub(super) fn convert_typeck_constraints<'tcx>(
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
