use rustc_data_structures::fx::FxHashSet;
use rustc_middle::mir::{Body, Location, Statement, StatementKind, Terminator, TerminatorKind};
use rustc_middle::ty::{TyCtxt, TypeVisitable};
use rustc_mir_dataflow::points::PointIndex;

use super::{LocalizedOutlivesConstraint, LocalizedOutlivesConstraintSet};
use crate::constraints::OutlivesConstraint;
use crate::region_infer::values::LivenessValues;
use crate::type_check::Locations;
use crate::universal_regions::UniversalRegions;

/// Propagate loans throughout the subset graph at a given point (with some subtleties around the
/// location where effects start to be visible).
pub(super) fn convert_typeck_constraints<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    liveness: &LivenessValues,
    outlives_constraints: impl Iterator<Item = OutlivesConstraint<'tcx>>,
    universal_regions: &UniversalRegions<'tcx>,
    localized_outlives_constraints: &mut LocalizedOutlivesConstraintSet,
) {
    for outlives_constraint in outlives_constraints {
        match outlives_constraint.locations {
            Locations::All(_) => {
                // We don't turn constraints holding at all points into physical edges at every
                // point in the graph. They are encoded into *traversal* instead: a given node's
                // successors will combine these logical edges with the regular, physical, localized
                // edges.
                continue;
            }

            Locations::Single(location) => {
                // This constraint is marked as holding at one location, we localize it to that
                // location or its successor, depending on the corresponding MIR
                // statement/terminator. Unfortunately, they all show up from typeck as coming "on
                // entry", so for now we modify them to take effects that should apply "on exit"
                // into account.
                //
                // FIXME: this approach is subtle, complicated, and hard to test, so we should track
                // this information better in MIR typeck instead, for example with a new `Locations`
                // variant that contains which node is crossing over between entry and exit.
                let point = liveness.point_from_location(location);
                let localized_constraint = if let Some(stmt) =
                    body[location.block].statements.get(location.statement_index)
                {
                    localize_statement_constraint(
                        tcx,
                        body,
                        stmt,
                        &outlives_constraint,
                        point,
                        universal_regions,
                    )
                } else {
                    assert_eq!(location.statement_index, body[location.block].statements.len());
                    let terminator = body[location.block].terminator();
                    localize_terminator_constraint(
                        tcx,
                        body,
                        terminator,
                        liveness,
                        &outlives_constraint,
                        point,
                        universal_regions,
                    )
                };
                localized_outlives_constraints.push(localized_constraint);
            }
        }
    }
}

/// For a given outlives constraint arising from a MIR statement, localize the constraint with the
/// needed CFG `from`-`to` intra-block nodes.
fn localize_statement_constraint<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    stmt: &Statement<'tcx>,
    outlives_constraint: &OutlivesConstraint<'tcx>,
    current_point: PointIndex,
    universal_regions: &UniversalRegions<'tcx>,
) -> LocalizedOutlivesConstraint {
    match &stmt.kind {
        StatementKind::Assign(box (lhs, rhs)) => {
            // To create localized outlives constraints without midpoints, we rely on the property
            // that no input regions from the RHS of the assignment will flow into themselves: they
            // should not appear in the output regions in the LHS. We believe this to be true by
            // construction of the MIR, via temporaries, and assert it here.
            //
            // We think we don't need midpoints because:
            // - every LHS Place has a unique set of regions that don't appear elsewhere
            // - this implies that for them to be part of the RHS, the same Place must be read and
            //   written
            // - and that should be impossible in MIR
            //
            // When we have a more complete implementation in the future, tested with crater, etc,
            // we can remove this assertion. It's a debug assert because it can be expensive.
            debug_assert!(
                {
                    let mut lhs_regions = FxHashSet::default();
                    tcx.for_each_free_region(lhs, |region| {
                        let region = universal_regions.to_region_vid(region);
                        lhs_regions.insert(region);
                    });

                    let mut rhs_regions = FxHashSet::default();
                    tcx.for_each_free_region(rhs, |region| {
                        let region = universal_regions.to_region_vid(region);
                        rhs_regions.insert(region);
                    });

                    // The intersection between LHS and RHS regions should be empty.
                    lhs_regions.is_disjoint(&rhs_regions)
                },
                "there should be no common regions between the LHS and RHS of an assignment"
            );

            let lhs_ty = body.local_decls[lhs.local].ty;
            let successor_point = current_point;
            compute_constraint_direction(
                tcx,
                outlives_constraint,
                &lhs_ty,
                current_point,
                successor_point,
                universal_regions,
            )
        }
        _ => {
            // For the other cases, we localize an outlives constraint to where it arises.
            LocalizedOutlivesConstraint {
                source: outlives_constraint.sup,
                from: current_point,
                target: outlives_constraint.sub,
                to: current_point,
            }
        }
    }
}

/// For a given outlives constraint arising from a MIR terminator, localize the constraint with the
/// needed CFG `from`-`to` inter-block nodes.
fn localize_terminator_constraint<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    terminator: &Terminator<'tcx>,
    liveness: &LivenessValues,
    outlives_constraint: &OutlivesConstraint<'tcx>,
    current_point: PointIndex,
    universal_regions: &UniversalRegions<'tcx>,
) -> LocalizedOutlivesConstraint {
    // FIXME: check if other terminators need the same handling as `Call`s, in particular
    // Assert/Yield/Drop. A handful of tests are failing with Drop related issues, as well as some
    // coroutine tests, and that may be why.
    match &terminator.kind {
        // FIXME: also handle diverging calls.
        TerminatorKind::Call { destination, target: Some(target), .. } => {
            // Calls are similar to assignments, and thus follow the same pattern. If there is a
            // target for the call we also relate what flows into the destination here to entry to
            // that successor.
            let destination_ty = destination.ty(&body.local_decls, tcx);
            let successor_location = Location { block: *target, statement_index: 0 };
            let successor_point = liveness.point_from_location(successor_location);
            compute_constraint_direction(
                tcx,
                outlives_constraint,
                &destination_ty,
                current_point,
                successor_point,
                universal_regions,
            )
        }
        _ => {
            // Typeck constraints guide loans between regions at the current point, so we do that in
            // the general case, and liveness will take care of making them flow to the terminator's
            // successors.
            LocalizedOutlivesConstraint {
                source: outlives_constraint.sup,
                from: current_point,
                target: outlives_constraint.sub,
                to: current_point,
            }
        }
    }
}

/// For a given outlives constraint and CFG edge, returns the localized constraint with the
/// appropriate `from`-`to` direction. This is computed according to whether the constraint flows to
/// or from a free region in the given `value`, some kind of result for an effectful operation, like
/// the LHS of an assignment.
fn compute_constraint_direction<'tcx>(
    tcx: TyCtxt<'tcx>,
    outlives_constraint: &OutlivesConstraint<'tcx>,
    value: &impl TypeVisitable<TyCtxt<'tcx>>,
    current_point: PointIndex,
    successor_point: PointIndex,
    universal_regions: &UniversalRegions<'tcx>,
) -> LocalizedOutlivesConstraint {
    let mut to = current_point;
    let mut from = current_point;
    tcx.for_each_free_region(value, |region| {
        let region = universal_regions.to_region_vid(region);
        if region == outlives_constraint.sub {
            // This constraint flows into the result, its effects start becoming visible on exit.
            to = successor_point;
        } else if region == outlives_constraint.sup {
            // This constraint flows from the result, its effects start becoming visible on exit.
            from = successor_point;
        }
    });

    LocalizedOutlivesConstraint {
        source: outlives_constraint.sup,
        from,
        target: outlives_constraint.sub,
        to,
    }
}
