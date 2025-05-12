use std::ops::ControlFlow;

use rustc_abi::FieldIdx;
use rustc_data_structures::graph::dominators::Dominators;
use rustc_middle::mir::{BasicBlock, Body, Location, Place, PlaceRef, ProjectionElem};
use rustc_middle::ty::TyCtxt;
use tracing::debug;

use crate::borrow_set::{BorrowData, BorrowSet, TwoPhaseActivation};
use crate::{AccessDepth, BorrowIndex, places_conflict};

/// Encapsulates the idea of iterating over every borrow that involves a particular path
pub(super) fn each_borrow_involving_path<'tcx, F, I, S>(
    s: &mut S,
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    access_place: (AccessDepth, Place<'tcx>),
    borrow_set: &BorrowSet<'tcx>,
    is_candidate: I,
    mut op: F,
) where
    F: FnMut(&mut S, BorrowIndex, &BorrowData<'tcx>) -> ControlFlow<()>,
    I: Fn(BorrowIndex) -> bool,
{
    let (access, place) = access_place;

    // The number of candidates can be large, but borrows for different locals cannot conflict with
    // each other, so we restrict the working set a priori.
    let Some(borrows_for_place_base) = borrow_set.local_map.get(&place.local) else { return };

    // check for loan restricting path P being used. Accounts for
    // borrows of P, P.a.b, etc.
    for &i in borrows_for_place_base {
        if !is_candidate(i) {
            continue;
        }
        let borrowed = &borrow_set[i];

        if places_conflict::borrow_conflicts_with_place(
            tcx,
            body,
            borrowed.borrowed_place,
            borrowed.kind,
            place.as_ref(),
            access,
            places_conflict::PlaceConflictBias::Overlap,
        ) {
            debug!(
                "each_borrow_involving_path: {:?} @ {:?} vs. {:?}/{:?}",
                i, borrowed, place, access
            );
            let ctrl = op(s, i, borrowed);
            if matches!(ctrl, ControlFlow::Break(_)) {
                return;
            }
        }
    }
}

pub(super) fn is_active<'tcx>(
    dominators: &Dominators<BasicBlock>,
    borrow_data: &BorrowData<'tcx>,
    location: Location,
) -> bool {
    debug!("is_active(borrow_data={:?}, location={:?})", borrow_data, location);

    let activation_location = match borrow_data.activation_location {
        // If this is not a 2-phase borrow, it is always active.
        TwoPhaseActivation::NotTwoPhase => return true,
        // And if the unique 2-phase use is not an activation, then it is *never* active.
        TwoPhaseActivation::NotActivated => return false,
        // Otherwise, we derive info from the activation point `loc`:
        TwoPhaseActivation::ActivatedAt(loc) => loc,
    };

    // Otherwise, it is active for every location *except* in between
    // the reservation and the activation:
    //
    //       X
    //      /
    //     R      <--+ Except for this
    //    / \        | diamond
    //    \ /        |
    //     A  <------+
    //     |
    //     Z
    //
    // Note that we assume that:
    // - the reservation R dominates the activation A
    // - the activation A post-dominates the reservation R (ignoring unwinding edges).
    //
    // This means that there can't be an edge that leaves A and
    // comes back into that diamond unless it passes through R.
    //
    // Suboptimal: In some cases, this code walks the dominator
    // tree twice when it only has to be walked once. I am
    // lazy. -nmatsakis

    // If dominated by the activation A, then it is active. The
    // activation occurs upon entering the point A, so this is
    // also true if location == activation_location.
    if activation_location.dominates(location, dominators) {
        return true;
    }

    // The reservation starts *on exiting* the reservation block,
    // so check if the location is dominated by R.successor. If so,
    // this point falls in between the reservation and location.
    let reserve_location = borrow_data.reserve_location.successor_within_block();
    if reserve_location.dominates(location, dominators) {
        false
    } else {
        // Otherwise, this point is outside the diamond, so
        // consider the borrow active. This could happen for
        // example if the borrow remains active around a loop (in
        // which case it would be active also for the point R,
        // which would generate an error).
        true
    }
}

/// Determines if a given borrow is borrowing local data
/// This is called for all Yield expressions on movable coroutines
pub(super) fn borrow_of_local_data(place: Place<'_>) -> bool {
    // Reborrow of already borrowed data is ignored
    // Any errors will be caught on the initial borrow
    !place.is_indirect()
}

/// If `place` is a field projection, and the field is being projected from a closure type,
/// then returns the index of the field being projected. Note that this closure will always
/// be `self` in the current MIR, because that is the only time we directly access the fields
/// of a closure type.
pub(crate) fn is_upvar_field_projection<'tcx>(
    tcx: TyCtxt<'tcx>,
    upvars: &[&rustc_middle::ty::CapturedPlace<'tcx>],
    place_ref: PlaceRef<'tcx>,
    body: &Body<'tcx>,
) -> Option<FieldIdx> {
    let mut place_ref = place_ref;
    let mut by_ref = false;

    if let Some((place_base, ProjectionElem::Deref)) = place_ref.last_projection() {
        place_ref = place_base;
        by_ref = true;
    }

    match place_ref.last_projection() {
        Some((place_base, ProjectionElem::Field(field, _ty))) => {
            let base_ty = place_base.ty(body, tcx).ty;
            if (base_ty.is_closure() || base_ty.is_coroutine() || base_ty.is_coroutine_closure())
                && (!by_ref || upvars[field.index()].is_by_ref())
            {
                Some(field)
            } else {
                None
            }
        }
        _ => None,
    }
}
