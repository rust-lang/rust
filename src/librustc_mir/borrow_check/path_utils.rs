// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use borrow_check::borrow_set::{BorrowSet, BorrowData, TwoPhaseActivation};
use borrow_check::places_conflict;
use borrow_check::Context;
use borrow_check::AccessDepth;
use dataflow::indexes::BorrowIndex;
use rustc::mir::{BasicBlock, Location, Mir, Place};
use rustc::mir::{ProjectionElem, BorrowKind};
use rustc::ty::TyCtxt;
use rustc_data_structures::graph::dominators::Dominators;

/// Returns true if the borrow represented by `kind` is
/// allowed to be split into separate Reservation and
/// Activation phases.
pub(super) fn allow_two_phase_borrow<'a, 'tcx, 'gcx: 'tcx>(
    tcx: &TyCtxt<'a, 'gcx, 'tcx>,
    kind: BorrowKind
) -> bool {
    tcx.two_phase_borrows()
        && (kind.allows_two_phase_borrow()
            || tcx.sess.opts.debugging_opts.two_phase_beyond_autoref)
}

/// Control for the path borrow checking code
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(super) enum Control {
    Continue,
    Break,
}

/// Encapsulates the idea of iterating over every borrow that involves a particular path
pub(super) fn each_borrow_involving_path<'a, 'tcx, 'gcx: 'tcx, F, I, S> (
    s: &mut S,
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    mir: &Mir<'tcx>,
    _context: Context,
    access_place: (AccessDepth, &Place<'tcx>),
    borrow_set: &BorrowSet<'tcx>,
    candidates: I,
    mut op: F,
) where
    F: FnMut(&mut S, BorrowIndex, &BorrowData<'tcx>) -> Control,
    I: Iterator<Item=BorrowIndex>
{
    let (access, place) = access_place;

    // FIXME: analogous code in check_loans first maps `place` to
    // its base_path.

    // check for loan restricting path P being used. Accounts for
    // borrows of P, P.a.b, etc.
    for i in candidates {
        let borrowed = &borrow_set[i];

        if places_conflict::borrow_conflicts_with_place(
            tcx,
            mir,
            &borrowed.borrowed_place,
            borrowed.kind,
            place,
            access,
            places_conflict::PlaceConflictBias::Overlap,
        ) {
            debug!(
                "each_borrow_involving_path: {:?} @ {:?} vs. {:?}/{:?}",
                i, borrowed, place, access
            );
            let ctrl = op(s, i, borrowed);
            if ctrl == Control::Break {
                return;
            }
        }
    }
}

pub(super) fn is_active<'tcx>(
    dominators: &Dominators<BasicBlock>,
    borrow_data: &BorrowData<'tcx>,
    location: Location
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
/// This is called for all Yield statements on movable generators
pub(super) fn borrow_of_local_data<'tcx>(place: &Place<'tcx>) -> bool {
    match place {
        Place::Promoted(_) |
        Place::Static(..) => false,
        Place::Local(..) => true,
        Place::Projection(box proj) => {
            match proj.elem {
                // Reborrow of already borrowed data is ignored
                // Any errors will be caught on the initial borrow
                ProjectionElem::Deref => false,

                // For interior references and downcasts, find out if the base is local
                ProjectionElem::Field(..)
                    | ProjectionElem::Index(..)
                    | ProjectionElem::ConstantIndex { .. }
                | ProjectionElem::Subslice { .. }
                | ProjectionElem::Downcast(..) => borrow_of_local_data(&proj.base),
            }
        }
    }
}
