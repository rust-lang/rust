// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// Returns true if the borrow represented by `kind` is
/// allowed to be split into separate Reservation and
/// Activation phases.
use borrow_check::ArtificialField;
use borrow_check::borrow_set::{BorrowSet, BorrowData, TwoPhaseUse};
use borrow_check::{Context, Overlap};
use borrow_check::{ShallowOrDeep, Deep, Shallow};
use dataflow::indexes::BorrowIndex;
use rustc::hir;
use rustc::mir::{BasicBlock, Location, Mir, Place};
use rustc::mir::{Projection, ProjectionElem, BorrowKind};
use rustc::ty::{self, TyCtxt};
use rustc_data_structures::control_flow_graph::dominators::Dominators;
use rustc_data_structures::small_vec::SmallVec;
use std::iter;

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
    access_place: (ShallowOrDeep, &Place<'tcx>),
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

        if places_conflict(tcx, mir, &borrowed.borrowed_place, place, access) {
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

pub(super) fn places_conflict<'a, 'gcx: 'tcx, 'tcx>(
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    mir: &Mir<'tcx>,
    borrow_place: &Place<'tcx>,
    access_place: &Place<'tcx>,
    access: ShallowOrDeep,
) -> bool {
    debug!(
        "places_conflict({:?},{:?},{:?})",
        borrow_place, access_place, access
    );

    let borrow_components = place_elements(borrow_place);
    let access_components = place_elements(access_place);
    debug!(
        "places_conflict: components {:?} / {:?}",
        borrow_components, access_components
    );

    let borrow_components = borrow_components
        .into_iter()
        .map(Some)
        .chain(iter::repeat(None));
    let access_components = access_components
        .into_iter()
        .map(Some)
        .chain(iter::repeat(None));
    // The borrowck rules for proving disjointness are applied from the "root" of the
    // borrow forwards, iterating over "similar" projections in lockstep until
    // we can prove overlap one way or another. Essentially, we treat `Overlap` as
    // a monoid and report a conflict if the product ends up not being `Disjoint`.
    //
    // At each step, if we didn't run out of borrow or place, we know that our elements
    // have the same type, and that they only overlap if they are the identical.
    //
    // For example, if we are comparing these:
    // BORROW:  (*x1[2].y).z.a
    // ACCESS:  (*x1[i].y).w.b
    //
    // Then our steps are:
    //       x1         |   x1          -- places are the same
    //       x1[2]      |   x1[i]       -- equal or disjoint (disjoint if indexes differ)
    //       x1[2].y    |   x1[i].y     -- equal or disjoint
    //      *x1[2].y    |  *x1[i].y     -- equal or disjoint
    //     (*x1[2].y).z | (*x1[i].y).w  -- we are disjoint and don't need to check more!
    //
    // Because `zip` does potentially bad things to the iterator inside, this loop
    // also handles the case where the access might be a *prefix* of the borrow, e.g.
    //
    // BORROW:  (*x1[2].y).z.a
    // ACCESS:  x1[i].y
    //
    // Then our steps are:
    //       x1         |   x1          -- places are the same
    //       x1[2]      |   x1[i]       -- equal or disjoint (disjoint if indexes differ)
    //       x1[2].y    |   x1[i].y     -- equal or disjoint
    //
    // -- here we run out of access - the borrow can access a part of it. If this
    // is a full deep access, then we *know* the borrow conflicts with it. However,
    // if the access is shallow, then we can proceed:
    //
    //       x1[2].y    | (*x1[i].y)    -- a deref! the access can't get past this, so we
    //                                     are disjoint
    //
    // Our invariant is, that at each step of the iteration:
    //  - If we didn't run out of access to match, our borrow and access are comparable
    //    and either equal or disjoint.
    //  - If we did run out of accesss, the borrow can access a part of it.
    for (borrow_c, access_c) in borrow_components.zip(access_components) {
        // loop invariant: borrow_c is always either equal to access_c or disjoint from it.
        debug!("places_conflict: {:?} vs. {:?}", borrow_c, access_c);
        match (borrow_c, access_c) {
            (None, _) => {
                // If we didn't run out of access, the borrow can access all of our
                // place (e.g. a borrow of `a.b` with an access to `a.b.c`),
                // so we have a conflict.
                //
                // If we did, then we still know that the borrow can access a *part*
                // of our place that our access cares about (a borrow of `a.b.c`
                // with an access to `a.b`), so we still have a conflict.
                //
                // FIXME: Differs from AST-borrowck; includes drive-by fix
                // to #38899. Will probably need back-compat mode flag.
                debug!("places_conflict: full borrow, CONFLICT");
                return true;
            }
            (Some(borrow_c), None) => {
                // We know that the borrow can access a part of our place. This
                // is a conflict if that is a part our access cares about.

                let (base, elem) = match borrow_c {
                    Place::Projection(box Projection { base, elem }) => (base, elem),
                    _ => bug!("place has no base?"),
                };
                let base_ty = base.ty(mir, tcx).to_ty(tcx);

                match (elem, &base_ty.sty, access) {
                    (_, _, Shallow(Some(ArtificialField::Discriminant)))
                        | (_, _, Shallow(Some(ArtificialField::ArrayLength))) => {
                            // The discriminant and array length are like
                            // additional fields on the type; they do not
                            // overlap any existing data there. Furthermore,
                            // they cannot actually be a prefix of any
                            // borrowed place (at least in MIR as it is
                            // currently.)
                            //
                            // e.g. a (mutable) borrow of `a[5]` while we read the
                            // array length of `a`.
                            debug!("places_conflict: implicit field");
                            return false;
                        }

                    (ProjectionElem::Deref, _, Shallow(None)) => {
                        // e.g. a borrow of `*x.y` while we shallowly access `x.y` or some
                        // prefix thereof - the shallow access can't touch anything behind
                        // the pointer.
                        debug!("places_conflict: shallow access behind ptr");
                        return false;
                    }
                    (
                        ProjectionElem::Deref,
                        ty::TyRef( _, _, hir::MutImmutable),
                        _,
                    ) => {
                        // the borrow goes through a dereference of a shared reference.
                        //
                        // I'm not sure why we are tracking these borrows - shared
                        // references can *always* be aliased, which means the
                        // permission check already account for this borrow.
                        debug!("places_conflict: behind a shared ref");
                        return false;
                    }

                    (ProjectionElem::Deref, _, Deep)
                        | (ProjectionElem::Field { .. }, _, _)
                        | (ProjectionElem::Index { .. }, _, _)
                        | (ProjectionElem::ConstantIndex { .. }, _, _)
                        | (ProjectionElem::Subslice { .. }, _, _)
                        | (ProjectionElem::Downcast { .. }, _, _) => {
                            // Recursive case. This can still be disjoint on a
                            // further iteration if this a shallow access and
                            // there's a deref later on, e.g. a borrow
                            // of `*x.y` while accessing `x`.
                        }
                }
            }
            (Some(borrow_c), Some(access_c)) => {
                match place_element_conflict(tcx, mir, &borrow_c, access_c) {
                    Overlap::Arbitrary => {
                        // We have encountered different fields of potentially
                        // the same union - the borrow now partially overlaps.
                        //
                        // There is no *easy* way of comparing the fields
                        // further on, because they might have different types
                        // (e.g. borrows of `u.a.0` and `u.b.y` where `.0` and
                        // `.y` come from different structs).
                        //
                        // We could try to do some things here - e.g. count
                        // dereferences - but that's probably not a good
                        // idea, at least for now, so just give up and
                        // report a conflict. This is unsafe code anyway so
                        // the user could always use raw pointers.
                        debug!("places_conflict: arbitrary -> conflict");
                        return true;
                    }
                    Overlap::EqualOrDisjoint => {
                        // This is the recursive case - proceed to the next element.
                    }
                    Overlap::Disjoint => {
                        // We have proven the borrow disjoint - further
                        // projections will remain disjoint.
                        debug!("places_conflict: disjoint");
                        return false;
                    }
                }
            }
        }
    }
    unreachable!("iter::repeat returned None")
}

/// Return all the prefixes of `place` in reverse order, including
/// downcasts.
fn place_elements<'a, 'tcx>(place: &'a Place<'tcx>) -> SmallVec<[&'a Place<'tcx>; 8]> {
    let mut result = SmallVec::new();
    let mut place = place;
    loop {
        result.push(place);
        match place {
            Place::Projection(interior) => {
                place = &interior.base;
            }
            Place::Local(_) | Place::Static(_) => {
                result.reverse();
                return result;
            }
        }
    }
}

// Given that the bases of `elem1` and `elem2` are always either equal
// or disjoint (and have the same type!), return the overlap situation
// between `elem1` and `elem2`.
fn place_element_conflict<'a, 'gcx: 'tcx, 'tcx>(
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    mir: &Mir<'tcx>,
    elem1: &Place<'tcx>,
    elem2: &Place<'tcx>
) -> Overlap {
    match (elem1, elem2) {
        (Place::Local(l1), Place::Local(l2)) => {
            if l1 == l2 {
                // the same local - base case, equal
                debug!("place_element_conflict: DISJOINT-OR-EQ-LOCAL");
                Overlap::EqualOrDisjoint
            } else {
                // different locals - base case, disjoint
                debug!("place_element_conflict: DISJOINT-LOCAL");
                Overlap::Disjoint
            }
        }
        (Place::Static(static1), Place::Static(static2)) => {
            if static1.def_id != static2.def_id {
                debug!("place_element_conflict: DISJOINT-STATIC");
                Overlap::Disjoint
            } else if tcx.is_static(static1.def_id) ==
                        Some(hir::Mutability::MutMutable) {
                // We ignore mutable statics - they can only be unsafe code.
                debug!("place_element_conflict: IGNORE-STATIC-MUT");
                Overlap::Disjoint
            } else {
                debug!("place_element_conflict: DISJOINT-OR-EQ-STATIC");
                Overlap::EqualOrDisjoint
            }
        }
        (Place::Local(_), Place::Static(_)) | (Place::Static(_), Place::Local(_)) => {
            debug!("place_element_conflict: DISJOINT-STATIC-LOCAL");
            Overlap::Disjoint
        }
        (Place::Projection(pi1), Place::Projection(pi2)) => {
            match (&pi1.elem, &pi2.elem) {
                (ProjectionElem::Deref, ProjectionElem::Deref) => {
                    // derefs (e.g. `*x` vs. `*x`) - recur.
                    debug!("place_element_conflict: DISJOINT-OR-EQ-DEREF");
                    Overlap::EqualOrDisjoint
                }
                (ProjectionElem::Field(f1, _), ProjectionElem::Field(f2, _)) => {
                    if f1 == f2 {
                        // same field (e.g. `a.y` vs. `a.y`) - recur.
                        debug!("place_element_conflict: DISJOINT-OR-EQ-FIELD");
                        Overlap::EqualOrDisjoint
                    } else {
                        let ty = pi1.base.ty(mir, tcx).to_ty(tcx);
                        match ty.sty {
                            ty::TyAdt(def, _) if def.is_union() => {
                                // Different fields of a union, we are basically stuck.
                                debug!("place_element_conflict: STUCK-UNION");
                                Overlap::Arbitrary
                            }
                            _ => {
                                // Different fields of a struct (`a.x` vs. `a.y`). Disjoint!
                                debug!("place_element_conflict: DISJOINT-FIELD");
                                Overlap::Disjoint
                            }
                        }
                    }
                }
                (ProjectionElem::Downcast(_, v1), ProjectionElem::Downcast(_, v2)) => {
                    // different variants are treated as having disjoint fields,
                    // even if they occupy the same "space", because it's
                    // impossible for 2 variants of the same enum to exist
                    // (and therefore, to be borrowed) at the same time.
                    //
                    // Note that this is different from unions - we *do* allow
                    // this code to compile:
                    //
                    // ```
                    // fn foo(x: &mut Result<i32, i32>) {
                    //     let mut v = None;
                    //     if let Ok(ref mut a) = *x {
                    //         v = Some(a);
                    //     }
                    //     // here, you would *think* that the
                    //     // *entirety* of `x` would be borrowed,
                    //     // but in fact only the `Ok` variant is,
                    //     // so the `Err` variant is *entirely free*:
                    //     if let Err(ref mut a) = *x {
                    //         v = Some(a);
                    //     }
                    //     drop(v);
                    // }
                    // ```
                    if v1 == v2 {
                        debug!("place_element_conflict: DISJOINT-OR-EQ-FIELD");
                        Overlap::EqualOrDisjoint
                    } else {
                        debug!("place_element_conflict: DISJOINT-FIELD");
                        Overlap::Disjoint
                    }
                }
                (ProjectionElem::Index(..), ProjectionElem::Index(..))
                | (ProjectionElem::Index(..), ProjectionElem::ConstantIndex { .. })
                | (ProjectionElem::Index(..), ProjectionElem::Subslice { .. })
                | (ProjectionElem::ConstantIndex { .. }, ProjectionElem::Index(..))
                | (
                    ProjectionElem::ConstantIndex { .. },
                    ProjectionElem::ConstantIndex { .. },
                )
                | (ProjectionElem::ConstantIndex { .. }, ProjectionElem::Subslice { .. })
                | (ProjectionElem::Subslice { .. }, ProjectionElem::Index(..))
                | (ProjectionElem::Subslice { .. }, ProjectionElem::ConstantIndex { .. })
                | (ProjectionElem::Subslice { .. }, ProjectionElem::Subslice { .. }) => {
                    // Array indexes (`a[0]` vs. `a[i]`). These can either be disjoint
                    // (if the indexes differ) or equal (if they are the same), so this
                    // is the recursive case that gives "equal *or* disjoint" its meaning.
                    //
                    // Note that by construction, MIR at borrowck can't subdivide
                    // `Subslice` accesses (e.g. `a[2..3][i]` will never be present) - they
                    // are only present in slice patterns, and we "merge together" nested
                    // slice patterns. That means we don't have to think about these. It's
                    // probably a good idea to assert this somewhere, but I'm too lazy.
                    //
                    // FIXME(#8636) we might want to return Disjoint if
                    // both projections are constant and disjoint.
                    debug!("place_element_conflict: DISJOINT-OR-EQ-ARRAY");
                    Overlap::EqualOrDisjoint
                }

                (ProjectionElem::Deref, _)
                | (ProjectionElem::Field(..), _)
                | (ProjectionElem::Index(..), _)
                | (ProjectionElem::ConstantIndex { .. }, _)
                | (ProjectionElem::Subslice { .. }, _)
                | (ProjectionElem::Downcast(..), _) => bug!(
                    "mismatched projections in place_element_conflict: {:?} and {:?}",
                    elem1,
                    elem2
                ),
            }
        }
        (Place::Projection(_), _) | (_, Place::Projection(_)) => bug!(
            "unexpected elements in place_element_conflict: {:?} and {:?}",
            elem1,
            elem2
        ),
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
        None => return true,
        // And if the unique 2-phase use is not an activation, then it is *never* active.
        Some((TwoPhaseUse::SharedUse, _)) => return false,
        // Otherwise, we derive info from the activation point `v`:
        Some((TwoPhaseUse::MutActivate, v)) => v,
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
