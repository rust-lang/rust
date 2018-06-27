// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use borrow_check::ArtificialField;
use borrow_check::Overlap;
use borrow_check::{ShallowOrDeep, Deep, Shallow};
use rustc::hir;
use rustc::mir::{Mir, Place};
use rustc::mir::{Projection, ProjectionElem};
use rustc::ty::{self, TyCtxt};

pub(super) fn places_conflict<'gcx, 'tcx>(
    tcx: TyCtxt<'_, 'gcx, 'tcx>,
    mir: &Mir<'tcx>,
    borrow_place: &Place<'tcx>,
    access_place: &Place<'tcx>,
    access: ShallowOrDeep,
) -> bool {
    debug!(
        "places_conflict({:?},{:?},{:?})",
        borrow_place, access_place, access
    );

    unroll_place(borrow_place, None, |borrow_components| {
        unroll_place(access_place, None, |access_components| {
            place_components_conflict(tcx, mir, borrow_components, access_components, access)
        })
    })
}

fn place_components_conflict<'gcx, 'tcx>(
    tcx: TyCtxt<'_, 'gcx, 'tcx>,
    mir: &Mir<'tcx>,
    borrow_components: PlaceComponentsIter<'_, 'tcx>,
    access_components: PlaceComponentsIter<'_, 'tcx>,
    access: ShallowOrDeep,
) -> bool {
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
        if let Some(borrow_c) = borrow_c {
            if let Some(access_c) = access_c {
                // Borrow and access path both have more components.
                //
                // Examples:
                //
                // - borrow of `a.(...)`, access to `a.(...)`
                // - borrow of `a.(...)`, access to `b.(...)`
                //
                // Here we only see the components we have checked so
                // far (in our examples, just the first component). We
                // check whether the components being borrowed vs
                // accessed are disjoint (as in the second example,
                // but not the first).
                match place_element_conflict(tcx, mir, borrow_c, access_c) {
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
            } else {
                // Borrow path is longer than the access path. Examples:
                //
                // - borrow of `a.b.c`, access to `a.b`
                //
                // Here, we know that the borrow can access a part of
                // our place. This is a conflict if that is a part our
                // access cares about.

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
        } else {
            // Borrow path ran out but access path may not
            // have. Examples:
            //
            // - borrow of `a.b`, access to `a.b.c`
            // - borrow of `a.b`, access to `a.b`
            //
            // In the first example, where we didn't run out of
            // access, the borrow can access all of our place, so we
            // have a conflict.
            //
            // If the second example, where we did, then we still know
            // that the borrow can access a *part* of our place that
            // our access cares about, so we still have a conflict.
            //
            // FIXME: Differs from AST-borrowck; includes drive-by fix
            // to #38899. Will probably need back-compat mode flag.
            debug!("places_conflict: full borrow, CONFLICT");
            return true;
        }
    }
    unreachable!("iter::repeat returned None")
}

/// A linked list of places running up the stack; begins with the
/// innermost place and extends to projections (e.g., `a.b` would have
/// the place `a` with a "next" pointer to `a.b`).  Created by
/// `unroll_place`.
struct PlaceComponents<'p, 'tcx: 'p> {
    component: &'p Place<'tcx>,
    next: Option<&'p PlaceComponents<'p, 'tcx>>,
}

impl<'p, 'tcx> PlaceComponents<'p, 'tcx> {
    /// Converts a list of `Place` components into an iterator; this
    /// iterator yields up a never-ending stream of `Option<&Place>`.
    /// These begin with the "innermst" place and then with each
    /// projection therefrom. So given a place like `a.b.c` it would
    /// yield up:
    ///
    /// ```notrust
    /// Some(`a`), Some(`a.b`), Some(`a.b.c`), None, None, ...
    /// ```
    fn iter(&self) -> PlaceComponentsIter<'_, 'tcx> {
        PlaceComponentsIter { value: Some(self) }
    }
}

/// Iterator over components; see `PlaceComponents::iter` for more
/// information.
struct PlaceComponentsIter<'p, 'tcx: 'p> {
    value: Option<&'p PlaceComponents<'p, 'tcx>>
}

impl<'p, 'tcx> Iterator for PlaceComponentsIter<'p, 'tcx> {
    type Item = Option<&'p Place<'tcx>>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(&PlaceComponents { component, next }) = self.value {
            self.value = next;
            Some(Some(component))
        } else {
            Some(None)
        }
    }
}

/// Recursively "unroll" a place into a `PlaceComponents` list,
/// invoking `op` with a `PlaceComponentsIter`.
fn unroll_place<'tcx, R>(
    place: &Place<'tcx>,
    next: Option<&PlaceComponents<'_, 'tcx>>,
    op: impl FnOnce(PlaceComponentsIter<'_, 'tcx>) -> R
) -> R {
    match place {
        Place::Projection(interior) => {
            unroll_place(
                &interior.base,
                Some(&PlaceComponents {
                    component: place,
                    next,
                }),
                op,
            )
        }

        Place::Local(_) | Place::Static(_) => {
            let list = PlaceComponents {
                component: place,
                next,
            };
            op(list.iter())
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
