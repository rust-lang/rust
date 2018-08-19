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
use borrow_check::{Deep, Shallow, ShallowOrDeep};
use rustc::hir;
use rustc::mir::{Mir, Place, PlaceBase, PlaceElem, ProjectionElem};
use rustc::ty::{self, Slice, Ty, TyCtxt};
use std::cmp::max;

// FIXME(csmoe): rewrite place_conflict with slice

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

    let place_elements_conflict = |tcx: TyCtxt<'_, 'gcx, 'tcx>,
                                   mir: &Mir<'tcx>,
                                   borrow_place: &Place<'tcx>,
                                   access_place: &Place<'tcx>| {
        // Enumerate for later base_place generation
        let mut borrow_elems = borrow_place.elems.iter().cloned().enumerate();

        let mut access_elems = access_place.elems.iter().cloned();

        loop {
            if let Some((i, borrow_elem)) = borrow_elems.next() {
                let base_place = Place {
                    base: borrow_place.base,
                    elems: if i > 0 {
                        tcx.mk_place_elems(borrow_place.elems.iter().cloned().take(i))
                    } else {
                        Slice::empty()
                    },
                };
                let base_ty = base_place.ty(mir, tcx).to_ty(tcx);

                if let Some(access_elem) = access_elems.next() {
                    debug!("places_conflict: access_elem = {:?}", access_elem);

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
                    match place_element_conflict(base_ty, (&borrow_elem, &access_elem)) {
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

                    match (borrow_elem, &base_ty.sty, access) {
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
                        (ProjectionElem::Deref, ty::TyRef(_, _, hir::MutImmutable), _) => {
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
    };

    match place_base_conflict(tcx, &borrow_place.base, &access_place.base) {
        // if the place.base disjoint, further projections will remain disjoint.
        Overlap::Disjoint => false,
        // process to projections to check further conflict.
        Overlap::EqualOrDisjoint => place_elements_conflict(tcx, mir, borrow_place, access_place),
        // place.base overlap is obvious, no Abitrary.
        _ => unreachable!(),
    }
}

fn place_base_conflict<'a, 'gcx: 'tcx, 'tcx>(
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    base1: &PlaceBase<'tcx>,
    base2: &PlaceBase<'tcx>,
) -> Overlap {
    match (base1, base2) {
        (PlaceBase::Local(l1), PlaceBase::Local(l2)) => {
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
        (PlaceBase::Static(static1), PlaceBase::Static(static2)) => {
            if static1.def_id != static2.def_id {
                debug!("place_element_conflict: DISJOINT-STATIC");
                Overlap::Disjoint
            } else if tcx.is_static(static1.def_id) == Some(hir::Mutability::MutMutable) {
                // We ignore mutable statics - they can only be unsafe code.
                debug!("place_element_conflict: IGNORE-STATIC-MUT");
                Overlap::Disjoint
            } else {
                debug!("place_element_conflict: DISJOINT-OR-EQ-STATIC");
                Overlap::EqualOrDisjoint
            }
        }
        (PlaceBase::Promoted(p1), PlaceBase::Promoted(p2)) => {
            if p1.0 == p2.0 {
                if let ty::TyArray(_, size) = p1.1.sty {
                    if size.unwrap_usize(tcx) == 0 {
                        // Ignore conflicts with promoted [T; 0].
                        debug!("place_element_conflict: IGNORE-LEN-0-PROMOTED");
                        return Overlap::Disjoint;
                    }
                }
                // the same promoted - base case, equal
                debug!("place_element_conflict: DISJOINT-OR-EQ-PROMOTED");
                Overlap::EqualOrDisjoint
            } else {
                // different promoteds - base case, disjoint
                debug!("place_element_conflict: DISJOINT-PROMOTED");
                Overlap::Disjoint
            }
        }
        (PlaceBase::Local(_), PlaceBase::Promoted(_))
        | (PlaceBase::Promoted(_), PlaceBase::Local(_))
        | (PlaceBase::Promoted(_), PlaceBase::Static(_))
        | (PlaceBase::Static(_), PlaceBase::Promoted(_))
        | (PlaceBase::Local(_), PlaceBase::Static(_))
        | (PlaceBase::Static(_), PlaceBase::Local(_)) => {
            debug!("place_element_conflict: DISJOINT-STATIC-LOCAL-PROMOTED");
            Overlap::Disjoint
        }
    }
}

// Given that the bases of `elem1` and `elem2` are always either equal
// or disjoint (and have the same type!), return the overlap situation
// between `elem1` and `elem2`.
fn place_element_conflict<'tcx>(
    base_ty: Ty<'tcx>,
    (elem1, elem2): (&PlaceElem<'tcx>, &PlaceElem<'tcx>),
) -> Overlap {
    match (elem1, elem2) {
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
                match base_ty.sty {
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
        | (ProjectionElem::Subslice { .. }, ProjectionElem::Index(..)) => {
            // Array indexes (`a[0]` vs. `a[i]`). These can either be disjoint
            // (if the indexes differ) or equal (if they are the same), so this
            // is the recursive case that gives "equal *or* disjoint" its meaning.
            debug!("place_element_conflict: DISJOINT-OR-EQ-ARRAY-INDEX");
            Overlap::EqualOrDisjoint
        }
        (
            ProjectionElem::ConstantIndex {
                offset: o1,
                min_length: _,
                from_end: false,
            },
            ProjectionElem::ConstantIndex {
                offset: o2,
                min_length: _,
                from_end: false,
            },
        )
        | (
            ProjectionElem::ConstantIndex {
                offset: o1,
                min_length: _,
                from_end: true,
            },
            ProjectionElem::ConstantIndex {
                offset: o2,
                min_length: _,
                from_end: true,
            },
        ) => {
            if o1 == o2 {
                debug!("place_element_conflict: DISJOINT-OR-EQ-ARRAY-CONSTANT-INDEX");
                Overlap::EqualOrDisjoint
            } else {
                debug!("place_element_conflict: DISJOINT-ARRAY-CONSTANT-INDEX");
                Overlap::Disjoint
            }
        }
        (
            ProjectionElem::ConstantIndex {
                offset: offset_from_begin,
                min_length: min_length1,
                from_end: false,
            },
            ProjectionElem::ConstantIndex {
                offset: offset_from_end,
                min_length: min_length2,
                from_end: true,
            },
        )
        | (
            ProjectionElem::ConstantIndex {
                offset: offset_from_end,
                min_length: min_length1,
                from_end: true,
            },
            ProjectionElem::ConstantIndex {
                offset: offset_from_begin,
                min_length: min_length2,
                from_end: false,
            },
        ) => {
            // both patterns matched so it must be at least the greater of the two
            let min_length = max(min_length1, min_length2);
            // `offset_from_end` can be in range `[1..min_length]`, 1 indicates the last
            // element (like -1 in Python) and `min_length` the first.
            // Therefore, `min_length - offset_from_end` gives the minimal possible
            // offset from the beginning
            if *offset_from_begin >= min_length - offset_from_end {
                debug!("place_element_conflict: DISJOINT-OR-EQ-ARRAY-CONSTANT-INDEX-FE");
                Overlap::EqualOrDisjoint
            } else {
                debug!("place_element_conflict: DISJOINT-ARRAY-CONSTANT-INDEX-FE");
                Overlap::Disjoint
            }
        }
        (
            ProjectionElem::ConstantIndex {
                offset,
                min_length: _,
                from_end: false,
            },
            ProjectionElem::Subslice { from, .. },
        )
        | (
            ProjectionElem::Subslice { from, .. },
            ProjectionElem::ConstantIndex {
                offset,
                min_length: _,
                from_end: false,
            },
        ) => {
            if offset >= from {
                debug!("place_element_conflict: DISJOINT-OR-EQ-ARRAY-CONSTANT-INDEX-SUBSLICE");
                Overlap::EqualOrDisjoint
            } else {
                debug!("place_element_conflict: DISJOINT-ARRAY-CONSTANT-INDEX-SUBSLICE");
                Overlap::Disjoint
            }
        }
        (
            ProjectionElem::ConstantIndex {
                offset,
                min_length: _,
                from_end: true,
            },
            ProjectionElem::Subslice { from: _, to },
        )
        | (
            ProjectionElem::Subslice { from: _, to },
            ProjectionElem::ConstantIndex {
                offset,
                min_length: _,
                from_end: true,
            },
        ) => {
            if offset > to {
                debug!(
                    "place_element_conflict: \
                     DISJOINT-OR-EQ-ARRAY-CONSTANT-INDEX-SUBSLICE-FE"
                );
                Overlap::EqualOrDisjoint
            } else {
                debug!("place_element_conflict: DISJOINT-ARRAY-CONSTANT-INDEX-SUBSLICE-FE");
                Overlap::Disjoint
            }
        }
        (ProjectionElem::Subslice { .. }, ProjectionElem::Subslice { .. }) => {
            debug!("place_element_conflict: DISJOINT-OR-EQ-ARRAY-SUBSLICES");
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
