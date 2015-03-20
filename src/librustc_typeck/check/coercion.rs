// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Type Coercion
//!
//! Under certain circumstances we will coerce from one type to another,
//! for example by auto-borrowing.  This occurs in situations where the
//! compiler has a firm 'expected type' that was supplied from the user,
//! and where the actual type is similar to that expected type in purpose
//! but not in representation (so actual subtyping is inappropriate).
//!
//! ## Reborrowing
//!
//! Note that if we are expecting a reference, we will *reborrow*
//! even if the argument provided was already a reference.  This is
//! useful for freezing mut/const things (that is, when the expected is &T
//! but you have &const T or &mut T) and also for avoiding the linearity
//! of mut things (when the expected is &mut T and you have &mut T).  See
//! the various `src/test/run-pass/coerce-reborrow-*.rs` tests for
//! examples of where this is useful.
//!
//! ## Subtle note
//!
//! When deciding what type coercions to consider, we do not attempt to
//! resolve any type variables we may encounter.  This is because `b`
//! represents the expected type "as the user wrote it", meaning that if
//! the user defined a generic function like
//!
//!    fn foo<A>(a: A, b: A) { ... }
//!
//! and then we wrote `foo(&1, @2)`, we will not auto-borrow
//! either argument.  In older code we went to some lengths to
//! resolve the `b` variable, which could mean that we'd
//! auto-borrow later arguments but not earlier ones, which
//! seems very confusing.
//!
//! ## Subtler note
//!
//! However, right now, if the user manually specifies the
//! values for the type variables, as so:
//!
//!    foo::<&int>(@1, @2)
//!
//! then we *will* auto-borrow, because we can't distinguish this from a
//! function that declared `&int`.  This is inconsistent but it's easiest
//! at the moment. The right thing to do, I think, is to consider the
//! *unsubstituted* type when deciding whether to auto-borrow, but the
//! *substituted* type when considering the bounds and so forth. But most
//! of our methods don't give access to the unsubstituted type, and
//! rightly so because they'd be error-prone.  So maybe the thing to do is
//! to actually determine the kind of coercions that should occur
//! separately and pass them in.  Or maybe it's ok as is.  Anyway, it's
//! sort of a minor point so I've opted to leave it for later---after all
//! we may want to adjust precisely when coercions occur.

use check::{autoderef, FnCtxt, NoPreference, PreferMutLvalue, UnresolvedTypeAction};

use middle::infer::{self, cres, Coercion, TypeTrace};
use middle::infer::combine::Combine;
use middle::infer::sub::Sub;
use middle::subst;
use middle::ty::{AutoDerefRef, AdjustDerefRef};
use middle::ty::{self, mt, Ty};
use util::common::indent;
use util::ppaux;
use util::ppaux::Repr;

use syntax::ast;

struct Coerce<'a, 'tcx: 'a> {
    fcx: &'a FnCtxt<'a, 'tcx>,
    trace: TypeTrace<'tcx>
}

type CoerceResult<'tcx> = cres<'tcx, Option<ty::AutoAdjustment<'tcx>>>;

impl<'f, 'tcx> Coerce<'f, 'tcx> {
    fn tcx(&self) -> &ty::ctxt<'tcx> {
        self.fcx.tcx()
    }

    fn subtype(&self, a: Ty<'tcx>, b: Ty<'tcx>) -> CoerceResult<'tcx> {
        let sub = Sub(self.fcx.infcx().combine_fields(false, self.trace.clone()));
        try!(sub.tys(a, b));
        Ok(None) // No coercion required.
    }

    fn outlives(&self, a: ty::Region, b: ty::Region) -> cres<'tcx, ()> {
        let sub = Sub(self.fcx.infcx().combine_fields(false, self.trace.clone()));
        try!(sub.regions(b, a));
        Ok(())
    }

    fn unpack_actual_value<T, F>(&self, a: Ty<'tcx>, f: F) -> T where
        F: FnOnce(Ty<'tcx>) -> T,
    {
        f(self.fcx.infcx().shallow_resolve(a))
    }

    fn coerce(&self,
              expr_a: &ast::Expr,
              a: Ty<'tcx>,
              b: Ty<'tcx>)
              -> CoerceResult<'tcx> {
        debug!("Coerce.tys({} => {})",
               a.repr(self.tcx()),
               b.repr(self.tcx()));

        // Consider coercing the subtype to a DST
        let unsize = self.unpack_actual_value(a, |a| {
            self.coerce_unsized(a, b)
        });
        if unsize.is_ok() {
            return unsize;
        }

        // Examine the supertype and consider auto-borrowing.
        //
        // Note: does not attempt to resolve type variables we encounter.
        // See above for details.
        match b.sty {
            ty::ty_ptr(mt_b) => {
                return self.unpack_actual_value(a, |a| {
                    self.coerce_unsafe_ptr(a, b, mt_b.mutbl)
                });
            }

            ty::ty_rptr(_, mt_b) => {
                return self.unpack_actual_value(a, |a| {
                    self.coerce_borrowed_pointer(expr_a, a, b, mt_b.mutbl)
                });
            }

            _ => {}
        }

        self.unpack_actual_value(a, |a| {
            match a.sty {
                ty::ty_bare_fn(Some(_), a_f) => {
                    // Function items are coercible to any closure
                    // type; function pointers are not (that would
                    // require double indirection).
                    self.coerce_from_fn_item(a, a_f, b)
                }
                ty::ty_bare_fn(None, a_f) => {
                    // We permit coercion of fn pointers to drop the
                    // unsafe qualifier.
                    self.coerce_from_fn_pointer(a, a_f, b)
                }
                _ => {
                    // Otherwise, just use subtyping rules.
                    self.subtype(a, b)
                }
            }
        })
    }

    /// Reborrows `&mut A` to `&mut B` and `&(mut) A` to `&B`.
    /// To match `A` with `B`, autoderef will be performed,
    /// calling `deref`/`deref_mut` where necessary.
    fn coerce_borrowed_pointer(&self,
                               expr_a: &ast::Expr,
                               a: Ty<'tcx>,
                               b: Ty<'tcx>,
                               mutbl_b: ast::Mutability)
                               -> CoerceResult<'tcx> {
        debug!("coerce_borrowed_pointer(a={}, b={})",
               a.repr(self.tcx()),
               b.repr(self.tcx()));

        // If we have a parameter of type `&M T_a` and the value
        // provided is `expr`, we will be adding an implicit borrow,
        // meaning that we convert `f(expr)` to `f(&M *expr)`.  Therefore,
        // to type check, we will construct the type that `&M*expr` would
        // yield.

        match a.sty {
            ty::ty_rptr(_, mt_a) => {
                try!(coerce_mutbls(mt_a.mutbl, mutbl_b));
            }
            _ => return self.subtype(a, b)
        }

        let coercion = Coercion(self.trace.clone());
        let r_borrow = self.fcx.infcx().next_region_var(coercion);
        let r_borrow = self.tcx().mk_region(r_borrow);
        let autoref = Some(ty::AutoPtr(r_borrow, mutbl_b));

        let lvalue_pref = match mutbl_b {
            ast::MutMutable => PreferMutLvalue,
            ast::MutImmutable => NoPreference
        };
        let mut first_error = None;
        let (_, autoderefs, success) = autoderef(self.fcx,
                                                 expr_a.span,
                                                 a,
                                                 Some(expr_a),
                                                 UnresolvedTypeAction::Ignore,
                                                 lvalue_pref,
                                                 |inner_ty, autoderef| {
            if autoderef == 0 {
                // Don't let this pass, otherwise it would cause
                // &T to autoref to &&T.
                return None;
            }
            let ty = ty::mk_rptr(self.tcx(), r_borrow,
                                 mt {ty: inner_ty, mutbl: mutbl_b});
            if let Err(err) = self.fcx.infcx().try(|_| self.subtype(ty, b)) {
                if first_error.is_none() {
                    first_error = Some(err);
                }
                None
            } else {
                Some(())
            }
        });

        match success {
            Some(_) => {
                Ok(Some(AdjustDerefRef(AutoDerefRef {
                    autoderefs: autoderefs,
                    autoref: autoref,
                    unsize: None
                })))
            }
            None => {
                // Return original error as if overloaded deref was never
                // attempted, to avoid irrelevant/confusing error messages.
                Err(first_error.expect("coerce_borrowed_pointer failed with no error?"))
            }
        }
    }


    // &[T, ..n] or &mut [T, ..n] -> &[T]
    // or &mut [T, ..n] -> &mut [T]
    // or &Concrete -> &Trait, etc.
    fn coerce_unsized(&self,
                      a: Ty<'tcx>,
                      b: Ty<'tcx>)
                      -> CoerceResult<'tcx> {
        debug!("coerce_unsized(a={}, b={})",
               a.repr(self.tcx()),
               b.repr(self.tcx()));

        // Note, we want to avoid unnecessary unsizing. We don't want to coerce to
        // a DST unless we have to. This currently comes out in the wash since
        // we can't unify [T] with U. But to properly support DST, we need to allow
        // that, at which point we will need extra checks on b here.

        let (reborrow, unsize) = match (&a.sty, &b.sty) {
            (&ty::ty_rptr(_, mt_a), &ty::ty_rptr(_, mt_b)) => {
                if let Some(unsize) = self.unsize_ty(mt_a.ty, mt_b.ty) {
                    try!(coerce_mutbls(mt_a.mutbl, mt_b.mutbl));

                    let coercion = Coercion(self.trace.clone());
                    let r_borrow = self.fcx.infcx().next_region_var(coercion);
                    let region = self.tcx().mk_region(r_borrow);
                    (Some(ty::AutoPtr(region, mt_b.mutbl)), unsize)
                } else {
                    return Err(ty::terr_mismatch);
                }
            }
            (&ty::ty_rptr(_, mt_a), &ty::ty_ptr(mt_b)) => {
                if let Some(unsize) = self.unsize_ty(mt_a.ty, mt_b.ty) {
                    try!(coerce_mutbls(mt_a.mutbl, mt_b.mutbl));
                    (Some(ty::AutoUnsafe(mt_b.mutbl)), unsize)
                } else {
                    return Err(ty::terr_mismatch);
                }
            }
            (&ty::ty_uniq(t_a), &ty::ty_uniq(t_b)) => {
                if let Some(mut unsize) = self.unsize_ty(t_a, t_b) {
                    unsize.target = ty::mk_uniq(self.tcx(), unsize.target);
                    (None, unsize)
                } else {
                    return Err(ty::terr_mismatch);
                }
            }
            _ => return Err(ty::terr_mismatch)
        };

        let ty = ty::adjust_ty_for_autoref(self.tcx(), unsize.target, reborrow);
        try!(self.fcx.infcx().try(|_| self.subtype(ty, b)));
        let adjustment = AutoDerefRef {
            autoderefs: if reborrow.is_some() { 1 } else { 0 },
            autoref: reborrow,
            unsize: Some(ty::AutoUnsize {
                target: ty,
                ..unsize
            })
        };
        debug!("Success, coerced with {}", adjustment.repr(self.tcx()));
        Ok(Some(AdjustDerefRef(adjustment)))
    }

    // Takes a type and returns an unsized version.
    // E.g., `[T, ..n]` -> `[T]`.
    fn unsize_ty(&self,
                 ty_a: Ty<'tcx>,
                 ty_b: Ty<'tcx>)
                 -> Option<ty::AutoUnsize<'tcx>> {
        let tcx = self.tcx();

        self.unpack_actual_value(ty_a, |a| self.unpack_actual_value(ty_b, |b| {
            debug!("unsize_ty(a={}, b={})", a.repr(self.tcx()), b.repr(self.tcx()));
            match (&a.sty, &b.sty) {
                (&ty::ty_vec(t_a, Some(_)), &ty::ty_vec(_, None)) => {
                    let ty = ty::mk_vec(tcx, t_a, None);
                    Some(ty::AutoUnsize {
                        leaf_source: a,
                        leaf_target: ty,
                        target: ty
                    })
                }
                (&ty::ty_trait(ref data_a), &ty::ty_trait(ref data_b)) => {
                    // Upcasts permit two things:
                    //
                    // 1. Dropping builtin bounds, e.g. `Foo+Send` to `Foo`
                    // 2. Tightening the region bound, e.g. `Foo+'a` to `Foo+'b` if `'a : 'b`
                    //
                    // Note that neither of these changes requires any
                    // change at runtime.  Eventually this will be
                    // generalized.
                    //
                    // We always upcast when we can because of reason
                    // #2 (region bounds).
                    if data_a.bounds.builtin_bounds.is_superset(&data_b.bounds.builtin_bounds) {
                        // construct a type `a1` which is a version of
                        // `a` using the upcast bounds from `b`
                        let bounds_a1 = ty::ExistentialBounds {
                            // From type b
                            region_bound: data_b.bounds.region_bound,
                            builtin_bounds: data_b.bounds.builtin_bounds,

                            // From type a
                            projection_bounds: data_a.bounds.projection_bounds.clone(),
                        };
                        let ty_a1 = ty::mk_trait(tcx, data_a.principal.clone(), bounds_a1);

                        // relate `a1` to `b`
                        let result = self.fcx.infcx().try(|_| {
                            // it's ok to upcast from Foo+'a to Foo+'b so long as 'a : 'b
                            try!(self.outlives(data_a.bounds.region_bound,
                                               data_b.bounds.region_bound));
                            self.subtype(ty_a1, ty_b)
                        });

                        // if that was successful, we have a coercion
                        match result {
                            Ok(_) => Some(ty::AutoUnsize {
                                leaf_source: a,
                                leaf_target: ty_b,
                                target: ty_b
                            }),
                            Err(_) => None,
                        }
                    } else {
                        None
                    }
                }
                (_, &ty::ty_trait(_)) => {
                    Some(ty::AutoUnsize {
                        leaf_source: a,
                        leaf_target: ty_b,
                        target: ty_b
                    })
                }
                (&ty::ty_struct(did_a, substs_a), &ty::ty_struct(did_b, substs_b))
                  if did_a == did_b => {
                    debug!("unsizing a struct");
                    // Try unsizing each type param in turn to see if we end up with ty_b.
                    let ty_substs_a = substs_a.types.get_slice(subst::TypeSpace);
                    let ty_substs_b = substs_b.types.get_slice(subst::TypeSpace);
                    assert!(ty_substs_a.len() == ty_substs_b.len());

                    let tps = ty_substs_a.iter().zip(ty_substs_b.iter()).enumerate();
                    for (i, (&tp_a, &tp_b)) in tps {
                        if self.fcx.infcx().try(|_| self.subtype(tp_a, tp_b)).is_ok() {
                            continue;
                        }
                        if let Some(unsize) = self.unsize_ty(tp_a, tp_b) {
                            // Check that the whole types match.
                            let mut new_substs = substs_a.clone();
                            let new_tp = unsize.target;
                            new_substs.types.get_mut_slice(subst::TypeSpace)[i] = new_tp;
                            let ty = ty::mk_struct(tcx, did_a, tcx.mk_substs(new_substs));
                            if self.fcx.infcx().try(|_| self.subtype(ty, ty_b)).is_err() {
                                debug!("Unsized type parameter '{}', but still \
                                        could not match types {} and {}",
                                        ppaux::ty_to_string(tcx, tp_a),
                                        ppaux::ty_to_string(tcx, ty),
                                        ppaux::ty_to_string(tcx, ty_b));
                                // We can only unsize a single type parameter, so
                                // if we unsize one and it doesn't give us the
                                // type we want, then we won't succeed later.
                                break;
                            }

                            return Some(ty::AutoUnsize { target: ty, ..unsize });
                        }
                    }
                    None
                }
                _ => None
            }
        }))
    }

    fn coerce_from_fn_pointer(&self,
                           a: Ty<'tcx>,
                           fn_ty_a: &'tcx ty::BareFnTy<'tcx>,
                           b: Ty<'tcx>)
                           -> CoerceResult<'tcx>
    {
        /*!
         * Attempts to coerce from the type of a Rust function item
         * into a closure or a `proc`.
         */

        self.unpack_actual_value(b, |b| {
            debug!("coerce_from_fn_pointer(a={}, b={})",
                   a.repr(self.tcx()), b.repr(self.tcx()));

            if let ty::ty_bare_fn(None, fn_ty_b) = b.sty {
                match (fn_ty_a.unsafety, fn_ty_b.unsafety) {
                    (ast::Unsafety::Normal, ast::Unsafety::Unsafe) => {
                        let unsafe_a = self.tcx().safe_to_unsafe_fn_ty(fn_ty_a);
                        try!(self.subtype(unsafe_a, b));
                        return Ok(Some(ty::AdjustUnsafeFnPointer));
                    }
                    _ => {}
                }
            }
            self.subtype(a, b)
        })
    }

    fn coerce_from_fn_item(&self,
                           a: Ty<'tcx>,
                           fn_ty_a: &'tcx ty::BareFnTy<'tcx>,
                           b: Ty<'tcx>)
                           -> CoerceResult<'tcx> {
        /*!
         * Attempts to coerce from the type of a Rust function item
         * into a closure or a `proc`.
         */

        self.unpack_actual_value(b, |b| {
            debug!("coerce_from_fn_item(a={}, b={})",
                   a.repr(self.tcx()), b.repr(self.tcx()));

            match b.sty {
                ty::ty_bare_fn(None, _) => {
                    let a_fn_pointer = ty::mk_bare_fn(self.tcx(), None, fn_ty_a);
                    try!(self.subtype(a_fn_pointer, b));
                    Ok(Some(ty::AdjustReifyFnPointer))
                }
                _ => self.subtype(a, b)
            }
        })
    }

    fn coerce_unsafe_ptr(&self,
                         a: Ty<'tcx>,
                         b: Ty<'tcx>,
                         mutbl_b: ast::Mutability)
                         -> CoerceResult<'tcx> {
        debug!("coerce_unsafe_ptr(a={}, b={})",
               a.repr(self.tcx()),
               b.repr(self.tcx()));

        let mt_a = match a.sty {
            ty::ty_rptr(_, mt) | ty::ty_ptr(mt) => mt,
            _ => {
                return self.subtype(a, b);
            }
        };

        // Check that the types which they point at are compatible.
        let a_unsafe = ty::mk_ptr(self.tcx(), ty::mt{ mutbl: mutbl_b, ty: mt_a.ty });
        try!(self.subtype(a_unsafe, b));
        try!(coerce_mutbls(mt_a.mutbl, mutbl_b));

        // Although references and unsafe ptrs have the same
        // representation, we still register an AutoDerefRef so that
        // regionck knows that the region for `a` must be valid here.
        Ok(Some(AdjustDerefRef(AutoDerefRef {
            autoderefs: 1,
            autoref: Some(ty::AutoUnsafe(mutbl_b)),
            unsize: None
        })))
    }
}

pub fn mk_assignty<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                             expr: &ast::Expr,
                             a: Ty<'tcx>,
                             b: Ty<'tcx>)
                             -> cres<'tcx, ()> {
    debug!("mk_assignty({} -> {})", a.repr(fcx.tcx()), b.repr(fcx.tcx()));
    let adjustment = try!(indent(|| {
        fcx.infcx().commit_if_ok(|| {
            let origin = infer::ExprAssignable(expr.span);
            Coerce {
                fcx: fcx,
                trace: infer::TypeTrace::types(origin, false, a, b)
            }.coerce(expr, a, b)
        })
    }));
    if let Some(adjustment) = adjustment {
        fcx.write_adjustment(expr.id, expr.span, adjustment);
    }
    Ok(())
}

fn coerce_mutbls<'tcx>(from_mutbl: ast::Mutability,
                       to_mutbl: ast::Mutability)
                       -> CoerceResult<'tcx> {
    match (from_mutbl, to_mutbl) {
        (ast::MutMutable, ast::MutMutable) |
        (ast::MutImmutable, ast::MutImmutable) |
        (ast::MutMutable, ast::MutImmutable) => Ok(None),
        (ast::MutImmutable, ast::MutMutable) => Err(ty::terr_mutability)
    }
}
