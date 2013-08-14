// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

# Type Coercion

Under certain circumstances we will coerce from one type to another,
for example by auto-borrowing.  This occurs in situations where the
compiler has a firm 'expected type' that was supplied from the user,
and where the actual type is similar to that expected type in purpose
but not in representation (so actual subtyping is inappropriate).

## Reborrowing

Note that if we are expecting a borrowed pointer, we will *reborrow*
even if the argument provided was already a borrowed pointer.  This is
useful for freezing mut/const things (that is, when the expected is &T
but you have &const T or &mut T) and also for avoiding the linearity
of mut things (when the expected is &mut T and you have &mut T).  See
the various `src/test/run-pass/coerce-reborrow-*.rs` tests for
examples of where this is useful.

## Subtle note

When deciding what type coercions to consider, we do not attempt to
resolve any type variables we may encounter.  This is because `b`
represents the expected type "as the user wrote it", meaning that if
the user defined a generic function like

   fn foo<A>(a: A, b: A) { ... }

and then we wrote `foo(&1, @2)`, we will not auto-borrow
either argument.  In older code we went to some lengths to
resolve the `b` variable, which could mean that we'd
auto-borrow later arguments but not earlier ones, which
seems very confusing.

## Subtler note

However, right now, if the user manually specifies the
values for the type variables, as so:

   foo::<&int>(@1, @2)

then we *will* auto-borrow, because we can't distinguish this from a
function that declared `&int`.  This is inconsistent but it's easiest
at the moment. The right thing to do, I think, is to consider the
*unsubstituted* type when deciding whether to auto-borrow, but the
*substituted* type when considering the bounds and so forth. But most
of our methods don't give access to the unsubstituted type, and
rightly so because they'd be error-prone.  So maybe the thing to do is
to actually determine the kind of coercions that should occur
separately and pass them in.  Or maybe it's ok as is.  Anyway, it's
sort of a minor point so I've opted to leave it for later---after all
we may want to adjust precisely when coercions occur.

*/


use middle::ty::{AutoPtr, AutoBorrowVec, AutoBorrowFn, AutoBorrowObj};
use middle::ty::{AutoDerefRef};
use middle::ty::{vstore_slice, vstore_box, vstore_uniq};
use middle::ty::{mt};
use middle::ty;
use middle::typeck::infer::{CoerceResult, resolve_type, Coercion};
use middle::typeck::infer::combine::CombineFields;
use middle::typeck::infer::sub::Sub;
use middle::typeck::infer::to_str::InferStr;
use middle::typeck::infer::resolve::try_resolve_tvar_shallow;
use util::common::indenter;

use syntax::ast::m_imm;
use syntax::ast;

// Note: Coerce is not actually a combiner, in that it does not
// conform to the same interface, though it performs a similar
// function.
pub struct Coerce(CombineFields);

impl Coerce {
    pub fn tys(&self, a: ty::t, b: ty::t) -> CoerceResult {
        debug!("Coerce.tys(%s => %s)",
               a.inf_str(self.infcx),
               b.inf_str(self.infcx));
        let _indent = indenter();

        // Examine the supertype and consider auto-borrowing.
        //
        // Note: does not attempt to resolve type variables we encounter.
        // See above for details.
        match ty::get(b).sty {
            ty::ty_rptr(_, mt_b) => {
                return do self.unpack_actual_value(a) |sty_a| {
                    self.coerce_borrowed_pointer(a, sty_a, b, mt_b)
                };
            }

            ty::ty_estr(vstore_slice(_)) => {
                return do self.unpack_actual_value(a) |sty_a| {
                    self.coerce_borrowed_string(a, sty_a, b)
                };
            }

            ty::ty_evec(mt_b, vstore_slice(_)) => {
                return do self.unpack_actual_value(a) |sty_a| {
                    self.coerce_borrowed_vector(a, sty_a, b, mt_b)
                };
            }

            ty::ty_closure(ty::ClosureTy {sigil: ast::BorrowedSigil, _}) => {
                return do self.unpack_actual_value(a) |sty_a| {
                    self.coerce_borrowed_fn(a, sty_a, b)
                };
            }

            ty::ty_trait(_, _, ty::RegionTraitStore(*), m, _) => {
                return do self.unpack_actual_value(a) |sty_a| {
                    self.coerce_borrowed_object(a, sty_a, b, m)
                };
            }

            ty::ty_ptr(mt_b) => {
                return do self.unpack_actual_value(a) |sty_a| {
                    self.coerce_unsafe_ptr(a, sty_a, b, mt_b)
                };
            }

            _ => {}
        }

        do self.unpack_actual_value(a) |sty_a| {
            match *sty_a {
                ty::ty_bare_fn(ref a_f) => {
                    // Bare functions are coercable to any closure type.
                    //
                    // FIXME(#3320) this should go away and be
                    // replaced with proper inference, got a patch
                    // underway - ndm
                    self.coerce_from_bare_fn(a, a_f, b)
                }
                _ => {
                    // Otherwise, just use subtyping rules.
                    self.subtype(a, b)
                }
            }
        }
    }

    pub fn subtype(&self, a: ty::t, b: ty::t) -> CoerceResult {
        match Sub(**self).tys(a, b) {
            Ok(_) => Ok(None),         // No coercion required.
            Err(ref e) => Err(*e)
        }
    }

    pub fn unpack_actual_value(&self,
                               a: ty::t,
                               f: &fn(&ty::sty) -> CoerceResult)
                               -> CoerceResult {
        match resolve_type(self.infcx, a, try_resolve_tvar_shallow) {
            Ok(t) => {
                f(&ty::get(t).sty)
            }
            Err(e) => {
                self.infcx.tcx.sess.span_bug(
                    self.trace.origin.span(),
                    fmt!("Failed to resolve even without \
                          any force options: %?", e));
            }
        }
    }

    pub fn coerce_borrowed_pointer(&self,
                                   a: ty::t,
                                   sty_a: &ty::sty,
                                   b: ty::t,
                                   mt_b: ty::mt)
                                   -> CoerceResult {
        debug!("coerce_borrowed_pointer(a=%s, sty_a=%?, b=%s, mt_b=%?)",
               a.inf_str(self.infcx), sty_a,
               b.inf_str(self.infcx), mt_b);

        // If we have a parameter of type `&M T_a` and the value
        // provided is `expr`, we will be adding an implicit borrow,
        // meaning that we convert `f(expr)` to `f(&M *expr)`.  Therefore,
        // to type check, we will construct the type that `&M*expr` would
        // yield.

        let sub = Sub(**self);
        let r_borrow = self.infcx.next_region_var(Coercion(self.trace));

        let inner_ty = match *sty_a {
            ty::ty_box(mt_a) => mt_a.ty,
            ty::ty_uniq(mt_a) => mt_a.ty,
            ty::ty_rptr(_, mt_a) => mt_a.ty,
            _ => {
                return self.subtype(a, b);
            }
        };

        let a_borrowed = ty::mk_rptr(self.infcx.tcx,
                                     r_borrow,
                                     mt {ty: inner_ty, mutbl: mt_b.mutbl});
        if_ok!(sub.tys(a_borrowed, b));
        Ok(Some(@AutoDerefRef(AutoDerefRef {
            autoderefs: 1,
            autoref: Some(AutoPtr(r_borrow, mt_b.mutbl))
        })))
    }

    pub fn coerce_borrowed_string(&self,
                                  a: ty::t,
                                  sty_a: &ty::sty,
                                  b: ty::t)
                                  -> CoerceResult {
        debug!("coerce_borrowed_string(a=%s, sty_a=%?, b=%s)",
               a.inf_str(self.infcx), sty_a,
               b.inf_str(self.infcx));

        match *sty_a {
            ty::ty_estr(vstore_box) |
            ty::ty_estr(vstore_uniq) => {}
            _ => {
                return self.subtype(a, b);
            }
        };

        let r_a = self.infcx.next_region_var(Coercion(self.trace));
        let a_borrowed = ty::mk_estr(self.infcx.tcx, vstore_slice(r_a));
        if_ok!(self.subtype(a_borrowed, b));
        Ok(Some(@AutoDerefRef(AutoDerefRef {
            autoderefs: 0,
            autoref: Some(AutoBorrowVec(r_a, m_imm))
        })))
    }

    pub fn coerce_borrowed_vector(&self,
                                  a: ty::t,
                                  sty_a: &ty::sty,
                                  b: ty::t,
                                  mt_b: ty::mt)
                                  -> CoerceResult {
        debug!("coerce_borrowed_vector(a=%s, sty_a=%?, b=%s)",
               a.inf_str(self.infcx), sty_a,
               b.inf_str(self.infcx));

        let sub = Sub(**self);
        let r_borrow = self.infcx.next_region_var(Coercion(self.trace));
        let ty_inner = match *sty_a {
            ty::ty_evec(mt, _) => mt.ty,
            _ => {
                return self.subtype(a, b);
            }
        };

        let a_borrowed = ty::mk_evec(self.infcx.tcx,
                                     mt {ty: ty_inner, mutbl: mt_b.mutbl},
                                     vstore_slice(r_borrow));
        if_ok!(sub.tys(a_borrowed, b));
        Ok(Some(@AutoDerefRef(AutoDerefRef {
            autoderefs: 0,
            autoref: Some(AutoBorrowVec(r_borrow, mt_b.mutbl))
        })))
    }

    fn coerce_borrowed_object(&self,
                              a: ty::t,
                              sty_a: &ty::sty,
                              b: ty::t,
                              b_mutbl: ast::mutability) -> CoerceResult
    {
        debug!("coerce_borrowed_object(a=%s, sty_a=%?, b=%s)",
               a.inf_str(self.infcx), sty_a,
               b.inf_str(self.infcx));

        let tcx = self.infcx.tcx;
        let r_a = self.infcx.next_region_var(Coercion(self.trace));

        let a_borrowed = match *sty_a {
            ty::ty_trait(did, ref substs, _, _, b) => {
                ty::mk_trait(tcx, did, substs.clone(),
                             ty::RegionTraitStore(r_a), b_mutbl, b)
            }
            _ => {
                return self.subtype(a, b);
            }
        };

        if_ok!(self.subtype(a_borrowed, b));
        Ok(Some(@AutoDerefRef(AutoDerefRef {
            autoderefs: 0,
            autoref: Some(AutoBorrowObj(r_a, b_mutbl))
        })))
    }

    pub fn coerce_borrowed_fn(&self,
                              a: ty::t,
                              sty_a: &ty::sty,
                              b: ty::t)
                              -> CoerceResult {
        debug!("coerce_borrowed_fn(a=%s, sty_a=%?, b=%s)",
               a.inf_str(self.infcx), sty_a,
               b.inf_str(self.infcx));

        let fn_ty = match *sty_a {
            ty::ty_closure(ref f) if f.sigil == ast::ManagedSigil ||
                                     f.sigil == ast::OwnedSigil => {
                (*f).clone()
            }
            ty::ty_bare_fn(ref f) => {
                return self.coerce_from_bare_fn(a, f, b);
            }
            _ => {
                return self.subtype(a, b);
            }
        };

        let r_borrow = self.infcx.next_region_var(Coercion(self.trace));
        let a_borrowed = ty::mk_closure(
            self.infcx.tcx,
            ty::ClosureTy {
                sigil: ast::BorrowedSigil,
                region: r_borrow,
                ..fn_ty
            });

        if_ok!(self.subtype(a_borrowed, b));
        Ok(Some(@AutoDerefRef(AutoDerefRef {
            autoderefs: 0,
            autoref: Some(AutoBorrowFn(r_borrow))
        })))
    }

    pub fn coerce_from_bare_fn(&self,
                               a: ty::t,
                               fn_ty_a: &ty::BareFnTy,
                               b: ty::t)
                               -> CoerceResult {
        do self.unpack_actual_value(b) |sty_b| {
            self.coerce_from_bare_fn_post_unpack(a, fn_ty_a, b, sty_b)
        }
    }

    pub fn coerce_from_bare_fn_post_unpack(&self,
                                           a: ty::t,
                                           fn_ty_a: &ty::BareFnTy,
                                           b: ty::t,
                                           sty_b: &ty::sty)
                                           -> CoerceResult {
        /*!
         *
         * Attempts to coerce from a bare Rust function (`extern
         * "rust" fn`) into a closure.
         */

        debug!("coerce_from_bare_fn(a=%s, b=%s)",
               a.inf_str(self.infcx), b.inf_str(self.infcx));

        if !fn_ty_a.abis.is_rust() {
            return self.subtype(a, b);
        }

        let fn_ty_b = match *sty_b {
            ty::ty_closure(ref f) => (*f).clone(),
            _ => return self.subtype(a, b),
        };

        let adj = @ty::AutoAddEnv(fn_ty_b.region, fn_ty_b.sigil);
        let a_closure = ty::mk_closure(self.infcx.tcx,
                                       ty::ClosureTy {
                                            sig: fn_ty_a.sig.clone(),
                                            ..fn_ty_b
                                       });
        if_ok!(self.subtype(a_closure, b));
        Ok(Some(adj))
    }

    pub fn coerce_unsafe_ptr(&self,
                             a: ty::t,
                             sty_a: &ty::sty,
                             b: ty::t,
                             mt_b: ty::mt)
                             -> CoerceResult {
        debug!("coerce_unsafe_ptr(a=%s, sty_a=%?, b=%s)",
               a.inf_str(self.infcx), sty_a,
               b.inf_str(self.infcx));

        let mt_a = match *sty_a {
            ty::ty_rptr(_, mt) => mt,
            _ => {
                return self.subtype(a, b);
            }
        };

        // check that the types which they point at are compatible
        let a_unsafe = ty::mk_ptr(self.infcx.tcx, mt_a);
        if_ok!(self.subtype(a_unsafe, b));

        // although borrowed ptrs and unsafe ptrs have the same
        // representation, we still register an AutoDerefRef so that
        // regionck knows that the region for `a` must be valid here
        Ok(Some(@AutoDerefRef(AutoDerefRef {
            autoderefs: 1,
            autoref: Some(ty::AutoUnsafe(mt_b.mutbl))
        })))
    }
}
