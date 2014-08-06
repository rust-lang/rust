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

Note that if we are expecting a reference, we will *reborrow*
even if the argument provided was already a reference.  This is
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

use middle::subst;
use middle::ty::{AutoPtr, AutoDerefRef, AutoUnsize};
use middle::ty::{mt};
use middle::ty;
use middle::typeck::infer::{CoerceResult, resolve_type, Coercion};
use middle::typeck::infer::combine::{CombineFields, Combine};
use middle::typeck::infer::sub::Sub;
use middle::typeck::infer::resolve::try_resolve_tvar_shallow;
use util::common::indenter;
use util::ppaux;
use util::ppaux::Repr;

use syntax::abi;
use syntax::ast;

// Note: Coerce is not actually a combiner, in that it does not
// conform to the same interface, though it performs a similar
// function.
pub struct Coerce<'f>(pub CombineFields<'f>);

impl<'f> Coerce<'f> {
    pub fn get_ref<'a>(&'a self) -> &'a CombineFields<'f> {
        let Coerce(ref v) = *self; v
    }

    pub fn tys(&self, a: ty::t, b: ty::t) -> CoerceResult {
        debug!("Coerce.tys({} => {})",
               a.repr(self.get_ref().infcx.tcx),
               b.repr(self.get_ref().infcx.tcx));
        let _indent = indenter();

        // Special case: if the subtype is a sized array literal (`[T, ..n]`),
        // then it would get auto-borrowed to `&[T, ..n]` and then DST-ified
        // to `&[T]`. Doing it all at once makes the target code a bit more
        // efficient and spares us from having to handle multiple coercions.
        match ty::get(b).sty {
            ty::ty_rptr(_, mt_b) => {
                match ty::get(mt_b.ty).sty {
                    ty::ty_vec(_, None) => {
                        let unsize_and_ref = self.unpack_actual_value(a, |sty_a| {
                            self.coerce_unsized_with_borrow(a, sty_a, b, mt_b.mutbl)
                        });
                        if unsize_and_ref.is_ok() {
                            return unsize_and_ref;
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }

        // Consider coercing the subtype to a DST
        let unsize = self.unpack_actual_value(a, |sty_a| {
            self.coerce_unsized(a, sty_a, b)
        });
        if unsize.is_ok() {
            return unsize;
        }

        // Examine the supertype and consider auto-borrowing.
        //
        // Note: does not attempt to resolve type variables we encounter.
        // See above for details.
        match ty::get(b).sty {
            ty::ty_rptr(_, mt_b) => {
                match ty::get(mt_b.ty).sty {
                    ty::ty_str => {
                        return self.unpack_actual_value(a, |sty_a| {
                            self.coerce_borrowed_pointer(a, sty_a, b, ast::MutImmutable)
                        });
                    }

                    ty::ty_trait(..) => {
                        let result = self.unpack_actual_value(a, |sty_a| {
                            self.coerce_borrowed_object(a, sty_a, b, mt_b.mutbl)
                        });

                        match result {
                            Ok(t) => return Ok(t),
                            Err(..) => {}
                        }
                    }

                    _ => {
                        return self.unpack_actual_value(a, |sty_a| {
                            self.coerce_borrowed_pointer(a, sty_a, b, mt_b.mutbl)
                        });
                    }
                };
            }

            ty::ty_closure(box ty::ClosureTy {
                    store: ty::RegionTraitStore(..),
                    ..
                }) => {
                return self.unpack_actual_value(a, |sty_a| {
                    self.coerce_borrowed_fn(a, sty_a, b)
                });
            }

            ty::ty_ptr(mt_b) => {
                return self.unpack_actual_value(a, |sty_a| {
                    self.coerce_unsafe_ptr(a, sty_a, b, mt_b)
                });
            }

            _ => {}
        }

        self.unpack_actual_value(a, |sty_a| {
            match *sty_a {
                ty::ty_bare_fn(ref a_f) => {
                    // Bare functions are coercible to any closure type.
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
        })
    }

    pub fn subtype(&self, a: ty::t, b: ty::t) -> CoerceResult {
        match Sub(self.get_ref().clone()).tys(a, b) {
            Ok(_) => Ok(None),         // No coercion required.
            Err(ref e) => Err(*e)
        }
    }

    pub fn unpack_actual_value<T>(&self, a: ty::t, f: |&ty::sty| -> T)
                                  -> T {
        match resolve_type(self.get_ref().infcx, None,
                           a, try_resolve_tvar_shallow) {
            Ok(t) => {
                f(&ty::get(t).sty)
            }
            Err(e) => {
                self.get_ref().infcx.tcx.sess.span_bug(
                    self.get_ref().trace.origin.span(),
                    format!("failed to resolve even without \
                             any force options: {:?}", e).as_slice());
            }
        }
    }

    // ~T -> &T or &mut T -> &T (including where T = [U] or str)
    pub fn coerce_borrowed_pointer(&self,
                                   a: ty::t,
                                   sty_a: &ty::sty,
                                   b: ty::t,
                                   mutbl_b: ast::Mutability)
                                   -> CoerceResult {
        debug!("coerce_borrowed_pointer(a={}, sty_a={:?}, b={})",
               a.repr(self.get_ref().infcx.tcx), sty_a,
               b.repr(self.get_ref().infcx.tcx));

        // If we have a parameter of type `&M T_a` and the value
        // provided is `expr`, we will be adding an implicit borrow,
        // meaning that we convert `f(expr)` to `f(&M *expr)`.  Therefore,
        // to type check, we will construct the type that `&M*expr` would
        // yield.

        let sub = Sub(self.get_ref().clone());
        let coercion = Coercion(self.get_ref().trace.clone());
        let r_borrow = self.get_ref().infcx.next_region_var(coercion);

        let inner_ty = match *sty_a {
            ty::ty_box(_) | ty::ty_uniq(_) => return Err(ty::terr_mismatch),
            ty::ty_rptr(_, mt_a) => mt_a.ty,
            _ => {
                return self.subtype(a, b);
            }
        };

        let a_borrowed = ty::mk_rptr(self.get_ref().infcx.tcx,
                                     r_borrow,
                                     mt {ty: inner_ty, mutbl: mutbl_b});
        if_ok!(sub.tys(a_borrowed, b));

        Ok(Some(AutoDerefRef(AutoDerefRef {
            autoderefs: 1,
            autoref: Some(AutoPtr(r_borrow, mutbl_b, None))
        })))
    }

    // [T, ..n] -> &[T] or &mut [T]
    fn coerce_unsized_with_borrow(&self,
                                  a: ty::t,
                                  sty_a: &ty::sty,
                                  b: ty::t,
                                  mutbl_b: ast::Mutability)
                                  -> CoerceResult {
        debug!("coerce_unsized_with_borrow(a={}, sty_a={:?}, b={})",
               a.repr(self.get_ref().infcx.tcx), sty_a,
               b.repr(self.get_ref().infcx.tcx));

        match *sty_a {
            ty::ty_vec(t_a, Some(len)) => {
                let sub = Sub(self.get_ref().clone());
                let coercion = Coercion(self.get_ref().trace.clone());
                let r_borrow = self.get_ref().infcx.next_region_var(coercion);
                let unsized_ty = ty::mk_slice(self.get_ref().infcx.tcx, r_borrow,
                                              mt {ty: t_a, mutbl: mutbl_b});
                if_ok!(self.get_ref().infcx.try(|| sub.tys(unsized_ty, b)));
                Ok(Some(AutoDerefRef(AutoDerefRef {
                    autoderefs: 0,
                    autoref: Some(ty::AutoPtr(r_borrow,
                                              mutbl_b,
                                              Some(box AutoUnsize(ty::UnsizeLength(len)))))
                })))
            }
            _ => Err(ty::terr_mismatch)
        }
    }

    // &[T, ..n] or &mut [T, ..n] -> &[T]
    // or &mut [T, ..n] -> &mut [T]
    // or &Concrete -> &Trait, etc.
    fn coerce_unsized(&self,
                      a: ty::t,
                      sty_a: &ty::sty,
                      b: ty::t)
                      -> CoerceResult {
        debug!("coerce_unsized(a={}, sty_a={:?}, b={})",
               a.repr(self.get_ref().infcx.tcx), sty_a,
               b.repr(self.get_ref().infcx.tcx));

        // Note, we want to avoid unnecessary unsizing. We don't want to coerce to
        // a DST unless we have to. This currently comes out in the wash since
        // we can't unify [T] with U. But to properly support DST, we need to allow
        // that, at which point we will need extra checks on b here.

        let sub = Sub(self.get_ref().clone());

        let sty_b = &ty::get(b).sty;
        match (sty_a, sty_b) {
            (&ty::ty_uniq(_), &ty::ty_rptr(..)) => Err(ty::terr_mismatch),
            (&ty::ty_rptr(_, ty::mt{ty: t_a, ..}), &ty::ty_rptr(_, mt_b)) => {
                self.unpack_actual_value(t_a, |sty_a| {
                    match self.unsize_ty(sty_a, mt_b.ty) {
                        Some((ty, kind)) => {
                            let coercion = Coercion(self.get_ref().trace.clone());
                            let r_borrow = self.get_ref().infcx.next_region_var(coercion);
                            let ty = ty::mk_rptr(self.get_ref().infcx.tcx,
                                                 r_borrow,
                                                 ty::mt{ty: ty, mutbl: mt_b.mutbl});
                            if_ok!(self.get_ref().infcx.try(|| sub.tys(ty, b)));
                            debug!("Success, coerced with AutoDerefRef(1, \
                                    AutoPtr(AutoUnsize({:?})))", kind);
                            Ok(Some(AutoDerefRef(AutoDerefRef {
                                autoderefs: 1,
                                autoref: Some(ty::AutoPtr(r_borrow, mt_b.mutbl,
                                                          Some(box AutoUnsize(kind))))
                            })))
                        }
                        _ => Err(ty::terr_mismatch)
                    }
                })
            }
            (&ty::ty_uniq(t_a), &ty::ty_uniq(t_b)) => {
                self.unpack_actual_value(t_a, |sty_a| {
                    match self.unsize_ty(sty_a, t_b) {
                        Some((ty, kind)) => {
                            let ty = ty::mk_uniq(self.get_ref().infcx.tcx, ty);
                            if_ok!(self.get_ref().infcx.try(|| sub.tys(ty, b)));
                            debug!("Success, coerced with AutoDerefRef(1, \
                                    AutoUnsizeUniq({:?}))", kind);
                            Ok(Some(AutoDerefRef(AutoDerefRef {
                                autoderefs: 1,
                                autoref: Some(ty::AutoUnsizeUniq(kind))
                            })))
                        }
                        _ => Err(ty::terr_mismatch)
                    }
                })
            }
            _ => Err(ty::terr_mismatch)
        }
    }

    // Takes a type and returns an unsized version along with the adjustment
    // performed to unsize it.
    // E.g., `[T, ..n]` -> `([T], UnsizeLength(n))`
    fn unsize_ty(&self,
                 sty_a: &ty::sty,
                 ty_b: ty::t)
                 -> Option<(ty::t, ty::UnsizeKind)> {
        debug!("unsize_ty(sty_a={:?}", sty_a);

        let tcx = self.get_ref().infcx.tcx;

        self.unpack_actual_value(ty_b, |sty_b|
            match (sty_a, sty_b) {
                (&ty::ty_vec(t_a, Some(len)), _) => {
                    let ty = ty::mk_vec(tcx, t_a, None);
                    Some((ty, ty::UnsizeLength(len)))
                }
                (&ty::ty_trait(..), &ty::ty_trait(..)) => None,
                (_, &ty::ty_trait(box ty::TyTrait { def_id, ref substs, bounds })) => {
                    let ty = ty::mk_trait(tcx,
                                          def_id,
                                          substs.clone(),
                                          bounds);
                    Some((ty, ty::UnsizeVtable(bounds,
                                               def_id,
                                               substs.clone())))
                }
                (&ty::ty_struct(did_a, ref substs_a), &ty::ty_struct(did_b, ref substs_b))
                  if did_a == did_b => {
                    debug!("unsizing a struct");
                    // Try unsizing each type param in turn to see if we end up with ty_b.
                    let ty_substs_a = substs_a.types.get_slice(subst::TypeSpace);
                    let ty_substs_b = substs_b.types.get_slice(subst::TypeSpace);
                    assert!(ty_substs_a.len() == ty_substs_b.len());

                    let sub = Sub(self.get_ref().clone());

                    let mut result = None;
                    let mut tps = ty_substs_a.iter().zip(ty_substs_b.iter()).enumerate();
                    for (i, (tp_a, tp_b)) in tps {
                        if self.get_ref().infcx.try(|| sub.tys(*tp_a, *tp_b)).is_ok() {
                            continue;
                        }
                        match self.unpack_actual_value(*tp_a, |tp| self.unsize_ty(tp, *tp_b)) {
                            Some((new_tp, k)) => {
                                // Check that the whole types match.
                                let mut new_substs = substs_a.clone();
                                new_substs.types.get_mut_slice(subst::TypeSpace)[i] = new_tp;
                                let ty = ty::mk_struct(tcx, did_a, new_substs);
                                if self.get_ref().infcx.try(|| sub.tys(ty, ty_b)).is_err() {
                                    debug!("Unsized type parameter '{}', but still \
                                            could not match types {} and {}",
                                           ppaux::ty_to_string(tcx, *tp_a),
                                           ppaux::ty_to_string(tcx, ty),
                                           ppaux::ty_to_string(tcx, ty_b));
                                    // We can only unsize a single type parameter, so
                                    // if we unsize one and it doesn't give us the
                                    // type we want, then we won't succeed later.
                                    break;
                                }

                                result = Some((ty, ty::UnsizeStruct(box k, i)));
                                break;
                            }
                            None => {}
                        }
                    }
                    result
                }
                _ => None
            }
        )
    }

    fn coerce_borrowed_object(&self,
                              a: ty::t,
                              sty_a: &ty::sty,
                              b: ty::t,
                              b_mutbl: ast::Mutability) -> CoerceResult
    {
        debug!("coerce_borrowed_object(a={}, sty_a={:?}, b={})",
               a.repr(self.get_ref().infcx.tcx), sty_a,
               b.repr(self.get_ref().infcx.tcx));

        let tcx = self.get_ref().infcx.tcx;
        let coercion = Coercion(self.get_ref().trace.clone());
        let r_a = self.get_ref().infcx.next_region_var(coercion);

        let a_borrowed = match *sty_a {
            ty::ty_uniq(ty) | ty::ty_rptr(_, ty::mt{ty, ..}) => match ty::get(ty).sty {
                ty::ty_trait(box ty::TyTrait {
                        def_id,
                        ref substs,
                        bounds,
                        ..
                    }) => {
                    let tr = ty::mk_trait(tcx, def_id, substs.clone(), bounds);
                    ty::mk_rptr(tcx, r_a, ty::mt{ mutbl: b_mutbl, ty: tr })
                }
                _ => {
                    return self.subtype(a, b);
                }
            },
            _ => {
                return self.subtype(a, b);
            }
        };

        if_ok!(self.subtype(a_borrowed, b));
        Ok(Some(AutoDerefRef(AutoDerefRef {
            autoderefs: 1,
            autoref: Some(AutoPtr(r_a, b_mutbl, None))
        })))
    }

    pub fn coerce_borrowed_fn(&self,
                              a: ty::t,
                              sty_a: &ty::sty,
                              b: ty::t)
                              -> CoerceResult {
        debug!("coerce_borrowed_fn(a={}, sty_a={:?}, b={})",
               a.repr(self.get_ref().infcx.tcx), sty_a,
               b.repr(self.get_ref().infcx.tcx));

        match *sty_a {
            ty::ty_bare_fn(ref f) => {
                self.coerce_from_bare_fn(a, f, b)
            }
            _ => {
                self.subtype(a, b)
            }
        }
    }

    fn coerce_from_bare_fn(&self, a: ty::t, fn_ty_a: &ty::BareFnTy, b: ty::t)
                           -> CoerceResult {
        /*!
         *
         * Attempts to coerce from a bare Rust function (`extern
         * "Rust" fn`) into a closure or a `proc`.
         */

        self.unpack_actual_value(b, |sty_b| {

            debug!("coerce_from_bare_fn(a={}, b={})",
                   a.repr(self.get_ref().infcx.tcx), b.repr(self.get_ref().infcx.tcx));

            if fn_ty_a.abi != abi::Rust || fn_ty_a.fn_style != ast::NormalFn {
                return self.subtype(a, b);
            }

            let fn_ty_b = match *sty_b {
                ty::ty_closure(ref f) => (*f).clone(),
                _ => return self.subtype(a, b)
            };

            let adj = ty::AutoAddEnv(fn_ty_b.store);
            let a_closure = ty::mk_closure(self.get_ref().infcx.tcx,
                                           ty::ClosureTy {
                                                sig: fn_ty_a.sig.clone(),
                                                .. *fn_ty_b
                                           });
            if_ok!(self.subtype(a_closure, b));
            Ok(Some(adj))
        })
    }

    pub fn coerce_unsafe_ptr(&self,
                             a: ty::t,
                             sty_a: &ty::sty,
                             b: ty::t,
                             mt_b: ty::mt)
                             -> CoerceResult {
        debug!("coerce_unsafe_ptr(a={}, sty_a={:?}, b={})",
               a.repr(self.get_ref().infcx.tcx), sty_a,
               b.repr(self.get_ref().infcx.tcx));

        let mt_a = match *sty_a {
            ty::ty_rptr(_, mt) => mt,
            _ => {
                return self.subtype(a, b);
            }
        };

        // check that the types which they point at are compatible
        let a_unsafe = ty::mk_ptr(self.get_ref().infcx.tcx, mt_a);
        if_ok!(self.subtype(a_unsafe, b));

        // although references and unsafe ptrs have the same
        // representation, we still register an AutoDerefRef so that
        // regionck knows that the region for `a` must be valid here
        Ok(Some(AutoDerefRef(AutoDerefRef {
            autoderefs: 1,
            autoref: Some(ty::AutoUnsafe(mt_b.mutbl))
        })))
    }
}
