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

use check::{autoderef, impl_self_ty, FnCtxt, LvaluePreference, UnresolvedTypeAction};

use middle::infer::{self, Coercion};
use middle::subst::Substs;
use middle::traits::{self, ObligationCause};
use middle::traits::{predicate_for_trait_def, report_selection_error};
use middle::ty::{AutoDerefRef, AdjustDerefRef};
use middle::ty::{self, TypeAndMut, Ty, TypeError};
use middle::ty_relate::RelateResult;

use std::collections::VecDeque;
use syntax::ast;

struct Coerce<'a, 'tcx: 'a> {
    fcx: &'a FnCtxt<'a, 'tcx>,
    origin: infer::TypeOrigin
}

type CoerceResult<'tcx> = RelateResult<'tcx, Option<ty::AutoAdjustment<'tcx>>>;

/// The result of an attempt at coercing a Source type to a
/// Target type via unsizing (`Source: CoerceUnsized<Target>`).
/// If successful, all `CoerceUnsized` and `Unsized` obligations were selected.
/// Other obligations, such as `T: Trait` for `&T -> &Trait`, are provided
/// alongside the adjustment, to be enforced later.
type CoerceUnsizedResult<'tcx> = Result<(AutoDerefRef<'tcx>,
                                         Vec<traits::PredicateObligation<'tcx>>),
                                        CoerceUnsizedError>;

#[derive(PartialEq, Eq)]
enum CoerceUnsizedError {
    /// Source definitely does not implement `CoerceUnsized<Target>`.
    Unapplicable,

    /// Source might implement `CoerceUnsized<Target>`.
    Ambiguous,
}

impl<'f, 'tcx> Coerce<'f, 'tcx> {
    fn tcx(&self) -> &ty::ctxt<'tcx> {
        self.fcx.tcx()
    }

    fn subtype(&self, a: Ty<'tcx>, b: Ty<'tcx>) -> CoerceResult<'tcx> {
        try!(self.fcx.infcx().sub_types(false, self.origin.clone(), a, b));
        Ok(None) // No coercion required.
    }

    fn coerce(&self,
              expr_a: &ast::Expr,
              a: Ty<'tcx>,
              b: Ty<'tcx>)
              -> CoerceResult<'tcx> {
        let a = self.fcx.infcx().shallow_resolve(a);
        debug!("coerce({:?} => {:?})", a, b);

        // Consider coercing the subtype to a DST
        let unsize = self.fcx.infcx().commit_if_ok(|_| self.coerce_unsized(a, b));
        if let Ok((adjustment, leftover_predicates)) = unsize {
            for obligation in leftover_predicates {
                self.fcx.register_predicate(obligation);
            }
            return Ok(Some(AdjustDerefRef(adjustment)));
        }

        // Examine the supertype and consider auto-borrowing.
        //
        // Note: does not attempt to resolve type variables we encounter.
        // See above for details.
        match b.sty {
            ty::TyRawPtr(mt_b) => {
                return self.coerce_unsafe_ptr(a, b, mt_b.mutbl);
            }

            ty::TyRef(_, mt_b) => {
                return self.coerce_borrowed_pointer(expr_a, a, b, mt_b.mutbl);
            }

            _ => {}
        }

        let b = self.fcx.infcx().shallow_resolve(b);
        match a.sty {
            ty::TyBareFn(Some(_), a_f) => {
                // Function items are coercible to any closure
                // type; function pointers are not (that would
                // require double indirection).
                return self.coerce_from_fn_item(a, a_f, b);
            }
            ty::TyBareFn(None, a_f) => {
                // We permit coercion of fn pointers to drop the
                // unsafe qualifier.
                return self.coerce_from_fn_pointer(a, a_f, b);
            }
            _ => {}
        }

        // Attempt to generalize the expected type in hopes of an unsizing
        // coercion, where an intermediary stop-gap is usually necessary.
        // This is the case with trait method calls where the returned type
        // was not inferred, e.g. `Make::make(x: T): Box<Trait>`, if `Make`
        // has many implementations. Unsizing coercion will be ambiguous
        // and subtyping would result in a selection failure, if `Box<Trait>`
        // does not implement `Make`, but `Box<T>` does. The stop-gap fix
        // is `Make::make(x: T): Box<T>: Box<Trait>`.
        // In that same case, the following generalization will attempt to
        // apply `Box<_>` to the otherwise unconstrained `Make::make` return
        // type and trigger selection, hoping to get the unambiguous source
        // type `Box<T>` for the coercion to `Box<Trait>`.
        // Subtyping rules are used if the generalization and second attempt
        // at coercions through unsizing do not apply.

        // The first unsizing coercion must have failed due to ambiguity.
        if unsize.err() != Some(CoerceUnsizedError::Ambiguous) {
            return self.subtype(a, b);
        }

        // The target type needs to be a structure or Box<T>.
        // The only other types that implement CoerceUnsized are
        // references and pointers and those have multiple forms,
        // such as `*mut T -> *const Trait`.
        match b.sty {
            ty::TyBox(_) | ty::TyStruct(..) => {}
            _ => return self.subtype(a, b)
        }

        // Construct a `Target: CoerceUnsized<Target>` predicate.
        let trait_predicate = ty::Binder(ty::TraitRef {
            def_id: self.tcx().lang_items.coerce_unsized_trait().unwrap(),
            substs: self.tcx().mk_substs(Substs::new_trait(vec![b], vec![], b))
        }).to_poly_trait_predicate();

        // Select `Target: CoerceUnsized<Target>`.
        let mut selcx = traits::SelectionContext::new(self.fcx.infcx());
        let cause = ObligationCause::misc(self.origin.span(), self.fcx.body_id);
        let obligation = traits::Obligation::new(cause, trait_predicate);
        if let Ok(Some(traits::VtableImpl(i))) = selcx.select(&obligation) {
            // There is a single applicable impl. If `Target = P<Trait>`, then
            // the `Self` of this impl is some kind of supertype of `P<Trait>`,
            // most likely `P<T> forall T: Unsize<U>`.
            // This `Self` type, when all impl type parameters have been
            // substituted with fresh inference variables (e.g. `P<_>`),
            // will unify with the target type and all possible source types
            // for a coercion.
            // It can thus be used as a supertype of the source type,
            // the generalized form that can allow fulfilling pending
            // obligations and ultimately an unsizing coercion.
            let success = self.fcx.infcx().commit_if_ok(|_| {
                self.subtype(a, impl_self_ty(self.fcx,
                                             self.origin.span(),
                                              i.impl_def_id).ty)
            });

            if success.is_ok() {
                // Select pending obligations to constrain the
                // source type further, and resolve it again.
                self.fcx.select_obligations_where_possible();
                let a = self.fcx.infcx().shallow_resolve(a);

                // Finally, attempt a coercion by unsizing again,
                // now that the types are (hopefully) better known.
                let unsize = self.fcx.infcx().commit_if_ok(|_| self.coerce_unsized(a, b));
                if let Ok((adjustment, leftover_predicates)) = unsize {
                    for obligation in leftover_predicates {
                        self.fcx.register_predicate(obligation);
                    }
                    return Ok(Some(AdjustDerefRef(adjustment)));
                }
            }
        }

        self.subtype(a, b)
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
        debug!("coerce_borrowed_pointer(a={:?}, b={:?})",
               a,
               b);

        // If we have a parameter of type `&M T_a` and the value
        // provided is `expr`, we will be adding an implicit borrow,
        // meaning that we convert `f(expr)` to `f(&M *expr)`.  Therefore,
        // to type check, we will construct the type that `&M*expr` would
        // yield.

        match a.sty {
            ty::TyRef(_, mt_a) => {
                try!(coerce_mutbls(mt_a.mutbl, mutbl_b));
            }
            _ => return self.subtype(a, b)
        }

        let coercion = Coercion(self.origin.span());
        let r_borrow = self.fcx.infcx().next_region_var(coercion);
        let r_borrow = self.tcx().mk_region(r_borrow);
        let autoref = Some(ty::AutoPtr(r_borrow, mutbl_b));

        let lvalue_pref = LvaluePreference::from_mutbl(mutbl_b);
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
            let ty = self.tcx().mk_ref(r_borrow,
                                        TypeAndMut {ty: inner_ty, mutbl: mutbl_b});
            if let Err(err) = self.subtype(ty, b) {
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


    // &[T; n] or &mut [T; n] -> &[T]
    // or &mut [T; n] -> &mut [T]
    // or &Concrete -> &Trait, etc.
    fn coerce_unsized(&self,
                      source: Ty<'tcx>,
                      target: Ty<'tcx>)
                      -> CoerceUnsizedResult<'tcx> {
        debug!("coerce_unsized(source={:?}, target={:?})",
               source,
               target);

        let traits = (self.tcx().lang_items.unsize_trait(),
                      self.tcx().lang_items.coerce_unsized_trait());
        let (unsize_did, coerce_unsized_did) = if let (Some(u), Some(cu)) = traits {
            (u, cu)
        } else {
            debug!("Missing Unsize or CoerceUnsized traits");
            return Err(CoerceUnsizedError::Unapplicable);
        };

        // Note, we want to avoid unnecessary unsizing. We don't want to coerce to
        // a DST unless we have to. This currently comes out in the wash since
        // we can't unify [T] with U. But to properly support DST, we need to allow
        // that, at which point we will need extra checks on the target here.

        // Handle reborrows before selecting `Source: CoerceUnsized<Target>`.
        let (source, reborrow) = match (&source.sty, &target.sty) {
            (&ty::TyRef(_, mt_a), &ty::TyRef(_, mt_b)) => {
                if coerce_mutbls(mt_a.mutbl, mt_b.mutbl).is_err() {
                    return Err(CoerceUnsizedError::Unapplicable);
                }

                let coercion = Coercion(self.origin.span());
                let r_borrow = self.fcx.infcx().next_region_var(coercion);
                let region = self.tcx().mk_region(r_borrow);
                (mt_a.ty, Some(ty::AutoPtr(region, mt_b.mutbl)))
            }
            (&ty::TyRef(_, mt_a), &ty::TyRawPtr(mt_b)) => {
                if coerce_mutbls(mt_a.mutbl, mt_b.mutbl).is_err() {
                    return Err(CoerceUnsizedError::Unapplicable);
                }
                (mt_a.ty, Some(ty::AutoUnsafe(mt_b.mutbl)))
            }
            _ => (source, None)
        };
        let source = source.adjust_for_autoref(self.tcx(), reborrow);

        let mut selcx = traits::SelectionContext::new(self.fcx.infcx());

        // Use a FIFO queue for this custom fulfillment procedure.
        let mut queue = VecDeque::new();
        let mut leftover_predicates = vec![];

        // Create an obligation for `Source: CoerceUnsized<Target>`.
        let cause = ObligationCause::misc(self.origin.span(), self.fcx.body_id);
        queue.push_back(predicate_for_trait_def(self.tcx(),
                                                cause,
                                                coerce_unsized_did,
                                                0,
                                                source,
                                                vec![target]));

        // Keep resolving `CoerceUnsized` and `Unsize` predicates to avoid
        // emitting a coercion in cases like `Foo<$1>` -> `Foo<$2>`, where
        // inference might unify those two inner type variables later.
        let traits = [coerce_unsized_did, unsize_did];
        while let Some(obligation) = queue.pop_front() {
            debug!("coerce_unsized resolve step: {:?}", obligation);
            let trait_ref =  match obligation.predicate {
                ty::Predicate::Trait(ref tr) if traits.contains(&tr.def_id()) => {
                    tr.clone()
                }
                _ => {
                    leftover_predicates.push(obligation);
                    continue;
                }
            };
            match selcx.select(&obligation.with(trait_ref)) {
                // Uncertain or unimplemented.
                Ok(None)  => {
                    debug!("coerce_unsized: early return (Ambiguous)");
                    return Err(CoerceUnsizedError::Ambiguous);
                }
                Err(traits::Unimplemented) => {
                    debug!("coerce_unsized: early return (Unapplicable)");
                    return Err(CoerceUnsizedError::Unapplicable);
                }

                // Object safety violations or miscellaneous.
                Err(err) => {
                    report_selection_error(self.fcx.infcx(), &obligation, &err);
                    // Treat this like an obligation and follow through
                    // with the unsizing - the lack of a coercion should
                    // be silent, as it causes a type mismatch later.
                }

                Ok(Some(vtable)) => {
                    for obligation in vtable.nested_obligations() {
                        queue.push_back(obligation);
                    }
                }
            }
        }

        let adjustment = AutoDerefRef {
            autoderefs: if reborrow.is_some() { 1 } else { 0 },
            autoref: reborrow,
            unsize: Some(target)
        };
        debug!("Success, coerced with {:?}", adjustment);
        Ok((adjustment, leftover_predicates))
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

        debug!("coerce_from_fn_pointer(a={:?}, b={:?})",
                a, b);

        if let ty::TyBareFn(None, fn_ty_b) = b.sty {
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

        debug!("coerce_from_fn_item(a={:?}, b={:?})",
                a, b);

        match b.sty {
            ty::TyBareFn(None, _) => {
                let a_fn_pointer = self.tcx().mk_fn(None, fn_ty_a);
                try!(self.subtype(a_fn_pointer, b));
                Ok(Some(ty::AdjustReifyFnPointer))
            }
            _ => self.subtype(a, b)
        }
    }

    fn coerce_unsafe_ptr(&self,
                         a: Ty<'tcx>,
                         b: Ty<'tcx>,
                         mutbl_b: ast::Mutability)
                         -> CoerceResult<'tcx> {
        debug!("coerce_unsafe_ptr(a={:?}, b={:?})",
               a,
               b);

        let (is_ref, mt_a) = match a.sty {
            ty::TyRef(_, mt) => (true, mt),
            ty::TyRawPtr(mt) => (false, mt),
            _ => {
                return self.subtype(a, b);
            }
        };

        // Check that the types which they point at are compatible.
        let a_unsafe = self.tcx().mk_ptr(ty::TypeAndMut{ mutbl: mutbl_b, ty: mt_a.ty });
        try!(self.fcx.infcx().commit_if_ok(|_| {
            try!(self.subtype(a_unsafe, b));
            coerce_mutbls(mt_a.mutbl, mutbl_b)
        }));

        // Although references and unsafe ptrs have the same
        // representation, we still register an AutoDerefRef so that
        // regionck knows that the region for `a` must be valid here.
        if is_ref {
            Ok(Some(AdjustDerefRef(AutoDerefRef {
                autoderefs: 1,
                autoref: Some(ty::AutoUnsafe(mutbl_b)),
                unsize: None
            })))
        } else {
            Ok(None)
        }
    }
}

pub fn mk_assignty<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                             expr: &ast::Expr,
                             a: Ty<'tcx>,
                             b: Ty<'tcx>)
                             -> RelateResult<'tcx, ()> {
    debug!("mk_assignty({:?} -> {:?})", a, b);
    let coerce = Coerce {
        fcx: fcx,
        origin: infer::ExprAssignable(expr.span)
    };

    if let Some(adjustment) = try!(coerce.coerce(expr, a, b)) {
        debug!("Success, coerced with {:?}", adjustment);
        fcx.write_adjustment(expr.id, adjustment);
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
        (ast::MutImmutable, ast::MutMutable) => Err(TypeError::Mutability)
    }
}
