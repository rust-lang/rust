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

use check::FnCtxt;

use rustc::hir;
use rustc::infer::{Coercion, InferOk, TypeTrace};
use rustc::traits::{self, ObligationCause, ObligationCauseCode};
use rustc::ty::adjustment::{Adjustment, Adjust, AutoBorrow};
use rustc::ty::{self, LvaluePreference, TypeAndMut, Ty};
use rustc::ty::fold::TypeFoldable;
use rustc::ty::error::TypeError;
use rustc::ty::relate::RelateResult;
use util::common::indent;

use std::cell::RefCell;
use std::collections::VecDeque;
use std::ops::Deref;

struct Coerce<'a, 'gcx: 'a + 'tcx, 'tcx: 'a> {
    fcx: &'a FnCtxt<'a, 'gcx, 'tcx>,
    cause: ObligationCause<'tcx>,
    use_lub: bool,
    unsizing_obligations: RefCell<Vec<traits::PredicateObligation<'tcx>>>,
}

impl<'a, 'gcx, 'tcx> Deref for Coerce<'a, 'gcx, 'tcx> {
    type Target = FnCtxt<'a, 'gcx, 'tcx>;
    fn deref(&self) -> &Self::Target {
        &self.fcx
    }
}

type CoerceResult<'tcx> = RelateResult<'tcx, (Ty<'tcx>, Adjust<'tcx>)>;

fn coerce_mutbls<'tcx>(from_mutbl: hir::Mutability,
                       to_mutbl: hir::Mutability)
                       -> RelateResult<'tcx, ()> {
    match (from_mutbl, to_mutbl) {
        (hir::MutMutable, hir::MutMutable) |
        (hir::MutImmutable, hir::MutImmutable) |
        (hir::MutMutable, hir::MutImmutable) => Ok(()),
        (hir::MutImmutable, hir::MutMutable) => Err(TypeError::Mutability),
    }
}

impl<'f, 'gcx, 'tcx> Coerce<'f, 'gcx, 'tcx> {
    fn new(fcx: &'f FnCtxt<'f, 'gcx, 'tcx>, cause: ObligationCause<'tcx>) -> Self {
        Coerce {
            fcx: fcx,
            cause: cause,
            use_lub: false,
            unsizing_obligations: RefCell::new(vec![]),
        }
    }

    fn unify(&self, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        self.commit_if_ok(|_| {
            let trace = TypeTrace::types(&self.cause, false, a, b);
            if self.use_lub {
                self.lub(false, trace, &a, &b)
                    .map(|ok| self.register_infer_ok_obligations(ok))
            } else {
                self.sub(false, trace, &a, &b)
                    .map(|InferOk { value, obligations }| {
                        self.fcx.register_predicates(obligations);
                        value
                    })
            }
        })
    }

    /// Unify two types (using sub or lub) and produce a noop coercion.
    fn unify_and_identity(&self, a: Ty<'tcx>, b: Ty<'tcx>) -> CoerceResult<'tcx> {
        self.unify(&a, &b).and_then(|ty| self.identity(ty))
    }

    /// Synthesize an identity adjustment.
    fn identity(&self, ty: Ty<'tcx>) -> CoerceResult<'tcx> {
        Ok((ty, Adjust::DerefRef {
            autoderefs: 0,
            autoref: None,
            unsize: false,
        }))
    }

    fn coerce<'a, E, I>(&self, exprs: &E, a: Ty<'tcx>, b: Ty<'tcx>) -> CoerceResult<'tcx>
        where E: Fn() -> I,
              I: IntoIterator<Item = &'a hir::Expr>
    {

        let a = self.shallow_resolve(a);
        debug!("Coerce.tys({:?} => {:?})", a, b);

        // Just ignore error types.
        if a.references_error() || b.references_error() {
            return self.identity(b);
        }

        if a.is_never() {
            return Ok((b, Adjust::NeverToAny));
        }

        // Consider coercing the subtype to a DST
        let unsize = self.coerce_unsized(a, b);
        if unsize.is_ok() {
            return unsize;
        }

        // Examine the supertype and consider auto-borrowing.
        //
        // Note: does not attempt to resolve type variables we encounter.
        // See above for details.
        match b.sty {
            ty::TyRawPtr(mt_b) => {
                return self.coerce_unsafe_ptr(a, b, mt_b.mutbl);
            }

            ty::TyRef(r_b, mt_b) => {
                return self.coerce_borrowed_pointer(exprs, a, b, r_b, mt_b);
            }

            _ => {}
        }

        match a.sty {
            ty::TyFnDef(.., a_f) => {
                // Function items are coercible to any closure
                // type; function pointers are not (that would
                // require double indirection).
                // Additionally, we permit coercion of function
                // items to drop the unsafe qualifier.
                self.coerce_from_fn_item(a, a_f, b)
            }
            ty::TyFnPtr(a_f) => {
                // We permit coercion of fn pointers to drop the
                // unsafe qualifier.
                self.coerce_from_fn_pointer(a, a_f, b)
            }
            _ => {
                // Otherwise, just use unification rules.
                self.unify_and_identity(a, b)
            }
        }
    }

    /// Reborrows `&mut A` to `&mut B` and `&(mut) A` to `&B`.
    /// To match `A` with `B`, autoderef will be performed,
    /// calling `deref`/`deref_mut` where necessary.
    fn coerce_borrowed_pointer<'a, E, I>(&self,
                                         exprs: &E,
                                         a: Ty<'tcx>,
                                         b: Ty<'tcx>,
                                         r_b: &'tcx ty::Region,
                                         mt_b: TypeAndMut<'tcx>)
                                         -> CoerceResult<'tcx>
        where E: Fn() -> I,
              I: IntoIterator<Item = &'a hir::Expr>
    {

        debug!("coerce_borrowed_pointer(a={:?}, b={:?})", a, b);

        // If we have a parameter of type `&M T_a` and the value
        // provided is `expr`, we will be adding an implicit borrow,
        // meaning that we convert `f(expr)` to `f(&M *expr)`.  Therefore,
        // to type check, we will construct the type that `&M*expr` would
        // yield.

        let (r_a, mt_a) = match a.sty {
            ty::TyRef(r_a, mt_a) => {
                coerce_mutbls(mt_a.mutbl, mt_b.mutbl)?;
                (r_a, mt_a)
            }
            _ => return self.unify_and_identity(a, b),
        };

        let span = self.cause.span;

        let mut first_error = None;
        let mut r_borrow_var = None;
        let mut autoderef = self.autoderef(span, a);
        let mut success = None;

        for (referent_ty, autoderefs) in autoderef.by_ref() {
            if autoderefs == 0 {
                // Don't let this pass, otherwise it would cause
                // &T to autoref to &&T.
                continue;
            }

            // At this point, we have deref'd `a` to `referent_ty`.  So
            // imagine we are coercing from `&'a mut Vec<T>` to `&'b mut [T]`.
            // In the autoderef loop for `&'a mut Vec<T>`, we would get
            // three callbacks:
            //
            // - `&'a mut Vec<T>` -- 0 derefs, just ignore it
            // - `Vec<T>` -- 1 deref
            // - `[T]` -- 2 deref
            //
            // At each point after the first callback, we want to
            // check to see whether this would match out target type
            // (`&'b mut [T]`) if we autoref'd it. We can't just
            // compare the referent types, though, because we still
            // have to consider the mutability. E.g., in the case
            // we've been considering, we have an `&mut` reference, so
            // the `T` in `[T]` needs to be unified with equality.
            //
            // Therefore, we construct reference types reflecting what
            // the types will be after we do the final auto-ref and
            // compare those. Note that this means we use the target
            // mutability [1], since it may be that we are coercing
            // from `&mut T` to `&U`.
            //
            // One fine point concerns the region that we use. We
            // choose the region such that the region of the final
            // type that results from `unify` will be the region we
            // want for the autoref:
            //
            // - if in sub mode, that means we want to use `'b` (the
            //   region from the target reference) for both
            //   pointers [2]. This is because sub mode (somewhat
            //   arbitrarily) returns the subtype region.  In the case
            //   where we are coercing to a target type, we know we
            //   want to use that target type region (`'b`) because --
            //   for the program to type-check -- it must be the
            //   smaller of the two.
            //   - One fine point. It may be surprising that we can
            //     use `'b` without relating `'a` and `'b`. The reason
            //     that this is ok is that what we produce is
            //     effectively a `&'b *x` expression (if you could
            //     annotate the region of a borrow), and regionck has
            //     code that adds edges from the region of a borrow
            //     (`'b`, here) into the regions in the borrowed
            //     expression (`*x`, here).  (Search for "link".)
            // - if in lub mode, things can get fairly complicated. The
            //   easiest thing is just to make a fresh
            //   region variable [4], which effectively means we defer
            //   the decision to region inference (and regionck, which will add
            //   some more edges to this variable). However, this can wind up
            //   creating a crippling number of variables in some cases --
            //   e.g. #32278 -- so we optimize one particular case [3].
            //   Let me try to explain with some examples:
            //   - The "running example" above represents the simple case,
            //     where we have one `&` reference at the outer level and
            //     ownership all the rest of the way down. In this case,
            //     we want `LUB('a, 'b)` as the resulting region.
            //   - However, if there are nested borrows, that region is
            //     too strong. Consider a coercion from `&'a &'x Rc<T>` to
            //     `&'b T`. In this case, `'a` is actually irrelevant.
            //     The pointer we want is `LUB('x, 'b`). If we choose `LUB('a,'b)`
            //     we get spurious errors (`run-pass/regions-lub-ref-ref-rc.rs`).
            //     (The errors actually show up in borrowck, typically, because
            //     this extra edge causes the region `'a` to be inferred to something
            //     too big, which then results in borrowck errors.)
            //   - We could track the innermost shared reference, but there is already
            //     code in regionck that has the job of creating links between
            //     the region of a borrow and the regions in the thing being
            //     borrowed (here, `'a` and `'x`), and it knows how to handle
            //     all the various cases. So instead we just make a region variable
            //     and let regionck figure it out.
            let r = if !self.use_lub {
                r_b // [2] above
            } else if autoderefs == 1 {
                r_a // [3] above
            } else {
                if r_borrow_var.is_none() {
                    // create var lazilly, at most once
                    let coercion = Coercion(span);
                    let r = self.next_region_var(coercion);
                    r_borrow_var = Some(r); // [4] above
                }
                r_borrow_var.unwrap()
            };
            let derefd_ty_a = self.tcx.mk_ref(r,
                                              TypeAndMut {
                                                  ty: referent_ty,
                                                  mutbl: mt_b.mutbl, // [1] above
                                              });
            match self.unify(derefd_ty_a, b) {
                Ok(ty) => {
                    success = Some((ty, autoderefs));
                    break;
                }
                Err(err) => {
                    if first_error.is_none() {
                        first_error = Some(err);
                    }
                }
            }
        }

        // Extract type or return an error. We return the first error
        // we got, which should be from relating the "base" type
        // (e.g., in example above, the failure from relating `Vec<T>`
        // to the target type), since that should be the least
        // confusing.
        let (ty, autoderefs) = match success {
            Some(d) => d,
            None => {
                let err = first_error.expect("coerce_borrowed_pointer had no error");
                debug!("coerce_borrowed_pointer: failed with err = {:?}", err);
                return Err(err);
            }
        };

        // This commits the obligations to the fulfillcx. After this succeeds,
        // this snapshot can't be rolled back.
        autoderef.finalize(LvaluePreference::from_mutbl(mt_b.mutbl), exprs());

        // Now apply the autoref. We have to extract the region out of
        // the final ref type we got.
        if ty == a && mt_a.mutbl == hir::MutImmutable && autoderefs == 1 {
            // As a special case, if we would produce `&'a *x`, that's
            // a total no-op. We end up with the type `&'a T` just as
            // we started with.  In that case, just skip it
            // altogether. This is just an optimization.
            //
            // Note that for `&mut`, we DO want to reborrow --
            // otherwise, this would be a move, which might be an
            // error. For example `foo(self.x)` where `self` and
            // `self.x` both have `&mut `type would be a move of
            // `self.x`, but we auto-coerce it to `foo(&mut *self.x)`,
            // which is a borrow.
            assert_eq!(mt_b.mutbl, hir::MutImmutable); // can only coerce &T -> &U
            return self.identity(ty);
        }
        let r_borrow = match ty.sty {
            ty::TyRef(r_borrow, _) => r_borrow,
            _ => span_bug!(span, "expected a ref type, got {:?}", ty),
        };
        let autoref = Some(AutoBorrow::Ref(r_borrow, mt_b.mutbl));
        debug!("coerce_borrowed_pointer: succeeded ty={:?} autoderefs={:?} autoref={:?}",
               ty,
               autoderefs,
               autoref);
        Ok((ty, Adjust::DerefRef {
            autoderefs: autoderefs,
            autoref: autoref,
            unsize: false,
        }))
    }


    // &[T; n] or &mut [T; n] -> &[T]
    // or &mut [T; n] -> &mut [T]
    // or &Concrete -> &Trait, etc.
    fn coerce_unsized(&self, source: Ty<'tcx>, target: Ty<'tcx>) -> CoerceResult<'tcx> {
        debug!("coerce_unsized(source={:?}, target={:?})", source, target);

        let traits = (self.tcx.lang_items.unsize_trait(),
                      self.tcx.lang_items.coerce_unsized_trait());
        let (unsize_did, coerce_unsized_did) = if let (Some(u), Some(cu)) = traits {
            (u, cu)
        } else {
            debug!("Missing Unsize or CoerceUnsized traits");
            return Err(TypeError::Mismatch);
        };

        // Note, we want to avoid unnecessary unsizing. We don't want to coerce to
        // a DST unless we have to. This currently comes out in the wash since
        // we can't unify [T] with U. But to properly support DST, we need to allow
        // that, at which point we will need extra checks on the target here.

        // Handle reborrows before selecting `Source: CoerceUnsized<Target>`.
        let (source, reborrow) = match (&source.sty, &target.sty) {
            (&ty::TyRef(_, mt_a), &ty::TyRef(_, mt_b)) => {
                coerce_mutbls(mt_a.mutbl, mt_b.mutbl)?;

                let coercion = Coercion(self.cause.span);
                let r_borrow = self.next_region_var(coercion);
                (mt_a.ty, Some(AutoBorrow::Ref(r_borrow, mt_b.mutbl)))
            }
            (&ty::TyRef(_, mt_a), &ty::TyRawPtr(mt_b)) => {
                coerce_mutbls(mt_a.mutbl, mt_b.mutbl)?;
                (mt_a.ty, Some(AutoBorrow::RawPtr(mt_b.mutbl)))
            }
            _ => (source, None),
        };
        let source = source.adjust_for_autoref(self.tcx, reborrow);

        let mut selcx = traits::SelectionContext::new(self);

        // Use a FIFO queue for this custom fulfillment procedure.
        let mut queue = VecDeque::new();
        let mut leftover_predicates = vec![];

        // Create an obligation for `Source: CoerceUnsized<Target>`.
        let cause = ObligationCause::misc(self.cause.span, self.body_id);
        queue.push_back(self.tcx
            .predicate_for_trait_def(cause, coerce_unsized_did, 0, source, &[target]));

        // Keep resolving `CoerceUnsized` and `Unsize` predicates to avoid
        // emitting a coercion in cases like `Foo<$1>` -> `Foo<$2>`, where
        // inference might unify those two inner type variables later.
        let traits = [coerce_unsized_did, unsize_did];
        while let Some(obligation) = queue.pop_front() {
            debug!("coerce_unsized resolve step: {:?}", obligation);
            let trait_ref = match obligation.predicate {
                ty::Predicate::Trait(ref tr) if traits.contains(&tr.def_id()) => tr.clone(),
                _ => {
                    leftover_predicates.push(obligation);
                    continue;
                }
            };
            match selcx.select(&obligation.with(trait_ref)) {
                // Uncertain or unimplemented.
                Ok(None) |
                Err(traits::Unimplemented) => {
                    debug!("coerce_unsized: early return - can't prove obligation");
                    return Err(TypeError::Mismatch);
                }

                // Object safety violations or miscellaneous.
                Err(err) => {
                    self.report_selection_error(&obligation, &err);
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

        *self.unsizing_obligations.borrow_mut() = leftover_predicates;

        let adjustment = Adjust::DerefRef {
            autoderefs: if reborrow.is_some() { 1 } else { 0 },
            autoref: reborrow,
            unsize: true,
        };
        debug!("Success, coerced with {:?}", adjustment);
        Ok((target, adjustment))
    }

    fn coerce_from_safe_fn(&self,
                           a: Ty<'tcx>,
                           fn_ty_a: &'tcx ty::BareFnTy<'tcx>,
                           b: Ty<'tcx>)
                           -> CoerceResult<'tcx> {
        if let ty::TyFnPtr(fn_ty_b) = b.sty {
            match (fn_ty_a.unsafety, fn_ty_b.unsafety) {
                (hir::Unsafety::Normal, hir::Unsafety::Unsafe) => {
                    let unsafe_a = self.tcx.safe_to_unsafe_fn_ty(fn_ty_a);
                    return self.unify_and_identity(unsafe_a, b)
                        .map(|(ty, _)| (ty, Adjust::UnsafeFnPointer));
                }
                _ => {}
            }
        }
        self.unify_and_identity(a, b)
    }

    fn coerce_from_fn_pointer(&self,
                              a: Ty<'tcx>,
                              fn_ty_a: &'tcx ty::BareFnTy<'tcx>,
                              b: Ty<'tcx>)
                              -> CoerceResult<'tcx> {
        //! Attempts to coerce from the type of a Rust function item
        //! into a closure or a `proc`.
        //!

        let b = self.shallow_resolve(b);
        debug!("coerce_from_fn_pointer(a={:?}, b={:?})", a, b);

        self.coerce_from_safe_fn(a, fn_ty_a, b)
    }

    fn coerce_from_fn_item(&self,
                           a: Ty<'tcx>,
                           fn_ty_a: &'tcx ty::BareFnTy<'tcx>,
                           b: Ty<'tcx>)
                           -> CoerceResult<'tcx> {
        //! Attempts to coerce from the type of a Rust function item
        //! into a closure or a `proc`.
        //!

        let b = self.shallow_resolve(b);
        debug!("coerce_from_fn_item(a={:?}, b={:?})", a, b);

        match b.sty {
            ty::TyFnPtr(_) => {
                let a_fn_pointer = self.tcx.mk_fn_ptr(fn_ty_a);
                self.coerce_from_safe_fn(a_fn_pointer, fn_ty_a, b)
                    .map(|(ty, _)| (ty, Adjust::ReifyFnPointer))
            }
            _ => self.unify_and_identity(a, b),
        }
    }

    fn coerce_unsafe_ptr(&self,
                         a: Ty<'tcx>,
                         b: Ty<'tcx>,
                         mutbl_b: hir::Mutability)
                         -> CoerceResult<'tcx> {
        debug!("coerce_unsafe_ptr(a={:?}, b={:?})", a, b);

        let (is_ref, mt_a) = match a.sty {
            ty::TyRef(_, mt) => (true, mt),
            ty::TyRawPtr(mt) => (false, mt),
            _ => {
                return self.unify_and_identity(a, b);
            }
        };

        // Check that the types which they point at are compatible.
        let a_unsafe = self.tcx.mk_ptr(ty::TypeAndMut {
            mutbl: mutbl_b,
            ty: mt_a.ty,
        });
        let (ty, noop) = self.unify_and_identity(a_unsafe, b)?;
        coerce_mutbls(mt_a.mutbl, mutbl_b)?;

        // Although references and unsafe ptrs have the same
        // representation, we still register an Adjust::DerefRef so that
        // regionck knows that the region for `a` must be valid here.
        Ok((ty,
            if is_ref {
                Adjust::DerefRef {
                    autoderefs: 1,
                    autoref: Some(AutoBorrow::RawPtr(mutbl_b)),
                    unsize: false,
                }
            } else if mt_a.mutbl != mutbl_b {
                Adjust::MutToConstPointer
            } else {
                noop
            }))
    }
}

fn apply<'a, 'b, 'gcx, 'tcx, E, I>(coerce: &mut Coerce<'a, 'gcx, 'tcx>,
                                   exprs: &E,
                                   a: Ty<'tcx>,
                                   b: Ty<'tcx>)
                                   -> RelateResult<'tcx, Adjustment<'tcx>>
    where E: Fn() -> I,
          I: IntoIterator<Item = &'b hir::Expr>
{

    let (ty, adjust) = indent(|| coerce.coerce(exprs, a, b))?;

    let fcx = coerce.fcx;
    if let Adjust::DerefRef { unsize: true, .. } = adjust {
        let mut obligations = coerce.unsizing_obligations.borrow_mut();
        for obligation in obligations.drain(..) {
            fcx.register_predicate(obligation);
        }
    }

    Ok(Adjustment {
        kind: adjust,
        target: ty
    })
}

impl<'a, 'gcx, 'tcx> FnCtxt<'a, 'gcx, 'tcx> {
    /// Attempt to coerce an expression to a type, and return the
    /// adjusted type of the expression, if successful.
    /// Adjustments are only recorded if the coercion succeeded.
    /// The expressions *must not* have any pre-existing adjustments.
    pub fn try_coerce(&self,
                      expr: &hir::Expr,
                      expr_ty: Ty<'tcx>,
                      target: Ty<'tcx>)
                      -> RelateResult<'tcx, Ty<'tcx>> {
        let source = self.resolve_type_vars_with_obligations(expr_ty);
        debug!("coercion::try({:?}: {:?} -> {:?})", expr, source, target);

        let cause = self.cause(expr.span, ObligationCauseCode::ExprAssignable);
        let mut coerce = Coerce::new(self, cause);
        self.commit_if_ok(|_| {
            let adjustment = apply(&mut coerce, &|| Some(expr), source, target)?;
            if !adjustment.is_identity() {
                debug!("Success, coerced with {:?}", adjustment);
                match self.tables.borrow().adjustments.get(&expr.id) {
                    None |
                    Some(&Adjustment { kind: Adjust::NeverToAny, .. }) => (),
                    _ => bug!("expr already has an adjustment on it!"),
                };
                self.write_adjustment(expr.id, adjustment);
            }
            Ok(adjustment.target)
        })
    }

    /// Given some expressions, their known unified type and another expression,
    /// tries to unify the types, potentially inserting coercions on any of the
    /// provided expressions and returns their LUB (aka "common supertype").
    pub fn try_find_coercion_lub<'b, E, I>(&self,
                                           cause: &ObligationCause<'tcx>,
                                           exprs: E,
                                           prev_ty: Ty<'tcx>,
                                           new: &'b hir::Expr,
                                           new_ty: Ty<'tcx>)
                                           -> RelateResult<'tcx, Ty<'tcx>>
        where E: Fn() -> I,
              I: IntoIterator<Item = &'b hir::Expr>
    {

        let prev_ty = self.resolve_type_vars_with_obligations(prev_ty);
        let new_ty = self.resolve_type_vars_with_obligations(new_ty);
        debug!("coercion::try_find_lub({:?}, {:?})", prev_ty, new_ty);

        let trace = TypeTrace::types(cause, true, prev_ty, new_ty);

        // Special-case that coercion alone cannot handle:
        // Two function item types of differing IDs or Substs.
        match (&prev_ty.sty, &new_ty.sty) {
            (&ty::TyFnDef(a_def_id, a_substs, a_fty), &ty::TyFnDef(b_def_id, b_substs, b_fty)) => {
                // The signature must always match.
                let fty = self.lub(true, trace.clone(), &a_fty, &b_fty)
                              .map(|ok| self.register_infer_ok_obligations(ok))?;

                if a_def_id == b_def_id {
                    // Same function, maybe the parameters match.
                    let substs = self.commit_if_ok(|_| {
                        self.lub(true, trace.clone(), &a_substs, &b_substs)
                            .map(|ok| self.register_infer_ok_obligations(ok))
                    });

                    if let Ok(substs) = substs {
                        // We have a LUB of prev_ty and new_ty, just return it.
                        return Ok(self.tcx.mk_fn_def(a_def_id, substs, fty));
                    }
                }

                // Reify both sides and return the reified fn pointer type.
                let fn_ptr = self.tcx.mk_fn_ptr(fty);
                for expr in exprs().into_iter().chain(Some(new)) {
                    // No adjustments can produce a fn item, so this should never trip.
                    assert!(!self.tables.borrow().adjustments.contains_key(&expr.id));
                    self.write_adjustment(expr.id, Adjustment {
                        kind: Adjust::ReifyFnPointer,
                        target: fn_ptr
                    });
                }
                return Ok(fn_ptr);
            }
            _ => {}
        }

        let mut coerce = Coerce::new(self, cause.clone());
        coerce.use_lub = true;

        // First try to coerce the new expression to the type of the previous ones,
        // but only if the new expression has no coercion already applied to it.
        let mut first_error = None;
        if !self.tables.borrow().adjustments.contains_key(&new.id) {
            let result = self.commit_if_ok(|_| apply(&mut coerce, &|| Some(new), new_ty, prev_ty));
            match result {
                Ok(adjustment) => {
                    if !adjustment.is_identity() {
                        self.write_adjustment(new.id, adjustment);
                    }
                    return Ok(adjustment.target);
                }
                Err(e) => first_error = Some(e),
            }
        }

        // Then try to coerce the previous expressions to the type of the new one.
        // This requires ensuring there are no coercions applied to *any* of the
        // previous expressions, other than noop reborrows (ignoring lifetimes).
        for expr in exprs() {
            let noop = match self.tables.borrow().adjustments.get(&expr.id).map(|adj| adj.kind) {
                Some(Adjust::DerefRef {
                    autoderefs: 1,
                    autoref: Some(AutoBorrow::Ref(_, mutbl_adj)),
                    unsize: false
                }) => {
                    match self.node_ty(expr.id).sty {
                        ty::TyRef(_, mt_orig) => {
                            // Reborrow that we can safely ignore.
                            mutbl_adj == mt_orig.mutbl
                        }
                        _ => false,
                    }
                }
                Some(Adjust::NeverToAny) => true,
                Some(_) => false,
                None => true,
            };

            if !noop {
                return self.commit_if_ok(|_| {
                    self.lub(true, trace.clone(), &prev_ty, &new_ty)
                        .map(|ok| self.register_infer_ok_obligations(ok))
                });
            }
        }

        match self.commit_if_ok(|_| apply(&mut coerce, &exprs, prev_ty, new_ty)) {
            Err(_) => {
                // Avoid giving strange errors on failed attempts.
                if let Some(e) = first_error {
                    Err(e)
                } else {
                    self.commit_if_ok(|_| {
                        self.lub(true, trace, &prev_ty, &new_ty)
                            .map(|ok| self.register_infer_ok_obligations(ok))
                    })
                }
            }
            Ok(adjustment) => {
                if !adjustment.is_identity() {
                    let mut tables = self.tables.borrow_mut();
                    for expr in exprs() {
                        if let Some(&mut Adjustment {
                            kind: Adjust::NeverToAny,
                            ref mut target
                        }) = tables.adjustments.get_mut(&expr.id) {
                            *target = adjustment.target;
                            continue;
                        }
                        tables.adjustments.insert(expr.id, adjustment);
                    }
                }
                Ok(adjustment.target)
            }
        }
    }
}
