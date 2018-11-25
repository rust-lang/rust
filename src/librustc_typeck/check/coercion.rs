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

use check::{FnCtxt, Needs};
use errors::DiagnosticBuilder;
use rustc::hir;
use rustc::hir::def_id::DefId;
use rustc::infer::{Coercion, InferResult, InferOk};
use rustc::infer::type_variable::TypeVariableOrigin;
use rustc::traits::{self, ObligationCause, ObligationCauseCode};
use rustc::ty::adjustment::{Adjustment, Adjust, AllowTwoPhase, AutoBorrow, AutoBorrowMutability};
use rustc::ty::{self, TypeAndMut, Ty, ClosureSubsts};
use rustc::ty::fold::TypeFoldable;
use rustc::ty::error::TypeError;
use rustc::ty::relate::RelateResult;
use smallvec::{smallvec, SmallVec};
use std::ops::Deref;
use syntax::feature_gate;
use syntax::ptr::P;
use syntax_pos;

struct Coerce<'a, 'gcx: 'a + 'tcx, 'tcx: 'a> {
    fcx: &'a FnCtxt<'a, 'gcx, 'tcx>,
    cause: ObligationCause<'tcx>,
    use_lub: bool,
    /// Determines whether or not allow_two_phase_borrow is set on any
    /// autoref adjustments we create while coercing. We don't want to
    /// allow deref coercions to create two-phase borrows, at least initially,
    /// but we do need two-phase borrows for function argument reborrows.
    /// See #47489 and #48598
    /// See docs on the "AllowTwoPhase" type for a more detailed discussion
    allow_two_phase: AllowTwoPhase,
}

impl<'a, 'gcx, 'tcx> Deref for Coerce<'a, 'gcx, 'tcx> {
    type Target = FnCtxt<'a, 'gcx, 'tcx>;
    fn deref(&self) -> &Self::Target {
        &self.fcx
    }
}

type CoerceResult<'tcx> = InferResult<'tcx, (Vec<Adjustment<'tcx>>, Ty<'tcx>)>;

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

fn identity(_: Ty) -> Vec<Adjustment> { vec![] }

fn simple<'tcx>(kind: Adjust<'tcx>) -> impl FnOnce(Ty<'tcx>) -> Vec<Adjustment<'tcx>> {
    move |target| vec![Adjustment { kind, target }]
}

fn success<'tcx>(adj: Vec<Adjustment<'tcx>>,
                 target: Ty<'tcx>,
                 obligations: traits::PredicateObligations<'tcx>)
                 -> CoerceResult<'tcx> {
    Ok(InferOk {
        value: (adj, target),
        obligations
    })
}

impl<'f, 'gcx, 'tcx> Coerce<'f, 'gcx, 'tcx> {
    fn new(fcx: &'f FnCtxt<'f, 'gcx, 'tcx>,
           cause: ObligationCause<'tcx>,
           allow_two_phase: AllowTwoPhase) -> Self {
        Coerce {
            fcx,
            cause,
            allow_two_phase,
            use_lub: false,
        }
    }

    fn unify(&self, a: Ty<'tcx>, b: Ty<'tcx>) -> InferResult<'tcx, Ty<'tcx>> {
        self.commit_if_ok(|_| {
            if self.use_lub {
                self.at(&self.cause, self.fcx.param_env).lub(b, a)
            } else {
                self.at(&self.cause, self.fcx.param_env)
                    .sup(b, a)
                    .map(|InferOk { value: (), obligations }| InferOk { value: a, obligations })
            }
        })
    }

    /// Unify two types (using sub or lub) and produce a specific coercion.
    fn unify_and<F>(&self, a: Ty<'tcx>, b: Ty<'tcx>, f: F)
                    -> CoerceResult<'tcx>
        where F: FnOnce(Ty<'tcx>) -> Vec<Adjustment<'tcx>>
    {
        self.unify(&a, &b).and_then(|InferOk { value: ty, obligations }| {
            success(f(ty), ty, obligations)
        })
    }

    fn coerce(&self, a: Ty<'tcx>, b: Ty<'tcx>) -> CoerceResult<'tcx> {
        let a = self.shallow_resolve(a);
        debug!("Coerce.tys({:?} => {:?})", a, b);

        // Just ignore error types.
        if a.references_error() || b.references_error() {
            return success(vec![], b, vec![]);
        }

        if a.is_never() {
            // Subtle: If we are coercing from `!` to `?T`, where `?T` is an unbound
            // type variable, we want `?T` to fallback to `!` if not
            // otherwise constrained. An example where this arises:
            //
            //     let _: Option<?T> = Some({ return; });
            //
            // here, we would coerce from `!` to `?T`.
            let b = self.shallow_resolve(b);
            return if self.shallow_resolve(b).is_ty_var() {
                // micro-optimization: no need for this if `b` is
                // already resolved in some way.
                let diverging_ty = self.next_diverging_ty_var(
                    TypeVariableOrigin::AdjustmentType(self.cause.span));
                self.unify_and(&b, &diverging_ty, simple(Adjust::NeverToAny))
            } else {
                success(simple(Adjust::NeverToAny)(b), b, vec![])
            };
        }

        // Consider coercing the subtype to a DST
        //
        // NOTE: this is wrapped in a `commit_if_ok` because it creates
        // a "spurious" type variable, and we don't want to have that
        // type variable in memory if the coercion fails.
        let unsize = self.commit_if_ok(|_| self.coerce_unsized(a, b));
        if unsize.is_ok() {
            debug!("coerce: unsize successful");
            return unsize;
        }
        debug!("coerce: unsize failed");

        // Examine the supertype and consider auto-borrowing.
        //
        // Note: does not attempt to resolve type variables we encounter.
        // See above for details.
        match b.sty {
            ty::RawPtr(mt_b) => {
                return self.coerce_unsafe_ptr(a, b, mt_b.mutbl);
            }

            ty::Ref(r_b, ty, mutbl) => {
                let mt_b = ty::TypeAndMut { ty, mutbl };
                return self.coerce_borrowed_pointer(a, b, r_b, mt_b);
            }

            _ => {}
        }

        match a.sty {
            ty::FnDef(..) => {
                // Function items are coercible to any closure
                // type; function pointers are not (that would
                // require double indirection).
                // Additionally, we permit coercion of function
                // items to drop the unsafe qualifier.
                self.coerce_from_fn_item(a, b)
            }
            ty::FnPtr(a_f) => {
                // We permit coercion of fn pointers to drop the
                // unsafe qualifier.
                self.coerce_from_fn_pointer(a, a_f, b)
            }
            ty::Closure(def_id_a, substs_a) => {
                // Non-capturing closures are coercible to
                // function pointers
                self.coerce_closure_to_fn(a, def_id_a, substs_a, b)
            }
            _ => {
                // Otherwise, just use unification rules.
                self.unify_and(a, b, identity)
            }
        }
    }

    /// Reborrows `&mut A` to `&mut B` and `&(mut) A` to `&B`.
    /// To match `A` with `B`, autoderef will be performed,
    /// calling `deref`/`deref_mut` where necessary.
    fn coerce_borrowed_pointer(&self,
                               a: Ty<'tcx>,
                               b: Ty<'tcx>,
                               r_b: ty::Region<'tcx>,
                               mt_b: TypeAndMut<'tcx>)
                               -> CoerceResult<'tcx>
    {
        debug!("coerce_borrowed_pointer(a={:?}, b={:?})", a, b);

        // If we have a parameter of type `&M T_a` and the value
        // provided is `expr`, we will be adding an implicit borrow,
        // meaning that we convert `f(expr)` to `f(&M *expr)`.  Therefore,
        // to type check, we will construct the type that `&M*expr` would
        // yield.

        let (r_a, mt_a) = match a.sty {
            ty::Ref(r_a, ty, mutbl) => {
                let mt_a = ty::TypeAndMut { ty, mutbl };
                coerce_mutbls(mt_a.mutbl, mt_b.mutbl)?;
                (r_a, mt_a)
            }
            _ => return self.unify_and(a, b, identity),
        };

        let span = self.cause.span;

        let mut first_error = None;
        let mut r_borrow_var = None;
        let mut autoderef = self.autoderef(span, a);
        let mut found = None;

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
            //   e.g., #32278 -- so we optimize one particular case [3].
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
                Ok(ok) => {
                    found = Some(ok);
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
        let InferOk { value: ty, mut obligations } = match found {
            Some(d) => d,
            None => {
                let err = first_error.expect("coerce_borrowed_pointer had no error");
                debug!("coerce_borrowed_pointer: failed with err = {:?}", err);
                return Err(err);
            }
        };

        if ty == a && mt_a.mutbl == hir::MutImmutable && autoderef.step_count() == 1 {
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
            return success(vec![], ty, obligations);
        }

        let needs = Needs::maybe_mut_place(mt_b.mutbl);
        let InferOk { value: mut adjustments, obligations: o }
            = autoderef.adjust_steps_as_infer_ok(needs);
        obligations.extend(o);
        obligations.extend(autoderef.into_obligations());

        // Now apply the autoref. We have to extract the region out of
        // the final ref type we got.
        let r_borrow = match ty.sty {
            ty::Ref(r_borrow, _, _) => r_borrow,
            _ => span_bug!(span, "expected a ref type, got {:?}", ty),
        };
        let mutbl = match mt_b.mutbl {
            hir::MutImmutable => AutoBorrowMutability::Immutable,
            hir::MutMutable => AutoBorrowMutability::Mutable {
                allow_two_phase_borrow: self.allow_two_phase,
            }
        };
        adjustments.push(Adjustment {
            kind: Adjust::Borrow(AutoBorrow::Ref(r_borrow, mutbl)),
            target: ty
        });

        debug!("coerce_borrowed_pointer: succeeded ty={:?} adjustments={:?}",
               ty,
               adjustments);

        success(adjustments, ty, obligations)
    }


    // &[T; n] or &mut [T; n] -> &[T]
    // or &mut [T; n] -> &mut [T]
    // or &Concrete -> &Trait, etc.
    fn coerce_unsized(&self, source: Ty<'tcx>, target: Ty<'tcx>) -> CoerceResult<'tcx> {
        debug!("coerce_unsized(source={:?}, target={:?})", source, target);

        let traits = (self.tcx.lang_items().unsize_trait(),
                      self.tcx.lang_items().coerce_unsized_trait());
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
        let reborrow = match (&source.sty, &target.sty) {
            (&ty::Ref(_, ty_a, mutbl_a), &ty::Ref(_, _, mutbl_b)) => {
                coerce_mutbls(mutbl_a, mutbl_b)?;

                let coercion = Coercion(self.cause.span);
                let r_borrow = self.next_region_var(coercion);
                let mutbl = match mutbl_b {
                    hir::MutImmutable => AutoBorrowMutability::Immutable,
                    hir::MutMutable => AutoBorrowMutability::Mutable {
                        // We don't allow two-phase borrows here, at least for initial
                        // implementation. If it happens that this coercion is a function argument,
                        // the reborrow in coerce_borrowed_ptr will pick it up.
                        allow_two_phase_borrow: AllowTwoPhase::No,
                    }
                };
                Some((Adjustment {
                    kind: Adjust::Deref(None),
                    target: ty_a
                }, Adjustment {
                    kind: Adjust::Borrow(AutoBorrow::Ref(r_borrow, mutbl)),
                    target:  self.tcx.mk_ref(r_borrow, ty::TypeAndMut {
                        mutbl: mutbl_b,
                        ty: ty_a
                    })
                }))
            }
            (&ty::Ref(_, ty_a, mt_a), &ty::RawPtr(ty::TypeAndMut { mutbl: mt_b, .. })) => {
                coerce_mutbls(mt_a, mt_b)?;

                Some((Adjustment {
                    kind: Adjust::Deref(None),
                    target: ty_a
                }, Adjustment {
                    kind: Adjust::Borrow(AutoBorrow::RawPtr(mt_b)),
                    target:  self.tcx.mk_ptr(ty::TypeAndMut {
                        mutbl: mt_b,
                        ty: ty_a
                    })
                }))
            }
            _ => None,
        };
        let coerce_source = reborrow.as_ref().map_or(source, |&(_, ref r)| r.target);

        // Setup either a subtyping or a LUB relationship between
        // the `CoerceUnsized` target type and the expected type.
        // We only have the latter, so we use an inference variable
        // for the former and let type inference do the rest.
        let origin = TypeVariableOrigin::MiscVariable(self.cause.span);
        let coerce_target = self.next_ty_var(origin);
        let mut coercion = self.unify_and(coerce_target, target, |target| {
            let unsize = Adjustment {
                kind: Adjust::Unsize,
                target
            };
            match reborrow {
                None => vec![unsize],
                Some((ref deref, ref autoref)) => {
                    vec![deref.clone(), autoref.clone(), unsize]
                }
            }
        })?;

        let mut selcx = traits::SelectionContext::new(self);

        // Create an obligation for `Source: CoerceUnsized<Target>`.
        let cause = ObligationCause::misc(self.cause.span, self.body_id);

        // Use a FIFO queue for this custom fulfillment procedure.
        //
        // A Vec (or SmallVec) is not a natural choice for a queue. However,
        // this code path is hot, and this queue usually has a max length of 1
        // and almost never more than 3. By using a SmallVec we avoid an
        // allocation, at the (very small) cost of (occasionally) having to
        // shift subsequent elements down when removing the front element.
        let mut queue: SmallVec<[_; 4]> =
            smallvec![self.tcx.predicate_for_trait_def(self.fcx.param_env,
                                                       cause,
                                                       coerce_unsized_did,
                                                       0,
                                                       coerce_source,
                                                       &[coerce_target.into()])];

        let mut has_unsized_tuple_coercion = false;

        // Keep resolving `CoerceUnsized` and `Unsize` predicates to avoid
        // emitting a coercion in cases like `Foo<$1>` -> `Foo<$2>`, where
        // inference might unify those two inner type variables later.
        let traits = [coerce_unsized_did, unsize_did];
        while !queue.is_empty() {
            let obligation = queue.remove(0);
            debug!("coerce_unsized resolve step: {:?}", obligation);
            let trait_ref = match obligation.predicate {
                ty::Predicate::Trait(ref tr) if traits.contains(&tr.def_id()) => {
                    if unsize_did == tr.def_id() {
                        let sty = &tr.skip_binder().input_types().nth(1).unwrap().sty;
                        if let ty::Tuple(..) = sty {
                            debug!("coerce_unsized: found unsized tuple coercion");
                            has_unsized_tuple_coercion = true;
                        }
                    }
                    tr.clone()
                }
                _ => {
                    coercion.obligations.push(obligation);
                    continue;
                }
            };
            match selcx.select(&obligation.with(trait_ref)) {
                // Uncertain or unimplemented.
                Ok(None) => {
                    if trait_ref.def_id() == unsize_did {
                        let trait_ref = self.resolve_type_vars_if_possible(&trait_ref);
                        let self_ty = trait_ref.skip_binder().self_ty();
                        let unsize_ty = trait_ref.skip_binder().input_types().nth(1).unwrap();
                        debug!("coerce_unsized: ambiguous unsize case for {:?}", trait_ref);
                        match (&self_ty.sty, &unsize_ty.sty) {
                            (ty::Infer(ty::TyVar(v)),
                             ty::Dynamic(..)) if self.type_var_is_sized(*v) => {
                                debug!("coerce_unsized: have sized infer {:?}", v);
                                coercion.obligations.push(obligation);
                                // `$0: Unsize<dyn Trait>` where we know that `$0: Sized`, try going
                                // for unsizing.
                            }
                            _ => {
                                // Some other case for `$0: Unsize<Something>`. Note that we
                                // hit this case even if `Something` is a sized type, so just
                                // don't do the coercion.
                                debug!("coerce_unsized: ambiguous unsize");
                                return Err(TypeError::Mismatch);
                            }
                        }
                    } else {
                        debug!("coerce_unsized: early return - ambiguous");
                        return Err(TypeError::Mismatch);
                    }
                }
                Err(traits::Unimplemented) => {
                    debug!("coerce_unsized: early return - can't prove obligation");
                    return Err(TypeError::Mismatch);
                }

                // Object safety violations or miscellaneous.
                Err(err) => {
                    self.report_selection_error(&obligation, &err, false);
                    // Treat this like an obligation and follow through
                    // with the unsizing - the lack of a coercion should
                    // be silent, as it causes a type mismatch later.
                }

                Ok(Some(vtable)) => {
                    queue.extend(vtable.nested_obligations())
                }
            }
        }

        if has_unsized_tuple_coercion && !self.tcx.features().unsized_tuple_coercion {
            feature_gate::emit_feature_err(&self.tcx.sess.parse_sess,
                                           "unsized_tuple_coercion",
                                           self.cause.span,
                                           feature_gate::GateIssue::Language,
                                           feature_gate::EXPLAIN_UNSIZED_TUPLE_COERCION);
        }

        Ok(coercion)
    }

    fn coerce_from_safe_fn<F, G>(&self,
                                 a: Ty<'tcx>,
                                 fn_ty_a: ty::PolyFnSig<'tcx>,
                                 b: Ty<'tcx>,
                                 to_unsafe: F,
                                 normal: G)
                                 -> CoerceResult<'tcx>
        where F: FnOnce(Ty<'tcx>) -> Vec<Adjustment<'tcx>>,
              G: FnOnce(Ty<'tcx>) -> Vec<Adjustment<'tcx>>
    {
        if let ty::FnPtr(fn_ty_b) = b.sty {
            if let (hir::Unsafety::Normal, hir::Unsafety::Unsafe)
                = (fn_ty_a.unsafety(), fn_ty_b.unsafety())
            {
                let unsafe_a = self.tcx.safe_to_unsafe_fn_ty(fn_ty_a);
                return self.unify_and(unsafe_a, b, to_unsafe);
            }
        }
        self.unify_and(a, b, normal)
    }

    fn coerce_from_fn_pointer(&self,
                              a: Ty<'tcx>,
                              fn_ty_a: ty::PolyFnSig<'tcx>,
                              b: Ty<'tcx>)
                              -> CoerceResult<'tcx> {
        //! Attempts to coerce from the type of a Rust function item
        //! into a closure or a `proc`.
        //!

        let b = self.shallow_resolve(b);
        debug!("coerce_from_fn_pointer(a={:?}, b={:?})", a, b);

        self.coerce_from_safe_fn(a, fn_ty_a, b,
            simple(Adjust::UnsafeFnPointer), identity)
    }

    fn coerce_from_fn_item(&self,
                           a: Ty<'tcx>,
                           b: Ty<'tcx>)
                           -> CoerceResult<'tcx> {
        //! Attempts to coerce from the type of a Rust function item
        //! into a closure or a `proc`.

        let b = self.shallow_resolve(b);
        debug!("coerce_from_fn_item(a={:?}, b={:?})", a, b);

        match b.sty {
            ty::FnPtr(_) => {
                let a_sig = a.fn_sig(self.tcx);
                let InferOk { value: a_sig, mut obligations } =
                    self.normalize_associated_types_in_as_infer_ok(self.cause.span, &a_sig);

                let a_fn_pointer = self.tcx.mk_fn_ptr(a_sig);
                let InferOk { value, obligations: o2 } = self.coerce_from_safe_fn(
                    a_fn_pointer,
                    a_sig,
                    b,
                    |unsafe_ty| {
                        vec![
                            Adjustment { kind: Adjust::ReifyFnPointer, target: a_fn_pointer },
                            Adjustment { kind: Adjust::UnsafeFnPointer, target: unsafe_ty },
                        ]
                    },
                    simple(Adjust::ReifyFnPointer)
                )?;

                obligations.extend(o2);
                Ok(InferOk { value, obligations })
            }
            _ => self.unify_and(a, b, identity),
        }
    }

    fn coerce_closure_to_fn(&self,
                           a: Ty<'tcx>,
                           def_id_a: DefId,
                           substs_a: ClosureSubsts<'tcx>,
                           b: Ty<'tcx>)
                           -> CoerceResult<'tcx> {
        //! Attempts to coerce from the type of a non-capturing closure
        //! into a function pointer.
        //!

        let b = self.shallow_resolve(b);

        let node_id_a = self.tcx.hir().as_local_node_id(def_id_a).unwrap();
        match b.sty {
            ty::FnPtr(_) if self.tcx.with_freevars(node_id_a, |v| v.is_empty()) => {
                // We coerce the closure, which has fn type
                //     `extern "rust-call" fn((arg0,arg1,...)) -> _`
                // to
                //     `fn(arg0,arg1,...) -> _`
                let sig = self.closure_sig(def_id_a, substs_a);
                let pointer_ty = self.tcx.coerce_closure_fn_ty(sig);
                debug!("coerce_closure_to_fn(a={:?}, b={:?}, pty={:?})",
                       a, b, pointer_ty);
                self.unify_and(pointer_ty, b, simple(Adjust::ClosureFnPointer))
            }
            _ => self.unify_and(a, b, identity),
        }
    }

    fn coerce_unsafe_ptr(&self,
                         a: Ty<'tcx>,
                         b: Ty<'tcx>,
                         mutbl_b: hir::Mutability)
                         -> CoerceResult<'tcx> {
        debug!("coerce_unsafe_ptr(a={:?}, b={:?})", a, b);

        let (is_ref, mt_a) = match a.sty {
            ty::Ref(_, ty, mutbl) => (true, ty::TypeAndMut { ty, mutbl }),
            ty::RawPtr(mt) => (false, mt),
            _ => return self.unify_and(a, b, identity)
        };

        // Check that the types which they point at are compatible.
        let a_unsafe = self.tcx.mk_ptr(ty::TypeAndMut {
            mutbl: mutbl_b,
            ty: mt_a.ty,
        });
        coerce_mutbls(mt_a.mutbl, mutbl_b)?;
        // Although references and unsafe ptrs have the same
        // representation, we still register an Adjust::DerefRef so that
        // regionck knows that the region for `a` must be valid here.
        if is_ref {
            self.unify_and(a_unsafe, b, |target| {
                vec![Adjustment {
                    kind: Adjust::Deref(None),
                    target: mt_a.ty
                }, Adjustment {
                    kind: Adjust::Borrow(AutoBorrow::RawPtr(mutbl_b)),
                    target
                }]
            })
        } else if mt_a.mutbl != mutbl_b {
            self.unify_and(a_unsafe, b, simple(Adjust::MutToConstPointer))
        } else {
            self.unify_and(a_unsafe, b, identity)
        }
    }
}

impl<'a, 'gcx, 'tcx> FnCtxt<'a, 'gcx, 'tcx> {
    /// Attempt to coerce an expression to a type, and return the
    /// adjusted type of the expression, if successful.
    /// Adjustments are only recorded if the coercion succeeded.
    /// The expressions *must not* have any pre-existing adjustments.
    pub fn try_coerce(&self,
                      expr: &hir::Expr,
                      expr_ty: Ty<'tcx>,
                      target: Ty<'tcx>,
                      allow_two_phase: AllowTwoPhase)
                      -> RelateResult<'tcx, Ty<'tcx>> {
        let source = self.resolve_type_vars_with_obligations(expr_ty);
        debug!("coercion::try({:?}: {:?} -> {:?})", expr, source, target);

        let cause = self.cause(expr.span, ObligationCauseCode::ExprAssignable);
        let coerce = Coerce::new(self, cause, allow_two_phase);
        let ok = self.commit_if_ok(|_| coerce.coerce(source, target))?;

        let (adjustments, _) = self.register_infer_ok_obligations(ok);
        self.apply_adjustments(expr, adjustments);
        Ok(target)
    }

    /// Same as `try_coerce()`, but without side-effects.
    pub fn can_coerce(&self, expr_ty: Ty<'tcx>, target: Ty<'tcx>) -> bool {
        let source = self.resolve_type_vars_with_obligations(expr_ty);
        debug!("coercion::can({:?} -> {:?})", source, target);

        let cause = self.cause(syntax_pos::DUMMY_SP, ObligationCauseCode::ExprAssignable);
        // We don't ever need two-phase here since we throw out the result of the coercion
        let coerce = Coerce::new(self, cause, AllowTwoPhase::No);
        self.probe(|_| coerce.coerce(source, target)).is_ok()
    }

    /// Given some expressions, their known unified type and another expression,
    /// tries to unify the types, potentially inserting coercions on any of the
    /// provided expressions and returns their LUB (aka "common supertype").
    ///
    /// This is really an internal helper. From outside the coercion
    /// module, you should instantiate a `CoerceMany` instance.
    fn try_find_coercion_lub<E>(&self,
                                cause: &ObligationCause<'tcx>,
                                exprs: &[E],
                                prev_ty: Ty<'tcx>,
                                new: &hir::Expr,
                                new_ty: Ty<'tcx>)
                                -> RelateResult<'tcx, Ty<'tcx>>
        where E: AsCoercionSite
    {
        let prev_ty = self.resolve_type_vars_with_obligations(prev_ty);
        let new_ty = self.resolve_type_vars_with_obligations(new_ty);
        debug!("coercion::try_find_coercion_lub({:?}, {:?})", prev_ty, new_ty);

        // Special-case that coercion alone cannot handle:
        // Two function item types of differing IDs or Substs.
        if let (&ty::FnDef(..), &ty::FnDef(..)) = (&prev_ty.sty, &new_ty.sty) {
            // Don't reify if the function types have a LUB, i.e., they
            // are the same function and their parameters have a LUB.
            let lub_ty = self.commit_if_ok(|_| {
                self.at(cause, self.param_env)
                    .lub(prev_ty, new_ty)
            }).map(|ok| self.register_infer_ok_obligations(ok));

            if lub_ty.is_ok() {
                // We have a LUB of prev_ty and new_ty, just return it.
                return lub_ty;
            }

            // The signature must match.
            let a_sig = prev_ty.fn_sig(self.tcx);
            let a_sig = self.normalize_associated_types_in(new.span, &a_sig);
            let b_sig = new_ty.fn_sig(self.tcx);
            let b_sig = self.normalize_associated_types_in(new.span, &b_sig);
            let sig = self.at(cause, self.param_env)
                          .trace(prev_ty, new_ty)
                          .lub(&a_sig, &b_sig)
                          .map(|ok| self.register_infer_ok_obligations(ok))?;

            // Reify both sides and return the reified fn pointer type.
            let fn_ptr = self.tcx.mk_fn_ptr(sig);
            for expr in exprs.iter().map(|e| e.as_coercion_site()).chain(Some(new)) {
                // The only adjustment that can produce an fn item is
                // `NeverToAny`, so this should always be valid.
                self.apply_adjustments(expr, vec![Adjustment {
                    kind: Adjust::ReifyFnPointer,
                    target: fn_ptr
                }]);
            }
            return Ok(fn_ptr);
        }

        // Configure a Coerce instance to compute the LUB.
        // We don't allow two-phase borrows on any autorefs this creates since we
        // probably aren't processing function arguments here and even if we were,
        // they're going to get autorefed again anyway and we can apply 2-phase borrows
        // at that time.
        let mut coerce = Coerce::new(self, cause.clone(), AllowTwoPhase::No);
        coerce.use_lub = true;

        // First try to coerce the new expression to the type of the previous ones,
        // but only if the new expression has no coercion already applied to it.
        let mut first_error = None;
        if !self.tables.borrow().adjustments().contains_key(new.hir_id) {
            let result = self.commit_if_ok(|_| coerce.coerce(new_ty, prev_ty));
            match result {
                Ok(ok) => {
                    let (adjustments, target) = self.register_infer_ok_obligations(ok);
                    self.apply_adjustments(new, adjustments);
                    return Ok(target);
                }
                Err(e) => first_error = Some(e),
            }
        }

        // Then try to coerce the previous expressions to the type of the new one.
        // This requires ensuring there are no coercions applied to *any* of the
        // previous expressions, other than noop reborrows (ignoring lifetimes).
        for expr in exprs {
            let expr = expr.as_coercion_site();
            let noop = match self.tables.borrow().expr_adjustments(expr) {
                &[
                    Adjustment { kind: Adjust::Deref(_), .. },
                    Adjustment { kind: Adjust::Borrow(AutoBorrow::Ref(_, mutbl_adj)), .. }
                ] => {
                    match self.node_ty(expr.hir_id).sty {
                        ty::Ref(_, _, mt_orig) => {
                            let mutbl_adj: hir::Mutability = mutbl_adj.into();
                            // Reborrow that we can safely ignore, because
                            // the next adjustment can only be a Deref
                            // which will be merged into it.
                            mutbl_adj == mt_orig
                        }
                        _ => false,
                    }
                }
                &[Adjustment { kind: Adjust::NeverToAny, .. }] | &[] => true,
                _ => false,
            };

            if !noop {
                return self.commit_if_ok(|_|
                    self.at(cause, self.param_env)
                        .lub(prev_ty, new_ty)
                ).map(|ok| self.register_infer_ok_obligations(ok));
            }
        }

        match self.commit_if_ok(|_| coerce.coerce(prev_ty, new_ty)) {
            Err(_) => {
                // Avoid giving strange errors on failed attempts.
                if let Some(e) = first_error {
                    Err(e)
                } else {
                    self.commit_if_ok(|_|
                        self.at(cause, self.param_env)
                            .lub(prev_ty, new_ty)
                    ).map(|ok| self.register_infer_ok_obligations(ok))
                }
            }
            Ok(ok) => {
                let (adjustments, target) = self.register_infer_ok_obligations(ok);
                for expr in exprs {
                    let expr = expr.as_coercion_site();
                    self.apply_adjustments(expr, adjustments.clone());
                }
                Ok(target)
            }
        }
    }
}

/// CoerceMany encapsulates the pattern you should use when you have
/// many expressions that are all getting coerced to a common
/// type. This arises, for example, when you have a match (the result
/// of each arm is coerced to a common type). It also arises in less
/// obvious places, such as when you have many `break foo` expressions
/// that target the same loop, or the various `return` expressions in
/// a function.
///
/// The basic protocol is as follows:
///
/// - Instantiate the `CoerceMany` with an initial `expected_ty`.
///   This will also serve as the "starting LUB". The expectation is
///   that this type is something which all of the expressions *must*
///   be coercible to. Use a fresh type variable if needed.
/// - For each expression whose result is to be coerced, invoke `coerce()` with.
///   - In some cases we wish to coerce "non-expressions" whose types are implicitly
///     unit. This happens for example if you have a `break` with no expression,
///     or an `if` with no `else`. In that case, invoke `coerce_forced_unit()`.
///   - `coerce()` and `coerce_forced_unit()` may report errors. They hide this
///     from you so that you don't have to worry your pretty head about it.
///     But if an error is reported, the final type will be `err`.
///   - Invoking `coerce()` may cause us to go and adjust the "adjustments" on
///     previously coerced expressions.
/// - When all done, invoke `complete()`. This will return the LUB of
///   all your expressions.
///   - WARNING: I don't believe this final type is guaranteed to be
///     related to your initial `expected_ty` in any particular way,
///     although it will typically be a subtype, so you should check it.
///   - Invoking `complete()` may cause us to go and adjust the "adjustments" on
///     previously coerced expressions.
///
/// Example:
///
/// ```
/// let mut coerce = CoerceMany::new(expected_ty);
/// for expr in exprs {
///     let expr_ty = fcx.check_expr_with_expectation(expr, expected);
///     coerce.coerce(fcx, &cause, expr, expr_ty);
/// }
/// let final_ty = coerce.complete(fcx);
/// ```
pub struct CoerceMany<'gcx, 'tcx, 'exprs, E>
    where 'gcx: 'tcx, E: 'exprs + AsCoercionSite,
{
    expected_ty: Ty<'tcx>,
    final_ty: Option<Ty<'tcx>>,
    expressions: Expressions<'gcx, 'exprs, E>,
    pushed: usize,
}

/// The type of a `CoerceMany` that is storing up the expressions into
/// a buffer. We use this in `check/mod.rs` for things like `break`.
pub type DynamicCoerceMany<'gcx, 'tcx> = CoerceMany<'gcx, 'tcx, 'gcx, P<hir::Expr>>;

enum Expressions<'gcx, 'exprs, E>
    where E: 'exprs + AsCoercionSite,
{
    Dynamic(Vec<&'gcx hir::Expr>),
    UpFront(&'exprs [E]),
}

impl<'gcx, 'tcx, 'exprs, E> CoerceMany<'gcx, 'tcx, 'exprs, E>
    where 'gcx: 'tcx, E: 'exprs + AsCoercionSite,
{
    /// The usual case; collect the set of expressions dynamically.
    /// If the full set of coercion sites is known before hand,
    /// consider `with_coercion_sites()` instead to avoid allocation.
    pub fn new(expected_ty: Ty<'tcx>) -> Self {
        Self::make(expected_ty, Expressions::Dynamic(vec![]))
    }

    /// As an optimization, you can create a `CoerceMany` with a
    /// pre-existing slice of expressions. In this case, you are
    /// expected to pass each element in the slice to `coerce(...)` in
    /// order. This is used with arrays in particular to avoid
    /// needlessly cloning the slice.
    pub fn with_coercion_sites(expected_ty: Ty<'tcx>,
                               coercion_sites: &'exprs [E])
                               -> Self {
        Self::make(expected_ty, Expressions::UpFront(coercion_sites))
    }

    fn make(expected_ty: Ty<'tcx>, expressions: Expressions<'gcx, 'exprs, E>) -> Self {
        CoerceMany {
            expected_ty,
            final_ty: None,
            expressions,
            pushed: 0,
        }
    }

    /// Return the "expected type" with which this coercion was
    /// constructed.  This represents the "downward propagated" type
    /// that was given to us at the start of typing whatever construct
    /// we are typing (e.g., the match expression).
    ///
    /// Typically, this is used as the expected type when
    /// type-checking each of the alternative expressions whose types
    /// we are trying to merge.
    pub fn expected_ty(&self) -> Ty<'tcx> {
        self.expected_ty
    }

    /// Returns the current "merged type", representing our best-guess
    /// at the LUB of the expressions we've seen so far (if any). This
    /// isn't *final* until you call `self.final()`, which will return
    /// the merged type.
    pub fn merged_ty(&self) -> Ty<'tcx> {
        self.final_ty.unwrap_or(self.expected_ty)
    }

    /// Indicates that the value generated by `expression`, which is
    /// of type `expression_ty`, is one of the possibilities that we
    /// could coerce from. This will record `expression`, and later
    /// calls to `coerce` may come back and add adjustments and things
    /// if necessary.
    pub fn coerce<'a>(&mut self,
                      fcx: &FnCtxt<'a, 'gcx, 'tcx>,
                      cause: &ObligationCause<'tcx>,
                      expression: &'gcx hir::Expr,
                      expression_ty: Ty<'tcx>)
    {
        self.coerce_inner(fcx,
                          cause,
                          Some(expression),
                          expression_ty,
                          None, false)
    }

    /// Indicates that one of the inputs is a "forced unit". This
    /// occurs in a case like `if foo { ... };`, where the missing else
    /// generates a "forced unit". Another example is a `loop { break;
    /// }`, where the `break` has no argument expression. We treat
    /// these cases slightly differently for error-reporting
    /// purposes. Note that these tend to correspond to cases where
    /// the `()` expression is implicit in the source, and hence we do
    /// not take an expression argument.
    ///
    /// The `augment_error` gives you a chance to extend the error
    /// message, in case any results (e.g., we use this to suggest
    /// removing a `;`).
    pub fn coerce_forced_unit<'a>(&mut self,
                                  fcx: &FnCtxt<'a, 'gcx, 'tcx>,
                                  cause: &ObligationCause<'tcx>,
                                  augment_error: &mut dyn FnMut(&mut DiagnosticBuilder),
                                  label_unit_as_expected: bool)
    {
        self.coerce_inner(fcx,
                          cause,
                          None,
                          fcx.tcx.mk_unit(),
                          Some(augment_error),
                          label_unit_as_expected)
    }

    /// The inner coercion "engine". If `expression` is `None`, this
    /// is a forced-unit case, and hence `expression_ty` must be
    /// `Nil`.
    fn coerce_inner<'a>(&mut self,
                        fcx: &FnCtxt<'a, 'gcx, 'tcx>,
                        cause: &ObligationCause<'tcx>,
                        expression: Option<&'gcx hir::Expr>,
                        mut expression_ty: Ty<'tcx>,
                        augment_error: Option<&mut dyn FnMut(&mut DiagnosticBuilder)>,
                        label_expression_as_expected: bool)
    {
        // Incorporate whatever type inference information we have
        // until now; in principle we might also want to process
        // pending obligations, but doing so should only improve
        // compatibility (hopefully that is true) by helping us
        // uncover never types better.
        if expression_ty.is_ty_var() {
            expression_ty = fcx.infcx.shallow_resolve(expression_ty);
        }

        // If we see any error types, just propagate that error
        // upwards.
        if expression_ty.references_error() || self.merged_ty().references_error() {
            self.final_ty = Some(fcx.tcx.types.err);
            return;
        }

        // Handle the actual type unification etc.
        let result = if let Some(expression) = expression {
            if self.pushed == 0 {
                // Special-case the first expression we are coercing.
                // To be honest, I'm not entirely sure why we do this.
                // We don't allow two-phase borrows, see comment in try_find_coercion_lub for why
                fcx.try_coerce(expression, expression_ty, self.expected_ty, AllowTwoPhase::No)
            } else {
                match self.expressions {
                    Expressions::Dynamic(ref exprs) =>
                        fcx.try_find_coercion_lub(cause,
                                                  exprs,
                                                  self.merged_ty(),
                                                  expression,
                                                  expression_ty),
                    Expressions::UpFront(ref coercion_sites) =>
                        fcx.try_find_coercion_lub(cause,
                                                  &coercion_sites[0..self.pushed],
                                                  self.merged_ty(),
                                                  expression,
                                                  expression_ty),
                }
            }
        } else {
            // this is a hack for cases where we default to `()` because
            // the expression etc has been omitted from the source. An
            // example is an `if let` without an else:
            //
            //     if let Some(x) = ... { }
            //
            // we wind up with a second match arm that is like `_ =>
            // ()`.  That is the case we are considering here. We take
            // a different path to get the right "expected, found"
            // message and so forth (and because we know that
            // `expression_ty` will be unit).
            //
            // Another example is `break` with no argument expression.
            assert!(expression_ty.is_unit());
            assert!(expression_ty.is_unit(), "if let hack without unit type");
            fcx.at(cause, fcx.param_env)
               .eq_exp(label_expression_as_expected, expression_ty, self.merged_ty())
               .map(|infer_ok| {
                   fcx.register_infer_ok_obligations(infer_ok);
                   expression_ty
               })
        };

        match result {
            Ok(v) => {
                self.final_ty = Some(v);
                if let Some(e) = expression {
                    match self.expressions {
                        Expressions::Dynamic(ref mut buffer) => buffer.push(e),
                        Expressions::UpFront(coercion_sites) => {
                            // if the user gave us an array to validate, check that we got
                            // the next expression in the list, as expected
                            assert_eq!(coercion_sites[self.pushed].as_coercion_site().id, e.id);
                        }
                    }
                    self.pushed += 1;
                }
            }
            Err(err) => {
                let (expected, found) = if label_expression_as_expected {
                    // In the case where this is a "forced unit", like
                    // `break`, we want to call the `()` "expected"
                    // since it is implied by the syntax.
                    // (Note: not all force-units work this way.)"
                    (expression_ty, self.final_ty.unwrap_or(self.expected_ty))
                } else {
                    // Otherwise, the "expected" type for error
                    // reporting is the current unification type,
                    // which is basically the LUB of the expressions
                    // we've seen so far (combined with the expected
                    // type)
                    (self.final_ty.unwrap_or(self.expected_ty), expression_ty)
                };

                let mut db;
                match cause.code {
                    ObligationCauseCode::ReturnNoExpression => {
                        db = struct_span_err!(
                            fcx.tcx.sess, cause.span, E0069,
                            "`return;` in a function whose return type is not `()`");
                        db.span_label(cause.span, "return type is not ()");
                    }
                    ObligationCauseCode::BlockTailExpression(blk_id) => {
                        db = fcx.report_mismatched_types(cause, expected, found, err);

                        let expr = expression.unwrap_or_else(|| {
                            span_bug!(cause.span,
                                      "supposed to be part of a block tail expression, but the \
                                       expression is empty");
                        });
                        fcx.suggest_mismatched_types_on_tail(
                            &mut db,
                            expr,
                            expected,
                            found,
                            cause.span,
                            blk_id,
                        );
                    }
                    _ => {
                        db = fcx.report_mismatched_types(cause, expected, found, err);
                    }
                }

                if let Some(augment_error) = augment_error {
                    augment_error(&mut db);
                }

                db.emit();

                self.final_ty = Some(fcx.tcx.types.err);
            }
        }
    }

    pub fn complete<'a>(self, fcx: &FnCtxt<'a, 'gcx, 'tcx>) -> Ty<'tcx> {
        if let Some(final_ty) = self.final_ty {
            final_ty
        } else {
            // If we only had inputs that were of type `!` (or no
            // inputs at all), then the final type is `!`.
            assert_eq!(self.pushed, 0);
            fcx.tcx.types.never
        }
    }
}

/// Something that can be converted into an expression to which we can
/// apply a coercion.
pub trait AsCoercionSite {
    fn as_coercion_site(&self) -> &hir::Expr;
}

impl AsCoercionSite for hir::Expr {
    fn as_coercion_site(&self) -> &hir::Expr {
        self
    }
}

impl AsCoercionSite for P<hir::Expr> {
    fn as_coercion_site(&self) -> &hir::Expr {
        self
    }
}

impl<'a, T> AsCoercionSite for &'a T
    where T: AsCoercionSite
{
    fn as_coercion_site(&self) -> &hir::Expr {
        (**self).as_coercion_site()
    }
}

impl AsCoercionSite for ! {
    fn as_coercion_site(&self) -> &hir::Expr {
        unreachable!()
    }
}

impl AsCoercionSite for hir::Arm {
    fn as_coercion_site(&self) -> &hir::Expr {
        &self.body
    }
}
