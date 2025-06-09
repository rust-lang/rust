//! # Type Coercion
//!
//! Under certain circumstances we will coerce from one type to another,
//! for example by auto-borrowing. This occurs in situations where the
//! compiler has a firm 'expected type' that was supplied from the user,
//! and where the actual type is similar to that expected type in purpose
//! but not in representation (so actual subtyping is inappropriate).
//!
//! ## Reborrowing
//!
//! Note that if we are expecting a reference, we will *reborrow*
//! even if the argument provided was already a reference. This is
//! useful for freezing mut things (that is, when the expected type is &T
//! but you have &mut T) and also for avoiding the linearity
//! of mut things (when the expected is &mut T and you have &mut T). See
//! the various `tests/ui/coerce/*.rs` tests for
//! examples of where this is useful.
//!
//! ## Subtle note
//!
//! When inferring the generic arguments of functions, the argument
//! order is relevant, which can lead to the following edge case:
//!
//! ```ignore (illustrative)
//! fn foo<T>(a: T, b: T) {
//!     // ...
//! }
//!
//! foo(&7i32, &mut 7i32);
//! // This compiles, as we first infer `T` to be `&i32`,
//! // and then coerce `&mut 7i32` to `&7i32`.
//!
//! foo(&mut 7i32, &7i32);
//! // This does not compile, as we first infer `T` to be `&mut i32`
//! // and are then unable to coerce `&7i32` to `&mut i32`.
//! ```

use std::ops::Deref;

use rustc_attr_data_structures::InlineAttr;
use rustc_errors::codes::*;
use rustc_errors::{Applicability, Diag, struct_span_code_err};
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir_analysis::hir_ty_lowering::HirTyLowerer;
use rustc_infer::infer::relate::RelateResult;
use rustc_infer::infer::{Coercion, DefineOpaqueTypes, InferOk, InferResult};
use rustc_infer::traits::{
    IfExpressionCause, MatchExpressionArmCause, Obligation, PredicateObligation,
    PredicateObligations, SelectionError,
};
use rustc_middle::span_bug;
use rustc_middle::ty::adjustment::{
    Adjust, Adjustment, AllowTwoPhase, AutoBorrow, AutoBorrowMutability, PointerCoercion,
};
use rustc_middle::ty::error::TypeError;
use rustc_middle::ty::{self, GenericArgsRef, Ty, TyCtxt, TypeVisitableExt};
use rustc_span::{BytePos, DUMMY_SP, DesugaringKind, Span};
use rustc_trait_selection::infer::InferCtxtExt as _;
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt;
use rustc_trait_selection::traits::{
    self, NormalizeExt, ObligationCause, ObligationCauseCode, ObligationCtxt,
};
use smallvec::{SmallVec, smallvec};
use tracing::{debug, instrument};

use crate::FnCtxt;
use crate::errors::SuggestBoxingForReturnImplTrait;

struct Coerce<'a, 'tcx> {
    fcx: &'a FnCtxt<'a, 'tcx>,
    cause: ObligationCause<'tcx>,
    use_lub: bool,
    /// Determines whether or not allow_two_phase_borrow is set on any
    /// autoref adjustments we create while coercing. We don't want to
    /// allow deref coercions to create two-phase borrows, at least initially,
    /// but we do need two-phase borrows for function argument reborrows.
    /// See #47489 and #48598
    /// See docs on the "AllowTwoPhase" type for a more detailed discussion
    allow_two_phase: AllowTwoPhase,
    /// Whether we allow `NeverToAny` coercions. This is unsound if we're
    /// coercing a place expression without it counting as a read in the MIR.
    /// This is a side-effect of HIR not really having a great distinction
    /// between places and values.
    coerce_never: bool,
}

impl<'a, 'tcx> Deref for Coerce<'a, 'tcx> {
    type Target = FnCtxt<'a, 'tcx>;
    fn deref(&self) -> &Self::Target {
        self.fcx
    }
}

type CoerceResult<'tcx> = InferResult<'tcx, (Vec<Adjustment<'tcx>>, Ty<'tcx>)>;

/// Coercing a mutable reference to an immutable works, while
/// coercing `&T` to `&mut T` should be forbidden.
fn coerce_mutbls<'tcx>(
    from_mutbl: hir::Mutability,
    to_mutbl: hir::Mutability,
) -> RelateResult<'tcx, ()> {
    if from_mutbl >= to_mutbl { Ok(()) } else { Err(TypeError::Mutability) }
}

/// This always returns `Ok(...)`.
fn success<'tcx>(
    adj: Vec<Adjustment<'tcx>>,
    target: Ty<'tcx>,
    obligations: PredicateObligations<'tcx>,
) -> CoerceResult<'tcx> {
    Ok(InferOk { value: (adj, target), obligations })
}

impl<'f, 'tcx> Coerce<'f, 'tcx> {
    fn new(
        fcx: &'f FnCtxt<'f, 'tcx>,
        cause: ObligationCause<'tcx>,
        allow_two_phase: AllowTwoPhase,
        coerce_never: bool,
    ) -> Self {
        Coerce { fcx, cause, allow_two_phase, use_lub: false, coerce_never }
    }

    fn unify_raw(&self, a: Ty<'tcx>, b: Ty<'tcx>) -> InferResult<'tcx, Ty<'tcx>> {
        debug!("unify(a: {:?}, b: {:?}, use_lub: {})", a, b, self.use_lub);
        self.commit_if_ok(|_| {
            let at = self.at(&self.cause, self.fcx.param_env);

            let res = if self.use_lub {
                at.lub(b, a)
            } else {
                at.sup(DefineOpaqueTypes::Yes, b, a)
                    .map(|InferOk { value: (), obligations }| InferOk { value: b, obligations })
            };

            // In the new solver, lazy norm may allow us to shallowly equate
            // more types, but we emit possibly impossible-to-satisfy obligations.
            // Filter these cases out to make sure our coercion is more accurate.
            match res {
                Ok(InferOk { value, obligations }) if self.next_trait_solver() => {
                    let ocx = ObligationCtxt::new(self);
                    ocx.register_obligations(obligations);
                    if ocx.select_where_possible().is_empty() {
                        Ok(InferOk { value, obligations: ocx.into_pending_obligations() })
                    } else {
                        Err(TypeError::Mismatch)
                    }
                }
                res => res,
            }
        })
    }

    /// Unify two types (using sub or lub).
    fn unify(&self, a: Ty<'tcx>, b: Ty<'tcx>) -> CoerceResult<'tcx> {
        self.unify_raw(a, b)
            .and_then(|InferOk { value: ty, obligations }| success(vec![], ty, obligations))
    }

    /// Unify two types (using sub or lub) and produce a specific coercion.
    fn unify_and(
        &self,
        a: Ty<'tcx>,
        b: Ty<'tcx>,
        adjustments: impl IntoIterator<Item = Adjustment<'tcx>>,
        final_adjustment: Adjust,
    ) -> CoerceResult<'tcx> {
        self.unify_raw(a, b).and_then(|InferOk { value: ty, obligations }| {
            success(
                adjustments
                    .into_iter()
                    .chain(std::iter::once(Adjustment { target: ty, kind: final_adjustment }))
                    .collect(),
                ty,
                obligations,
            )
        })
    }

    #[instrument(skip(self))]
    fn coerce(&self, a: Ty<'tcx>, b: Ty<'tcx>) -> CoerceResult<'tcx> {
        // First, remove any resolved type variables (at the top level, at least):
        let a = self.shallow_resolve(a);
        let b = self.shallow_resolve(b);
        debug!("Coerce.tys({:?} => {:?})", a, b);

        // Coercing from `!` to any type is allowed:
        if a.is_never() {
            if self.coerce_never {
                return success(
                    vec![Adjustment { kind: Adjust::NeverToAny, target: b }],
                    b,
                    PredicateObligations::new(),
                );
            } else {
                // Otherwise the only coercion we can do is unification.
                return self.unify(a, b);
            }
        }

        // Coercing *from* an unresolved inference variable means that
        // we have no information about the source type. This will always
        // ultimately fall back to some form of subtyping.
        if a.is_ty_var() {
            return self.coerce_from_inference_variable(a, b);
        }

        // Consider coercing the subtype to a DST
        //
        // NOTE: this is wrapped in a `commit_if_ok` because it creates
        // a "spurious" type variable, and we don't want to have that
        // type variable in memory if the coercion fails.
        let unsize = self.commit_if_ok(|_| self.coerce_unsized(a, b));
        match unsize {
            Ok(_) => {
                debug!("coerce: unsize successful");
                return unsize;
            }
            Err(error) => {
                debug!(?error, "coerce: unsize failed");
            }
        }

        // Examine the supertype and consider type-specific coercions, such
        // as auto-borrowing, coercing pointer mutability, a `dyn*` coercion,
        // or pin-ergonomics.
        match *b.kind() {
            ty::RawPtr(_, b_mutbl) => {
                return self.coerce_raw_ptr(a, b, b_mutbl);
            }
            ty::Ref(r_b, _, mutbl_b) => {
                return self.coerce_borrowed_pointer(a, b, r_b, mutbl_b);
            }
            ty::Dynamic(predicates, region, ty::DynStar) if self.tcx.features().dyn_star() => {
                return self.coerce_dyn_star(a, b, predicates, region);
            }
            ty::Adt(pin, _)
                if self.tcx.features().pin_ergonomics()
                    && self.tcx.is_lang_item(pin.did(), hir::LangItem::Pin) =>
            {
                let pin_coerce = self.commit_if_ok(|_| self.coerce_pin_ref(a, b));
                if pin_coerce.is_ok() {
                    return pin_coerce;
                }
            }
            _ => {}
        }

        match *a.kind() {
            ty::FnDef(..) => {
                // Function items are coercible to any closure
                // type; function pointers are not (that would
                // require double indirection).
                // Additionally, we permit coercion of function
                // items to drop the unsafe qualifier.
                self.coerce_from_fn_item(a, b)
            }
            ty::FnPtr(a_sig_tys, a_hdr) => {
                // We permit coercion of fn pointers to drop the
                // unsafe qualifier.
                self.coerce_from_fn_pointer(a_sig_tys.with(a_hdr), b)
            }
            ty::Closure(closure_def_id_a, args_a) => {
                // Non-capturing closures are coercible to
                // function pointers or unsafe function pointers.
                // It cannot convert closures that require unsafe.
                self.coerce_closure_to_fn(a, closure_def_id_a, args_a, b)
            }
            _ => {
                // Otherwise, just use unification rules.
                self.unify(a, b)
            }
        }
    }

    /// Coercing *from* an inference variable. In this case, we have no information
    /// about the source type, so we can't really do a true coercion and we always
    /// fall back to subtyping (`unify_and`).
    fn coerce_from_inference_variable(&self, a: Ty<'tcx>, b: Ty<'tcx>) -> CoerceResult<'tcx> {
        debug!("coerce_from_inference_variable(a={:?}, b={:?})", a, b);
        assert!(a.is_ty_var() && self.shallow_resolve(a) == a);
        assert!(self.shallow_resolve(b) == b);

        if b.is_ty_var() {
            // Two unresolved type variables: create a `Coerce` predicate.
            let target_ty = if self.use_lub { self.next_ty_var(self.cause.span) } else { b };

            let mut obligations = PredicateObligations::with_capacity(2);
            for &source_ty in &[a, b] {
                if source_ty != target_ty {
                    obligations.push(Obligation::new(
                        self.tcx(),
                        self.cause.clone(),
                        self.param_env,
                        ty::Binder::dummy(ty::PredicateKind::Coerce(ty::CoercePredicate {
                            a: source_ty,
                            b: target_ty,
                        })),
                    ));
                }
            }

            debug!(
                "coerce_from_inference_variable: two inference variables, target_ty={:?}, obligations={:?}",
                target_ty, obligations
            );
            success(vec![], target_ty, obligations)
        } else {
            // One unresolved type variable: just apply subtyping, we may be able
            // to do something useful.
            self.unify(a, b)
        }
    }

    /// Reborrows `&mut A` to `&mut B` and `&(mut) A` to `&B`.
    /// To match `A` with `B`, autoderef will be performed,
    /// calling `deref`/`deref_mut` where necessary.
    fn coerce_borrowed_pointer(
        &self,
        a: Ty<'tcx>,
        b: Ty<'tcx>,
        r_b: ty::Region<'tcx>,
        mutbl_b: hir::Mutability,
    ) -> CoerceResult<'tcx> {
        debug!("coerce_borrowed_pointer(a={:?}, b={:?})", a, b);

        // If we have a parameter of type `&M T_a` and the value
        // provided is `expr`, we will be adding an implicit borrow,
        // meaning that we convert `f(expr)` to `f(&M *expr)`. Therefore,
        // to type check, we will construct the type that `&M*expr` would
        // yield.

        let (r_a, mt_a) = match *a.kind() {
            ty::Ref(r_a, ty, mutbl) => {
                let mt_a = ty::TypeAndMut { ty, mutbl };
                coerce_mutbls(mt_a.mutbl, mutbl_b)?;
                (r_a, mt_a)
            }
            _ => return self.unify(a, b),
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

            // At this point, we have deref'd `a` to `referent_ty`. So
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
            //   arbitrarily) returns the subtype region. In the case
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
            //     expression (`*x`, here). (Search for "link".)
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
            //     we get spurious errors (`ui/regions-lub-ref-ref-rc.rs`).
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
                    // create var lazily, at most once
                    let coercion = Coercion(span);
                    let r = self.next_region_var(coercion);
                    r_borrow_var = Some(r); // [4] above
                }
                r_borrow_var.unwrap()
            };
            let derefd_ty_a = Ty::new_ref(
                self.tcx,
                r,
                referent_ty,
                mutbl_b, // [1] above
            );
            match self.unify_raw(derefd_ty_a, b) {
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
        let Some(InferOk { value: ty, mut obligations }) = found else {
            if let Some(first_error) = first_error {
                debug!("coerce_borrowed_pointer: failed with err = {:?}", first_error);
                return Err(first_error);
            } else {
                // This may happen in the new trait solver since autoderef requires
                // the pointee to be structurally normalizable, or else it'll just bail.
                // So when we have a type like `&<not well formed>`, then we get no
                // autoderef steps (even though there should be at least one). That means
                // we get no type mismatches, since the loop above just exits early.
                return Err(TypeError::Mismatch);
            }
        };

        if ty == a && mt_a.mutbl.is_not() && autoderef.step_count() == 1 {
            // As a special case, if we would produce `&'a *x`, that's
            // a total no-op. We end up with the type `&'a T` just as
            // we started with. In that case, just skip it
            // altogether. This is just an optimization.
            //
            // Note that for `&mut`, we DO want to reborrow --
            // otherwise, this would be a move, which might be an
            // error. For example `foo(self.x)` where `self` and
            // `self.x` both have `&mut `type would be a move of
            // `self.x`, but we auto-coerce it to `foo(&mut *self.x)`,
            // which is a borrow.
            assert!(mutbl_b.is_not()); // can only coerce &T -> &U
            return success(vec![], ty, obligations);
        }

        let InferOk { value: mut adjustments, obligations: o } =
            self.adjust_steps_as_infer_ok(&autoderef);
        obligations.extend(o);
        obligations.extend(autoderef.into_obligations());

        // Now apply the autoref. We have to extract the region out of
        // the final ref type we got.
        let ty::Ref(..) = ty.kind() else {
            span_bug!(span, "expected a ref type, got {:?}", ty);
        };
        let mutbl = AutoBorrowMutability::new(mutbl_b, self.allow_two_phase);
        adjustments.push(Adjustment { kind: Adjust::Borrow(AutoBorrow::Ref(mutbl)), target: ty });

        debug!("coerce_borrowed_pointer: succeeded ty={:?} adjustments={:?}", ty, adjustments);

        success(adjustments, ty, obligations)
    }

    /// Performs [unsized coercion] by emulating a fulfillment loop on a
    /// `CoerceUnsized` goal until all `CoerceUnsized` and `Unsize` goals
    /// are successfully selected.
    ///
    /// [unsized coercion](https://doc.rust-lang.org/reference/type-coercions.html#unsized-coercions)
    #[instrument(skip(self), level = "debug")]
    fn coerce_unsized(&self, mut source: Ty<'tcx>, mut target: Ty<'tcx>) -> CoerceResult<'tcx> {
        source = self.shallow_resolve(source);
        target = self.shallow_resolve(target);
        debug!(?source, ?target);

        // We don't apply any coercions incase either the source or target
        // aren't sufficiently well known but tend to instead just equate
        // them both.
        if source.is_ty_var() {
            debug!("coerce_unsized: source is a TyVar, bailing out");
            return Err(TypeError::Mismatch);
        }
        if target.is_ty_var() {
            debug!("coerce_unsized: target is a TyVar, bailing out");
            return Err(TypeError::Mismatch);
        }

        let traits =
            (self.tcx.lang_items().unsize_trait(), self.tcx.lang_items().coerce_unsized_trait());
        let (Some(unsize_did), Some(coerce_unsized_did)) = traits else {
            debug!("missing Unsize or CoerceUnsized traits");
            return Err(TypeError::Mismatch);
        };

        // Note, we want to avoid unnecessary unsizing. We don't want to coerce to
        // a DST unless we have to. This currently comes out in the wash since
        // we can't unify [T] with U. But to properly support DST, we need to allow
        // that, at which point we will need extra checks on the target here.

        // Handle reborrows before selecting `Source: CoerceUnsized<Target>`.
        let reborrow = match (source.kind(), target.kind()) {
            (&ty::Ref(_, ty_a, mutbl_a), &ty::Ref(_, _, mutbl_b)) => {
                coerce_mutbls(mutbl_a, mutbl_b)?;

                let coercion = Coercion(self.cause.span);
                let r_borrow = self.next_region_var(coercion);

                // We don't allow two-phase borrows here, at least for initial
                // implementation. If it happens that this coercion is a function argument,
                // the reborrow in coerce_borrowed_ptr will pick it up.
                let mutbl = AutoBorrowMutability::new(mutbl_b, AllowTwoPhase::No);

                Some((
                    Adjustment { kind: Adjust::Deref(None), target: ty_a },
                    Adjustment {
                        kind: Adjust::Borrow(AutoBorrow::Ref(mutbl)),
                        target: Ty::new_ref(self.tcx, r_borrow, ty_a, mutbl_b),
                    },
                ))
            }
            (&ty::Ref(_, ty_a, mt_a), &ty::RawPtr(_, mt_b)) => {
                coerce_mutbls(mt_a, mt_b)?;

                Some((
                    Adjustment { kind: Adjust::Deref(None), target: ty_a },
                    Adjustment {
                        kind: Adjust::Borrow(AutoBorrow::RawPtr(mt_b)),
                        target: Ty::new_ptr(self.tcx, ty_a, mt_b),
                    },
                ))
            }
            _ => None,
        };
        let coerce_source = reborrow.as_ref().map_or(source, |(_, r)| r.target);

        // Setup either a subtyping or a LUB relationship between
        // the `CoerceUnsized` target type and the expected type.
        // We only have the latter, so we use an inference variable
        // for the former and let type inference do the rest.
        let coerce_target = self.next_ty_var(self.cause.span);

        let mut coercion = self.unify_and(
            coerce_target,
            target,
            reborrow.into_iter().flat_map(|(deref, autoref)| [deref, autoref]),
            Adjust::Pointer(PointerCoercion::Unsize),
        )?;

        let mut selcx = traits::SelectionContext::new(self);

        // Create an obligation for `Source: CoerceUnsized<Target>`.
        let cause = self.cause(self.cause.span, ObligationCauseCode::Coercion { source, target });

        // Use a FIFO queue for this custom fulfillment procedure.
        //
        // A Vec (or SmallVec) is not a natural choice for a queue. However,
        // this code path is hot, and this queue usually has a max length of 1
        // and almost never more than 3. By using a SmallVec we avoid an
        // allocation, at the (very small) cost of (occasionally) having to
        // shift subsequent elements down when removing the front element.
        let mut queue: SmallVec<[PredicateObligation<'tcx>; 4]> = smallvec![Obligation::new(
            self.tcx,
            cause,
            self.fcx.param_env,
            ty::TraitRef::new(self.tcx, coerce_unsized_did, [coerce_source, coerce_target])
        )];

        // Keep resolving `CoerceUnsized` and `Unsize` predicates to avoid
        // emitting a coercion in cases like `Foo<$1>` -> `Foo<$2>`, where
        // inference might unify those two inner type variables later.
        let traits = [coerce_unsized_did, unsize_did];
        while !queue.is_empty() {
            let obligation = queue.remove(0);
            let trait_pred = match obligation.predicate.kind().no_bound_vars() {
                Some(ty::PredicateKind::Clause(ty::ClauseKind::Trait(trait_pred)))
                    if traits.contains(&trait_pred.def_id()) =>
                {
                    self.resolve_vars_if_possible(trait_pred)
                }
                // Eagerly process alias-relate obligations in new trait solver,
                // since these can be emitted in the process of solving trait goals,
                // but we need to constrain vars before processing goals mentioning
                // them.
                Some(ty::PredicateKind::AliasRelate(..)) => {
                    let ocx = ObligationCtxt::new(self);
                    ocx.register_obligation(obligation);
                    if !ocx.select_where_possible().is_empty() {
                        return Err(TypeError::Mismatch);
                    }
                    coercion.obligations.extend(ocx.into_pending_obligations());
                    continue;
                }
                _ => {
                    coercion.obligations.push(obligation);
                    continue;
                }
            };
            debug!("coerce_unsized resolve step: {:?}", trait_pred);
            match selcx.select(&obligation.with(selcx.tcx(), trait_pred)) {
                // Uncertain or unimplemented.
                Ok(None) => {
                    if trait_pred.def_id() == unsize_did {
                        let self_ty = trait_pred.self_ty();
                        let unsize_ty = trait_pred.trait_ref.args[1].expect_ty();
                        debug!("coerce_unsized: ambiguous unsize case for {:?}", trait_pred);
                        match (self_ty.kind(), unsize_ty.kind()) {
                            (&ty::Infer(ty::TyVar(v)), ty::Dynamic(..))
                                if self.type_var_is_sized(v) =>
                            {
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

                Err(SelectionError::TraitDynIncompatible(_)) => {
                    // Dyn compatibility errors in coercion will *always* be due to the
                    // fact that the RHS of the coercion is a non-dyn compatible `dyn Trait`
                    // writen in source somewhere (otherwise we will never have lowered
                    // the dyn trait from HIR to middle).
                    //
                    // There's no reason to emit yet another dyn compatibility error,
                    // especially since the span will differ slightly and thus not be
                    // deduplicated at all!
                    self.fcx.set_tainted_by_errors(
                        self.fcx
                            .dcx()
                            .span_delayed_bug(self.cause.span, "dyn compatibility during coercion"),
                    );
                }
                Err(err) => {
                    let guar = self.err_ctxt().report_selection_error(
                        obligation.clone(),
                        &obligation,
                        &err,
                    );
                    self.fcx.set_tainted_by_errors(guar);
                    // Treat this like an obligation and follow through
                    // with the unsizing - the lack of a coercion should
                    // be silent, as it causes a type mismatch later.
                }

                Ok(Some(impl_source)) => queue.extend(impl_source.nested_obligations()),
            }
        }

        Ok(coercion)
    }

    fn coerce_dyn_star(
        &self,
        a: Ty<'tcx>,
        b: Ty<'tcx>,
        predicates: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
        b_region: ty::Region<'tcx>,
    ) -> CoerceResult<'tcx> {
        if !self.tcx.features().dyn_star() {
            return Err(TypeError::Mismatch);
        }

        // FIXME(dyn_star): We should probably allow things like casting from
        // `dyn* Foo + Send` to `dyn* Foo`.
        if let ty::Dynamic(a_data, _, ty::DynStar) = a.kind()
            && let ty::Dynamic(b_data, _, ty::DynStar) = b.kind()
            && a_data.principal_def_id() == b_data.principal_def_id()
        {
            return self.unify(a, b);
        }

        // Check the obligations of the cast -- for example, when casting
        // `usize` to `dyn* Clone + 'static`:
        let obligations = predicates
            .iter()
            .map(|predicate| {
                // For each existential predicate (e.g., `?Self: Clone`) instantiate
                // the type of the expression (e.g., `usize` in our example above)
                // and then require that the resulting predicate (e.g., `usize: Clone`)
                // holds (it does).
                let predicate = predicate.with_self_ty(self.tcx, a);
                Obligation::new(self.tcx, self.cause.clone(), self.param_env, predicate)
            })
            .chain([
                // Enforce the region bound (e.g., `usize: 'static`, in our example).
                Obligation::new(
                    self.tcx,
                    self.cause.clone(),
                    self.param_env,
                    ty::Binder::dummy(ty::PredicateKind::Clause(ty::ClauseKind::TypeOutlives(
                        ty::OutlivesPredicate(a, b_region),
                    ))),
                ),
                // Enforce that the type is `usize`/pointer-sized.
                Obligation::new(
                    self.tcx,
                    self.cause.clone(),
                    self.param_env,
                    ty::TraitRef::new(
                        self.tcx,
                        self.tcx.require_lang_item(hir::LangItem::PointerLike, self.cause.span),
                        [a],
                    ),
                ),
            ])
            .collect();

        Ok(InferOk {
            value: (
                vec![Adjustment { kind: Adjust::Pointer(PointerCoercion::DynStar), target: b }],
                b,
            ),
            obligations,
        })
    }

    /// Applies reborrowing for `Pin`
    ///
    /// We currently only support reborrowing `Pin<&mut T>` as `Pin<&mut T>`. This is accomplished
    /// by inserting a call to `Pin::as_mut` during MIR building.
    ///
    /// In the future we might want to support other reborrowing coercions, such as:
    /// - `Pin<&mut T>` as `Pin<&T>`
    /// - `Pin<&T>` as `Pin<&T>`
    /// - `Pin<Box<T>>` as `Pin<&T>`
    /// - `Pin<Box<T>>` as `Pin<&mut T>`
    #[instrument(skip(self), level = "trace")]
    fn coerce_pin_ref(&self, a: Ty<'tcx>, b: Ty<'tcx>) -> CoerceResult<'tcx> {
        // We need to make sure the two types are compatible for coercion.
        // Then we will build a ReborrowPin adjustment and return that as an InferOk.

        // Right now we can only reborrow if this is a `Pin<&mut T>`.
        let extract_pin_mut = |ty: Ty<'tcx>| {
            // Get the T out of Pin<T>
            let (pin, ty) = match ty.kind() {
                ty::Adt(pin, args) if self.tcx.is_lang_item(pin.did(), hir::LangItem::Pin) => {
                    (*pin, args[0].expect_ty())
                }
                _ => {
                    debug!("can't reborrow {:?} as pinned", ty);
                    return Err(TypeError::Mismatch);
                }
            };
            // Make sure the T is something we understand (just `&mut U` for now)
            match ty.kind() {
                ty::Ref(region, ty, mutbl) => Ok((pin, *region, *ty, *mutbl)),
                _ => {
                    debug!("can't reborrow pin of inner type {:?}", ty);
                    Err(TypeError::Mismatch)
                }
            }
        };

        let (pin, a_region, a_ty, mut_a) = extract_pin_mut(a)?;
        let (_, _, _b_ty, mut_b) = extract_pin_mut(b)?;

        coerce_mutbls(mut_a, mut_b)?;

        // update a with b's mutability since we'll be coercing mutability
        let a = Ty::new_adt(
            self.tcx,
            pin,
            self.tcx.mk_args(&[Ty::new_ref(self.tcx, a_region, a_ty, mut_b).into()]),
        );

        // To complete the reborrow, we need to make sure we can unify the inner types, and if so we
        // add the adjustments.
        self.unify_and(a, b, [], Adjust::ReborrowPin(mut_b))
    }

    fn coerce_from_safe_fn(
        &self,
        fn_ty_a: ty::PolyFnSig<'tcx>,
        b: Ty<'tcx>,
        adjustment: Option<Adjust>,
    ) -> CoerceResult<'tcx> {
        self.commit_if_ok(|snapshot| {
            let outer_universe = self.infcx.universe();

            let result = if let ty::FnPtr(_, hdr_b) = b.kind()
                && fn_ty_a.safety().is_safe()
                && hdr_b.safety.is_unsafe()
            {
                let unsafe_a = self.tcx.safe_to_unsafe_fn_ty(fn_ty_a);
                self.unify_and(
                    unsafe_a,
                    b,
                    adjustment
                        .map(|kind| Adjustment { kind, target: Ty::new_fn_ptr(self.tcx, fn_ty_a) }),
                    Adjust::Pointer(PointerCoercion::UnsafeFnPointer),
                )
            } else {
                let a = Ty::new_fn_ptr(self.tcx, fn_ty_a);
                match adjustment {
                    Some(adjust) => self.unify_and(a, b, [], adjust),
                    None => self.unify(a, b),
                }
            };

            // FIXME(#73154): This is a hack. Currently LUB can generate
            // unsolvable constraints. Additionally, it returns `a`
            // unconditionally, even when the "LUB" is `b`. In the future, we
            // want the coerced type to be the actual supertype of these two,
            // but for now, we want to just error to ensure we don't lock
            // ourselves into a specific behavior with NLL.
            self.leak_check(outer_universe, Some(snapshot))?;

            result
        })
    }

    fn coerce_from_fn_pointer(
        &self,
        fn_ty_a: ty::PolyFnSig<'tcx>,
        b: Ty<'tcx>,
    ) -> CoerceResult<'tcx> {
        //! Attempts to coerce from the type of a Rust function item
        //! into a closure or a `proc`.
        //!

        let b = self.shallow_resolve(b);
        debug!(?fn_ty_a, ?b, "coerce_from_fn_pointer");

        self.coerce_from_safe_fn(fn_ty_a, b, None)
    }

    fn coerce_from_fn_item(&self, a: Ty<'tcx>, b: Ty<'tcx>) -> CoerceResult<'tcx> {
        //! Attempts to coerce from the type of a Rust function item
        //! into a closure or a `proc`.

        let b = self.shallow_resolve(b);
        let InferOk { value: b, mut obligations } =
            self.at(&self.cause, self.param_env).normalize(b);
        debug!("coerce_from_fn_item(a={:?}, b={:?})", a, b);

        match b.kind() {
            ty::FnPtr(_, b_hdr) => {
                let mut a_sig = a.fn_sig(self.tcx);
                if let ty::FnDef(def_id, _) = *a.kind() {
                    // Intrinsics are not coercible to function pointers
                    if self.tcx.intrinsic(def_id).is_some() {
                        return Err(TypeError::IntrinsicCast);
                    }

                    let fn_attrs = self.tcx.codegen_fn_attrs(def_id);
                    if matches!(fn_attrs.inline, InlineAttr::Force { .. }) {
                        return Err(TypeError::ForceInlineCast);
                    }

                    if b_hdr.safety.is_safe()
                        && self.tcx.codegen_fn_attrs(def_id).safe_target_features
                    {
                        // Allow the coercion if the current function has all the features that would be
                        // needed to call the coercee safely.
                        if let Some(safe_sig) = self.tcx.adjust_target_feature_sig(
                            def_id,
                            a_sig,
                            self.fcx.body_id.into(),
                        ) {
                            a_sig = safe_sig;
                        } else {
                            return Err(TypeError::TargetFeatureCast(def_id));
                        }
                    }
                }

                let InferOk { value: a_sig, obligations: o1 } =
                    self.at(&self.cause, self.param_env).normalize(a_sig);
                obligations.extend(o1);

                let InferOk { value, obligations: o2 } = self.coerce_from_safe_fn(
                    a_sig,
                    b,
                    Some(Adjust::Pointer(PointerCoercion::ReifyFnPointer)),
                )?;

                obligations.extend(o2);
                Ok(InferOk { value, obligations })
            }
            _ => self.unify(a, b),
        }
    }

    fn coerce_closure_to_fn(
        &self,
        a: Ty<'tcx>,
        closure_def_id_a: DefId,
        args_a: GenericArgsRef<'tcx>,
        b: Ty<'tcx>,
    ) -> CoerceResult<'tcx> {
        //! Attempts to coerce from the type of a non-capturing closure
        //! into a function pointer.
        //!

        let b = self.shallow_resolve(b);

        match b.kind() {
            // At this point we haven't done capture analysis, which means
            // that the ClosureArgs just contains an inference variable instead
            // of tuple of captured types.
            //
            // All we care here is if any variable is being captured and not the exact paths,
            // so we check `upvars_mentioned` for root variables being captured.
            ty::FnPtr(_, hdr)
                if self
                    .tcx
                    .upvars_mentioned(closure_def_id_a.expect_local())
                    .is_none_or(|u| u.is_empty()) =>
            {
                // We coerce the closure, which has fn type
                //     `extern "rust-call" fn((arg0,arg1,...)) -> _`
                // to
                //     `fn(arg0,arg1,...) -> _`
                // or
                //     `unsafe fn(arg0,arg1,...) -> _`
                let closure_sig = args_a.as_closure().sig();
                let safety = hdr.safety;
                let pointer_ty =
                    Ty::new_fn_ptr(self.tcx, self.tcx.signature_unclosure(closure_sig, safety));
                debug!("coerce_closure_to_fn(a={:?}, b={:?}, pty={:?})", a, b, pointer_ty);
                self.unify_and(
                    pointer_ty,
                    b,
                    [],
                    Adjust::Pointer(PointerCoercion::ClosureFnPointer(safety)),
                )
            }
            _ => self.unify(a, b),
        }
    }

    fn coerce_raw_ptr(
        &self,
        a: Ty<'tcx>,
        b: Ty<'tcx>,
        mutbl_b: hir::Mutability,
    ) -> CoerceResult<'tcx> {
        debug!("coerce_raw_ptr(a={:?}, b={:?})", a, b);

        let (is_ref, mt_a) = match *a.kind() {
            ty::Ref(_, ty, mutbl) => (true, ty::TypeAndMut { ty, mutbl }),
            ty::RawPtr(ty, mutbl) => (false, ty::TypeAndMut { ty, mutbl }),
            _ => return self.unify(a, b),
        };
        coerce_mutbls(mt_a.mutbl, mutbl_b)?;

        // Check that the types which they point at are compatible.
        let a_raw = Ty::new_ptr(self.tcx, mt_a.ty, mutbl_b);
        // Although references and raw ptrs have the same
        // representation, we still register an Adjust::DerefRef so that
        // regionck knows that the region for `a` must be valid here.
        if is_ref {
            self.unify_and(
                a_raw,
                b,
                [Adjustment { kind: Adjust::Deref(None), target: mt_a.ty }],
                Adjust::Borrow(AutoBorrow::RawPtr(mutbl_b)),
            )
        } else if mt_a.mutbl != mutbl_b {
            self.unify_and(a_raw, b, [], Adjust::Pointer(PointerCoercion::MutToConstPointer))
        } else {
            self.unify(a_raw, b)
        }
    }
}

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    /// Attempt to coerce an expression to a type, and return the
    /// adjusted type of the expression, if successful.
    /// Adjustments are only recorded if the coercion succeeded.
    /// The expressions *must not* have any preexisting adjustments.
    pub(crate) fn coerce(
        &self,
        expr: &'tcx hir::Expr<'tcx>,
        expr_ty: Ty<'tcx>,
        mut target: Ty<'tcx>,
        allow_two_phase: AllowTwoPhase,
        cause: Option<ObligationCause<'tcx>>,
    ) -> RelateResult<'tcx, Ty<'tcx>> {
        let source = self.try_structurally_resolve_type(expr.span, expr_ty);
        if self.next_trait_solver() {
            target = self.try_structurally_resolve_type(
                cause.as_ref().map_or(expr.span, |cause| cause.span),
                target,
            );
        }
        debug!("coercion::try({:?}: {:?} -> {:?})", expr, source, target);

        let cause =
            cause.unwrap_or_else(|| self.cause(expr.span, ObligationCauseCode::ExprAssignable));
        let coerce = Coerce::new(
            self,
            cause,
            allow_two_phase,
            self.expr_guaranteed_to_constitute_read_for_never(expr),
        );
        let ok = self.commit_if_ok(|_| coerce.coerce(source, target))?;

        let (adjustments, _) = self.register_infer_ok_obligations(ok);
        self.apply_adjustments(expr, adjustments);
        Ok(if let Err(guar) = expr_ty.error_reported() {
            Ty::new_error(self.tcx, guar)
        } else {
            target
        })
    }

    /// Probe whether `expr_ty` can be coerced to `target_ty`. This has no side-effects,
    /// and may return false positives if types are not yet fully constrained by inference.
    ///
    /// Returns false if the coercion is not possible, or if the coercion creates any
    /// sub-obligations that result in errors.
    ///
    /// This should only be used for diagnostics.
    pub(crate) fn may_coerce(&self, expr_ty: Ty<'tcx>, target_ty: Ty<'tcx>) -> bool {
        let cause = self.cause(DUMMY_SP, ObligationCauseCode::ExprAssignable);
        // We don't ever need two-phase here since we throw out the result of the coercion.
        // We also just always set `coerce_never` to true, since this is a heuristic.
        let coerce = Coerce::new(self, cause.clone(), AllowTwoPhase::No, true);
        self.probe(|_| {
            // Make sure to structurally resolve the types, since we use
            // the `TyKind`s heavily in coercion.
            let ocx = ObligationCtxt::new(self);
            let structurally_resolve = |ty| {
                let ty = self.shallow_resolve(ty);
                if self.next_trait_solver()
                    && let ty::Alias(..) = ty.kind()
                {
                    ocx.structurally_normalize_ty(&cause, self.param_env, ty)
                } else {
                    Ok(ty)
                }
            };
            let Ok(expr_ty) = structurally_resolve(expr_ty) else {
                return false;
            };
            let Ok(target_ty) = structurally_resolve(target_ty) else {
                return false;
            };

            let Ok(ok) = coerce.coerce(expr_ty, target_ty) else {
                return false;
            };
            ocx.register_obligations(ok.obligations);
            ocx.select_where_possible().is_empty()
        })
    }

    /// Given a type and a target type, this function will calculate and return
    /// how many dereference steps needed to coerce `expr_ty` to `target`. If
    /// it's not possible, return `None`.
    pub(crate) fn deref_steps_for_suggestion(
        &self,
        expr_ty: Ty<'tcx>,
        target: Ty<'tcx>,
    ) -> Option<usize> {
        let cause = self.cause(DUMMY_SP, ObligationCauseCode::ExprAssignable);
        // We don't ever need two-phase here since we throw out the result of the coercion.
        let coerce = Coerce::new(self, cause, AllowTwoPhase::No, true);
        coerce.autoderef(DUMMY_SP, expr_ty).find_map(|(ty, steps)| {
            self.probe(|_| coerce.unify_raw(ty, target)).ok().map(|_| steps)
        })
    }

    /// Given a type, this function will calculate and return the type given
    /// for `<Ty as Deref>::Target` only if `Ty` also implements `DerefMut`.
    ///
    /// This function is for diagnostics only, since it does not register
    /// trait or region sub-obligations. (presumably we could, but it's not
    /// particularly important for diagnostics...)
    pub(crate) fn deref_once_mutably_for_diagnostic(&self, expr_ty: Ty<'tcx>) -> Option<Ty<'tcx>> {
        self.autoderef(DUMMY_SP, expr_ty).silence_errors().nth(1).and_then(|(deref_ty, _)| {
            self.infcx
                .type_implements_trait(
                    self.tcx.lang_items().deref_mut_trait()?,
                    [expr_ty],
                    self.param_env,
                )
                .may_apply()
                .then_some(deref_ty)
        })
    }

    /// Given some expressions, their known unified type and another expression,
    /// tries to unify the types, potentially inserting coercions on any of the
    /// provided expressions and returns their LUB (aka "common supertype").
    ///
    /// This is really an internal helper. From outside the coercion
    /// module, you should instantiate a `CoerceMany` instance.
    fn try_find_coercion_lub<E>(
        &self,
        cause: &ObligationCause<'tcx>,
        exprs: &[E],
        prev_ty: Ty<'tcx>,
        new: &hir::Expr<'_>,
        new_ty: Ty<'tcx>,
    ) -> RelateResult<'tcx, Ty<'tcx>>
    where
        E: AsCoercionSite,
    {
        let prev_ty = self.try_structurally_resolve_type(cause.span, prev_ty);
        let new_ty = self.try_structurally_resolve_type(new.span, new_ty);
        debug!(
            "coercion::try_find_coercion_lub({:?}, {:?}, exprs={:?} exprs)",
            prev_ty,
            new_ty,
            exprs.len()
        );

        // The following check fixes #88097, where the compiler erroneously
        // attempted to coerce a closure type to itself via a function pointer.
        if prev_ty == new_ty {
            return Ok(prev_ty);
        }

        let is_force_inline = |ty: Ty<'tcx>| {
            if let ty::FnDef(did, _) = ty.kind() {
                matches!(self.tcx.codegen_fn_attrs(did).inline, InlineAttr::Force { .. })
            } else {
                false
            }
        };
        if is_force_inline(prev_ty) || is_force_inline(new_ty) {
            return Err(TypeError::ForceInlineCast);
        }

        // Special-case that coercion alone cannot handle:
        // Function items or non-capturing closures of differing IDs or GenericArgs.
        let (a_sig, b_sig) = {
            let is_capturing_closure = |ty: Ty<'tcx>| {
                if let &ty::Closure(closure_def_id, _args) = ty.kind() {
                    self.tcx.upvars_mentioned(closure_def_id.expect_local()).is_some()
                } else {
                    false
                }
            };
            if is_capturing_closure(prev_ty) || is_capturing_closure(new_ty) {
                (None, None)
            } else {
                match (prev_ty.kind(), new_ty.kind()) {
                    (ty::FnDef(..), ty::FnDef(..)) => {
                        // Don't reify if the function types have a LUB, i.e., they
                        // are the same function and their parameters have a LUB.
                        match self.commit_if_ok(|_| {
                            // We need to eagerly handle nested obligations due to lazy norm.
                            if self.next_trait_solver() {
                                let ocx = ObligationCtxt::new(self);
                                let value = ocx.lub(cause, self.param_env, prev_ty, new_ty)?;
                                if ocx.select_where_possible().is_empty() {
                                    Ok(InferOk {
                                        value,
                                        obligations: ocx.into_pending_obligations(),
                                    })
                                } else {
                                    Err(TypeError::Mismatch)
                                }
                            } else {
                                self.at(cause, self.param_env).lub(prev_ty, new_ty)
                            }
                        }) {
                            // We have a LUB of prev_ty and new_ty, just return it.
                            Ok(ok) => return Ok(self.register_infer_ok_obligations(ok)),
                            Err(_) => {
                                (Some(prev_ty.fn_sig(self.tcx)), Some(new_ty.fn_sig(self.tcx)))
                            }
                        }
                    }
                    (ty::Closure(_, args), ty::FnDef(..)) => {
                        let b_sig = new_ty.fn_sig(self.tcx);
                        let a_sig =
                            self.tcx.signature_unclosure(args.as_closure().sig(), b_sig.safety());
                        (Some(a_sig), Some(b_sig))
                    }
                    (ty::FnDef(..), ty::Closure(_, args)) => {
                        let a_sig = prev_ty.fn_sig(self.tcx);
                        let b_sig =
                            self.tcx.signature_unclosure(args.as_closure().sig(), a_sig.safety());
                        (Some(a_sig), Some(b_sig))
                    }
                    (ty::Closure(_, args_a), ty::Closure(_, args_b)) => (
                        Some(
                            self.tcx
                                .signature_unclosure(args_a.as_closure().sig(), hir::Safety::Safe),
                        ),
                        Some(
                            self.tcx
                                .signature_unclosure(args_b.as_closure().sig(), hir::Safety::Safe),
                        ),
                    ),
                    _ => (None, None),
                }
            }
        };
        if let (Some(a_sig), Some(b_sig)) = (a_sig, b_sig) {
            // The signature must match.
            let (a_sig, b_sig) = self.normalize(new.span, (a_sig, b_sig));
            let sig = self
                .at(cause, self.param_env)
                .lub(a_sig, b_sig)
                .map(|ok| self.register_infer_ok_obligations(ok))?;

            // Reify both sides and return the reified fn pointer type.
            let fn_ptr = Ty::new_fn_ptr(self.tcx, sig);
            let prev_adjustment = match prev_ty.kind() {
                ty::Closure(..) => {
                    Adjust::Pointer(PointerCoercion::ClosureFnPointer(a_sig.safety()))
                }
                ty::FnDef(..) => Adjust::Pointer(PointerCoercion::ReifyFnPointer),
                _ => span_bug!(cause.span, "should not try to coerce a {prev_ty} to a fn pointer"),
            };
            let next_adjustment = match new_ty.kind() {
                ty::Closure(..) => {
                    Adjust::Pointer(PointerCoercion::ClosureFnPointer(b_sig.safety()))
                }
                ty::FnDef(..) => Adjust::Pointer(PointerCoercion::ReifyFnPointer),
                _ => span_bug!(new.span, "should not try to coerce a {new_ty} to a fn pointer"),
            };
            for expr in exprs.iter().map(|e| e.as_coercion_site()) {
                self.apply_adjustments(
                    expr,
                    vec![Adjustment { kind: prev_adjustment.clone(), target: fn_ptr }],
                );
            }
            self.apply_adjustments(new, vec![Adjustment { kind: next_adjustment, target: fn_ptr }]);
            return Ok(fn_ptr);
        }

        // Configure a Coerce instance to compute the LUB.
        // We don't allow two-phase borrows on any autorefs this creates since we
        // probably aren't processing function arguments here and even if we were,
        // they're going to get autorefed again anyway and we can apply 2-phase borrows
        // at that time.
        //
        // NOTE: we set `coerce_never` to `true` here because coercion LUBs only
        // operate on values and not places, so a never coercion is valid.
        let mut coerce = Coerce::new(self, cause.clone(), AllowTwoPhase::No, true);
        coerce.use_lub = true;

        // First try to coerce the new expression to the type of the previous ones,
        // but only if the new expression has no coercion already applied to it.
        let mut first_error = None;
        if !self.typeck_results.borrow().adjustments().contains_key(new.hir_id) {
            let result = self.commit_if_ok(|_| coerce.coerce(new_ty, prev_ty));
            match result {
                Ok(ok) => {
                    let (adjustments, target) = self.register_infer_ok_obligations(ok);
                    self.apply_adjustments(new, adjustments);
                    debug!(
                        "coercion::try_find_coercion_lub: was able to coerce from new type {:?} to previous type {:?} ({:?})",
                        new_ty, prev_ty, target
                    );
                    return Ok(target);
                }
                Err(e) => first_error = Some(e),
            }
        }

        match self.commit_if_ok(|_| coerce.coerce(prev_ty, new_ty)) {
            Err(_) => {
                // Avoid giving strange errors on failed attempts.
                if let Some(e) = first_error {
                    Err(e)
                } else {
                    Err(self
                        .commit_if_ok(|_| self.at(cause, self.param_env).lub(prev_ty, new_ty))
                        .unwrap_err())
                }
            }
            Ok(ok) => {
                let (adjustments, target) = self.register_infer_ok_obligations(ok);
                for expr in exprs {
                    let expr = expr.as_coercion_site();
                    self.apply_adjustments(expr, adjustments.clone());
                }
                debug!(
                    "coercion::try_find_coercion_lub: was able to coerce previous type {:?} to new type {:?} ({:?})",
                    prev_ty, new_ty, target
                );
                Ok(target)
            }
        }
    }
}

/// Check whether `ty` can be coerced to `output_ty`.
/// Used from clippy.
pub fn can_coerce<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    body_id: LocalDefId,
    ty: Ty<'tcx>,
    output_ty: Ty<'tcx>,
) -> bool {
    let root_ctxt = crate::typeck_root_ctxt::TypeckRootCtxt::new(tcx, body_id);
    let fn_ctxt = FnCtxt::new(&root_ctxt, param_env, body_id);
    fn_ctxt.may_coerce(ty, output_ty)
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
/// ```ignore (illustrative)
/// let mut coerce = CoerceMany::new(expected_ty);
/// for expr in exprs {
///     let expr_ty = fcx.check_expr_with_expectation(expr, expected);
///     coerce.coerce(fcx, &cause, expr, expr_ty);
/// }
/// let final_ty = coerce.complete(fcx);
/// ```
pub(crate) struct CoerceMany<'tcx, 'exprs, E: AsCoercionSite> {
    expected_ty: Ty<'tcx>,
    final_ty: Option<Ty<'tcx>>,
    expressions: Expressions<'tcx, 'exprs, E>,
    pushed: usize,
}

/// The type of a `CoerceMany` that is storing up the expressions into
/// a buffer. We use this in `check/mod.rs` for things like `break`.
pub(crate) type DynamicCoerceMany<'tcx> = CoerceMany<'tcx, 'tcx, &'tcx hir::Expr<'tcx>>;

enum Expressions<'tcx, 'exprs, E: AsCoercionSite> {
    Dynamic(Vec<&'tcx hir::Expr<'tcx>>),
    UpFront(&'exprs [E]),
}

impl<'tcx, 'exprs, E: AsCoercionSite> CoerceMany<'tcx, 'exprs, E> {
    /// The usual case; collect the set of expressions dynamically.
    /// If the full set of coercion sites is known before hand,
    /// consider `with_coercion_sites()` instead to avoid allocation.
    pub(crate) fn new(expected_ty: Ty<'tcx>) -> Self {
        Self::make(expected_ty, Expressions::Dynamic(vec![]))
    }

    /// As an optimization, you can create a `CoerceMany` with a
    /// preexisting slice of expressions. In this case, you are
    /// expected to pass each element in the slice to `coerce(...)` in
    /// order. This is used with arrays in particular to avoid
    /// needlessly cloning the slice.
    pub(crate) fn with_coercion_sites(expected_ty: Ty<'tcx>, coercion_sites: &'exprs [E]) -> Self {
        Self::make(expected_ty, Expressions::UpFront(coercion_sites))
    }

    fn make(expected_ty: Ty<'tcx>, expressions: Expressions<'tcx, 'exprs, E>) -> Self {
        CoerceMany { expected_ty, final_ty: None, expressions, pushed: 0 }
    }

    /// Returns the "expected type" with which this coercion was
    /// constructed. This represents the "downward propagated" type
    /// that was given to us at the start of typing whatever construct
    /// we are typing (e.g., the match expression).
    ///
    /// Typically, this is used as the expected type when
    /// type-checking each of the alternative expressions whose types
    /// we are trying to merge.
    pub(crate) fn expected_ty(&self) -> Ty<'tcx> {
        self.expected_ty
    }

    /// Returns the current "merged type", representing our best-guess
    /// at the LUB of the expressions we've seen so far (if any). This
    /// isn't *final* until you call `self.complete()`, which will return
    /// the merged type.
    pub(crate) fn merged_ty(&self) -> Ty<'tcx> {
        self.final_ty.unwrap_or(self.expected_ty)
    }

    /// Indicates that the value generated by `expression`, which is
    /// of type `expression_ty`, is one of the possibilities that we
    /// could coerce from. This will record `expression`, and later
    /// calls to `coerce` may come back and add adjustments and things
    /// if necessary.
    pub(crate) fn coerce<'a>(
        &mut self,
        fcx: &FnCtxt<'a, 'tcx>,
        cause: &ObligationCause<'tcx>,
        expression: &'tcx hir::Expr<'tcx>,
        expression_ty: Ty<'tcx>,
    ) {
        self.coerce_inner(fcx, cause, Some(expression), expression_ty, |_| {}, false)
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
    pub(crate) fn coerce_forced_unit<'a>(
        &mut self,
        fcx: &FnCtxt<'a, 'tcx>,
        cause: &ObligationCause<'tcx>,
        augment_error: impl FnOnce(&mut Diag<'_>),
        label_unit_as_expected: bool,
    ) {
        self.coerce_inner(
            fcx,
            cause,
            None,
            fcx.tcx.types.unit,
            augment_error,
            label_unit_as_expected,
        )
    }

    /// The inner coercion "engine". If `expression` is `None`, this
    /// is a forced-unit case, and hence `expression_ty` must be
    /// `Nil`.
    #[instrument(skip(self, fcx, augment_error, label_expression_as_expected), level = "debug")]
    pub(crate) fn coerce_inner<'a>(
        &mut self,
        fcx: &FnCtxt<'a, 'tcx>,
        cause: &ObligationCause<'tcx>,
        expression: Option<&'tcx hir::Expr<'tcx>>,
        mut expression_ty: Ty<'tcx>,
        augment_error: impl FnOnce(&mut Diag<'_>),
        label_expression_as_expected: bool,
    ) {
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
        if let Err(guar) = (expression_ty, self.merged_ty()).error_reported() {
            self.final_ty = Some(Ty::new_error(fcx.tcx, guar));
            return;
        }

        let (expected, found) = if label_expression_as_expected {
            // In the case where this is a "forced unit", like
            // `break`, we want to call the `()` "expected"
            // since it is implied by the syntax.
            // (Note: not all force-units work this way.)"
            (expression_ty, self.merged_ty())
        } else {
            // Otherwise, the "expected" type for error
            // reporting is the current unification type,
            // which is basically the LUB of the expressions
            // we've seen so far (combined with the expected
            // type)
            (self.merged_ty(), expression_ty)
        };

        // Handle the actual type unification etc.
        let result = if let Some(expression) = expression {
            if self.pushed == 0 {
                // Special-case the first expression we are coercing.
                // To be honest, I'm not entirely sure why we do this.
                // We don't allow two-phase borrows, see comment in try_find_coercion_lub for why
                fcx.coerce(
                    expression,
                    expression_ty,
                    self.expected_ty,
                    AllowTwoPhase::No,
                    Some(cause.clone()),
                )
            } else {
                match self.expressions {
                    Expressions::Dynamic(ref exprs) => fcx.try_find_coercion_lub(
                        cause,
                        exprs,
                        self.merged_ty(),
                        expression,
                        expression_ty,
                    ),
                    Expressions::UpFront(coercion_sites) => fcx.try_find_coercion_lub(
                        cause,
                        &coercion_sites[0..self.pushed],
                        self.merged_ty(),
                        expression,
                        expression_ty,
                    ),
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
            // ()`. That is the case we are considering here. We take
            // a different path to get the right "expected, found"
            // message and so forth (and because we know that
            // `expression_ty` will be unit).
            //
            // Another example is `break` with no argument expression.
            assert!(expression_ty.is_unit(), "if let hack without unit type");
            fcx.at(cause, fcx.param_env)
                .eq(
                    // needed for tests/ui/type-alias-impl-trait/issue-65679-inst-opaque-ty-from-val-twice.rs
                    DefineOpaqueTypes::Yes,
                    expected,
                    found,
                )
                .map(|infer_ok| {
                    fcx.register_infer_ok_obligations(infer_ok);
                    expression_ty
                })
        };

        debug!(?result);
        match result {
            Ok(v) => {
                self.final_ty = Some(v);
                if let Some(e) = expression {
                    match self.expressions {
                        Expressions::Dynamic(ref mut buffer) => buffer.push(e),
                        Expressions::UpFront(coercion_sites) => {
                            // if the user gave us an array to validate, check that we got
                            // the next expression in the list, as expected
                            assert_eq!(
                                coercion_sites[self.pushed].as_coercion_site().hir_id,
                                e.hir_id
                            );
                        }
                    }
                    self.pushed += 1;
                }
            }
            Err(coercion_error) => {
                // Mark that we've failed to coerce the types here to suppress
                // any superfluous errors we might encounter while trying to
                // emit or provide suggestions on how to fix the initial error.
                fcx.set_tainted_by_errors(
                    fcx.dcx().span_delayed_bug(cause.span, "coercion error but no error emitted"),
                );
                let (expected, found) = fcx.resolve_vars_if_possible((expected, found));

                let mut err;
                let mut unsized_return = false;
                match *cause.code() {
                    ObligationCauseCode::ReturnNoExpression => {
                        err = struct_span_code_err!(
                            fcx.dcx(),
                            cause.span,
                            E0069,
                            "`return;` in a function whose return type is not `()`"
                        );
                        if let Some(value) = fcx.err_ctxt().ty_kind_suggestion(fcx.param_env, found)
                        {
                            err.span_suggestion_verbose(
                                cause.span.shrink_to_hi(),
                                "give the `return` a value of the expected type",
                                format!(" {value}"),
                                Applicability::HasPlaceholders,
                            );
                        }
                        err.span_label(cause.span, "return type is not `()`");
                    }
                    ObligationCauseCode::BlockTailExpression(blk_id, ..) => {
                        err = self.report_return_mismatched_types(
                            cause,
                            expected,
                            found,
                            coercion_error,
                            fcx,
                            blk_id,
                            expression,
                        );
                        if !fcx.tcx.features().unsized_locals() {
                            unsized_return = self.is_return_ty_definitely_unsized(fcx);
                        }
                    }
                    ObligationCauseCode::ReturnValue(return_expr_id) => {
                        err = self.report_return_mismatched_types(
                            cause,
                            expected,
                            found,
                            coercion_error,
                            fcx,
                            return_expr_id,
                            expression,
                        );
                        if !fcx.tcx.features().unsized_locals() {
                            unsized_return = self.is_return_ty_definitely_unsized(fcx);
                        }
                    }
                    ObligationCauseCode::MatchExpressionArm(box MatchExpressionArmCause {
                        arm_span,
                        arm_ty,
                        prior_arm_ty,
                        ref prior_non_diverging_arms,
                        tail_defines_return_position_impl_trait: Some(rpit_def_id),
                        ..
                    }) => {
                        err = fcx.err_ctxt().report_mismatched_types(
                            cause,
                            fcx.param_env,
                            expected,
                            found,
                            coercion_error,
                        );
                        // Check that we're actually in the second or later arm
                        if prior_non_diverging_arms.len() > 0 {
                            self.suggest_boxing_tail_for_return_position_impl_trait(
                                fcx,
                                &mut err,
                                rpit_def_id,
                                arm_ty,
                                prior_arm_ty,
                                prior_non_diverging_arms
                                    .iter()
                                    .chain(std::iter::once(&arm_span))
                                    .copied(),
                            );
                        }
                    }
                    ObligationCauseCode::IfExpression(box IfExpressionCause {
                        then_id,
                        else_id,
                        then_ty,
                        else_ty,
                        tail_defines_return_position_impl_trait: Some(rpit_def_id),
                        ..
                    }) => {
                        err = fcx.err_ctxt().report_mismatched_types(
                            cause,
                            fcx.param_env,
                            expected,
                            found,
                            coercion_error,
                        );
                        let then_span = fcx.find_block_span_from_hir_id(then_id);
                        let else_span = fcx.find_block_span_from_hir_id(else_id);
                        // don't suggest wrapping either blocks in `if .. {} else {}`
                        let is_empty_arm = |id| {
                            let hir::Node::Block(blk) = fcx.tcx.hir_node(id) else {
                                return false;
                            };
                            if blk.expr.is_some() || !blk.stmts.is_empty() {
                                return false;
                            }
                            let Some((_, hir::Node::Expr(expr))) =
                                fcx.tcx.hir_parent_iter(id).nth(1)
                            else {
                                return false;
                            };
                            matches!(expr.kind, hir::ExprKind::If(..))
                        };
                        if !is_empty_arm(then_id) && !is_empty_arm(else_id) {
                            self.suggest_boxing_tail_for_return_position_impl_trait(
                                fcx,
                                &mut err,
                                rpit_def_id,
                                then_ty,
                                else_ty,
                                [then_span, else_span].into_iter(),
                            );
                        }
                    }
                    _ => {
                        err = fcx.err_ctxt().report_mismatched_types(
                            cause,
                            fcx.param_env,
                            expected,
                            found,
                            coercion_error,
                        );
                    }
                }

                augment_error(&mut err);

                if let Some(expr) = expression {
                    if let hir::ExprKind::Loop(
                        _,
                        _,
                        loop_src @ (hir::LoopSource::While | hir::LoopSource::ForLoop),
                        _,
                    ) = expr.kind
                    {
                        let loop_type = if loop_src == hir::LoopSource::While {
                            "`while` loops"
                        } else {
                            "`for` loops"
                        };

                        err.note(format!("{loop_type} evaluate to unit type `()`"));
                    }

                    fcx.emit_coerce_suggestions(
                        &mut err,
                        expr,
                        found,
                        expected,
                        None,
                        Some(coercion_error),
                    );
                }

                let reported = err.emit_unless(unsized_return);

                self.final_ty = Some(Ty::new_error(fcx.tcx, reported));
            }
        }
    }

    fn suggest_boxing_tail_for_return_position_impl_trait(
        &self,
        fcx: &FnCtxt<'_, 'tcx>,
        err: &mut Diag<'_>,
        rpit_def_id: LocalDefId,
        a_ty: Ty<'tcx>,
        b_ty: Ty<'tcx>,
        arm_spans: impl Iterator<Item = Span>,
    ) {
        let compatible = |ty: Ty<'tcx>| {
            fcx.probe(|_| {
                let ocx = ObligationCtxt::new(fcx);
                ocx.register_obligations(
                    fcx.tcx.item_self_bounds(rpit_def_id).iter_identity().filter_map(|clause| {
                        let predicate = clause
                            .kind()
                            .map_bound(|clause| match clause {
                                ty::ClauseKind::Trait(trait_pred) => Some(ty::ClauseKind::Trait(
                                    trait_pred.with_self_ty(fcx.tcx, ty),
                                )),
                                ty::ClauseKind::Projection(proj_pred) => Some(
                                    ty::ClauseKind::Projection(proj_pred.with_self_ty(fcx.tcx, ty)),
                                ),
                                _ => None,
                            })
                            .transpose()?;
                        Some(Obligation::new(
                            fcx.tcx,
                            ObligationCause::dummy(),
                            fcx.param_env,
                            predicate,
                        ))
                    }),
                );
                ocx.select_where_possible().is_empty()
            })
        };

        if !compatible(a_ty) || !compatible(b_ty) {
            return;
        }

        let rpid_def_span = fcx.tcx.def_span(rpit_def_id);
        err.subdiagnostic(SuggestBoxingForReturnImplTrait::ChangeReturnType {
            start_sp: rpid_def_span.with_hi(rpid_def_span.lo() + BytePos(4)),
            end_sp: rpid_def_span.shrink_to_hi(),
        });

        let (starts, ends) =
            arm_spans.map(|span| (span.shrink_to_lo(), span.shrink_to_hi())).unzip();
        err.subdiagnostic(SuggestBoxingForReturnImplTrait::BoxReturnExpr { starts, ends });
    }

    fn report_return_mismatched_types<'infcx>(
        &self,
        cause: &ObligationCause<'tcx>,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
        ty_err: TypeError<'tcx>,
        fcx: &'infcx FnCtxt<'_, 'tcx>,
        block_or_return_id: hir::HirId,
        expression: Option<&'tcx hir::Expr<'tcx>>,
    ) -> Diag<'infcx> {
        let mut err =
            fcx.err_ctxt().report_mismatched_types(cause, fcx.param_env, expected, found, ty_err);

        let due_to_block = matches!(fcx.tcx.hir_node(block_or_return_id), hir::Node::Block(..));
        let parent = fcx.tcx.parent_hir_node(block_or_return_id);
        if let Some(expr) = expression
            && let hir::Node::Expr(&hir::Expr {
                kind: hir::ExprKind::Closure(&hir::Closure { body, .. }),
                ..
            }) = parent
        {
            let needs_block =
                !matches!(fcx.tcx.hir_body(body).value.kind, hir::ExprKind::Block(..));
            fcx.suggest_missing_semicolon(&mut err, expr, expected, needs_block, true);
        }
        // Verify that this is a tail expression of a function, otherwise the
        // label pointing out the cause for the type coercion will be wrong
        // as prior return coercions would not be relevant (#57664).
        if let Some(expr) = expression
            && due_to_block
        {
            fcx.suggest_missing_semicolon(&mut err, expr, expected, false, false);
            let pointing_at_return_type = fcx.suggest_mismatched_types_on_tail(
                &mut err,
                expr,
                expected,
                found,
                block_or_return_id,
            );
            if let Some(cond_expr) = fcx.tcx.hir_get_if_cause(expr.hir_id)
                && expected.is_unit()
                && !pointing_at_return_type
                // If the block is from an external macro or try (`?`) desugaring, then
                // do not suggest adding a semicolon, because there's nowhere to put it.
                // See issues #81943 and #87051.
                && matches!(
                    cond_expr.span.desugaring_kind(),
                    None | Some(DesugaringKind::WhileLoop)
                )
                && !cond_expr.span.in_external_macro(fcx.tcx.sess.source_map())
                && !matches!(
                    cond_expr.kind,
                    hir::ExprKind::Match(.., hir::MatchSource::TryDesugar(_))
                )
            {
                err.span_label(cond_expr.span, "expected this to be `()`");
                if expr.can_have_side_effects() {
                    fcx.suggest_semicolon_at_end(cond_expr.span, &mut err);
                }
            }
        };

        // If this is due to an explicit `return`, suggest adding a return type.
        if let Some((fn_id, fn_decl)) = fcx.get_fn_decl(block_or_return_id)
            && !due_to_block
        {
            fcx.suggest_missing_return_type(&mut err, fn_decl, expected, found, fn_id);
        }

        // If this is due to a block, then maybe we forgot a `return`/`break`.
        if due_to_block
            && let Some(expr) = expression
            && let Some(parent_fn_decl) =
                fcx.tcx.hir_fn_decl_by_hir_id(fcx.tcx.local_def_id_to_hir_id(fcx.body_id))
        {
            fcx.suggest_missing_break_or_return_expr(
                &mut err,
                expr,
                parent_fn_decl,
                expected,
                found,
                block_or_return_id,
                fcx.body_id,
            );
        }

        let ret_coercion_span = fcx.ret_coercion_span.get();

        if let Some(sp) = ret_coercion_span
            // If the closure has an explicit return type annotation, or if
            // the closure's return type has been inferred from outside
            // requirements (such as an Fn* trait bound), then a type error
            // may occur at the first return expression we see in the closure
            // (if it conflicts with the declared return type). Skip adding a
            // note in this case, since it would be incorrect.
            && let Some(fn_sig) = fcx.body_fn_sig()
            && fn_sig.output().is_ty_var()
        {
            err.span_note(sp, format!("return type inferred to be `{expected}` here"));
        }

        err
    }

    /// Checks whether the return type is unsized via an obligation, which makes
    /// sure we consider `dyn Trait: Sized` where clauses, which are trivially
    /// false but technically valid for typeck.
    fn is_return_ty_definitely_unsized(&self, fcx: &FnCtxt<'_, 'tcx>) -> bool {
        if let Some(sig) = fcx.body_fn_sig() {
            !fcx.predicate_may_hold(&Obligation::new(
                fcx.tcx,
                ObligationCause::dummy(),
                fcx.param_env,
                ty::TraitRef::new(
                    fcx.tcx,
                    fcx.tcx.require_lang_item(hir::LangItem::Sized, DUMMY_SP),
                    [sig.output()],
                ),
            ))
        } else {
            false
        }
    }

    pub(crate) fn complete<'a>(self, fcx: &FnCtxt<'a, 'tcx>) -> Ty<'tcx> {
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
pub(crate) trait AsCoercionSite {
    fn as_coercion_site(&self) -> &hir::Expr<'_>;
}

impl AsCoercionSite for hir::Expr<'_> {
    fn as_coercion_site(&self) -> &hir::Expr<'_> {
        self
    }
}

impl<'a, T> AsCoercionSite for &'a T
where
    T: AsCoercionSite,
{
    fn as_coercion_site(&self) -> &hir::Expr<'_> {
        (**self).as_coercion_site()
    }
}

impl AsCoercionSite for ! {
    fn as_coercion_site(&self) -> &hir::Expr<'_> {
        *self
    }
}

impl AsCoercionSite for hir::Arm<'_> {
    fn as_coercion_site(&self) -> &hir::Expr<'_> {
        self.body
    }
}
