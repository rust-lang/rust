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

use rustc_hir::def_id::DefId;
use rustc_hir::{self as hir};
use rustc_hir_analysis::hir_ty_lowering::HirTyLowerer;
use rustc_infer::infer::relate::RelateResult;
use rustc_infer::infer::{
    DefineOpaqueTypes, InferCtxt, InferOk, InferResult, RegionVariableOrigin,
};
use rustc_infer::traits::{Obligation, PredicateObligation, PredicateObligations, SelectionError};
use rustc_middle::ty::adjustment::{
    Adjust, Adjustment, AllowTwoPhase, AutoBorrow, AutoBorrowMutability, DerefAdjustKind, PointerCoercion,
};
use rustc_middle::ty::error::TypeError;
use rustc_middle::ty::relate::{self, Relate, TypeRelation};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_trait_selection::solve::Goal;
use rustc_trait_selection::solve::inspect::InferCtxtProofTreeExt;
use rustc_trait_selection::traits::{
    self, ImplSource, NormalizeExt, ObligationCause, ObligationCauseCode, ObligationCtxt,
};
use smallvec::{SmallVec, smallvec};
use tracing::{debug, instrument};

use crate::FnCtxt;
use crate::coercion::CoerceVisitor;

pub(crate) struct CoerceGuidance<'a, 'tcx> {
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

impl<'a, 'tcx> Deref for CoerceGuidance<'a, 'tcx> {
    type Target = FnCtxt<'a, 'tcx>;
    fn deref(&self) -> &Self::Target {
        self.fcx
    }
}

/// Coercing a mutable reference to an immutable works, while
/// coercing `&T` to `&mut T` should be forbidden.
fn coerce_mutbls<'tcx>(
    from_mutbl: hir::Mutability,
    to_mutbl: hir::Mutability,
) -> RelateResult<'tcx, ()> {
    if from_mutbl >= to_mutbl { Ok(()) } else { Err(TypeError::Mutability) }
}

#[derive(Debug)]
enum ForceLeakCheck {
    Yes,
    No,
}

type CoerceGuidanceResult<'tcx> = InferResult<'tcx, ()>;

impl<'f, 'tcx> CoerceGuidance<'f, 'tcx> {
    pub(crate) fn new(
        fcx: &'f FnCtxt<'f, 'tcx>,
        cause: ObligationCause<'tcx>,
        allow_two_phase: AllowTwoPhase,
        coerce_never: bool,
    ) -> Self {
        CoerceGuidance { fcx, cause, allow_two_phase, use_lub: false, coerce_never }
    }

    #[tracing::instrument(skip(self), ret)]
    fn unify(
        &self,
        a: Ty<'tcx>,
        b: Ty<'tcx>,
        leak_check: ForceLeakCheck,
    ) -> CoerceGuidanceResult<'tcx> {
        debug!("unify(a: {:?}, b: {:?}, use_lub: {})", a, b, self.use_lub);
        let guidance = TypeGuidance { fcx: self.fcx }.relate(a, b)?;
        let at = self.at(&self.cause, self.fcx.param_env);
        at.sup(DefineOpaqueTypes::Yes, b, a)
    }

    #[instrument(skip(self), ret)]
    pub(crate) fn do_guidance(
        &self,
        source: Ty<'tcx>,
        target: Ty<'tcx>,
    ) -> CoerceGuidanceResult<'tcx> {
        // First, remove any resolved type variables (at the top level, at least):
        let source = self.shallow_resolve(source);
        let target = self.shallow_resolve(target);
        debug!("CoerceGuidance.tys({:?} => {:?})", source, target);

        // Coercing from `!` to any type is allowed, but gives no guidance
        if source.is_never() {
            if self.coerce_never {
                return Ok(InferOk { value: (), obligations: PredicateObligations::new() });
            } else {
                // Otherwise the only coercion we can do is unification.
                return self.unify(source, target, ForceLeakCheck::No);
            }
        }

        // Coercing *from* an unresolved inference variable means that
        // we have no information about the source type. This will always
        // ultimately fall back to some form of subtyping.
        if source.is_ty_var() {
            return Ok(InferOk { value: (), obligations: PredicateObligations::new() });
        }

        // Consider coercing the subtype to a DST
        //
        // NOTE: this is wrapped in a `commit_if_ok` because it creates
        // a "spurious" type variable, and we don't want to have that
        // type variable in memory if the coercion fails.
        let unsize = self.commit_if_ok(|_| self.coerce_unsized(source, target));
        match unsize {
            Ok(_) => {
                debug!("coerce: unsize successful");
                return unsize;
            }
            Err(error) => {
                debug!(?error, "coerce: unsize failed");
            }
        }

        // Examine the target type and consider type-specific coercions, such
        // as auto-borrowing, coercing pointer mutability, or pin-ergonomics.
        match *target.kind() {
            ty::RawPtr(_, b_mutbl) => {
                return self.coerce_to_raw_ptr(source, target, b_mutbl);
            }
            ty::Ref(r_b, _, mutbl_b) => {
                return self.coerce_to_ref(source, target, r_b, mutbl_b);
            }
            ty::Adt(pin, _)
                if self.tcx.features().pin_ergonomics()
                    && self.tcx.is_lang_item(pin.did(), hir::LangItem::Pin) =>
            {
                let pin_coerce = self.commit_if_ok(|_| self.coerce_to_pin_ref(source, target));
                if pin_coerce.is_ok() {
                    return pin_coerce;
                }
            }
            _ => {}
        }

        match *source.kind() {
            ty::FnDef(..) => {
                // Function items are coercible to any closure
                // type; function pointers are not (that would
                // require double indirection).
                // Additionally, we permit coercion of function
                // items to drop the unsafe qualifier.
                self.coerce_from_fn_item(source, target)
            }
            ty::FnPtr(a_sig_tys, a_hdr) => {
                // We permit coercion of fn pointers to drop the
                // unsafe qualifier.
                self.coerce_from_fn_pointer(source, a_sig_tys.with(a_hdr), target)
            }
            ty::Closure(..) => {
                // Non-capturing closures are coercible to
                // function pointers or unsafe function pointers.
                // It cannot convert closures that require unsafe.
                self.coerce_closure_to_fn(source, target)
            }
            _ => {
                let source = self.fcx.try_structurally_resolve_type(rustc_span::DUMMY_SP, source);
                let target = if self.fcx.next_trait_solver() {
                    self.fcx.try_structurally_resolve_type(rustc_span::DUMMY_SP, target)
                } else {
                    target
                };

                // Otherwise, just use unification rules.
                self.unify(source, target, ForceLeakCheck::No)
            }
        }
    }

    /// Handles coercing some arbitrary type `a` to some reference (`b`). This
    /// handles a few cases:
    /// - Introducing reborrows to give more flexible lifetimes
    /// - Deref coercions to allow `&T` to coerce to `&T::Target`
    /// - Coercing mutable references to immutable references
    /// These coercions can be freely intermixed, for example we are able to
    /// coerce `&mut T` to `&mut T::Target`.
    fn coerce_to_ref(
        &self,
        a: Ty<'tcx>,
        b: Ty<'tcx>,
        r_b: ty::Region<'tcx>,
        mutbl_b: hir::Mutability,
    ) -> CoerceGuidanceResult<'tcx> {
        debug!("coerce_to_ref(a={:?}, b={:?})", a, b);
        debug_assert!(self.shallow_resolve(a) == a);
        debug_assert!(self.shallow_resolve(b) == b);

        let (r_a, mt_a) = match *a.kind() {
            ty::Ref(r_a, ty, mutbl) => {
                coerce_mutbls(mutbl, mutbl_b)?;
                (r_a, ty::TypeAndMut { ty, mutbl })
            }
            _ => return self.unify(a, b, ForceLeakCheck::No),
        };

        // Look at each step in the `Deref` chain and check if
        // any of the autoref'd `Target` types unify with the
        // coercion target.
        //
        // For example when coercing from `&mut Vec<T>` to `&M [T]` we
        // have three deref steps:
        // 1. `&mut Vec<T>`, skip autoref
        // 2. `Vec<T>`, autoref'd ty: `&M Vec<T>`
        //     - `&M Vec<T>` does not unify with `&M [T]`
        // 3. `[T]`, autoref'd ty: `&M [T]`
        //     - `&M [T]` does unify with `&M [T]`
        let mut first_error = None;
        let mut r_borrow_var = None;
        let mut autoderef = self.autoderef(self.cause.span, a);
        let found = autoderef.by_ref().find_map(|(deref_ty, autoderefs)| {
            if autoderefs == 0 {
                // Don't autoref the first step as otherwise we'd allow
                // coercing `&T` to `&&T`.
                return None;
            }

            // The logic here really shouldn't exist. We don't care about free
            // lifetimes during HIR typeck. Unfortunately later parts of this
            // function rely on structural identity of the autoref'd deref'd ty.
            //
            // This means that what region we use here actually impacts whether
            // we emit a reborrow coercion or not which can affect diagnostics
            // and capture analysis (which in turn affects borrowck).
            let r = if !self.use_lub {
                r_b
            } else if autoderefs == 1 {
                r_a
            } else {
                if r_borrow_var.is_none() {
                    // create var lazily, at most once
                    let coercion = RegionVariableOrigin::Coercion(self.cause.span);
                    let r = self.next_region_var(coercion);
                    r_borrow_var = Some(r);
                }
                r_borrow_var.unwrap()
            };

            let autorefd_deref_ty = Ty::new_ref(self.tcx, r, deref_ty, mutbl_b);

            // Note that we unify the autoref'd `Target` type with `b` rather than
            // the `Target` type with the pointee of `b`. This is necessary
            // to properly account for the differing variances of the pointees
            // of `&` vs `&mut` references.
            match self.unify(autorefd_deref_ty, b, ForceLeakCheck::No) {
                Ok(ok) => Some(ok),
                Err(err) => {
                    if first_error.is_none() {
                        first_error = Some(err);
                    }
                    None
                }
            }
        });

        // Extract type or return an error. We return the first error
        // we got, which should be from relating the "base" type
        // (e.g., in example above, the failure from relating `Vec<T>`
        // to the target type), since that should be the least
        // confusing.
        let Some(InferOk { value: (), mut obligations }) = found else {
            if let Some(first_error) = first_error {
                debug!("coerce_to_ref: failed with err = {:?}", first_error);
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

        Ok(InferOk { value: (), obligations })
    }

    /// Performs [unsized coercion] by emulating a fulfillment loop on a
    /// `CoerceUnsized` goal until all `CoerceUnsized` and `Unsize` goals
    /// are successfully selected.
    ///
    /// [unsized coercion](https://doc.rust-lang.org/reference/type-coercions.html#unsized-coercions)
    #[instrument(skip(self), level = "debug")]
    fn coerce_unsized(&self, source: Ty<'tcx>, target: Ty<'tcx>) -> CoerceGuidanceResult<'tcx> {
        debug!(?source, ?target);
        debug_assert!(self.shallow_resolve(source) == source);
        debug_assert!(self.shallow_resolve(target) == target);

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

        // This is an optimization because coercion is one of the most common
        // operations that we do in typeck, since it happens at every assignment
        // and call arg (among other positions).
        //
        // These targets are known to never be RHS in `LHS: CoerceUnsized<RHS>`.
        // That's because these are built-in types for which a core-provided impl
        // doesn't exist, and for which a user-written impl is invalid.
        //
        // This is technically incomplete when users write impossible bounds like
        // `where T: CoerceUnsized<usize>`, for example, but that trait is unstable
        // and coercion is allowed to be incomplete. The only case where this matters
        // is impossible bounds.
        //
        // Note that some of these types implement `LHS: Unsize<RHS>`, but they
        // do not implement *`CoerceUnsized`* which is the root obligation of the
        // check below.
        match target.kind() {
            ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Infer(ty::IntVar(_) | ty::FloatVar(_))
            | ty::Str
            | ty::Array(_, _)
            | ty::Slice(_)
            | ty::FnDef(_, _)
            | ty::FnPtr(_, _)
            | ty::Dynamic(_, _)
            | ty::Closure(_, _)
            | ty::CoroutineClosure(_, _)
            | ty::Coroutine(_, _)
            | ty::CoroutineWitness(_, _)
            | ty::Never
            | ty::Tuple(_) => return Err(TypeError::Mismatch),
            _ => {}
        }
        // `&str: CoerceUnsized<&str>` does not hold but is encountered frequently
        // so we fast path bail out here
        if let ty::Ref(_, source_pointee, ty::Mutability::Not) = *source.kind()
            && source_pointee.is_str()
            && let ty::Ref(_, target_pointee, ty::Mutability::Not) = *target.kind()
            && target_pointee.is_str()
        {
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

                let coercion = RegionVariableOrigin::Coercion(self.cause.span);
                let r_borrow = self.next_region_var(coercion);

                // We don't allow two-phase borrows here, at least for initial
                // implementation. If it happens that this coercion is a function argument,
                // the reborrow in coerce_borrowed_ptr will pick it up.
                let mutbl = AutoBorrowMutability::new(mutbl_b, AllowTwoPhase::No);

                Some((
                    Adjustment { kind: Adjust::Deref(DerefAdjustKind::Builtin), target: ty_a },
                    Adjustment {
                        kind: Adjust::Borrow(AutoBorrow::Ref(mutbl)),
                        target: Ty::new_ref(self.tcx, r_borrow, ty_a, mutbl_b),
                    },
                ))
            }
            (&ty::Ref(_, ty_a, mt_a), &ty::RawPtr(_, mt_b)) => {
                coerce_mutbls(mt_a, mt_b)?;

                Some((
                    Adjustment { kind: Adjust::Deref(DerefAdjustKind::Builtin), target: ty_a },
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

        let mut coercion = self.unify(coerce_target, target, ForceLeakCheck::No)?;

        // Create an obligation for `Source: CoerceUnsized<Target>`.
        let cause = self.cause(self.cause.span, ObligationCauseCode::Coercion { source, target });
        let pred = ty::TraitRef::new(self.tcx, coerce_unsized_did, [coerce_source, coerce_target]);
        let obligation = Obligation::new(self.tcx, cause, self.fcx.param_env, pred);

        if self.next_trait_solver() {
            if self
                .infcx
                .visit_proof_tree(
                    Goal::new(self.tcx, self.param_env, pred),
                    &mut CoerceVisitor { fcx: self.fcx, span: self.cause.span, errored: false },
                )
                .is_break()
            {
                return Err(TypeError::Mismatch);
            }
        } else {
            self.coerce_unsized_old_solver(obligation, coerce_unsized_did, unsize_did)?;
        }

        Ok(coercion)
    }

    fn coerce_unsized_old_solver(
        &self,
        obligation: Obligation<'tcx, ty::Predicate<'tcx>>,
        coerce_unsized_did: DefId,
        unsize_did: DefId,
    ) -> Result<(), TypeError<'tcx>> {
        let mut selcx = traits::SelectionContext::new(self);
        // Use a FIFO queue for this custom fulfillment procedure.
        //
        // A Vec (or SmallVec) is not a natural choice for a queue. However,
        // this code path is hot, and this queue usually has a max length of 1
        // and almost never more than 3. By using a SmallVec we avoid an
        // allocation, at the (very small) cost of (occasionally) having to
        // shift subsequent elements down when removing the front element.
        let mut queue: SmallVec<[PredicateObligation<'tcx>; 4]> = smallvec![obligation];

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
                    if !ocx.try_evaluate_obligations().is_empty() {
                        return Err(TypeError::Mismatch);
                    }
                    continue;
                }
                _ => {
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
                Err(SelectionError::Unimplemented) => {
                    debug!("coerce_unsized: early return - can't prove obligation");
                    return Err(TypeError::Mismatch);
                }

                Err(SelectionError::TraitDynIncompatible(_)) => {
                    // Dyn compatibility errors in coercion will *always* be due to the
                    // fact that the RHS of the coercion is a non-dyn compatible `dyn Trait`
                    // written in source somewhere (otherwise we will never have lowered
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
                Ok(Some(ImplSource::UserDefined(impl_source))) => {
                    queue.extend(impl_source.nested);
                    // Certain incoherent `CoerceUnsized` implementations may cause ICEs,
                    // so check the impl's validity. Taint the body so that we don't try
                    // to evaluate these invalid coercions in CTFE. We only need to do this
                    // for local impls, since upstream impls should be valid.
                    if impl_source.impl_def_id.is_local()
                        && let Err(guar) =
                            self.tcx.ensure_ok().coerce_unsized_info(impl_source.impl_def_id)
                    {
                        self.fcx.set_tainted_by_errors(guar);
                    }
                }
                Ok(Some(impl_source)) => queue.extend(impl_source.nested_obligations()),
            }
        }

        Ok(())
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
    fn coerce_to_pin_ref(&self, a: Ty<'tcx>, b: Ty<'tcx>) -> CoerceGuidanceResult<'tcx> {
        debug_assert!(self.shallow_resolve(a) == a);
        debug_assert!(self.shallow_resolve(b) == b);

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
        self.unify(a, b, ForceLeakCheck::No)
    }

    fn coerce_from_fn_pointer(
        &self,
        a: Ty<'tcx>,
        a_sig: ty::PolyFnSig<'tcx>,
        b: Ty<'tcx>,
    ) -> CoerceGuidanceResult<'tcx> {
        debug!(?a_sig, ?b, "coerce_from_fn_pointer");
        debug_assert!(self.shallow_resolve(b) == b);

        match b.kind() {
            ty::FnPtr(_, b_hdr) if a_sig.safety().is_safe() && b_hdr.safety.is_unsafe() => {
                let a = self.tcx.safe_to_unsafe_fn_ty(a_sig);
                self.unify(a, b, ForceLeakCheck::Yes)
            }
            _ => self.unify(a, b, ForceLeakCheck::Yes),
        }
    }

    fn coerce_from_fn_item(&self, a: Ty<'tcx>, b: Ty<'tcx>) -> CoerceGuidanceResult<'tcx> {
        debug!("coerce_from_fn_item(a={:?}, b={:?})", a, b);
        debug_assert!(self.shallow_resolve(a) == a);
        debug_assert!(self.shallow_resolve(b) == b);

        match b.kind() {
            ty::FnPtr(_, b_hdr) => {
                let a_sig = self.sig_for_fn_def_coercion(a, Some(b_hdr.safety))?;

                let InferOk { value: a_sig, obligations: _ } =
                    self.at(&self.cause, self.param_env).normalize(a_sig);
                let a = Ty::new_fn_ptr(self.tcx, a_sig);

                self.unify(a, b, ForceLeakCheck::Yes)
            }
            _ => self.unify(a, b, ForceLeakCheck::No),
        }
    }

    /// Attempts to coerce from a closure to a function pointer. Fails
    /// if the closure has any upvars.
    fn coerce_closure_to_fn(&self, a: Ty<'tcx>, b: Ty<'tcx>) -> CoerceGuidanceResult<'tcx> {
        debug_assert!(self.shallow_resolve(a) == a);
        debug_assert!(self.shallow_resolve(b) == b);

        match b.kind() {
            ty::FnPtr(_, hdr) => {
                let terr = TypeError::Sorts(ty::error::ExpectedFound::new(a, b));
                let closure_sig = self.sig_for_closure_coercion(a, Some(hdr.safety), terr)?;
                let pointer_ty = Ty::new_fn_ptr(self.tcx, closure_sig);
                debug!("coerce_closure_to_fn(a={:?}, b={:?}, pty={:?})", a, b, pointer_ty);

                self.unify(pointer_ty, b, ForceLeakCheck::No)
            }
            _ => self.unify(a, b, ForceLeakCheck::No),
        }
    }

    fn coerce_to_raw_ptr(
        &self,
        a: Ty<'tcx>,
        b: Ty<'tcx>,
        mutbl_b: hir::Mutability,
    ) -> CoerceGuidanceResult<'tcx> {
        debug!("coerce_to_raw_ptr(a={:?}, b={:?})", a, b);
        debug_assert!(self.shallow_resolve(a) == a);
        debug_assert!(self.shallow_resolve(b) == b);

        let (is_ref, mt_a) = match *a.kind() {
            ty::Ref(_, ty, mutbl) => (true, ty::TypeAndMut { ty, mutbl }),
            ty::RawPtr(ty, mutbl) => (false, ty::TypeAndMut { ty, mutbl }),
            _ => return self.unify(a, b, ForceLeakCheck::No),
        };
        coerce_mutbls(mt_a.mutbl, mutbl_b)?;

        // Check that the types which they point at are compatible.
        let a_raw = Ty::new_ptr(self.tcx, mt_a.ty, mutbl_b);
        // Although references and raw ptrs have the same
        // representation, we still register an Adjust::DerefRef so that
        // regionck knows that the region for `a` must be valid here.
        if is_ref {
            self.unify(a_raw, b, ForceLeakCheck::No)
        } else if mt_a.mutbl != mutbl_b {
            self.unify(a_raw, b, ForceLeakCheck::No)
        } else {
            self.unify(a_raw, b, ForceLeakCheck::No)
        }
    }
}

struct TypeGuidance<'infcx, 'tcx> {
    fcx: &'infcx FnCtxt<'infcx, 'tcx>,
}

impl<'tcx> TypeRelation<TyCtxt<'tcx>> for TypeGuidance<'_, 'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.fcx.tcx
    }

    fn relate_ty_args(
        &mut self,
        a_ty: Ty<'tcx>,
        _: Ty<'tcx>,
        _: DefId,
        a_args: ty::GenericArgsRef<'tcx>,
        b_args: ty::GenericArgsRef<'tcx>,
        _: impl FnOnce(ty::GenericArgsRef<'tcx>) -> Ty<'tcx>,
    ) -> RelateResult<'tcx, Ty<'tcx>> {
        relate::relate_args_invariantly(self, a_args, b_args)?;
        Ok(a_ty)
    }

    fn relate_with_variance<T: Relate<TyCtxt<'tcx>>>(
        &mut self,
        _: ty::Variance,
        _: ty::VarianceDiagInfo<TyCtxt<'tcx>>,
        a: T,
        b: T,
    ) -> RelateResult<'tcx, T> {
        self.relate(a, b)
    }

    #[instrument(skip(self), level = "trace")]
    fn regions(
        &mut self,
        a: ty::Region<'tcx>,
        _b: ty::Region<'tcx>,
    ) -> RelateResult<'tcx, ty::Region<'tcx>> {
        Ok(a)
    }

    #[instrument(skip(self), level = "trace")]
    fn tys(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        if a == b {
            return Ok(a);
        }

        match (a.kind(), b.kind()) {
            (_, ty::Infer(_)) => Ok(a),
            (ty::Infer(ty::InferTy::TyVar(a_vid)), _) => {
                tracing::debug!(?b);
                let a_gen = self
                    .fcx
                    .generalize(
                        rustc_span::DUMMY_SP,
                        rustc_infer::infer::StructurallyRelateAliases::Yes,
                        *a_vid,
                        ty::Variance::Invariant,
                        b,
                    )?
                    .value_may_be_infer;
                tracing::debug!(?a_gen);
                if a_gen.is_ty_var() {
                    Ok(a)
                } else {
                    self.fcx.inner.borrow_mut().type_variables().instantiate(*a_vid, a_gen);
                    relate::structurally_relate_tys(self, a_gen, b)
                }
            }
            (ty::Infer(_), _) => Ok(b),

            _ => relate::structurally_relate_tys(self, a, b),
        }
    }

    #[instrument(skip(self), level = "trace")]
    fn consts(
        &mut self,
        a: ty::Const<'tcx>,
        b: ty::Const<'tcx>,
    ) -> RelateResult<'tcx, ty::Const<'tcx>> {
        if a == b {
            return Ok(a);
        }

        match (a.kind(), b.kind()) {
            (_, ty::ConstKind::Infer(_)) => Ok(a),
            (ty::ConstKind::Infer(_), _) => Ok(b),
            _ => relate::structurally_relate_consts(self, a, b),
        }
    }

    fn binders<T>(
        &mut self,
        a: ty::Binder<'tcx, T>,
        b: ty::Binder<'tcx, T>,
    ) -> RelateResult<'tcx, ty::Binder<'tcx, T>>
    where
        T: Relate<TyCtxt<'tcx>>,
    {
        Ok(a.rebind(self.relate(a.skip_binder(), b.skip_binder())?))
    }
}
