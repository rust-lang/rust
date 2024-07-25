//! Candidate assembly.
//!
//! The selection process begins by examining all in-scope impls,
//! caller obligations, and so forth and assembling a list of
//! candidates. See the [rustc dev guide] for more details.
//!
//! [rustc dev guide]:https://rustc-dev-guide.rust-lang.org/traits/resolution.html#candidate-assembly

use std::ops::ControlFlow;

use hir::def_id::DefId;
use hir::LangItem;
use rustc_data_structures::fx::{FxHashSet, FxIndexSet};
use rustc_hir as hir;
use rustc_infer::traits::ObligationCause;
use rustc_infer::traits::{Obligation, PolyTraitObligation, SelectionError};
use rustc_middle::ty::fast_reject::{DeepRejectCtxt, TreatParams};
use rustc_middle::ty::{self, ToPolyTraitRef, Ty, TypeVisitableExt};
use rustc_middle::{bug, span_bug};

use crate::traits;
use crate::traits::query::evaluate_obligation::InferCtxtExt;
use crate::traits::util;

use super::BuiltinImplConditions;
use super::SelectionCandidate::*;
use super::{SelectionCandidateSet, SelectionContext, TraitObligationStack};

impl<'cx, 'tcx> SelectionContext<'cx, 'tcx> {
    #[instrument(skip(self, stack), level = "debug")]
    pub(super) fn assemble_candidates<'o>(
        &mut self,
        stack: &TraitObligationStack<'o, 'tcx>,
    ) -> Result<SelectionCandidateSet<'tcx>, SelectionError<'tcx>> {
        let TraitObligationStack { obligation, .. } = *stack;
        let obligation = &Obligation {
            param_env: obligation.param_env,
            cause: obligation.cause.clone(),
            recursion_depth: obligation.recursion_depth,
            predicate: self.infcx.resolve_vars_if_possible(obligation.predicate),
        };

        if obligation.predicate.skip_binder().self_ty().is_ty_var() {
            debug!(ty = ?obligation.predicate.skip_binder().self_ty(), "ambiguous inference var or opaque type");
            // Self is a type variable (e.g., `_: AsRef<str>`).
            //
            // This is somewhat problematic, as the current scheme can't really
            // handle it turning to be a projection. This does end up as truly
            // ambiguous in most cases anyway.
            //
            // Take the fast path out - this also improves
            // performance by preventing assemble_candidates_from_impls from
            // matching every impl for this trait.
            return Ok(SelectionCandidateSet { vec: vec![], ambiguous: true });
        }

        let mut candidates = SelectionCandidateSet { vec: Vec::new(), ambiguous: false };

        // Negative trait predicates have different rules than positive trait predicates.
        if obligation.polarity() == ty::PredicatePolarity::Negative {
            self.assemble_candidates_for_trait_alias(obligation, &mut candidates);
            self.assemble_candidates_from_impls(obligation, &mut candidates);
            self.assemble_candidates_from_caller_bounds(stack, &mut candidates)?;
        } else {
            self.assemble_candidates_for_trait_alias(obligation, &mut candidates);

            // Other bounds. Consider both in-scope bounds from fn decl
            // and applicable impls. There is a certain set of precedence rules here.
            let def_id = obligation.predicate.def_id();
            let tcx = self.tcx();

            if tcx.is_lang_item(def_id, LangItem::Copy) {
                debug!(obligation_self_ty = ?obligation.predicate.skip_binder().self_ty());

                // User-defined copy impls are permitted, but only for
                // structs and enums.
                self.assemble_candidates_from_impls(obligation, &mut candidates);

                // For other types, we'll use the builtin rules.
                let copy_conditions = self.copy_clone_conditions(obligation);
                self.assemble_builtin_bound_candidates(copy_conditions, &mut candidates);
            } else if tcx.is_lang_item(def_id, LangItem::DiscriminantKind) {
                // `DiscriminantKind` is automatically implemented for every type.
                candidates.vec.push(BuiltinCandidate { has_nested: false });
            } else if tcx.is_lang_item(def_id, LangItem::AsyncDestruct) {
                // `AsyncDestruct` is automatically implemented for every type.
                candidates.vec.push(BuiltinCandidate { has_nested: false });
            } else if tcx.is_lang_item(def_id, LangItem::PointeeTrait) {
                // `Pointee` is automatically implemented for every type.
                candidates.vec.push(BuiltinCandidate { has_nested: false });
            } else if tcx.is_lang_item(def_id, LangItem::Sized) {
                // Sized is never implementable by end-users, it is
                // always automatically computed.

                // FIXME: Consider moving this check to the top level as it
                // may also be useful for predicates other than `Sized`
                // Error type cannot possibly implement `Sized` (fixes #123154)
                if let Err(e) = obligation.predicate.skip_binder().self_ty().error_reported() {
                    return Err(SelectionError::Overflow(e.into()));
                }

                let sized_conditions = self.sized_conditions(obligation);
                self.assemble_builtin_bound_candidates(sized_conditions, &mut candidates);
            } else if tcx.is_lang_item(def_id, LangItem::Unsize) {
                self.assemble_candidates_for_unsizing(obligation, &mut candidates);
            } else if tcx.is_lang_item(def_id, LangItem::Destruct) {
                self.assemble_const_destruct_candidates(obligation, &mut candidates);
            } else if tcx.is_lang_item(def_id, LangItem::TransmuteTrait) {
                // User-defined transmutability impls are permitted.
                self.assemble_candidates_from_impls(obligation, &mut candidates);
                self.assemble_candidates_for_transmutability(obligation, &mut candidates);
            } else if tcx.is_lang_item(def_id, LangItem::Tuple) {
                self.assemble_candidate_for_tuple(obligation, &mut candidates);
            } else if tcx.is_lang_item(def_id, LangItem::PointerLike) {
                self.assemble_candidate_for_pointer_like(obligation, &mut candidates);
            } else if tcx.is_lang_item(def_id, LangItem::FnPtrTrait) {
                self.assemble_candidates_for_fn_ptr_trait(obligation, &mut candidates);
            } else {
                if tcx.is_lang_item(def_id, LangItem::Clone) {
                    // Same builtin conditions as `Copy`, i.e., every type which has builtin support
                    // for `Copy` also has builtin support for `Clone`, and tuples/arrays of `Clone`
                    // types have builtin support for `Clone`.
                    let clone_conditions = self.copy_clone_conditions(obligation);
                    self.assemble_builtin_bound_candidates(clone_conditions, &mut candidates);
                }

                if tcx.is_lang_item(def_id, LangItem::Coroutine) {
                    self.assemble_coroutine_candidates(obligation, &mut candidates);
                } else if tcx.is_lang_item(def_id, LangItem::Future) {
                    self.assemble_future_candidates(obligation, &mut candidates);
                } else if tcx.is_lang_item(def_id, LangItem::Iterator) {
                    self.assemble_iterator_candidates(obligation, &mut candidates);
                } else if tcx.is_lang_item(def_id, LangItem::FusedIterator) {
                    self.assemble_fused_iterator_candidates(obligation, &mut candidates);
                } else if tcx.is_lang_item(def_id, LangItem::AsyncIterator) {
                    self.assemble_async_iterator_candidates(obligation, &mut candidates);
                } else if tcx.is_lang_item(def_id, LangItem::AsyncFnKindHelper) {
                    self.assemble_async_fn_kind_helper_candidates(obligation, &mut candidates);
                }

                // FIXME: Put these into `else if` blocks above, since they're built-in.
                self.assemble_closure_candidates(obligation, &mut candidates);
                self.assemble_async_closure_candidates(obligation, &mut candidates);
                self.assemble_fn_pointer_candidates(obligation, &mut candidates);

                self.assemble_candidates_from_impls(obligation, &mut candidates);
                self.assemble_candidates_from_object_ty(obligation, &mut candidates);
            }

            self.assemble_candidates_from_projected_tys(obligation, &mut candidates);
            self.assemble_candidates_from_caller_bounds(stack, &mut candidates)?;
            self.assemble_candidates_from_auto_impls(obligation, &mut candidates);
        }
        debug!("candidate list size: {}", candidates.vec.len());
        Ok(candidates)
    }

    #[instrument(level = "debug", skip(self, candidates))]
    fn assemble_candidates_from_projected_tys(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) {
        // Before we go into the whole placeholder thing, just
        // quickly check if the self-type is a projection at all.
        match obligation.predicate.skip_binder().trait_ref.self_ty().kind() {
            // Excluding IATs and type aliases here as they don't have meaningful item bounds.
            ty::Alias(ty::Projection | ty::Opaque, _) => {}
            ty::Infer(ty::TyVar(_)) => {
                span_bug!(
                    obligation.cause.span,
                    "Self=_ should have been handled by assemble_candidates"
                );
            }
            _ => return,
        }

        self.infcx.probe(|_| {
            let poly_trait_predicate = self.infcx.resolve_vars_if_possible(obligation.predicate);
            let placeholder_trait_predicate =
                self.infcx.enter_forall_and_leak_universe(poly_trait_predicate);

            // The bounds returned by `item_bounds` may contain duplicates after
            // normalization, so try to deduplicate when possible to avoid
            // unnecessary ambiguity.
            let mut distinct_normalized_bounds = FxHashSet::default();
            self.for_each_item_bound::<!>(
                placeholder_trait_predicate.self_ty(),
                |selcx, bound, idx| {
                    let Some(bound) = bound.as_trait_clause() else {
                        return ControlFlow::Continue(());
                    };
                    if bound.polarity() != placeholder_trait_predicate.polarity {
                        return ControlFlow::Continue(());
                    }

                    selcx.infcx.probe(|_| {
                        match selcx.match_normalize_trait_ref(
                            obligation,
                            placeholder_trait_predicate.trait_ref,
                            bound.to_poly_trait_ref(),
                        ) {
                            Ok(None) => {
                                candidates.vec.push(ProjectionCandidate(idx));
                            }
                            Ok(Some(normalized_trait))
                                if distinct_normalized_bounds.insert(normalized_trait) =>
                            {
                                candidates.vec.push(ProjectionCandidate(idx));
                            }
                            _ => {}
                        }
                    });

                    ControlFlow::Continue(())
                },
                // On ambiguity.
                || candidates.ambiguous = true,
            );
        });
    }

    /// Given an obligation like `<SomeTrait for T>`, searches the obligations that the caller
    /// supplied to find out whether it is listed among them.
    ///
    /// Never affects the inference environment.
    #[instrument(level = "debug", skip(self, stack, candidates))]
    fn assemble_candidates_from_caller_bounds<'o>(
        &mut self,
        stack: &TraitObligationStack<'o, 'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) -> Result<(), SelectionError<'tcx>> {
        debug!(?stack.obligation);

        // An error type will unify with anything. So, avoid
        // matching an error type with `ParamCandidate`.
        // This helps us avoid spurious errors like issue #121941.
        if stack.obligation.predicate.references_error() {
            return Ok(());
        }

        let bounds = stack
            .obligation
            .param_env
            .caller_bounds()
            .iter()
            .filter(|p| !p.references_error())
            .filter_map(|p| p.as_trait_clause())
            // Micro-optimization: filter out predicates relating to different traits.
            .filter(|p| p.def_id() == stack.obligation.predicate.def_id())
            .filter(|p| p.polarity() == stack.obligation.predicate.polarity());

        // Keep only those bounds which may apply, and propagate overflow if it occurs.
        for bound in bounds {
            // FIXME(oli-obk): it is suspicious that we are dropping the constness and
            // polarity here.
            let wc = self.where_clause_may_apply(stack, bound.map_bound(|t| t.trait_ref))?;
            if wc.may_apply() {
                candidates.vec.push(ParamCandidate(bound));
            }
        }

        Ok(())
    }

    fn assemble_coroutine_candidates(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) {
        // Okay to skip binder because the args on coroutine types never
        // touch bound regions, they just capture the in-scope
        // type/region parameters.
        let self_ty = obligation.self_ty().skip_binder();
        match self_ty.kind() {
            // `async`/`gen` constructs get lowered to a special kind of coroutine that
            // should *not* `impl Coroutine`.
            ty::Coroutine(did, ..) if self.tcx().is_general_coroutine(*did) => {
                debug!(?self_ty, ?obligation, "assemble_coroutine_candidates",);

                candidates.vec.push(CoroutineCandidate);
            }
            ty::Infer(ty::TyVar(_)) => {
                debug!("assemble_coroutine_candidates: ambiguous self-type");
                candidates.ambiguous = true;
            }
            _ => {}
        }
    }

    fn assemble_future_candidates(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) {
        let self_ty = obligation.self_ty().skip_binder();
        if let ty::Coroutine(did, ..) = self_ty.kind() {
            // async constructs get lowered to a special kind of coroutine that
            // should directly `impl Future`.
            if self.tcx().coroutine_is_async(*did) {
                debug!(?self_ty, ?obligation, "assemble_future_candidates",);

                candidates.vec.push(FutureCandidate);
            }
        }
    }

    fn assemble_iterator_candidates(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) {
        let self_ty = obligation.self_ty().skip_binder();
        // gen constructs get lowered to a special kind of coroutine that
        // should directly `impl Iterator`.
        if let ty::Coroutine(did, ..) = self_ty.kind()
            && self.tcx().coroutine_is_gen(*did)
        {
            debug!(?self_ty, ?obligation, "assemble_iterator_candidates",);

            candidates.vec.push(IteratorCandidate);
        }
    }

    fn assemble_fused_iterator_candidates(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) {
        let self_ty = obligation.self_ty().skip_binder();
        // gen constructs get lowered to a special kind of coroutine that
        // should directly `impl FusedIterator`.
        if let ty::Coroutine(did, ..) = self_ty.kind()
            && self.tcx().coroutine_is_gen(*did)
        {
            debug!(?self_ty, ?obligation, "assemble_fused_iterator_candidates",);

            candidates.vec.push(BuiltinCandidate { has_nested: false });
        }
    }

    fn assemble_async_iterator_candidates(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) {
        let self_ty = obligation.self_ty().skip_binder();
        if let ty::Coroutine(did, args) = *self_ty.kind() {
            // gen constructs get lowered to a special kind of coroutine that
            // should directly `impl AsyncIterator`.
            if self.tcx().coroutine_is_async_gen(did) {
                debug!(?self_ty, ?obligation, "assemble_iterator_candidates",);

                // Can only confirm this candidate if we have constrained
                // the `Yield` type to at least `Poll<Option<?0>>`..
                let ty::Adt(_poll_def, args) = *args.as_coroutine().yield_ty().kind() else {
                    candidates.ambiguous = true;
                    return;
                };
                let ty::Adt(_option_def, _) = *args.type_at(0).kind() else {
                    candidates.ambiguous = true;
                    return;
                };

                candidates.vec.push(AsyncIteratorCandidate);
            }
        }
    }

    /// Checks for the artificial impl that the compiler will create for an obligation like `X :
    /// FnMut<..>` where `X` is a closure type.
    ///
    /// Note: the type parameters on a closure candidate are modeled as *output* type
    /// parameters and hence do not affect whether this trait is a match or not. They will be
    /// unified during the confirmation step.
    fn assemble_closure_candidates(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) {
        let Some(kind) = self.tcx().fn_trait_kind_from_def_id(obligation.predicate.def_id()) else {
            return;
        };

        // Okay to skip binder because the args on closure types never
        // touch bound regions, they just capture the in-scope
        // type/region parameters
        let self_ty = obligation.self_ty().skip_binder();
        match *self_ty.kind() {
            ty::Closure(def_id, _) => {
                let is_const = self.tcx().is_const_fn_raw(def_id);
                debug!(?kind, ?obligation, "assemble_unboxed_candidates");
                match self.infcx.closure_kind(self_ty) {
                    Some(closure_kind) => {
                        debug!(?closure_kind, "assemble_unboxed_candidates");
                        if closure_kind.extends(kind) {
                            candidates.vec.push(ClosureCandidate { is_const });
                        }
                    }
                    None => {
                        if kind == ty::ClosureKind::FnOnce {
                            candidates.vec.push(ClosureCandidate { is_const });
                        } else {
                            candidates.ambiguous = true;
                        }
                    }
                }
            }
            ty::CoroutineClosure(def_id, args) => {
                let args = args.as_coroutine_closure();
                let is_const = self.tcx().is_const_fn_raw(def_id);
                if let Some(closure_kind) = self.infcx.closure_kind(self_ty)
                    // Ambiguity if upvars haven't been constrained yet
                    && !args.tupled_upvars_ty().is_ty_var()
                {
                    // A coroutine-closure implements `FnOnce` *always*, since it may
                    // always be called once. It additionally implements `Fn`/`FnMut`
                    // only if it has no upvars referencing the closure-env lifetime,
                    // and if the closure kind permits it.
                    if closure_kind.extends(kind) && !args.has_self_borrows() {
                        candidates.vec.push(ClosureCandidate { is_const });
                    } else if kind == ty::ClosureKind::FnOnce {
                        candidates.vec.push(ClosureCandidate { is_const });
                    }
                } else {
                    if kind == ty::ClosureKind::FnOnce {
                        candidates.vec.push(ClosureCandidate { is_const });
                    } else {
                        // This stays ambiguous until kind+upvars are determined.
                        candidates.ambiguous = true;
                    }
                }
            }
            ty::Infer(ty::TyVar(_)) => {
                debug!("assemble_unboxed_closure_candidates: ambiguous self-type");
                candidates.ambiguous = true;
            }
            _ => {}
        }
    }

    fn assemble_async_closure_candidates(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) {
        let Some(goal_kind) =
            self.tcx().async_fn_trait_kind_from_def_id(obligation.predicate.def_id())
        else {
            return;
        };

        match *obligation.self_ty().skip_binder().kind() {
            ty::CoroutineClosure(_, args) => {
                if let Some(closure_kind) =
                    args.as_coroutine_closure().kind_ty().to_opt_closure_kind()
                    && !closure_kind.extends(goal_kind)
                {
                    return;
                }
                candidates.vec.push(AsyncClosureCandidate);
            }
            // Closures and fn pointers implement `AsyncFn*` if their return types
            // implement `Future`, which is checked later.
            ty::Closure(_, args) => {
                if let Some(closure_kind) = args.as_closure().kind_ty().to_opt_closure_kind()
                    && !closure_kind.extends(goal_kind)
                {
                    return;
                }
                candidates.vec.push(AsyncClosureCandidate);
            }
            ty::FnDef(..) | ty::FnPtr(..) => {
                candidates.vec.push(AsyncClosureCandidate);
            }
            _ => {}
        }
    }

    fn assemble_async_fn_kind_helper_candidates(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) {
        let self_ty = obligation.self_ty().skip_binder();
        let target_kind_ty = obligation.predicate.skip_binder().trait_ref.args.type_at(1);

        // `to_opt_closure_kind` is kind of ICEy when it sees non-int types.
        if !(self_ty.is_integral() || self_ty.is_ty_var()) {
            return;
        }
        if !(target_kind_ty.is_integral() || self_ty.is_ty_var()) {
            return;
        }

        // Check that the self kind extends the goal kind. If it does,
        // then there's nothing else to check.
        if let Some(closure_kind) = self_ty.to_opt_closure_kind()
            && let Some(goal_kind) = target_kind_ty.to_opt_closure_kind()
        {
            if closure_kind.extends(goal_kind) {
                candidates.vec.push(AsyncFnKindHelperCandidate);
            }
        }
    }

    /// Implements one of the `Fn()` family for a fn pointer.
    fn assemble_fn_pointer_candidates(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) {
        // We provide impl of all fn traits for fn pointers.
        if !self.tcx().is_fn_trait(obligation.predicate.def_id()) {
            return;
        }

        // Keep this function in sync with extract_tupled_inputs_and_output_from_callable
        // until the old solver (and thus this function) is removed.

        // Okay to skip binder because what we are inspecting doesn't involve bound regions.
        let self_ty = obligation.self_ty().skip_binder();
        match *self_ty.kind() {
            ty::Infer(ty::TyVar(_)) => {
                debug!("assemble_fn_pointer_candidates: ambiguous self-type");
                candidates.ambiguous = true; // Could wind up being a fn() type.
            }
            // Provide an impl, but only for suitable `fn` pointers.
            ty::FnPtr(sig) => {
                if sig.is_fn_trait_compatible() {
                    candidates
                        .vec
                        .push(FnPointerCandidate { fn_host_effect: self.tcx().consts.true_ });
                }
            }
            // Provide an impl for suitable functions, rejecting `#[target_feature]` functions (RFC 2396).
            ty::FnDef(def_id, args) => {
                let tcx = self.tcx();
                if tcx.fn_sig(def_id).skip_binder().is_fn_trait_compatible()
                    && tcx.codegen_fn_attrs(def_id).target_features.is_empty()
                {
                    candidates.vec.push(FnPointerCandidate {
                        fn_host_effect: tcx
                            .generics_of(def_id)
                            .host_effect_index
                            .map_or(tcx.consts.true_, |idx| args.const_at(idx)),
                    });
                }
            }
            _ => {}
        }
    }

    /// Searches for impls that might apply to `obligation`.
    #[instrument(level = "debug", skip(self, candidates))]
    fn assemble_candidates_from_impls(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) {
        // Essentially any user-written impl will match with an error type,
        // so creating `ImplCandidates` isn't useful. However, we might
        // end up finding a candidate elsewhere (e.g. a `BuiltinCandidate` for `Sized`)
        // This helps us avoid overflow: see issue #72839
        // Since compilation is already guaranteed to fail, this is just
        // to try to show the 'nicest' possible errors to the user.
        // We don't check for errors in the `ParamEnv` - in practice,
        // it seems to cause us to be overly aggressive in deciding
        // to give up searching for candidates, leading to spurious errors.
        if obligation.predicate.references_error() {
            return;
        }

        let drcx = DeepRejectCtxt::new(self.tcx(), TreatParams::ForLookup);
        let obligation_args = obligation.predicate.skip_binder().trait_ref.args;
        self.tcx().for_each_relevant_impl(
            obligation.predicate.def_id(),
            obligation.predicate.skip_binder().trait_ref.self_ty(),
            |impl_def_id| {
                // Before we create the generic parameters and everything, first
                // consider a "quick reject". This avoids creating more types
                // and so forth that we need to.
                let impl_trait_header = self.tcx().impl_trait_header(impl_def_id).unwrap();
                if !drcx
                    .args_may_unify(obligation_args, impl_trait_header.trait_ref.skip_binder().args)
                {
                    return;
                }

                // For every `default impl`, there's always a non-default `impl`
                // that will *also* apply. There's no reason to register a candidate
                // for this impl, since it is *not* proof that the trait goal holds.
                if self.tcx().defaultness(impl_def_id).is_default() {
                    return;
                }

                if self.reject_fn_ptr_impls(
                    impl_def_id,
                    obligation,
                    impl_trait_header.trait_ref.skip_binder().self_ty(),
                ) {
                    return;
                }

                self.infcx.probe(|_| {
                    if let Ok(_args) = self.match_impl(impl_def_id, impl_trait_header, obligation) {
                        candidates.vec.push(ImplCandidate(impl_def_id));
                    }
                });
            },
        );
    }

    /// The various `impl<T: FnPtr> Trait for T` in libcore are more like builtin impls for all function items
    /// and function pointers and less like blanket impls. Rejecting them when they can't possibly apply (because
    /// the obligation's self-type does not implement `FnPtr`) avoids reporting that the self type does not implement
    /// `FnPtr`, when we wanted to report that it doesn't implement `Trait`.
    #[instrument(level = "trace", skip(self), ret)]
    fn reject_fn_ptr_impls(
        &mut self,
        impl_def_id: DefId,
        obligation: &PolyTraitObligation<'tcx>,
        impl_self_ty: Ty<'tcx>,
    ) -> bool {
        // Let `impl<T: FnPtr> Trait for Vec<T>` go through the normal rejection path.
        if !matches!(impl_self_ty.kind(), ty::Param(..)) {
            return false;
        }
        let Some(fn_ptr_trait) = self.tcx().lang_items().fn_ptr_trait() else {
            return false;
        };

        for &(predicate, _) in self.tcx().predicates_of(impl_def_id).predicates {
            let ty::ClauseKind::Trait(pred) = predicate.kind().skip_binder() else { continue };
            if fn_ptr_trait != pred.trait_ref.def_id {
                continue;
            }
            trace!(?pred);
            // Not the bound we're looking for
            if pred.self_ty() != impl_self_ty {
                continue;
            }

            match obligation.self_ty().skip_binder().kind() {
                // Fast path to avoid evaluating an obligation that trivially holds.
                // There may be more bounds, but these are checked by the regular path.
                ty::FnPtr(..) => return false,

                // These may potentially implement `FnPtr`
                ty::Placeholder(..)
                | ty::Dynamic(_, _, _)
                | ty::Alias(_, _)
                | ty::Infer(_)
                | ty::Param(..)
                | ty::Bound(_, _) => {}

                // These can't possibly implement `FnPtr` as they are concrete types
                // and not `FnPtr`
                ty::Bool
                | ty::Char
                | ty::Int(_)
                | ty::Uint(_)
                | ty::Float(_)
                | ty::Adt(_, _)
                | ty::Foreign(_)
                | ty::Str
                | ty::Array(_, _)
                | ty::Pat(_, _)
                | ty::Slice(_)
                | ty::RawPtr(_, _)
                | ty::Ref(_, _, _)
                | ty::Closure(..)
                | ty::CoroutineClosure(..)
                | ty::Coroutine(_, _)
                | ty::CoroutineWitness(..)
                | ty::Never
                | ty::Tuple(_)
                | ty::Error(_) => return true,
                // FIXME: Function definitions could actually implement `FnPtr` by
                // casting the ZST function def to a function pointer.
                ty::FnDef(_, _) => return true,
            }

            // Generic params can implement `FnPtr` if the predicate
            // holds within its own environment.
            let obligation = Obligation::new(
                self.tcx(),
                obligation.cause.clone(),
                obligation.param_env,
                self.tcx().mk_predicate(obligation.predicate.map_bound(|mut pred| {
                    pred.trait_ref =
                        ty::TraitRef::new(self.tcx(), fn_ptr_trait, [pred.trait_ref.self_ty()]);
                    ty::PredicateKind::Clause(ty::ClauseKind::Trait(pred))
                })),
            );
            if let Ok(r) = self.evaluate_root_obligation(&obligation) {
                if !r.may_apply() {
                    return true;
                }
            }
        }
        false
    }

    fn assemble_candidates_from_auto_impls(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) {
        // Okay to skip binder here because the tests we do below do not involve bound regions.
        let self_ty = obligation.self_ty().skip_binder();
        debug!(?self_ty, "assemble_candidates_from_auto_impls");

        let def_id = obligation.predicate.def_id();

        if self.tcx().trait_is_auto(def_id) {
            match *self_ty.kind() {
                ty::Dynamic(..) => {
                    // For object types, we don't know what the closed
                    // over types are. This means we conservatively
                    // say nothing; a candidate may be added by
                    // `assemble_candidates_from_object_ty`.
                }
                ty::Foreign(..) => {
                    // Since the contents of foreign types is unknown,
                    // we don't add any `..` impl. Default traits could
                    // still be provided by a manual implementation for
                    // this trait and type.
                }
                ty::Param(..)
                | ty::Alias(ty::Projection | ty::Inherent | ty::Weak, ..)
                | ty::Placeholder(..)
                | ty::Bound(..) => {
                    // In these cases, we don't know what the actual
                    // type is. Therefore, we cannot break it down
                    // into its constituent types. So we don't
                    // consider the `..` impl but instead just add no
                    // candidates: this means that typeck will only
                    // succeed if there is another reason to believe
                    // that this obligation holds. That could be a
                    // where-clause or, in the case of an object type,
                    // it could be that the object type lists the
                    // trait (e.g., `Foo+Send : Send`). See
                    // `ui/typeck/typeck-default-trait-impl-send-param.rs`
                    // for an example of a test case that exercises
                    // this path.
                }
                ty::Infer(ty::TyVar(_) | ty::IntVar(_) | ty::FloatVar(_)) => {
                    // The auto impl might apply; we don't know.
                    candidates.ambiguous = true;
                }
                ty::Coroutine(coroutine_def_id, _)
                    if self.tcx().is_lang_item(def_id, LangItem::Unpin) =>
                {
                    match self.tcx().coroutine_movability(coroutine_def_id) {
                        hir::Movability::Static => {
                            // Immovable coroutines are never `Unpin`, so
                            // suppress the normal auto-impl candidate for it.
                        }
                        hir::Movability::Movable => {
                            // Movable coroutines are always `Unpin`, so add an
                            // unconditional builtin candidate.
                            candidates.vec.push(BuiltinCandidate { has_nested: false });
                        }
                    }
                }

                ty::Infer(ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) => {
                    bug!(
                        "asked to assemble auto trait candidates of unexpected type: {:?}",
                        self_ty
                    );
                }

                ty::Alias(ty::Opaque, alias) => {
                    if candidates.vec.iter().any(|c| matches!(c, ProjectionCandidate(_))) {
                        // We do not generate an auto impl candidate for `impl Trait`s which already
                        // reference our auto trait.
                        //
                        // For example during candidate assembly for `impl Send: Send`, we don't have
                        // to look at the constituent types for this opaque types to figure out that this
                        // trivially holds.
                        //
                        // Note that this is only sound as projection candidates of opaque types
                        // are always applicable for auto traits.
                    } else if self.infcx.intercrate {
                        // We do not emit auto trait candidates for opaque types in coherence.
                        // Doing so can result in weird dependency cycles.
                        candidates.ambiguous = true;
                    } else if self.infcx.can_define_opaque_ty(alias.def_id) {
                        // We do not emit auto trait candidates for opaque types in their defining scope, as
                        // we need to know the hidden type first, which we can't reliably know within the defining
                        // scope.
                        candidates.ambiguous = true;
                    } else {
                        candidates.vec.push(AutoImplCandidate)
                    }
                }

                ty::Bool
                | ty::Char
                | ty::Int(_)
                | ty::Uint(_)
                | ty::Float(_)
                | ty::Str
                | ty::Array(_, _)
                | ty::Pat(_, _)
                | ty::Slice(_)
                | ty::Adt(..)
                | ty::RawPtr(_, _)
                | ty::Ref(..)
                | ty::FnDef(..)
                | ty::FnPtr(_)
                | ty::Closure(..)
                | ty::CoroutineClosure(..)
                | ty::Coroutine(..)
                | ty::Never
                | ty::Tuple(_)
                | ty::CoroutineWitness(..) => {
                    // Only consider auto impls if there are no manual impls for the root of `self_ty`.
                    //
                    // For example, we only consider auto candidates for `&i32: Auto` if no explicit impl
                    // for `&SomeType: Auto` exists. Due to E0321 the only crate where impls
                    // for `&SomeType: Auto` can be defined is the crate where `Auto` has been defined.
                    //
                    // Generally, we have to guarantee that for all `SimplifiedType`s the only crate
                    // which may define impls for that type is either the crate defining the type
                    // or the trait. This should be guaranteed by the orphan check.
                    let mut has_impl = false;
                    self.tcx().for_each_relevant_impl(def_id, self_ty, |_| has_impl = true);
                    if !has_impl {
                        candidates.vec.push(AutoImplCandidate)
                    }
                }
                ty::Error(_) => {} // do not add an auto trait impl for `ty::Error` for now.
            }
        }
    }

    /// Searches for impls that might apply to `obligation`.
    fn assemble_candidates_from_object_ty(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) {
        debug!(
            self_ty = ?obligation.self_ty().skip_binder(),
            "assemble_candidates_from_object_ty",
        );

        if !self.tcx().trait_def(obligation.predicate.def_id()).implement_via_object {
            return;
        }

        self.infcx.probe(|_snapshot| {
            let poly_trait_predicate = self.infcx.resolve_vars_if_possible(obligation.predicate);
            self.infcx.enter_forall(poly_trait_predicate, |placeholder_trait_predicate| {
                let self_ty = placeholder_trait_predicate.self_ty();
                let principal_trait_ref = match self_ty.kind() {
                    ty::Dynamic(data, ..) => {
                        if data.auto_traits().any(|did| did == obligation.predicate.def_id()) {
                            debug!(
                                "assemble_candidates_from_object_ty: matched builtin bound, \
                             pushing candidate"
                            );
                            candidates.vec.push(BuiltinObjectCandidate);
                            return;
                        }

                        if let Some(principal) = data.principal() {
                            if !self.infcx.tcx.features().object_safe_for_dispatch {
                                principal.with_self_ty(self.tcx(), self_ty)
                            } else if self.tcx().is_object_safe(principal.def_id()) {
                                principal.with_self_ty(self.tcx(), self_ty)
                            } else {
                                return;
                            }
                        } else {
                            // Only auto trait bounds exist.
                            return;
                        }
                    }
                    ty::Infer(ty::TyVar(_)) => {
                        debug!("assemble_candidates_from_object_ty: ambiguous");
                        candidates.ambiguous = true; // could wind up being an object type
                        return;
                    }
                    _ => return,
                };

                debug!(?principal_trait_ref, "assemble_candidates_from_object_ty");

                // Count only those upcast versions that match the trait-ref
                // we are looking for. Specifically, do not only check for the
                // correct trait, but also the correct type parameters.
                // For example, we may be trying to upcast `Foo` to `Bar<i32>`,
                // but `Foo` is declared as `trait Foo: Bar<u32>`.
                let candidate_supertraits = util::supertraits(self.tcx(), principal_trait_ref)
                    .enumerate()
                    .filter(|&(_, upcast_trait_ref)| {
                        self.infcx.probe(|_| {
                            self.match_normalize_trait_ref(
                                obligation,
                                placeholder_trait_predicate.trait_ref,
                                upcast_trait_ref,
                            )
                            .is_ok()
                        })
                    })
                    .map(|(idx, _)| ObjectCandidate(idx));

                candidates.vec.extend(candidate_supertraits);
            })
        })
    }

    /// Temporary migration for #89190
    fn need_migrate_deref_output_trait_object(
        &mut self,
        ty: Ty<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        cause: &ObligationCause<'tcx>,
    ) -> Option<ty::PolyExistentialTraitRef<'tcx>> {
        // Don't drop any candidates in intercrate mode, as it's incomplete.
        // (Not that it matters, since `Unsize` is not a stable trait.)
        if self.infcx.intercrate {
            return None;
        }

        let tcx = self.tcx();
        if tcx.features().trait_upcasting {
            return None;
        }

        // <ty as Deref>
        let trait_ref = ty::TraitRef::new(tcx, tcx.lang_items().deref_trait()?, [ty]);

        let obligation =
            traits::Obligation::new(tcx, cause.clone(), param_env, ty::Binder::dummy(trait_ref));
        if !self.infcx.predicate_may_hold(&obligation) {
            return None;
        }

        self.infcx.probe(|_| {
            let ty = traits::normalize_projection_ty(
                self,
                param_env,
                ty::AliasTy::new_from_args(tcx, tcx.lang_items().deref_target()?, trait_ref.args),
                cause.clone(),
                0,
                // We're *intentionally* throwing these away,
                // since we don't actually use them.
                &mut vec![],
            )
            .as_type()
            .unwrap();

            if let ty::Dynamic(data, ..) = ty.kind() { data.principal() } else { None }
        })
    }

    /// Searches for unsizing that might apply to `obligation`.
    fn assemble_candidates_for_unsizing(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) {
        // We currently never consider higher-ranked obligations e.g.
        // `for<'a> &'a T: Unsize<Trait+'a>` to be implemented. This is not
        // because they are a priori invalid, and we could potentially add support
        // for them later, it's just that there isn't really a strong need for it.
        // A `T: Unsize<U>` obligation is always used as part of a `T: CoerceUnsize<U>`
        // impl, and those are generally applied to concrete types.
        //
        // That said, one might try to write a fn with a where clause like
        //     for<'a> Foo<'a, T>: Unsize<Foo<'a, Trait>>
        // where the `'a` is kind of orthogonal to the relevant part of the `Unsize`.
        // Still, you'd be more likely to write that where clause as
        //     T: Trait
        // so it seems ok if we (conservatively) fail to accept that `Unsize`
        // obligation above. Should be possible to extend this in the future.
        let Some(source) = obligation.self_ty().no_bound_vars() else {
            // Don't add any candidates if there are bound regions.
            return;
        };
        let target = obligation.predicate.skip_binder().trait_ref.args.type_at(1);

        debug!(?source, ?target, "assemble_candidates_for_unsizing");

        match (source.kind(), target.kind()) {
            // Trait+Kx+'a -> Trait+Ky+'b (upcasts).
            (&ty::Dynamic(a_data, a_region, ty::Dyn), &ty::Dynamic(b_data, b_region, ty::Dyn)) => {
                // Upcast coercions permit several things:
                //
                // 1. Dropping auto traits, e.g., `Foo + Send` to `Foo`
                // 2. Tightening the region bound, e.g., `Foo + 'a` to `Foo + 'b` if `'a: 'b`
                // 3. Tightening trait to its super traits, eg. `Foo` to `Bar` if `Foo: Bar`
                //
                // Note that neither of the first two of these changes requires any
                // change at runtime. The third needs to change pointer metadata at runtime.
                //
                // We always perform upcasting coercions when we can because of reason
                // #2 (region bounds).
                let principal_def_id_a = a_data.principal_def_id();
                let principal_def_id_b = b_data.principal_def_id();
                if principal_def_id_a == principal_def_id_b {
                    // We may upcast to auto traits that are either explicitly listed in
                    // the object type's bounds, or implied by the principal trait ref's
                    // supertraits.
                    let a_auto_traits: FxIndexSet<DefId> = a_data
                        .auto_traits()
                        .chain(principal_def_id_a.into_iter().flat_map(|principal_def_id| {
                            self.tcx()
                                .supertrait_def_ids(principal_def_id)
                                .filter(|def_id| self.tcx().trait_is_auto(*def_id))
                        }))
                        .collect();
                    let auto_traits_compatible = b_data
                        .auto_traits()
                        // All of a's auto traits need to be in b's auto traits.
                        .all(|b| a_auto_traits.contains(&b));
                    if auto_traits_compatible {
                        candidates.vec.push(BuiltinUnsizeCandidate);
                    }
                } else if principal_def_id_a.is_some() && principal_def_id_b.is_some() {
                    // not casual unsizing, now check whether this is trait upcasting coercion.
                    let principal_a = a_data.principal().unwrap();
                    let target_trait_did = principal_def_id_b.unwrap();
                    let source_trait_ref = principal_a.with_self_ty(self.tcx(), source);
                    if let Some(deref_trait_ref) = self.need_migrate_deref_output_trait_object(
                        source,
                        obligation.param_env,
                        &obligation.cause,
                    ) {
                        if deref_trait_ref.def_id() == target_trait_did {
                            return;
                        }
                    }

                    for (idx, upcast_trait_ref) in
                        util::supertraits(self.tcx(), source_trait_ref).enumerate()
                    {
                        self.infcx.probe(|_| {
                            if upcast_trait_ref.def_id() == target_trait_did
                                && let Ok(nested) = self.match_upcast_principal(
                                    obligation,
                                    upcast_trait_ref,
                                    a_data,
                                    b_data,
                                    a_region,
                                    b_region,
                                )
                            {
                                if nested.is_none() {
                                    candidates.ambiguous = true;
                                }
                                candidates.vec.push(TraitUpcastingUnsizeCandidate(idx));
                            }
                        })
                    }
                }
            }

            // `T` -> `Trait`
            (_, &ty::Dynamic(_, _, ty::Dyn)) => {
                candidates.vec.push(BuiltinUnsizeCandidate);
            }

            // Ambiguous handling is below `T` -> `Trait`, because inference
            // variables can still implement `Unsize<Trait>` and nested
            // obligations will have the final say (likely deferred).
            (&ty::Infer(ty::TyVar(_)), _) | (_, &ty::Infer(ty::TyVar(_))) => {
                debug!("assemble_candidates_for_unsizing: ambiguous");
                candidates.ambiguous = true;
            }

            // `[T; n]` -> `[T]`
            (&ty::Array(..), &ty::Slice(_)) => {
                candidates.vec.push(BuiltinUnsizeCandidate);
            }

            // `Struct<T>` -> `Struct<U>`
            (&ty::Adt(def_id_a, _), &ty::Adt(def_id_b, _)) if def_id_a.is_struct() => {
                if def_id_a == def_id_b {
                    candidates.vec.push(BuiltinUnsizeCandidate);
                }
            }

            // `(.., T)` -> `(.., U)`
            (&ty::Tuple(tys_a), &ty::Tuple(tys_b)) => {
                if tys_a.len() == tys_b.len() {
                    candidates.vec.push(BuiltinUnsizeCandidate);
                }
            }

            _ => {}
        };
    }

    #[instrument(level = "debug", skip(self, obligation, candidates))]
    fn assemble_candidates_for_transmutability(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) {
        if obligation.predicate.has_non_region_param() {
            return;
        }

        if obligation.has_non_region_infer() {
            candidates.ambiguous = true;
            return;
        }

        candidates.vec.push(TransmutabilityCandidate);
    }

    #[instrument(level = "debug", skip(self, obligation, candidates))]
    fn assemble_candidates_for_trait_alias(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) {
        // Okay to skip binder here because the tests we do below do not involve bound regions.
        let self_ty = obligation.self_ty().skip_binder();
        debug!(?self_ty);

        let def_id = obligation.predicate.def_id();

        if self.tcx().is_trait_alias(def_id) {
            candidates.vec.push(TraitAliasCandidate);
        }
    }

    /// Assembles the trait which are built-in to the language itself:
    /// `Copy`, `Clone` and `Sized`.
    #[instrument(level = "debug", skip(self, candidates))]
    fn assemble_builtin_bound_candidates(
        &mut self,
        conditions: BuiltinImplConditions<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) {
        match conditions {
            BuiltinImplConditions::Where(nested) => {
                candidates
                    .vec
                    .push(BuiltinCandidate { has_nested: !nested.skip_binder().is_empty() });
            }
            BuiltinImplConditions::None => {}
            BuiltinImplConditions::Ambiguous => {
                candidates.ambiguous = true;
            }
        }
    }

    fn assemble_const_destruct_candidates(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) {
        // If the predicate is `~const Destruct` in a non-const environment, we don't actually need
        // to check anything. We'll short-circuit checking any obligations in confirmation, too.
        let Some(host_effect_index) =
            self.tcx().generics_of(obligation.predicate.def_id()).host_effect_index
        else {
            candidates.vec.push(BuiltinCandidate { has_nested: false });
            return;
        };
        // If the obligation has `host = true`, then the obligation is non-const and it's always
        // trivially implemented.
        if obligation.predicate.skip_binder().trait_ref.args.const_at(host_effect_index)
            == self.tcx().consts.true_
        {
            candidates.vec.push(BuiltinCandidate { has_nested: false });
            return;
        }

        let self_ty = self.infcx.shallow_resolve(obligation.self_ty().skip_binder());
        match self_ty.kind() {
            ty::Alias(..)
            | ty::Dynamic(..)
            | ty::Error(_)
            | ty::Bound(..)
            | ty::Param(_)
            | ty::Placeholder(_) => {
                // We don't know if these are `~const Destruct`, at least
                // not structurally... so don't push a candidate.
            }

            ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Infer(ty::IntVar(_))
            | ty::Infer(ty::FloatVar(_))
            | ty::Str
            | ty::RawPtr(_, _)
            | ty::Ref(..)
            | ty::FnDef(..)
            | ty::FnPtr(_)
            | ty::Never
            | ty::Foreign(_)
            | ty::Array(..)
            | ty::Pat(..)
            | ty::Slice(_)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Coroutine(..)
            | ty::Tuple(_)
            | ty::CoroutineWitness(..) => {
                // These are built-in, and cannot have a custom `impl const Destruct`.
                candidates.vec.push(ConstDestructCandidate(None));
            }

            ty::Adt(..) => {
                let mut relevant_impl = None;
                self.tcx().for_each_relevant_impl(
                    self.tcx().require_lang_item(LangItem::Drop, None),
                    obligation.predicate.skip_binder().trait_ref.self_ty(),
                    |impl_def_id| {
                        if let Some(old_impl_def_id) = relevant_impl {
                            self.tcx()
                                .dcx()
                                .struct_span_err(
                                    self.tcx().def_span(impl_def_id),
                                    "multiple drop impls found",
                                )
                                .with_span_note(
                                    self.tcx().def_span(old_impl_def_id),
                                    "other impl here",
                                )
                                .delay_as_bug();
                        }

                        relevant_impl = Some(impl_def_id);
                    },
                );

                if let Some(impl_def_id) = relevant_impl {
                    // Check that `impl Drop` is actually const, if there is a custom impl
                    if self.tcx().constness(impl_def_id) == hir::Constness::Const {
                        candidates.vec.push(ConstDestructCandidate(Some(impl_def_id)));
                    }
                } else {
                    // Otherwise check the ADT like a built-in type (structurally)
                    candidates.vec.push(ConstDestructCandidate(None));
                }
            }

            ty::Infer(_) => {
                candidates.ambiguous = true;
            }
        }
    }

    fn assemble_candidate_for_tuple(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) {
        let self_ty = self.infcx.shallow_resolve(obligation.self_ty().skip_binder());
        match self_ty.kind() {
            ty::Tuple(_) => {
                candidates.vec.push(BuiltinCandidate { has_nested: false });
            }
            ty::Infer(ty::TyVar(_)) => {
                candidates.ambiguous = true;
            }
            ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Adt(_, _)
            | ty::Foreign(_)
            | ty::Str
            | ty::Array(_, _)
            | ty::Slice(_)
            | ty::RawPtr(_, _)
            | ty::Ref(_, _, _)
            | ty::FnDef(_, _)
            | ty::Pat(_, _)
            | ty::FnPtr(_)
            | ty::Dynamic(_, _, _)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Coroutine(_, _)
            | ty::CoroutineWitness(..)
            | ty::Never
            | ty::Alias(..)
            | ty::Param(_)
            | ty::Bound(_, _)
            | ty::Error(_)
            | ty::Infer(_)
            | ty::Placeholder(_) => {}
        }
    }

    fn assemble_candidate_for_pointer_like(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) {
        // The regions of a type don't affect the size of the type
        let tcx = self.tcx();
        let self_ty = tcx.instantiate_bound_regions_with_erased(obligation.predicate.self_ty());
        // We should erase regions from both the param-env and type, since both
        // may have infer regions. Specifically, after canonicalizing and instantiating,
        // early bound regions turn into region vars in both the new and old solver.
        let key = tcx.erase_regions(obligation.param_env.and(self_ty));
        // But if there are inference variables, we have to wait until it's resolved.
        if key.has_non_region_infer() {
            candidates.ambiguous = true;
            return;
        }

        if let Ok(layout) = tcx.layout_of(key)
            && layout.layout.is_pointer_like(&tcx.data_layout)
        {
            candidates.vec.push(BuiltinCandidate { has_nested: false });
        }
    }

    fn assemble_candidates_for_fn_ptr_trait(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) {
        let self_ty = self.infcx.resolve_vars_if_possible(obligation.self_ty());

        match self_ty.skip_binder().kind() {
            ty::FnPtr(_) => candidates.vec.push(BuiltinCandidate { has_nested: false }),
            ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Adt(..)
            | ty::Foreign(..)
            | ty::Str
            | ty::Array(..)
            | ty::Pat(..)
            | ty::Slice(_)
            | ty::RawPtr(_, _)
            | ty::Ref(..)
            | ty::FnDef(..)
            | ty::Placeholder(..)
            | ty::Dynamic(..)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Coroutine(..)
            | ty::CoroutineWitness(..)
            | ty::Never
            | ty::Tuple(..)
            | ty::Alias(..)
            | ty::Param(..)
            | ty::Bound(..)
            | ty::Error(_)
            | ty::Infer(
                ty::InferTy::IntVar(_)
                | ty::InferTy::FloatVar(_)
                | ty::InferTy::FreshIntTy(_)
                | ty::InferTy::FreshFloatTy(_),
            ) => {}
            ty::Infer(ty::InferTy::TyVar(_) | ty::InferTy::FreshTy(_)) => {
                candidates.ambiguous = true;
            }
        }
    }
}
