// ignore-tidy-filelength

//! Candidate selection. See the [rustc dev guide] for more information on how this works.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/traits/resolution.html#selection

use self::EvaluationResult::*;
use self::SelectionCandidate::*;

use super::coherence::{self, Conflict};
use super::project;
use super::project::{
    normalize_with_depth, normalize_with_depth_to, Normalized, ProjectionCacheKey,
};
use super::util;
use super::util::{closure_trait_ref_and_return_type, predicate_for_trait_def};
use super::wf;
use super::DerivedObligationCause;
use super::Selection;
use super::SelectionResult;
use super::TraitNotObjectSafe;
use super::TraitQueryMode;
use super::{BuiltinDerivedObligation, ImplDerivedObligation, ObligationCauseCode};
use super::{ObjectCastObligation, Obligation};
use super::{ObligationCause, PredicateObligation, TraitObligation};
use super::{OutputTypeParameterMismatch, Overflow, SelectionError, Unimplemented};
use super::{
    VtableAutoImpl, VtableBuiltin, VtableClosure, VtableFnPointer, VtableGenerator, VtableImpl,
    VtableObject, VtableParam, VtableTraitAlias,
};
use super::{
    VtableAutoImplData, VtableBuiltinData, VtableClosureData, VtableFnPointerData,
    VtableGeneratorData, VtableImplData, VtableObjectData, VtableTraitAliasData,
};

use crate::infer::{CombinedSnapshot, InferCtxt, InferOk, PlaceholderMap, TypeFreshener};
use rustc::dep_graph::{DepKind, DepNodeIndex};
use rustc::middle::lang_items;
use rustc::ty::fast_reject;
use rustc::ty::relate::TypeRelation;
use rustc::ty::subst::{Subst, SubstsRef};
use rustc::ty::{self, ToPolyTraitRef, ToPredicate, Ty, TyCtxt, TypeFoldable, WithConstness};
use rustc_ast::attr;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_index::bit_set::GrowableBitSet;
use rustc_span::symbol::sym;
use rustc_target::spec::abi::Abi;

use std::cell::{Cell, RefCell};
use std::cmp;
use std::fmt::{self, Display};
use std::iter;
use std::rc::Rc;

pub use rustc::traits::select::*;

pub struct SelectionContext<'cx, 'tcx> {
    infcx: &'cx InferCtxt<'cx, 'tcx>,

    /// Freshener used specifically for entries on the obligation
    /// stack. This ensures that all entries on the stack at one time
    /// will have the same set of placeholder entries, which is
    /// important for checking for trait bounds that recursively
    /// require themselves.
    freshener: TypeFreshener<'cx, 'tcx>,

    /// If `true`, indicates that the evaluation should be conservative
    /// and consider the possibility of types outside this crate.
    /// This comes up primarily when resolving ambiguity. Imagine
    /// there is some trait reference `$0: Bar` where `$0` is an
    /// inference variable. If `intercrate` is true, then we can never
    /// say for sure that this reference is not implemented, even if
    /// there are *no impls at all for `Bar`*, because `$0` could be
    /// bound to some type that in a downstream crate that implements
    /// `Bar`. This is the suitable mode for coherence. Elsewhere,
    /// though, we set this to false, because we are only interested
    /// in types that the user could actually have written --- in
    /// other words, we consider `$0: Bar` to be unimplemented if
    /// there is no type that the user could *actually name* that
    /// would satisfy it. This avoids crippling inference, basically.
    intercrate: bool,

    intercrate_ambiguity_causes: Option<Vec<IntercrateAmbiguityCause>>,

    /// Controls whether or not to filter out negative impls when selecting.
    /// This is used in librustdoc to distinguish between the lack of an impl
    /// and a negative impl
    allow_negative_impls: bool,

    /// The mode that trait queries run in, which informs our error handling
    /// policy. In essence, canonicalized queries need their errors propagated
    /// rather than immediately reported because we do not have accurate spans.
    query_mode: TraitQueryMode,
}

// A stack that walks back up the stack frame.
struct TraitObligationStack<'prev, 'tcx> {
    obligation: &'prev TraitObligation<'tcx>,

    /// The trait ref from `obligation` but "freshened" with the
    /// selection-context's freshener. Used to check for recursion.
    fresh_trait_ref: ty::PolyTraitRef<'tcx>,

    /// Starts out equal to `depth` -- if, during evaluation, we
    /// encounter a cycle, then we will set this flag to the minimum
    /// depth of that cycle for all participants in the cycle. These
    /// participants will then forego caching their results. This is
    /// not the most efficient solution, but it addresses #60010. The
    /// problem we are trying to prevent:
    ///
    /// - If you have `A: AutoTrait` requires `B: AutoTrait` and `C: NonAutoTrait`
    /// - `B: AutoTrait` requires `A: AutoTrait` (coinductive cycle, ok)
    /// - `C: NonAutoTrait` requires `A: AutoTrait` (non-coinductive cycle, not ok)
    ///
    /// you don't want to cache that `B: AutoTrait` or `A: AutoTrait`
    /// is `EvaluatedToOk`; this is because they were only considered
    /// ok on the premise that if `A: AutoTrait` held, but we indeed
    /// encountered a problem (later on) with `A: AutoTrait. So we
    /// currently set a flag on the stack node for `B: AutoTrait` (as
    /// well as the second instance of `A: AutoTrait`) to suppress
    /// caching.
    ///
    /// This is a simple, targeted fix. A more-performant fix requires
    /// deeper changes, but would permit more caching: we could
    /// basically defer caching until we have fully evaluated the
    /// tree, and then cache the entire tree at once. In any case, the
    /// performance impact here shouldn't be so horrible: every time
    /// this is hit, we do cache at least one trait, so we only
    /// evaluate each member of a cycle up to N times, where N is the
    /// length of the cycle. This means the performance impact is
    /// bounded and we shouldn't have any terrible worst-cases.
    reached_depth: Cell<usize>,

    previous: TraitObligationStackList<'prev, 'tcx>,

    /// The number of parent frames plus one (thus, the topmost frame has depth 1).
    depth: usize,

    /// The depth-first number of this node in the search graph -- a
    /// pre-order index. Basically, a freshly incremented counter.
    dfn: usize,
}

struct SelectionCandidateSet<'tcx> {
    // A list of candidates that definitely apply to the current
    // obligation (meaning: types unify).
    vec: Vec<SelectionCandidate<'tcx>>,

    // If `true`, then there were candidates that might or might
    // not have applied, but we couldn't tell. This occurs when some
    // of the input types are type variables, in which case there are
    // various "builtin" rules that might or might not trigger.
    ambiguous: bool,
}

#[derive(PartialEq, Eq, Debug, Clone)]
struct EvaluatedCandidate<'tcx> {
    candidate: SelectionCandidate<'tcx>,
    evaluation: EvaluationResult,
}

/// When does the builtin impl for `T: Trait` apply?
enum BuiltinImplConditions<'tcx> {
    /// The impl is conditional on `T1, T2, ...: Trait`.
    Where(ty::Binder<Vec<Ty<'tcx>>>),
    /// There is no built-in impl. There may be some other
    /// candidate (a where-clause or user-defined impl).
    None,
    /// It is unknown whether there is an impl.
    Ambiguous,
}

impl<'cx, 'tcx> SelectionContext<'cx, 'tcx> {
    pub fn new(infcx: &'cx InferCtxt<'cx, 'tcx>) -> SelectionContext<'cx, 'tcx> {
        SelectionContext {
            infcx,
            freshener: infcx.freshener(),
            intercrate: false,
            intercrate_ambiguity_causes: None,
            allow_negative_impls: false,
            query_mode: TraitQueryMode::Standard,
        }
    }

    pub fn intercrate(infcx: &'cx InferCtxt<'cx, 'tcx>) -> SelectionContext<'cx, 'tcx> {
        SelectionContext {
            infcx,
            freshener: infcx.freshener(),
            intercrate: true,
            intercrate_ambiguity_causes: None,
            allow_negative_impls: false,
            query_mode: TraitQueryMode::Standard,
        }
    }

    pub fn with_negative(
        infcx: &'cx InferCtxt<'cx, 'tcx>,
        allow_negative_impls: bool,
    ) -> SelectionContext<'cx, 'tcx> {
        debug!("with_negative({:?})", allow_negative_impls);
        SelectionContext {
            infcx,
            freshener: infcx.freshener(),
            intercrate: false,
            intercrate_ambiguity_causes: None,
            allow_negative_impls,
            query_mode: TraitQueryMode::Standard,
        }
    }

    pub fn with_query_mode(
        infcx: &'cx InferCtxt<'cx, 'tcx>,
        query_mode: TraitQueryMode,
    ) -> SelectionContext<'cx, 'tcx> {
        debug!("with_query_mode({:?})", query_mode);
        SelectionContext {
            infcx,
            freshener: infcx.freshener(),
            intercrate: false,
            intercrate_ambiguity_causes: None,
            allow_negative_impls: false,
            query_mode,
        }
    }

    /// Enables tracking of intercrate ambiguity causes. These are
    /// used in coherence to give improved diagnostics. We don't do
    /// this until we detect a coherence error because it can lead to
    /// false overflow results (#47139) and because it costs
    /// computation time.
    pub fn enable_tracking_intercrate_ambiguity_causes(&mut self) {
        assert!(self.intercrate);
        assert!(self.intercrate_ambiguity_causes.is_none());
        self.intercrate_ambiguity_causes = Some(vec![]);
        debug!("selcx: enable_tracking_intercrate_ambiguity_causes");
    }

    /// Gets the intercrate ambiguity causes collected since tracking
    /// was enabled and disables tracking at the same time. If
    /// tracking is not enabled, just returns an empty vector.
    pub fn take_intercrate_ambiguity_causes(&mut self) -> Vec<IntercrateAmbiguityCause> {
        assert!(self.intercrate);
        self.intercrate_ambiguity_causes.take().unwrap_or(vec![])
    }

    pub fn infcx(&self) -> &'cx InferCtxt<'cx, 'tcx> {
        self.infcx
    }

    pub fn tcx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    pub fn closure_typer(&self) -> &'cx InferCtxt<'cx, 'tcx> {
        self.infcx
    }

    ///////////////////////////////////////////////////////////////////////////
    // Selection
    //
    // The selection phase tries to identify *how* an obligation will
    // be resolved. For example, it will identify which impl or
    // parameter bound is to be used. The process can be inconclusive
    // if the self type in the obligation is not fully inferred. Selection
    // can result in an error in one of two ways:
    //
    // 1. If no applicable impl or parameter bound can be found.
    // 2. If the output type parameters in the obligation do not match
    //    those specified by the impl/bound. For example, if the obligation
    //    is `Vec<Foo>: Iterable<Bar>`, but the impl specifies
    //    `impl<T> Iterable<T> for Vec<T>`, than an error would result.

    /// Attempts to satisfy the obligation. If successful, this will affect the surrounding
    /// type environment by performing unification.
    pub fn select(
        &mut self,
        obligation: &TraitObligation<'tcx>,
    ) -> SelectionResult<'tcx, Selection<'tcx>> {
        debug!("select({:?})", obligation);
        debug_assert!(!obligation.predicate.has_escaping_bound_vars());

        let pec = &ProvisionalEvaluationCache::default();
        let stack = self.push_stack(TraitObligationStackList::empty(pec), obligation);

        let candidate = match self.candidate_from_obligation(&stack) {
            Err(SelectionError::Overflow) => {
                // In standard mode, overflow must have been caught and reported
                // earlier.
                assert!(self.query_mode == TraitQueryMode::Canonical);
                return Err(SelectionError::Overflow);
            }
            Err(e) => {
                return Err(e);
            }
            Ok(None) => {
                return Ok(None);
            }
            Ok(Some(candidate)) => candidate,
        };

        match self.confirm_candidate(obligation, candidate) {
            Err(SelectionError::Overflow) => {
                assert!(self.query_mode == TraitQueryMode::Canonical);
                Err(SelectionError::Overflow)
            }
            Err(e) => Err(e),
            Ok(candidate) => Ok(Some(candidate)),
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // EVALUATION
    //
    // Tests whether an obligation can be selected or whether an impl
    // can be applied to particular types. It skips the "confirmation"
    // step and hence completely ignores output type parameters.
    //
    // The result is "true" if the obligation *may* hold and "false" if
    // we can be sure it does not.

    /// Evaluates whether the obligation `obligation` can be satisfied (by any means).
    pub fn predicate_may_hold_fatal(&mut self, obligation: &PredicateObligation<'tcx>) -> bool {
        debug!("predicate_may_hold_fatal({:?})", obligation);

        // This fatal query is a stopgap that should only be used in standard mode,
        // where we do not expect overflow to be propagated.
        assert!(self.query_mode == TraitQueryMode::Standard);

        self.evaluate_root_obligation(obligation)
            .expect("Overflow should be caught earlier in standard query mode")
            .may_apply()
    }

    /// Evaluates whether the obligation `obligation` can be satisfied
    /// and returns an `EvaluationResult`. This is meant for the
    /// *initial* call.
    pub fn evaluate_root_obligation(
        &mut self,
        obligation: &PredicateObligation<'tcx>,
    ) -> Result<EvaluationResult, OverflowError> {
        self.evaluation_probe(|this| {
            this.evaluate_predicate_recursively(
                TraitObligationStackList::empty(&ProvisionalEvaluationCache::default()),
                obligation.clone(),
            )
        })
    }

    fn evaluation_probe(
        &mut self,
        op: impl FnOnce(&mut Self) -> Result<EvaluationResult, OverflowError>,
    ) -> Result<EvaluationResult, OverflowError> {
        self.infcx.probe(|snapshot| -> Result<EvaluationResult, OverflowError> {
            let result = op(self)?;
            match self.infcx.region_constraints_added_in_snapshot(snapshot) {
                None => Ok(result),
                Some(_) => Ok(result.max(EvaluatedToOkModuloRegions)),
            }
        })
    }

    /// Evaluates the predicates in `predicates` recursively. Note that
    /// this applies projections in the predicates, and therefore
    /// is run within an inference probe.
    fn evaluate_predicates_recursively<'o, I>(
        &mut self,
        stack: TraitObligationStackList<'o, 'tcx>,
        predicates: I,
    ) -> Result<EvaluationResult, OverflowError>
    where
        I: IntoIterator<Item = PredicateObligation<'tcx>>,
    {
        let mut result = EvaluatedToOk;
        for obligation in predicates {
            let eval = self.evaluate_predicate_recursively(stack, obligation.clone())?;
            debug!("evaluate_predicate_recursively({:?}) = {:?}", obligation, eval);
            if let EvaluatedToErr = eval {
                // fast-path - EvaluatedToErr is the top of the lattice,
                // so we don't need to look on the other predicates.
                return Ok(EvaluatedToErr);
            } else {
                result = cmp::max(result, eval);
            }
        }
        Ok(result)
    }

    fn evaluate_predicate_recursively<'o>(
        &mut self,
        previous_stack: TraitObligationStackList<'o, 'tcx>,
        obligation: PredicateObligation<'tcx>,
    ) -> Result<EvaluationResult, OverflowError> {
        debug!(
            "evaluate_predicate_recursively(previous_stack={:?}, obligation={:?})",
            previous_stack.head(),
            obligation
        );

        // `previous_stack` stores a `TraitObligatiom`, while `obligation` is
        // a `PredicateObligation`. These are distinct types, so we can't
        // use any `Option` combinator method that would force them to be
        // the same.
        match previous_stack.head() {
            Some(h) => self.check_recursion_limit(&obligation, h.obligation)?,
            None => self.check_recursion_limit(&obligation, &obligation)?,
        }

        match obligation.predicate {
            ty::Predicate::Trait(ref t, _) => {
                debug_assert!(!t.has_escaping_bound_vars());
                let obligation = obligation.with(t.clone());
                self.evaluate_trait_predicate_recursively(previous_stack, obligation)
            }

            ty::Predicate::Subtype(ref p) => {
                // Does this code ever run?
                match self.infcx.subtype_predicate(&obligation.cause, obligation.param_env, p) {
                    Some(Ok(InferOk { mut obligations, .. })) => {
                        self.add_depth(obligations.iter_mut(), obligation.recursion_depth);
                        self.evaluate_predicates_recursively(
                            previous_stack,
                            obligations.into_iter(),
                        )
                    }
                    Some(Err(_)) => Ok(EvaluatedToErr),
                    None => Ok(EvaluatedToAmbig),
                }
            }

            ty::Predicate::WellFormed(ty) => match wf::obligations(
                self.infcx,
                obligation.param_env,
                obligation.cause.body_id,
                ty,
                obligation.cause.span,
            ) {
                Some(mut obligations) => {
                    self.add_depth(obligations.iter_mut(), obligation.recursion_depth);
                    self.evaluate_predicates_recursively(previous_stack, obligations.into_iter())
                }
                None => Ok(EvaluatedToAmbig),
            },

            ty::Predicate::TypeOutlives(..) | ty::Predicate::RegionOutlives(..) => {
                // We do not consider region relationships when evaluating trait matches.
                Ok(EvaluatedToOkModuloRegions)
            }

            ty::Predicate::ObjectSafe(trait_def_id) => {
                if self.tcx().is_object_safe(trait_def_id) {
                    Ok(EvaluatedToOk)
                } else {
                    Ok(EvaluatedToErr)
                }
            }

            ty::Predicate::Projection(ref data) => {
                let project_obligation = obligation.with(data.clone());
                match project::poly_project_and_unify_type(self, &project_obligation) {
                    Ok(Some(mut subobligations)) => {
                        self.add_depth(subobligations.iter_mut(), obligation.recursion_depth);
                        let result = self.evaluate_predicates_recursively(
                            previous_stack,
                            subobligations.into_iter(),
                        );
                        if let Some(key) =
                            ProjectionCacheKey::from_poly_projection_predicate(self, data)
                        {
                            self.infcx.inner.borrow_mut().projection_cache.complete(key);
                        }
                        result
                    }
                    Ok(None) => Ok(EvaluatedToAmbig),
                    Err(_) => Ok(EvaluatedToErr),
                }
            }

            ty::Predicate::ClosureKind(closure_def_id, closure_substs, kind) => {
                match self.infcx.closure_kind(closure_def_id, closure_substs) {
                    Some(closure_kind) => {
                        if closure_kind.extends(kind) {
                            Ok(EvaluatedToOk)
                        } else {
                            Ok(EvaluatedToErr)
                        }
                    }
                    None => Ok(EvaluatedToAmbig),
                }
            }

            ty::Predicate::ConstEvaluatable(def_id, substs) => {
                match self.tcx().const_eval_resolve(
                    obligation.param_env,
                    def_id,
                    substs,
                    None,
                    None,
                ) {
                    Ok(_) => Ok(EvaluatedToOk),
                    Err(_) => Ok(EvaluatedToErr),
                }
            }
        }
    }

    fn evaluate_trait_predicate_recursively<'o>(
        &mut self,
        previous_stack: TraitObligationStackList<'o, 'tcx>,
        mut obligation: TraitObligation<'tcx>,
    ) -> Result<EvaluationResult, OverflowError> {
        debug!("evaluate_trait_predicate_recursively({:?})", obligation);

        if !self.intercrate
            && obligation.is_global()
            && obligation.param_env.caller_bounds.iter().all(|bound| bound.needs_subst())
        {
            // If a param env has no global bounds, global obligations do not
            // depend on its particular value in order to work, so we can clear
            // out the param env and get better caching.
            debug!("evaluate_trait_predicate_recursively({:?}) - in global", obligation);
            obligation.param_env = obligation.param_env.without_caller_bounds();
        }

        let stack = self.push_stack(previous_stack, &obligation);
        let fresh_trait_ref = stack.fresh_trait_ref;
        if let Some(result) = self.check_evaluation_cache(obligation.param_env, fresh_trait_ref) {
            debug!("CACHE HIT: EVAL({:?})={:?}", fresh_trait_ref, result);
            return Ok(result);
        }

        if let Some(result) = stack.cache().get_provisional(fresh_trait_ref) {
            debug!("PROVISIONAL CACHE HIT: EVAL({:?})={:?}", fresh_trait_ref, result);
            stack.update_reached_depth(stack.cache().current_reached_depth());
            return Ok(result);
        }

        // Check if this is a match for something already on the
        // stack. If so, we don't want to insert the result into the
        // main cache (it is cycle dependent) nor the provisional
        // cache (which is meant for things that have completed but
        // for a "backedge" -- this result *is* the backedge).
        if let Some(cycle_result) = self.check_evaluation_cycle(&stack) {
            return Ok(cycle_result);
        }

        let (result, dep_node) = self.in_task(|this| this.evaluate_stack(&stack));
        let result = result?;

        if !result.must_apply_modulo_regions() {
            stack.cache().on_failure(stack.dfn);
        }

        let reached_depth = stack.reached_depth.get();
        if reached_depth >= stack.depth {
            debug!("CACHE MISS: EVAL({:?})={:?}", fresh_trait_ref, result);
            self.insert_evaluation_cache(obligation.param_env, fresh_trait_ref, dep_node, result);

            stack.cache().on_completion(stack.depth, |fresh_trait_ref, provisional_result| {
                self.insert_evaluation_cache(
                    obligation.param_env,
                    fresh_trait_ref,
                    dep_node,
                    provisional_result.max(result),
                );
            });
        } else {
            debug!("PROVISIONAL: {:?}={:?}", fresh_trait_ref, result);
            debug!(
                "evaluate_trait_predicate_recursively: caching provisionally because {:?} \
                 is a cycle participant (at depth {}, reached depth {})",
                fresh_trait_ref, stack.depth, reached_depth,
            );

            stack.cache().insert_provisional(stack.dfn, reached_depth, fresh_trait_ref, result);
        }

        Ok(result)
    }

    /// If there is any previous entry on the stack that precisely
    /// matches this obligation, then we can assume that the
    /// obligation is satisfied for now (still all other conditions
    /// must be met of course). One obvious case this comes up is
    /// marker traits like `Send`. Think of a linked list:
    ///
    ///    struct List<T> { data: T, next: Option<Box<List<T>>> }
    ///
    /// `Box<List<T>>` will be `Send` if `T` is `Send` and
    /// `Option<Box<List<T>>>` is `Send`, and in turn
    /// `Option<Box<List<T>>>` is `Send` if `Box<List<T>>` is
    /// `Send`.
    ///
    /// Note that we do this comparison using the `fresh_trait_ref`
    /// fields. Because these have all been freshened using
    /// `self.freshener`, we can be sure that (a) this will not
    /// affect the inferencer state and (b) that if we see two
    /// fresh regions with the same index, they refer to the same
    /// unbound type variable.
    fn check_evaluation_cycle(
        &mut self,
        stack: &TraitObligationStack<'_, 'tcx>,
    ) -> Option<EvaluationResult> {
        if let Some(cycle_depth) = stack
            .iter()
            .skip(1) // Skip top-most frame.
            .find(|prev| {
                stack.obligation.param_env == prev.obligation.param_env
                    && stack.fresh_trait_ref == prev.fresh_trait_ref
            })
            .map(|stack| stack.depth)
        {
            debug!(
                "evaluate_stack({:?}) --> recursive at depth {}",
                stack.fresh_trait_ref, cycle_depth,
            );

            // If we have a stack like `A B C D E A`, where the top of
            // the stack is the final `A`, then this will iterate over
            // `A, E, D, C, B` -- i.e., all the participants apart
            // from the cycle head. We mark them as participating in a
            // cycle. This suppresses caching for those nodes. See
            // `in_cycle` field for more details.
            stack.update_reached_depth(cycle_depth);

            // Subtle: when checking for a coinductive cycle, we do
            // not compare using the "freshened trait refs" (which
            // have erased regions) but rather the fully explicit
            // trait refs. This is important because it's only a cycle
            // if the regions match exactly.
            let cycle = stack.iter().skip(1).take_while(|s| s.depth >= cycle_depth);
            let cycle = cycle.map(|stack| {
                ty::Predicate::Trait(stack.obligation.predicate, hir::Constness::NotConst)
            });
            if self.coinductive_match(cycle) {
                debug!("evaluate_stack({:?}) --> recursive, coinductive", stack.fresh_trait_ref);
                Some(EvaluatedToOk)
            } else {
                debug!("evaluate_stack({:?}) --> recursive, inductive", stack.fresh_trait_ref);
                Some(EvaluatedToRecur)
            }
        } else {
            None
        }
    }

    fn evaluate_stack<'o>(
        &mut self,
        stack: &TraitObligationStack<'o, 'tcx>,
    ) -> Result<EvaluationResult, OverflowError> {
        // In intercrate mode, whenever any of the types are unbound,
        // there can always be an impl. Even if there are no impls in
        // this crate, perhaps the type would be unified with
        // something from another crate that does provide an impl.
        //
        // In intra mode, we must still be conservative. The reason is
        // that we want to avoid cycles. Imagine an impl like:
        //
        //     impl<T:Eq> Eq for Vec<T>
        //
        // and a trait reference like `$0 : Eq` where `$0` is an
        // unbound variable. When we evaluate this trait-reference, we
        // will unify `$0` with `Vec<$1>` (for some fresh variable
        // `$1`), on the condition that `$1 : Eq`. We will then wind
        // up with many candidates (since that are other `Eq` impls
        // that apply) and try to winnow things down. This results in
        // a recursive evaluation that `$1 : Eq` -- as you can
        // imagine, this is just where we started. To avoid that, we
        // check for unbound variables and return an ambiguous (hence possible)
        // match if we've seen this trait before.
        //
        // This suffices to allow chains like `FnMut` implemented in
        // terms of `Fn` etc, but we could probably make this more
        // precise still.
        let unbound_input_types =
            stack.fresh_trait_ref.skip_binder().input_types().any(|ty| ty.is_fresh());
        // This check was an imperfect workaround for a bug in the old
        // intercrate mode; it should be removed when that goes away.
        if unbound_input_types && self.intercrate {
            debug!(
                "evaluate_stack({:?}) --> unbound argument, intercrate -->  ambiguous",
                stack.fresh_trait_ref
            );
            // Heuristics: show the diagnostics when there are no candidates in crate.
            if self.intercrate_ambiguity_causes.is_some() {
                debug!("evaluate_stack: intercrate_ambiguity_causes is some");
                if let Ok(candidate_set) = self.assemble_candidates(stack) {
                    if !candidate_set.ambiguous && candidate_set.vec.is_empty() {
                        let trait_ref = stack.obligation.predicate.skip_binder().trait_ref;
                        let self_ty = trait_ref.self_ty();
                        let cause = IntercrateAmbiguityCause::DownstreamCrate {
                            trait_desc: trait_ref.print_only_trait_path().to_string(),
                            self_desc: if self_ty.has_concrete_skeleton() {
                                Some(self_ty.to_string())
                            } else {
                                None
                            },
                        };
                        debug!("evaluate_stack: pushing cause = {:?}", cause);
                        self.intercrate_ambiguity_causes.as_mut().unwrap().push(cause);
                    }
                }
            }
            return Ok(EvaluatedToAmbig);
        }
        if unbound_input_types
            && stack.iter().skip(1).any(|prev| {
                stack.obligation.param_env == prev.obligation.param_env
                    && self.match_fresh_trait_refs(
                        &stack.fresh_trait_ref,
                        &prev.fresh_trait_ref,
                        prev.obligation.param_env,
                    )
            })
        {
            debug!(
                "evaluate_stack({:?}) --> unbound argument, recursive --> giving up",
                stack.fresh_trait_ref
            );
            return Ok(EvaluatedToUnknown);
        }

        match self.candidate_from_obligation(stack) {
            Ok(Some(c)) => self.evaluate_candidate(stack, &c),
            Ok(None) => Ok(EvaluatedToAmbig),
            Err(Overflow) => Err(OverflowError),
            Err(..) => Ok(EvaluatedToErr),
        }
    }

    /// For defaulted traits, we use a co-inductive strategy to solve, so
    /// that recursion is ok. This routine returns `true` if the top of the
    /// stack (`cycle[0]`):
    ///
    /// - is a defaulted trait,
    /// - it also appears in the backtrace at some position `X`,
    /// - all the predicates at positions `X..` between `X` and the top are
    ///   also defaulted traits.
    pub fn coinductive_match<I>(&mut self, cycle: I) -> bool
    where
        I: Iterator<Item = ty::Predicate<'tcx>>,
    {
        let mut cycle = cycle;
        cycle.all(|predicate| self.coinductive_predicate(predicate))
    }

    fn coinductive_predicate(&self, predicate: ty::Predicate<'tcx>) -> bool {
        let result = match predicate {
            ty::Predicate::Trait(ref data, _) => self.tcx().trait_is_auto(data.def_id()),
            _ => false,
        };
        debug!("coinductive_predicate({:?}) = {:?}", predicate, result);
        result
    }

    /// Further evaluates `candidate` to decide whether all type parameters match and whether nested
    /// obligations are met. Returns whether `candidate` remains viable after this further
    /// scrutiny.
    fn evaluate_candidate<'o>(
        &mut self,
        stack: &TraitObligationStack<'o, 'tcx>,
        candidate: &SelectionCandidate<'tcx>,
    ) -> Result<EvaluationResult, OverflowError> {
        debug!(
            "evaluate_candidate: depth={} candidate={:?}",
            stack.obligation.recursion_depth, candidate
        );
        let result = self.evaluation_probe(|this| {
            let candidate = (*candidate).clone();
            match this.confirm_candidate(stack.obligation, candidate) {
                Ok(selection) => this.evaluate_predicates_recursively(
                    stack.list(),
                    selection.nested_obligations().into_iter(),
                ),
                Err(..) => Ok(EvaluatedToErr),
            }
        })?;
        debug!(
            "evaluate_candidate: depth={} result={:?}",
            stack.obligation.recursion_depth, result
        );
        Ok(result)
    }

    fn check_evaluation_cache(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        trait_ref: ty::PolyTraitRef<'tcx>,
    ) -> Option<EvaluationResult> {
        let tcx = self.tcx();
        if self.can_use_global_caches(param_env) {
            let cache = tcx.evaluation_cache.hashmap.borrow();
            if let Some(cached) = cache.get(&param_env.and(trait_ref)) {
                return Some(cached.get(tcx));
            }
        }
        self.infcx
            .evaluation_cache
            .hashmap
            .borrow()
            .get(&param_env.and(trait_ref))
            .map(|v| v.get(tcx))
    }

    fn insert_evaluation_cache(
        &mut self,
        param_env: ty::ParamEnv<'tcx>,
        trait_ref: ty::PolyTraitRef<'tcx>,
        dep_node: DepNodeIndex,
        result: EvaluationResult,
    ) {
        // Avoid caching results that depend on more than just the trait-ref
        // - the stack can create recursion.
        if result.is_stack_dependent() {
            return;
        }

        if self.can_use_global_caches(param_env) {
            if !trait_ref.has_local_value() {
                debug!(
                    "insert_evaluation_cache(trait_ref={:?}, candidate={:?}) global",
                    trait_ref, result,
                );
                // This may overwrite the cache with the same value
                // FIXME: Due to #50507 this overwrites the different values
                // This should be changed to use HashMapExt::insert_same
                // when that is fixed
                self.tcx()
                    .evaluation_cache
                    .hashmap
                    .borrow_mut()
                    .insert(param_env.and(trait_ref), WithDepNode::new(dep_node, result));
                return;
            }
        }

        debug!("insert_evaluation_cache(trait_ref={:?}, candidate={:?})", trait_ref, result,);
        self.infcx
            .evaluation_cache
            .hashmap
            .borrow_mut()
            .insert(param_env.and(trait_ref), WithDepNode::new(dep_node, result));
    }

    /// For various reasons, it's possible for a subobligation
    /// to have a *lower* recursion_depth than the obligation used to create it.
    /// Projection sub-obligations may be returned from the projection cache,
    /// which results in obligations with an 'old' `recursion_depth`.
    /// Additionally, methods like `wf::obligations` and
    /// `InferCtxt.subtype_predicate` produce subobligations without
    /// taking in a 'parent' depth, causing the generated subobligations
    /// to have a `recursion_depth` of `0`.
    ///
    /// To ensure that obligation_depth never decreasees, we force all subobligations
    /// to have at least the depth of the original obligation.
    fn add_depth<T: 'cx, I: Iterator<Item = &'cx mut Obligation<'tcx, T>>>(
        &self,
        it: I,
        min_depth: usize,
    ) {
        it.for_each(|o| o.recursion_depth = cmp::max(min_depth, o.recursion_depth) + 1);
    }

    /// Checks that the recursion limit has not been exceeded.
    ///
    /// The weird return type of this function allows it to be used with the `try` (`?`)
    /// operator within certain functions.
    fn check_recursion_limit<T: Display + TypeFoldable<'tcx>, V: Display + TypeFoldable<'tcx>>(
        &self,
        obligation: &Obligation<'tcx, T>,
        error_obligation: &Obligation<'tcx, V>,
    ) -> Result<(), OverflowError> {
        let recursion_limit = *self.infcx.tcx.sess.recursion_limit.get();
        if obligation.recursion_depth >= recursion_limit {
            match self.query_mode {
                TraitQueryMode::Standard => {
                    self.infcx().report_overflow_error(error_obligation, true);
                }
                TraitQueryMode::Canonical => {
                    return Err(OverflowError);
                }
            }
        }
        Ok(())
    }

    ///////////////////////////////////////////////////////////////////////////
    // CANDIDATE ASSEMBLY
    //
    // The selection process begins by examining all in-scope impls,
    // caller obligations, and so forth and assembling a list of
    // candidates. See the [rustc dev guide] for more details.
    //
    // [rustc dev guide]:
    // https://rustc-dev-guide.rust-lang.org/traits/resolution.html#candidate-assembly

    fn candidate_from_obligation<'o>(
        &mut self,
        stack: &TraitObligationStack<'o, 'tcx>,
    ) -> SelectionResult<'tcx, SelectionCandidate<'tcx>> {
        // Watch out for overflow. This intentionally bypasses (and does
        // not update) the cache.
        self.check_recursion_limit(&stack.obligation, &stack.obligation)?;

        // Check the cache. Note that we freshen the trait-ref
        // separately rather than using `stack.fresh_trait_ref` --
        // this is because we want the unbound variables to be
        // replaced with fresh types starting from index 0.
        let cache_fresh_trait_pred = self.infcx.freshen(stack.obligation.predicate.clone());
        debug!(
            "candidate_from_obligation(cache_fresh_trait_pred={:?}, obligation={:?})",
            cache_fresh_trait_pred, stack
        );
        debug_assert!(!stack.obligation.predicate.has_escaping_bound_vars());

        if let Some(c) =
            self.check_candidate_cache(stack.obligation.param_env, &cache_fresh_trait_pred)
        {
            debug!("CACHE HIT: SELECT({:?})={:?}", cache_fresh_trait_pred, c);
            return c;
        }

        // If no match, compute result and insert into cache.
        //
        // FIXME(nikomatsakis) -- this cache is not taking into
        // account cycles that may have occurred in forming the
        // candidate. I don't know of any specific problems that
        // result but it seems awfully suspicious.
        let (candidate, dep_node) =
            self.in_task(|this| this.candidate_from_obligation_no_cache(stack));

        debug!("CACHE MISS: SELECT({:?})={:?}", cache_fresh_trait_pred, candidate);
        self.insert_candidate_cache(
            stack.obligation.param_env,
            cache_fresh_trait_pred,
            dep_node,
            candidate.clone(),
        );
        candidate
    }

    fn in_task<OP, R>(&mut self, op: OP) -> (R, DepNodeIndex)
    where
        OP: FnOnce(&mut Self) -> R,
    {
        let (result, dep_node) =
            self.tcx().dep_graph.with_anon_task(DepKind::TraitSelect, || op(self));
        self.tcx().dep_graph.read_index(dep_node);
        (result, dep_node)
    }

    // Treat negative impls as unimplemented, and reservation impls as ambiguity.
    fn filter_negative_and_reservation_impls(
        &mut self,
        candidate: SelectionCandidate<'tcx>,
    ) -> SelectionResult<'tcx, SelectionCandidate<'tcx>> {
        if let ImplCandidate(def_id) = candidate {
            let tcx = self.tcx();
            match tcx.impl_polarity(def_id) {
                ty::ImplPolarity::Negative if !self.allow_negative_impls => {
                    return Err(Unimplemented);
                }
                ty::ImplPolarity::Reservation => {
                    if let Some(intercrate_ambiguity_clauses) =
                        &mut self.intercrate_ambiguity_causes
                    {
                        let attrs = tcx.get_attrs(def_id);
                        let attr = attr::find_by_name(&attrs, sym::rustc_reservation_impl);
                        let value = attr.and_then(|a| a.value_str());
                        if let Some(value) = value {
                            debug!(
                                "filter_negative_and_reservation_impls: \
                                 reservation impl ambiguity on {:?}",
                                def_id
                            );
                            intercrate_ambiguity_clauses.push(
                                IntercrateAmbiguityCause::ReservationImpl {
                                    message: value.to_string(),
                                },
                            );
                        }
                    }
                    return Ok(None);
                }
                _ => {}
            };
        }
        Ok(Some(candidate))
    }

    fn candidate_from_obligation_no_cache<'o>(
        &mut self,
        stack: &TraitObligationStack<'o, 'tcx>,
    ) -> SelectionResult<'tcx, SelectionCandidate<'tcx>> {
        if stack.obligation.predicate.references_error() {
            // If we encounter a `Error`, we generally prefer the
            // most "optimistic" result in response -- that is, the
            // one least likely to report downstream errors. But
            // because this routine is shared by coherence and by
            // trait selection, there isn't an obvious "right" choice
            // here in that respect, so we opt to just return
            // ambiguity and let the upstream clients sort it out.
            return Ok(None);
        }

        if let Some(conflict) = self.is_knowable(stack) {
            debug!("coherence stage: not knowable");
            if self.intercrate_ambiguity_causes.is_some() {
                debug!("evaluate_stack: intercrate_ambiguity_causes is some");
                // Heuristics: show the diagnostics when there are no candidates in crate.
                if let Ok(candidate_set) = self.assemble_candidates(stack) {
                    let mut no_candidates_apply = true;
                    {
                        let evaluated_candidates =
                            candidate_set.vec.iter().map(|c| self.evaluate_candidate(stack, &c));

                        for ec in evaluated_candidates {
                            match ec {
                                Ok(c) => {
                                    if c.may_apply() {
                                        no_candidates_apply = false;
                                        break;
                                    }
                                }
                                Err(e) => return Err(e.into()),
                            }
                        }
                    }

                    if !candidate_set.ambiguous && no_candidates_apply {
                        let trait_ref = stack.obligation.predicate.skip_binder().trait_ref;
                        let self_ty = trait_ref.self_ty();
                        let trait_desc = trait_ref.print_only_trait_path().to_string();
                        let self_desc = if self_ty.has_concrete_skeleton() {
                            Some(self_ty.to_string())
                        } else {
                            None
                        };
                        let cause = if let Conflict::Upstream = conflict {
                            IntercrateAmbiguityCause::UpstreamCrateUpdate { trait_desc, self_desc }
                        } else {
                            IntercrateAmbiguityCause::DownstreamCrate { trait_desc, self_desc }
                        };
                        debug!("evaluate_stack: pushing cause = {:?}", cause);
                        self.intercrate_ambiguity_causes.as_mut().unwrap().push(cause);
                    }
                }
            }
            return Ok(None);
        }

        let candidate_set = self.assemble_candidates(stack)?;

        if candidate_set.ambiguous {
            debug!("candidate set contains ambig");
            return Ok(None);
        }

        let mut candidates = candidate_set.vec;

        debug!("assembled {} candidates for {:?}: {:?}", candidates.len(), stack, candidates);

        // At this point, we know that each of the entries in the
        // candidate set is *individually* applicable. Now we have to
        // figure out if they contain mutual incompatibilities. This
        // frequently arises if we have an unconstrained input type --
        // for example, we are looking for `$0: Eq` where `$0` is some
        // unconstrained type variable. In that case, we'll get a
        // candidate which assumes $0 == int, one that assumes `$0 ==
        // usize`, etc. This spells an ambiguity.

        // If there is more than one candidate, first winnow them down
        // by considering extra conditions (nested obligations and so
        // forth). We don't winnow if there is exactly one
        // candidate. This is a relatively minor distinction but it
        // can lead to better inference and error-reporting. An
        // example would be if there was an impl:
        //
        //     impl<T:Clone> Vec<T> { fn push_clone(...) { ... } }
        //
        // and we were to see some code `foo.push_clone()` where `boo`
        // is a `Vec<Bar>` and `Bar` does not implement `Clone`.  If
        // we were to winnow, we'd wind up with zero candidates.
        // Instead, we select the right impl now but report "`Bar` does
        // not implement `Clone`".
        if candidates.len() == 1 {
            return self.filter_negative_and_reservation_impls(candidates.pop().unwrap());
        }

        // Winnow, but record the exact outcome of evaluation, which
        // is needed for specialization. Propagate overflow if it occurs.
        let mut candidates = candidates
            .into_iter()
            .map(|c| match self.evaluate_candidate(stack, &c) {
                Ok(eval) if eval.may_apply() => {
                    Ok(Some(EvaluatedCandidate { candidate: c, evaluation: eval }))
                }
                Ok(_) => Ok(None),
                Err(OverflowError) => Err(Overflow),
            })
            .flat_map(Result::transpose)
            .collect::<Result<Vec<_>, _>>()?;

        debug!("winnowed to {} candidates for {:?}: {:?}", candidates.len(), stack, candidates);

        let needs_infer = stack.obligation.predicate.needs_infer();

        // If there are STILL multiple candidates, we can further
        // reduce the list by dropping duplicates -- including
        // resolving specializations.
        if candidates.len() > 1 {
            let mut i = 0;
            while i < candidates.len() {
                let is_dup = (0..candidates.len()).filter(|&j| i != j).any(|j| {
                    self.candidate_should_be_dropped_in_favor_of(
                        &candidates[i],
                        &candidates[j],
                        needs_infer,
                    )
                });
                if is_dup {
                    debug!("Dropping candidate #{}/{}: {:?}", i, candidates.len(), candidates[i]);
                    candidates.swap_remove(i);
                } else {
                    debug!("Retaining candidate #{}/{}: {:?}", i, candidates.len(), candidates[i]);
                    i += 1;

                    // If there are *STILL* multiple candidates, give up
                    // and report ambiguity.
                    if i > 1 {
                        debug!("multiple matches, ambig");
                        return Ok(None);
                    }
                }
            }
        }

        // If there are *NO* candidates, then there are no impls --
        // that we know of, anyway. Note that in the case where there
        // are unbound type variables within the obligation, it might
        // be the case that you could still satisfy the obligation
        // from another crate by instantiating the type variables with
        // a type from another crate that does have an impl. This case
        // is checked for in `evaluate_stack` (and hence users
        // who might care about this case, like coherence, should use
        // that function).
        if candidates.is_empty() {
            return Err(Unimplemented);
        }

        // Just one candidate left.
        self.filter_negative_and_reservation_impls(candidates.pop().unwrap().candidate)
    }

    fn is_knowable<'o>(&mut self, stack: &TraitObligationStack<'o, 'tcx>) -> Option<Conflict> {
        debug!("is_knowable(intercrate={:?})", self.intercrate);

        if !self.intercrate {
            return None;
        }

        let obligation = &stack.obligation;
        let predicate = self.infcx().resolve_vars_if_possible(&obligation.predicate);

        // Okay to skip binder because of the nature of the
        // trait-ref-is-knowable check, which does not care about
        // bound regions.
        let trait_ref = predicate.skip_binder().trait_ref;

        coherence::trait_ref_is_knowable(self.tcx(), trait_ref)
    }

    /// Returns `true` if the global caches can be used.
    /// Do note that if the type itself is not in the
    /// global tcx, the local caches will be used.
    fn can_use_global_caches(&self, param_env: ty::ParamEnv<'tcx>) -> bool {
        // If there are any e.g. inference variables in the `ParamEnv`, then we
        // always use a cache local to this particular scope. Otherwise, we
        // switch to a global cache.
        if param_env.has_local_value() {
            return false;
        }

        // Avoid using the master cache during coherence and just rely
        // on the local cache. This effectively disables caching
        // during coherence. It is really just a simplification to
        // avoid us having to fear that coherence results "pollute"
        // the master cache. Since coherence executes pretty quickly,
        // it's not worth going to more trouble to increase the
        // hit-rate, I don't think.
        if self.intercrate {
            return false;
        }

        // Otherwise, we can use the global cache.
        true
    }

    fn check_candidate_cache(
        &mut self,
        param_env: ty::ParamEnv<'tcx>,
        cache_fresh_trait_pred: &ty::PolyTraitPredicate<'tcx>,
    ) -> Option<SelectionResult<'tcx, SelectionCandidate<'tcx>>> {
        let tcx = self.tcx();
        let trait_ref = &cache_fresh_trait_pred.skip_binder().trait_ref;
        if self.can_use_global_caches(param_env) {
            let cache = tcx.selection_cache.hashmap.borrow();
            if let Some(cached) = cache.get(&param_env.and(*trait_ref)) {
                return Some(cached.get(tcx));
            }
        }
        self.infcx
            .selection_cache
            .hashmap
            .borrow()
            .get(&param_env.and(*trait_ref))
            .map(|v| v.get(tcx))
    }

    /// Determines whether can we safely cache the result
    /// of selecting an obligation. This is almost always `true`,
    /// except when dealing with certain `ParamCandidate`s.
    ///
    /// Ordinarily, a `ParamCandidate` will contain no inference variables,
    /// since it was usually produced directly from a `DefId`. However,
    /// certain cases (currently only librustdoc's blanket impl finder),
    /// a `ParamEnv` may be explicitly constructed with inference types.
    /// When this is the case, we do *not* want to cache the resulting selection
    /// candidate. This is due to the fact that it might not always be possible
    /// to equate the obligation's trait ref and the candidate's trait ref,
    /// if more constraints end up getting added to an inference variable.
    ///
    /// Because of this, we always want to re-run the full selection
    /// process for our obligation the next time we see it, since
    /// we might end up picking a different `SelectionCandidate` (or none at all).
    fn can_cache_candidate(
        &self,
        result: &SelectionResult<'tcx, SelectionCandidate<'tcx>>,
    ) -> bool {
        match result {
            Ok(Some(SelectionCandidate::ParamCandidate(trait_ref))) => {
                !trait_ref.skip_binder().input_types().any(|t| t.walk().any(|t_| t_.is_ty_infer()))
            }
            _ => true,
        }
    }

    fn insert_candidate_cache(
        &mut self,
        param_env: ty::ParamEnv<'tcx>,
        cache_fresh_trait_pred: ty::PolyTraitPredicate<'tcx>,
        dep_node: DepNodeIndex,
        candidate: SelectionResult<'tcx, SelectionCandidate<'tcx>>,
    ) {
        let tcx = self.tcx();
        let trait_ref = cache_fresh_trait_pred.skip_binder().trait_ref;

        if !self.can_cache_candidate(&candidate) {
            debug!(
                "insert_candidate_cache(trait_ref={:?}, candidate={:?} -\
                 candidate is not cacheable",
                trait_ref, candidate
            );
            return;
        }

        if self.can_use_global_caches(param_env) {
            if let Err(Overflow) = candidate {
                // Don't cache overflow globally; we only produce this in certain modes.
            } else if !trait_ref.has_local_value() {
                if !candidate.has_local_value() {
                    debug!(
                        "insert_candidate_cache(trait_ref={:?}, candidate={:?}) global",
                        trait_ref, candidate,
                    );
                    // This may overwrite the cache with the same value.
                    tcx.selection_cache
                        .hashmap
                        .borrow_mut()
                        .insert(param_env.and(trait_ref), WithDepNode::new(dep_node, candidate));
                    return;
                }
            }
        }

        debug!(
            "insert_candidate_cache(trait_ref={:?}, candidate={:?}) local",
            trait_ref, candidate,
        );
        self.infcx
            .selection_cache
            .hashmap
            .borrow_mut()
            .insert(param_env.and(trait_ref), WithDepNode::new(dep_node, candidate));
    }

    fn assemble_candidates<'o>(
        &mut self,
        stack: &TraitObligationStack<'o, 'tcx>,
    ) -> Result<SelectionCandidateSet<'tcx>, SelectionError<'tcx>> {
        let TraitObligationStack { obligation, .. } = *stack;
        let obligation = &Obligation {
            param_env: obligation.param_env,
            cause: obligation.cause.clone(),
            recursion_depth: obligation.recursion_depth,
            predicate: self.infcx().resolve_vars_if_possible(&obligation.predicate),
        };

        if obligation.predicate.skip_binder().self_ty().is_ty_var() {
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

        self.assemble_candidates_for_trait_alias(obligation, &mut candidates)?;

        // Other bounds. Consider both in-scope bounds from fn decl
        // and applicable impls. There is a certain set of precedence rules here.
        let def_id = obligation.predicate.def_id();
        let lang_items = self.tcx().lang_items();

        if lang_items.copy_trait() == Some(def_id) {
            debug!("obligation self ty is {:?}", obligation.predicate.skip_binder().self_ty());

            // User-defined copy impls are permitted, but only for
            // structs and enums.
            self.assemble_candidates_from_impls(obligation, &mut candidates)?;

            // For other types, we'll use the builtin rules.
            let copy_conditions = self.copy_clone_conditions(obligation);
            self.assemble_builtin_bound_candidates(copy_conditions, &mut candidates)?;
        } else if lang_items.sized_trait() == Some(def_id) {
            // Sized is never implementable by end-users, it is
            // always automatically computed.
            let sized_conditions = self.sized_conditions(obligation);
            self.assemble_builtin_bound_candidates(sized_conditions, &mut candidates)?;
        } else if lang_items.unsize_trait() == Some(def_id) {
            self.assemble_candidates_for_unsizing(obligation, &mut candidates);
        } else {
            if lang_items.clone_trait() == Some(def_id) {
                // Same builtin conditions as `Copy`, i.e., every type which has builtin support
                // for `Copy` also has builtin support for `Clone`, and tuples/arrays of `Clone`
                // types have builtin support for `Clone`.
                let clone_conditions = self.copy_clone_conditions(obligation);
                self.assemble_builtin_bound_candidates(clone_conditions, &mut candidates)?;
            }

            self.assemble_generator_candidates(obligation, &mut candidates)?;
            self.assemble_closure_candidates(obligation, &mut candidates)?;
            self.assemble_fn_pointer_candidates(obligation, &mut candidates)?;
            self.assemble_candidates_from_impls(obligation, &mut candidates)?;
            self.assemble_candidates_from_object_ty(obligation, &mut candidates);
        }

        self.assemble_candidates_from_projected_tys(obligation, &mut candidates);
        self.assemble_candidates_from_caller_bounds(stack, &mut candidates)?;
        // Auto implementations have lower priority, so we only
        // consider triggering a default if there is no other impl that can apply.
        if candidates.vec.is_empty() {
            self.assemble_candidates_from_auto_impls(obligation, &mut candidates)?;
        }
        debug!("candidate list size: {}", candidates.vec.len());
        Ok(candidates)
    }

    fn assemble_candidates_from_projected_tys(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) {
        debug!("assemble_candidates_for_projected_tys({:?})", obligation);

        // Before we go into the whole placeholder thing, just
        // quickly check if the self-type is a projection at all.
        match obligation.predicate.skip_binder().trait_ref.self_ty().kind {
            ty::Projection(_) | ty::Opaque(..) => {}
            ty::Infer(ty::TyVar(_)) => {
                span_bug!(
                    obligation.cause.span,
                    "Self=_ should have been handled by assemble_candidates"
                );
            }
            _ => return,
        }

        let result = self.infcx.probe(|snapshot| {
            self.match_projection_obligation_against_definition_bounds(obligation, snapshot)
        });

        if result {
            candidates.vec.push(ProjectionCandidate);
        }
    }

    fn match_projection_obligation_against_definition_bounds(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        snapshot: &CombinedSnapshot<'_, 'tcx>,
    ) -> bool {
        let poly_trait_predicate = self.infcx().resolve_vars_if_possible(&obligation.predicate);
        let (placeholder_trait_predicate, placeholder_map) =
            self.infcx().replace_bound_vars_with_placeholders(&poly_trait_predicate);
        debug!(
            "match_projection_obligation_against_definition_bounds: \
             placeholder_trait_predicate={:?}",
            placeholder_trait_predicate,
        );

        let (def_id, substs) = match placeholder_trait_predicate.trait_ref.self_ty().kind {
            ty::Projection(ref data) => (data.trait_ref(self.tcx()).def_id, data.substs),
            ty::Opaque(def_id, substs) => (def_id, substs),
            _ => {
                span_bug!(
                    obligation.cause.span,
                    "match_projection_obligation_against_definition_bounds() called \
                     but self-ty is not a projection: {:?}",
                    placeholder_trait_predicate.trait_ref.self_ty()
                );
            }
        };
        debug!(
            "match_projection_obligation_against_definition_bounds: \
             def_id={:?}, substs={:?}",
            def_id, substs
        );

        let predicates_of = self.tcx().predicates_of(def_id);
        let bounds = predicates_of.instantiate(self.tcx(), substs);
        debug!(
            "match_projection_obligation_against_definition_bounds: \
             bounds={:?}",
            bounds
        );

        let elaborated_predicates = util::elaborate_predicates(self.tcx(), bounds.predicates);
        let matching_bound = elaborated_predicates.filter_to_traits().find(|bound| {
            self.infcx.probe(|_| {
                self.match_projection(
                    obligation,
                    bound.clone(),
                    placeholder_trait_predicate.trait_ref.clone(),
                    &placeholder_map,
                    snapshot,
                )
            })
        });

        debug!(
            "match_projection_obligation_against_definition_bounds: \
             matching_bound={:?}",
            matching_bound
        );
        match matching_bound {
            None => false,
            Some(bound) => {
                // Repeat the successful match, if any, this time outside of a probe.
                let result = self.match_projection(
                    obligation,
                    bound,
                    placeholder_trait_predicate.trait_ref.clone(),
                    &placeholder_map,
                    snapshot,
                );

                assert!(result);
                true
            }
        }
    }

    fn match_projection(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        trait_bound: ty::PolyTraitRef<'tcx>,
        placeholder_trait_ref: ty::TraitRef<'tcx>,
        placeholder_map: &PlaceholderMap<'tcx>,
        snapshot: &CombinedSnapshot<'_, 'tcx>,
    ) -> bool {
        debug_assert!(!placeholder_trait_ref.has_escaping_bound_vars());
        self.infcx
            .at(&obligation.cause, obligation.param_env)
            .sup(ty::Binder::dummy(placeholder_trait_ref), trait_bound)
            .is_ok()
            && self.infcx.leak_check(false, placeholder_map, snapshot).is_ok()
    }

    /// Given an obligation like `<SomeTrait for T>`, searches the obligations that the caller
    /// supplied to find out whether it is listed among them.
    ///
    /// Never affects the inference environment.
    fn assemble_candidates_from_caller_bounds<'o>(
        &mut self,
        stack: &TraitObligationStack<'o, 'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) -> Result<(), SelectionError<'tcx>> {
        debug!("assemble_candidates_from_caller_bounds({:?})", stack.obligation);

        let all_bounds = stack
            .obligation
            .param_env
            .caller_bounds
            .iter()
            .filter_map(|o| o.to_opt_poly_trait_ref());

        // Micro-optimization: filter out predicates relating to different traits.
        let matching_bounds =
            all_bounds.filter(|p| p.def_id() == stack.obligation.predicate.def_id());

        // Keep only those bounds which may apply, and propagate overflow if it occurs.
        let mut param_candidates = vec![];
        for bound in matching_bounds {
            let wc = self.evaluate_where_clause(stack, bound.clone())?;
            if wc.may_apply() {
                param_candidates.push(ParamCandidate(bound));
            }
        }

        candidates.vec.extend(param_candidates);

        Ok(())
    }

    fn evaluate_where_clause<'o>(
        &mut self,
        stack: &TraitObligationStack<'o, 'tcx>,
        where_clause_trait_ref: ty::PolyTraitRef<'tcx>,
    ) -> Result<EvaluationResult, OverflowError> {
        self.evaluation_probe(|this| {
            match this.match_where_clause_trait_ref(stack.obligation, where_clause_trait_ref) {
                Ok(obligations) => {
                    this.evaluate_predicates_recursively(stack.list(), obligations.into_iter())
                }
                Err(()) => Ok(EvaluatedToErr),
            }
        })
    }

    fn assemble_generator_candidates(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) -> Result<(), SelectionError<'tcx>> {
        if self.tcx().lang_items().gen_trait() != Some(obligation.predicate.def_id()) {
            return Ok(());
        }

        // Okay to skip binder because the substs on generator types never
        // touch bound regions, they just capture the in-scope
        // type/region parameters.
        let self_ty = *obligation.self_ty().skip_binder();
        match self_ty.kind {
            ty::Generator(..) => {
                debug!(
                    "assemble_generator_candidates: self_ty={:?} obligation={:?}",
                    self_ty, obligation
                );

                candidates.vec.push(GeneratorCandidate);
            }
            ty::Infer(ty::TyVar(_)) => {
                debug!("assemble_generator_candidates: ambiguous self-type");
                candidates.ambiguous = true;
            }
            _ => {}
        }

        Ok(())
    }

    /// Checks for the artificial impl that the compiler will create for an obligation like `X :
    /// FnMut<..>` where `X` is a closure type.
    ///
    /// Note: the type parameters on a closure candidate are modeled as *output* type
    /// parameters and hence do not affect whether this trait is a match or not. They will be
    /// unified during the confirmation step.
    fn assemble_closure_candidates(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) -> Result<(), SelectionError<'tcx>> {
        let kind = match self.tcx().fn_trait_kind_from_lang_item(obligation.predicate.def_id()) {
            Some(k) => k,
            None => {
                return Ok(());
            }
        };

        // Okay to skip binder because the substs on closure types never
        // touch bound regions, they just capture the in-scope
        // type/region parameters
        match obligation.self_ty().skip_binder().kind {
            ty::Closure(closure_def_id, closure_substs) => {
                debug!("assemble_unboxed_candidates: kind={:?} obligation={:?}", kind, obligation);
                match self.infcx.closure_kind(closure_def_id, closure_substs) {
                    Some(closure_kind) => {
                        debug!("assemble_unboxed_candidates: closure_kind = {:?}", closure_kind);
                        if closure_kind.extends(kind) {
                            candidates.vec.push(ClosureCandidate);
                        }
                    }
                    None => {
                        debug!("assemble_unboxed_candidates: closure_kind not yet known");
                        candidates.vec.push(ClosureCandidate);
                    }
                }
            }
            ty::Infer(ty::TyVar(_)) => {
                debug!("assemble_unboxed_closure_candidates: ambiguous self-type");
                candidates.ambiguous = true;
            }
            _ => {}
        }

        Ok(())
    }

    /// Implements one of the `Fn()` family for a fn pointer.
    fn assemble_fn_pointer_candidates(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) -> Result<(), SelectionError<'tcx>> {
        // We provide impl of all fn traits for fn pointers.
        if self.tcx().fn_trait_kind_from_lang_item(obligation.predicate.def_id()).is_none() {
            return Ok(());
        }

        // Okay to skip binder because what we are inspecting doesn't involve bound regions.
        let self_ty = *obligation.self_ty().skip_binder();
        match self_ty.kind {
            ty::Infer(ty::TyVar(_)) => {
                debug!("assemble_fn_pointer_candidates: ambiguous self-type");
                candidates.ambiguous = true; // Could wind up being a fn() type.
            }
            // Provide an impl, but only for suitable `fn` pointers.
            ty::FnDef(..) | ty::FnPtr(_) => {
                if let ty::FnSig {
                    unsafety: hir::Unsafety::Normal,
                    abi: Abi::Rust,
                    c_variadic: false,
                    ..
                } = self_ty.fn_sig(self.tcx()).skip_binder()
                {
                    candidates.vec.push(FnPointerCandidate);
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Searches for impls that might apply to `obligation`.
    fn assemble_candidates_from_impls(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) -> Result<(), SelectionError<'tcx>> {
        debug!("assemble_candidates_from_impls(obligation={:?})", obligation);

        self.tcx().for_each_relevant_impl(
            obligation.predicate.def_id(),
            obligation.predicate.skip_binder().trait_ref.self_ty(),
            |impl_def_id| {
                self.infcx.probe(|snapshot| {
                    if let Ok(_substs) = self.match_impl(impl_def_id, obligation, snapshot) {
                        candidates.vec.push(ImplCandidate(impl_def_id));
                    }
                });
            },
        );

        Ok(())
    }

    fn assemble_candidates_from_auto_impls(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) -> Result<(), SelectionError<'tcx>> {
        // Okay to skip binder here because the tests we do below do not involve bound regions.
        let self_ty = *obligation.self_ty().skip_binder();
        debug!("assemble_candidates_from_auto_impls(self_ty={:?})", self_ty);

        let def_id = obligation.predicate.def_id();

        if self.tcx().trait_is_auto(def_id) {
            match self_ty.kind {
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
                ty::Param(..) | ty::Projection(..) => {
                    // In these cases, we don't know what the actual
                    // type is.  Therefore, we cannot break it down
                    // into its constituent types. So we don't
                    // consider the `..` impl but instead just add no
                    // candidates: this means that typeck will only
                    // succeed if there is another reason to believe
                    // that this obligation holds. That could be a
                    // where-clause or, in the case of an object type,
                    // it could be that the object type lists the
                    // trait (e.g., `Foo+Send : Send`). See
                    // `compile-fail/typeck-default-trait-impl-send-param.rs`
                    // for an example of a test case that exercises
                    // this path.
                }
                ty::Infer(ty::TyVar(_)) => {
                    // The auto impl might apply; we don't know.
                    candidates.ambiguous = true;
                }
                ty::Generator(_, _, movability)
                    if self.tcx().lang_items().unpin_trait() == Some(def_id) =>
                {
                    match movability {
                        hir::Movability::Static => {
                            // Immovable generators are never `Unpin`, so
                            // suppress the normal auto-impl candidate for it.
                        }
                        hir::Movability::Movable => {
                            // Movable generators are always `Unpin`, so add an
                            // unconditional builtin candidate.
                            candidates.vec.push(BuiltinCandidate { has_nested: false });
                        }
                    }
                }

                _ => candidates.vec.push(AutoImplCandidate(def_id)),
            }
        }

        Ok(())
    }

    /// Searches for impls that might apply to `obligation`.
    fn assemble_candidates_from_object_ty(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) {
        debug!(
            "assemble_candidates_from_object_ty(self_ty={:?})",
            obligation.self_ty().skip_binder()
        );

        self.infcx.probe(|_snapshot| {
            // The code below doesn't care about regions, and the
            // self-ty here doesn't escape this probe, so just erase
            // any LBR.
            let self_ty = self.tcx().erase_late_bound_regions(&obligation.self_ty());
            let poly_trait_ref = match self_ty.kind {
                ty::Dynamic(ref data, ..) => {
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

            debug!("assemble_candidates_from_object_ty: poly_trait_ref={:?}", poly_trait_ref);

            // Count only those upcast versions that match the trait-ref
            // we are looking for. Specifically, do not only check for the
            // correct trait, but also the correct type parameters.
            // For example, we may be trying to upcast `Foo` to `Bar<i32>`,
            // but `Foo` is declared as `trait Foo: Bar<u32>`.
            let upcast_trait_refs = util::supertraits(self.tcx(), poly_trait_ref)
                .filter(|upcast_trait_ref| {
                    self.infcx
                        .probe(|_| self.match_poly_trait_ref(obligation, *upcast_trait_ref).is_ok())
                })
                .count();

            if upcast_trait_refs > 1 {
                // Can be upcast in many ways; need more type information.
                candidates.ambiguous = true;
            } else if upcast_trait_refs == 1 {
                candidates.vec.push(ObjectCandidate);
            }
        })
    }

    /// Searches for unsizing that might apply to `obligation`.
    fn assemble_candidates_for_unsizing(
        &mut self,
        obligation: &TraitObligation<'tcx>,
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
        let source = match obligation.self_ty().no_bound_vars() {
            Some(t) => t,
            None => {
                // Don't add any candidates if there are bound regions.
                return;
            }
        };
        let target = obligation.predicate.skip_binder().trait_ref.substs.type_at(1);

        debug!("assemble_candidates_for_unsizing(source={:?}, target={:?})", source, target);

        let may_apply = match (&source.kind, &target.kind) {
            // Trait+Kx+'a -> Trait+Ky+'b (upcasts).
            (&ty::Dynamic(ref data_a, ..), &ty::Dynamic(ref data_b, ..)) => {
                // Upcasts permit two things:
                //
                // 1. Dropping auto traits, e.g., `Foo + Send` to `Foo`
                // 2. Tightening the region bound, e.g., `Foo + 'a` to `Foo + 'b` if `'a: 'b`
                //
                // Note that neither of these changes requires any
                // change at runtime. Eventually this will be
                // generalized.
                //
                // We always upcast when we can because of reason
                // #2 (region bounds).
                data_a.principal_def_id() == data_b.principal_def_id()
                    && data_b
                        .auto_traits()
                        // All of a's auto traits need to be in b's auto traits.
                        .all(|b| data_a.auto_traits().any(|a| a == b))
            }

            // `T` -> `Trait`
            (_, &ty::Dynamic(..)) => true,

            // Ambiguous handling is below `T` -> `Trait`, because inference
            // variables can still implement `Unsize<Trait>` and nested
            // obligations will have the final say (likely deferred).
            (&ty::Infer(ty::TyVar(_)), _) | (_, &ty::Infer(ty::TyVar(_))) => {
                debug!("assemble_candidates_for_unsizing: ambiguous");
                candidates.ambiguous = true;
                false
            }

            // `[T; n]` -> `[T]`
            (&ty::Array(..), &ty::Slice(_)) => true,

            // `Struct<T>` -> `Struct<U>`
            (&ty::Adt(def_id_a, _), &ty::Adt(def_id_b, _)) if def_id_a.is_struct() => {
                def_id_a == def_id_b
            }

            // `(.., T)` -> `(.., U)`
            (&ty::Tuple(tys_a), &ty::Tuple(tys_b)) => tys_a.len() == tys_b.len(),

            _ => false,
        };

        if may_apply {
            candidates.vec.push(BuiltinUnsizeCandidate);
        }
    }

    fn assemble_candidates_for_trait_alias(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) -> Result<(), SelectionError<'tcx>> {
        // Okay to skip binder here because the tests we do below do not involve bound regions.
        let self_ty = *obligation.self_ty().skip_binder();
        debug!("assemble_candidates_for_trait_alias(self_ty={:?})", self_ty);

        let def_id = obligation.predicate.def_id();

        if self.tcx().is_trait_alias(def_id) {
            candidates.vec.push(TraitAliasCandidate(def_id));
        }

        Ok(())
    }

    ///////////////////////////////////////////////////////////////////////////
    // WINNOW
    //
    // Winnowing is the process of attempting to resolve ambiguity by
    // probing further. During the winnowing process, we unify all
    // type variables and then we also attempt to evaluate recursive
    // bounds to see if they are satisfied.

    /// Returns `true` if `victim` should be dropped in favor of
    /// `other`. Generally speaking we will drop duplicate
    /// candidates and prefer where-clause candidates.
    ///
    /// See the comment for "SelectionCandidate" for more details.
    fn candidate_should_be_dropped_in_favor_of(
        &mut self,
        victim: &EvaluatedCandidate<'tcx>,
        other: &EvaluatedCandidate<'tcx>,
        needs_infer: bool,
    ) -> bool {
        if victim.candidate == other.candidate {
            return true;
        }

        // Check if a bound would previously have been removed when normalizing
        // the param_env so that it can be given the lowest priority. See
        // #50825 for the motivation for this.
        let is_global =
            |cand: &ty::PolyTraitRef<'_>| cand.is_global() && !cand.has_late_bound_regions();

        match other.candidate {
            // Prefer `BuiltinCandidate { has_nested: false }` to anything else.
            // This is a fix for #53123 and prevents winnowing from accidentally extending the
            // lifetime of a variable.
            BuiltinCandidate { has_nested: false } => true,
            ParamCandidate(ref cand) => match victim.candidate {
                AutoImplCandidate(..) => {
                    bug!(
                        "default implementations shouldn't be recorded \
                         when there are other valid candidates"
                    );
                }
                // Prefer `BuiltinCandidate { has_nested: false }` to anything else.
                // This is a fix for #53123 and prevents winnowing from accidentally extending the
                // lifetime of a variable.
                BuiltinCandidate { has_nested: false } => false,
                ImplCandidate(..)
                | ClosureCandidate
                | GeneratorCandidate
                | FnPointerCandidate
                | BuiltinObjectCandidate
                | BuiltinUnsizeCandidate
                | BuiltinCandidate { .. }
                | TraitAliasCandidate(..) => {
                    // Global bounds from the where clause should be ignored
                    // here (see issue #50825). Otherwise, we have a where
                    // clause so don't go around looking for impls.
                    !is_global(cand)
                }
                ObjectCandidate | ProjectionCandidate => {
                    // Arbitrarily give param candidates priority
                    // over projection and object candidates.
                    !is_global(cand)
                }
                ParamCandidate(..) => false,
            },
            ObjectCandidate | ProjectionCandidate => match victim.candidate {
                AutoImplCandidate(..) => {
                    bug!(
                        "default implementations shouldn't be recorded \
                         when there are other valid candidates"
                    );
                }
                // Prefer `BuiltinCandidate { has_nested: false }` to anything else.
                // This is a fix for #53123 and prevents winnowing from accidentally extending the
                // lifetime of a variable.
                BuiltinCandidate { has_nested: false } => false,
                ImplCandidate(..)
                | ClosureCandidate
                | GeneratorCandidate
                | FnPointerCandidate
                | BuiltinObjectCandidate
                | BuiltinUnsizeCandidate
                | BuiltinCandidate { .. }
                | TraitAliasCandidate(..) => true,
                ObjectCandidate | ProjectionCandidate => {
                    // Arbitrarily give param candidates priority
                    // over projection and object candidates.
                    true
                }
                ParamCandidate(ref cand) => is_global(cand),
            },
            ImplCandidate(other_def) => {
                // See if we can toss out `victim` based on specialization.
                // This requires us to know *for sure* that the `other` impl applies
                // i.e., `EvaluatedToOk`.
                if other.evaluation.must_apply_modulo_regions() {
                    match victim.candidate {
                        ImplCandidate(victim_def) => {
                            let tcx = self.tcx();
                            if tcx.specializes((other_def, victim_def)) {
                                return true;
                            }
                            return match tcx.impls_are_allowed_to_overlap(other_def, victim_def) {
                                Some(ty::ImplOverlapKind::Permitted { marker: true }) => {
                                    // Subtle: If the predicate we are evaluating has inference
                                    // variables, do *not* allow discarding candidates due to
                                    // marker trait impls.
                                    //
                                    // Without this restriction, we could end up accidentally
                                    // constrainting inference variables based on an arbitrarily
                                    // chosen trait impl.
                                    //
                                    // Imagine we have the following code:
                                    //
                                    // ```rust
                                    // #[marker] trait MyTrait {}
                                    // impl MyTrait for u8 {}
                                    // impl MyTrait for bool {}
                                    // ```
                                    //
                                    // And we are evaluating the predicate `<_#0t as MyTrait>`.
                                    //
                                    // During selection, we will end up with one candidate for each
                                    // impl of `MyTrait`. If we were to discard one impl in favor
                                    // of the other, we would be left with one candidate, causing
                                    // us to "successfully" select the predicate, unifying
                                    // _#0t with (for example) `u8`.
                                    //
                                    // However, we have no reason to believe that this unification
                                    // is correct - we've essentially just picked an arbitrary
                                    // *possibility* for _#0t, and required that this be the *only*
                                    // possibility.
                                    //
                                    // Eventually, we will either:
                                    // 1) Unify all inference variables in the predicate through
                                    // some other means (e.g. type-checking of a function). We will
                                    // then be in a position to drop marker trait candidates
                                    // without constraining inference variables (since there are
                                    // none left to constrin)
                                    // 2) Be left with some unconstrained inference variables. We
                                    // will then correctly report an inference error, since the
                                    // existence of multiple marker trait impls tells us nothing
                                    // about which one should actually apply.
                                    !needs_infer
                                }
                                Some(_) => true,
                                None => false,
                            };
                        }
                        ParamCandidate(ref cand) => {
                            // Prefer the impl to a global where clause candidate.
                            return is_global(cand);
                        }
                        _ => (),
                    }
                }

                false
            }
            ClosureCandidate
            | GeneratorCandidate
            | FnPointerCandidate
            | BuiltinObjectCandidate
            | BuiltinUnsizeCandidate
            | BuiltinCandidate { has_nested: true } => {
                match victim.candidate {
                    ParamCandidate(ref cand) => {
                        // Prefer these to a global where-clause bound
                        // (see issue #50825).
                        is_global(cand) && other.evaluation.must_apply_modulo_regions()
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // BUILTIN BOUNDS
    //
    // These cover the traits that are built-in to the language
    // itself: `Copy`, `Clone` and `Sized`.

    fn assemble_builtin_bound_candidates(
        &mut self,
        conditions: BuiltinImplConditions<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) -> Result<(), SelectionError<'tcx>> {
        match conditions {
            BuiltinImplConditions::Where(nested) => {
                debug!("builtin_bound: nested={:?}", nested);
                candidates
                    .vec
                    .push(BuiltinCandidate { has_nested: !nested.skip_binder().is_empty() });
            }
            BuiltinImplConditions::None => {}
            BuiltinImplConditions::Ambiguous => {
                debug!("assemble_builtin_bound_candidates: ambiguous builtin");
                candidates.ambiguous = true;
            }
        }

        Ok(())
    }

    fn sized_conditions(
        &mut self,
        obligation: &TraitObligation<'tcx>,
    ) -> BuiltinImplConditions<'tcx> {
        use self::BuiltinImplConditions::{Ambiguous, None, Where};

        // NOTE: binder moved to (*)
        let self_ty = self.infcx.shallow_resolve(obligation.predicate.skip_binder().self_ty());

        match self_ty.kind {
            ty::Infer(ty::IntVar(_))
            | ty::Infer(ty::FloatVar(_))
            | ty::Uint(_)
            | ty::Int(_)
            | ty::Bool
            | ty::Float(_)
            | ty::FnDef(..)
            | ty::FnPtr(_)
            | ty::RawPtr(..)
            | ty::Char
            | ty::Ref(..)
            | ty::Generator(..)
            | ty::GeneratorWitness(..)
            | ty::Array(..)
            | ty::Closure(..)
            | ty::Never
            | ty::Error => {
                // safe for everything
                Where(ty::Binder::dummy(Vec::new()))
            }

            ty::Str | ty::Slice(_) | ty::Dynamic(..) | ty::Foreign(..) => None,

            ty::Tuple(tys) => {
                Where(ty::Binder::bind(tys.last().into_iter().map(|k| k.expect_ty()).collect()))
            }

            ty::Adt(def, substs) => {
                let sized_crit = def.sized_constraint(self.tcx());
                // (*) binder moved here
                Where(ty::Binder::bind(
                    sized_crit.iter().map(|ty| ty.subst(self.tcx(), substs)).collect(),
                ))
            }

            ty::Projection(_) | ty::Param(_) | ty::Opaque(..) => None,
            ty::Infer(ty::TyVar(_)) => Ambiguous,

            ty::UnnormalizedProjection(..)
            | ty::Placeholder(..)
            | ty::Bound(..)
            | ty::Infer(ty::FreshTy(_))
            | ty::Infer(ty::FreshIntTy(_))
            | ty::Infer(ty::FreshFloatTy(_)) => {
                bug!("asked to assemble builtin bounds of unexpected type: {:?}", self_ty);
            }
        }
    }

    fn copy_clone_conditions(
        &mut self,
        obligation: &TraitObligation<'tcx>,
    ) -> BuiltinImplConditions<'tcx> {
        // NOTE: binder moved to (*)
        let self_ty = self.infcx.shallow_resolve(obligation.predicate.skip_binder().self_ty());

        use self::BuiltinImplConditions::{Ambiguous, None, Where};

        match self_ty.kind {
            ty::Infer(ty::IntVar(_))
            | ty::Infer(ty::FloatVar(_))
            | ty::FnDef(..)
            | ty::FnPtr(_)
            | ty::Error => Where(ty::Binder::dummy(Vec::new())),

            ty::Uint(_)
            | ty::Int(_)
            | ty::Bool
            | ty::Float(_)
            | ty::Char
            | ty::RawPtr(..)
            | ty::Never
            | ty::Ref(_, _, hir::Mutability::Not) => {
                // Implementations provided in libcore
                None
            }

            ty::Dynamic(..)
            | ty::Str
            | ty::Slice(..)
            | ty::Generator(..)
            | ty::GeneratorWitness(..)
            | ty::Foreign(..)
            | ty::Ref(_, _, hir::Mutability::Mut) => None,

            ty::Array(element_ty, _) => {
                // (*) binder moved here
                Where(ty::Binder::bind(vec![element_ty]))
            }

            ty::Tuple(tys) => {
                // (*) binder moved here
                Where(ty::Binder::bind(tys.iter().map(|k| k.expect_ty()).collect()))
            }

            ty::Closure(def_id, substs) => {
                // (*) binder moved here
                Where(ty::Binder::bind(substs.as_closure().upvar_tys(def_id, self.tcx()).collect()))
            }

            ty::Adt(..) | ty::Projection(..) | ty::Param(..) | ty::Opaque(..) => {
                // Fallback to whatever user-defined impls exist in this case.
                None
            }

            ty::Infer(ty::TyVar(_)) => {
                // Unbound type variable. Might or might not have
                // applicable impls and so forth, depending on what
                // those type variables wind up being bound to.
                Ambiguous
            }

            ty::UnnormalizedProjection(..)
            | ty::Placeholder(..)
            | ty::Bound(..)
            | ty::Infer(ty::FreshTy(_))
            | ty::Infer(ty::FreshIntTy(_))
            | ty::Infer(ty::FreshFloatTy(_)) => {
                bug!("asked to assemble builtin bounds of unexpected type: {:?}", self_ty);
            }
        }
    }

    /// For default impls, we need to break apart a type into its
    /// "constituent types" -- meaning, the types that it contains.
    ///
    /// Here are some (simple) examples:
    ///
    /// ```
    /// (i32, u32) -> [i32, u32]
    /// Foo where struct Foo { x: i32, y: u32 } -> [i32, u32]
    /// Bar<i32> where struct Bar<T> { x: T, y: u32 } -> [i32, u32]
    /// Zed<i32> where enum Zed { A(T), B(u32) } -> [i32, u32]
    /// ```
    fn constituent_types_for_ty(&self, t: Ty<'tcx>) -> Vec<Ty<'tcx>> {
        match t.kind {
            ty::Uint(_)
            | ty::Int(_)
            | ty::Bool
            | ty::Float(_)
            | ty::FnDef(..)
            | ty::FnPtr(_)
            | ty::Str
            | ty::Error
            | ty::Infer(ty::IntVar(_))
            | ty::Infer(ty::FloatVar(_))
            | ty::Never
            | ty::Char => Vec::new(),

            ty::UnnormalizedProjection(..)
            | ty::Placeholder(..)
            | ty::Dynamic(..)
            | ty::Param(..)
            | ty::Foreign(..)
            | ty::Projection(..)
            | ty::Bound(..)
            | ty::Infer(ty::TyVar(_))
            | ty::Infer(ty::FreshTy(_))
            | ty::Infer(ty::FreshIntTy(_))
            | ty::Infer(ty::FreshFloatTy(_)) => {
                bug!("asked to assemble constituent types of unexpected type: {:?}", t);
            }

            ty::RawPtr(ty::TypeAndMut { ty: element_ty, .. }) | ty::Ref(_, element_ty, _) => {
                vec![element_ty]
            }

            ty::Array(element_ty, _) | ty::Slice(element_ty) => vec![element_ty],

            ty::Tuple(ref tys) => {
                // (T1, ..., Tn) -- meets any bound that all of T1...Tn meet
                tys.iter().map(|k| k.expect_ty()).collect()
            }

            ty::Closure(def_id, ref substs) => {
                substs.as_closure().upvar_tys(def_id, self.tcx()).collect()
            }

            ty::Generator(def_id, ref substs, _) => {
                let witness = substs.as_generator().witness(def_id, self.tcx());
                substs
                    .as_generator()
                    .upvar_tys(def_id, self.tcx())
                    .chain(iter::once(witness))
                    .collect()
            }

            ty::GeneratorWitness(types) => {
                // This is sound because no regions in the witness can refer to
                // the binder outside the witness. So we'll effectivly reuse
                // the implicit binder around the witness.
                types.skip_binder().to_vec()
            }

            // For `PhantomData<T>`, we pass `T`.
            ty::Adt(def, substs) if def.is_phantom_data() => substs.types().collect(),

            ty::Adt(def, substs) => def.all_fields().map(|f| f.ty(self.tcx(), substs)).collect(),

            ty::Opaque(def_id, substs) => {
                // We can resolve the `impl Trait` to its concrete type,
                // which enforces a DAG between the functions requiring
                // the auto trait bounds in question.
                vec![self.tcx().type_of(def_id).subst(self.tcx(), substs)]
            }
        }
    }

    fn collect_predicates_for_types(
        &mut self,
        param_env: ty::ParamEnv<'tcx>,
        cause: ObligationCause<'tcx>,
        recursion_depth: usize,
        trait_def_id: DefId,
        types: ty::Binder<Vec<Ty<'tcx>>>,
    ) -> Vec<PredicateObligation<'tcx>> {
        // Because the types were potentially derived from
        // higher-ranked obligations they may reference late-bound
        // regions. For example, `for<'a> Foo<&'a int> : Copy` would
        // yield a type like `for<'a> &'a int`. In general, we
        // maintain the invariant that we never manipulate bound
        // regions, so we have to process these bound regions somehow.
        //
        // The strategy is to:
        //
        // 1. Instantiate those regions to placeholder regions (e.g.,
        //    `for<'a> &'a int` becomes `&0 int`.
        // 2. Produce something like `&'0 int : Copy`
        // 3. Re-bind the regions back to `for<'a> &'a int : Copy`

        types
            .skip_binder()
            .iter()
            .flat_map(|ty| {
                // binder moved -\
                let ty: ty::Binder<Ty<'tcx>> = ty::Binder::bind(ty); // <----/

                self.infcx.commit_unconditionally(|_| {
                    let (skol_ty, _) = self.infcx.replace_bound_vars_with_placeholders(&ty);
                    let Normalized { value: normalized_ty, mut obligations } =
                        project::normalize_with_depth(
                            self,
                            param_env,
                            cause.clone(),
                            recursion_depth,
                            &skol_ty,
                        );
                    let skol_obligation = predicate_for_trait_def(
                        self.tcx(),
                        param_env,
                        cause.clone(),
                        trait_def_id,
                        recursion_depth,
                        normalized_ty,
                        &[],
                    );
                    obligations.push(skol_obligation);
                    obligations
                })
            })
            .collect()
    }

    ///////////////////////////////////////////////////////////////////////////
    // CONFIRMATION
    //
    // Confirmation unifies the output type parameters of the trait
    // with the values found in the obligation, possibly yielding a
    // type error.  See the [rustc dev guide] for more details.
    //
    // [rustc dev guide]:
    // https://rustc-dev-guide.rust-lang.org/traits/resolution.html#confirmation

    fn confirm_candidate(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        candidate: SelectionCandidate<'tcx>,
    ) -> Result<Selection<'tcx>, SelectionError<'tcx>> {
        debug!("confirm_candidate({:?}, {:?})", obligation, candidate);

        match candidate {
            BuiltinCandidate { has_nested } => {
                let data = self.confirm_builtin_candidate(obligation, has_nested);
                Ok(VtableBuiltin(data))
            }

            ParamCandidate(param) => {
                let obligations = self.confirm_param_candidate(obligation, param);
                Ok(VtableParam(obligations))
            }

            ImplCandidate(impl_def_id) => {
                Ok(VtableImpl(self.confirm_impl_candidate(obligation, impl_def_id)))
            }

            AutoImplCandidate(trait_def_id) => {
                let data = self.confirm_auto_impl_candidate(obligation, trait_def_id);
                Ok(VtableAutoImpl(data))
            }

            ProjectionCandidate => {
                self.confirm_projection_candidate(obligation);
                Ok(VtableParam(Vec::new()))
            }

            ClosureCandidate => {
                let vtable_closure = self.confirm_closure_candidate(obligation)?;
                Ok(VtableClosure(vtable_closure))
            }

            GeneratorCandidate => {
                let vtable_generator = self.confirm_generator_candidate(obligation)?;
                Ok(VtableGenerator(vtable_generator))
            }

            FnPointerCandidate => {
                let data = self.confirm_fn_pointer_candidate(obligation)?;
                Ok(VtableFnPointer(data))
            }

            TraitAliasCandidate(alias_def_id) => {
                let data = self.confirm_trait_alias_candidate(obligation, alias_def_id);
                Ok(VtableTraitAlias(data))
            }

            ObjectCandidate => {
                let data = self.confirm_object_candidate(obligation);
                Ok(VtableObject(data))
            }

            BuiltinObjectCandidate => {
                // This indicates something like `Trait + Send: Send`. In this case, we know that
                // this holds because that's what the object type is telling us, and there's really
                // no additional obligations to prove and no types in particular to unify, etc.
                Ok(VtableParam(Vec::new()))
            }

            BuiltinUnsizeCandidate => {
                let data = self.confirm_builtin_unsize_candidate(obligation)?;
                Ok(VtableBuiltin(data))
            }
        }
    }

    fn confirm_projection_candidate(&mut self, obligation: &TraitObligation<'tcx>) {
        self.infcx.commit_unconditionally(|snapshot| {
            let result =
                self.match_projection_obligation_against_definition_bounds(obligation, snapshot);
            assert!(result);
        })
    }

    fn confirm_param_candidate(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        param: ty::PolyTraitRef<'tcx>,
    ) -> Vec<PredicateObligation<'tcx>> {
        debug!("confirm_param_candidate({:?},{:?})", obligation, param);

        // During evaluation, we already checked that this
        // where-clause trait-ref could be unified with the obligation
        // trait-ref. Repeat that unification now without any
        // transactional boundary; it should not fail.
        match self.match_where_clause_trait_ref(obligation, param.clone()) {
            Ok(obligations) => obligations,
            Err(()) => {
                bug!(
                    "Where clause `{:?}` was applicable to `{:?}` but now is not",
                    param,
                    obligation
                );
            }
        }
    }

    fn confirm_builtin_candidate(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        has_nested: bool,
    ) -> VtableBuiltinData<PredicateObligation<'tcx>> {
        debug!("confirm_builtin_candidate({:?}, {:?})", obligation, has_nested);

        let lang_items = self.tcx().lang_items();
        let obligations = if has_nested {
            let trait_def = obligation.predicate.def_id();
            let conditions = if Some(trait_def) == lang_items.sized_trait() {
                self.sized_conditions(obligation)
            } else if Some(trait_def) == lang_items.copy_trait() {
                self.copy_clone_conditions(obligation)
            } else if Some(trait_def) == lang_items.clone_trait() {
                self.copy_clone_conditions(obligation)
            } else {
                bug!("unexpected builtin trait {:?}", trait_def)
            };
            let nested = match conditions {
                BuiltinImplConditions::Where(nested) => nested,
                _ => bug!("obligation {:?} had matched a builtin impl but now doesn't", obligation),
            };

            let cause = obligation.derived_cause(BuiltinDerivedObligation);
            self.collect_predicates_for_types(
                obligation.param_env,
                cause,
                obligation.recursion_depth + 1,
                trait_def,
                nested,
            )
        } else {
            vec![]
        };

        debug!("confirm_builtin_candidate: obligations={:?}", obligations);

        VtableBuiltinData { nested: obligations }
    }

    /// This handles the case where a `auto trait Foo` impl is being used.
    /// The idea is that the impl applies to `X : Foo` if the following conditions are met:
    ///
    /// 1. For each constituent type `Y` in `X`, `Y : Foo` holds
    /// 2. For each where-clause `C` declared on `Foo`, `[Self => X] C` holds.
    fn confirm_auto_impl_candidate(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        trait_def_id: DefId,
    ) -> VtableAutoImplData<PredicateObligation<'tcx>> {
        debug!("confirm_auto_impl_candidate({:?}, {:?})", obligation, trait_def_id);

        let types = obligation.predicate.map_bound(|inner| {
            let self_ty = self.infcx.shallow_resolve(inner.self_ty());
            self.constituent_types_for_ty(self_ty)
        });
        self.vtable_auto_impl(obligation, trait_def_id, types)
    }

    /// See `confirm_auto_impl_candidate`.
    fn vtable_auto_impl(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        trait_def_id: DefId,
        nested: ty::Binder<Vec<Ty<'tcx>>>,
    ) -> VtableAutoImplData<PredicateObligation<'tcx>> {
        debug!("vtable_auto_impl: nested={:?}", nested);

        let cause = obligation.derived_cause(BuiltinDerivedObligation);
        let mut obligations = self.collect_predicates_for_types(
            obligation.param_env,
            cause,
            obligation.recursion_depth + 1,
            trait_def_id,
            nested,
        );

        let trait_obligations: Vec<PredicateObligation<'_>> =
            self.infcx.commit_unconditionally(|_| {
                let poly_trait_ref = obligation.predicate.to_poly_trait_ref();
                let (trait_ref, _) =
                    self.infcx.replace_bound_vars_with_placeholders(&poly_trait_ref);
                let cause = obligation.derived_cause(ImplDerivedObligation);
                self.impl_or_trait_obligations(
                    cause,
                    obligation.recursion_depth + 1,
                    obligation.param_env,
                    trait_def_id,
                    &trait_ref.substs,
                )
            });

        // Adds the predicates from the trait.  Note that this contains a `Self: Trait`
        // predicate as usual.  It won't have any effect since auto traits are coinductive.
        obligations.extend(trait_obligations);

        debug!("vtable_auto_impl: obligations={:?}", obligations);

        VtableAutoImplData { trait_def_id, nested: obligations }
    }

    fn confirm_impl_candidate(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        impl_def_id: DefId,
    ) -> VtableImplData<'tcx, PredicateObligation<'tcx>> {
        debug!("confirm_impl_candidate({:?},{:?})", obligation, impl_def_id);

        // First, create the substitutions by matching the impl again,
        // this time not in a probe.
        self.infcx.commit_unconditionally(|snapshot| {
            let substs = self.rematch_impl(impl_def_id, obligation, snapshot);
            debug!("confirm_impl_candidate: substs={:?}", substs);
            let cause = obligation.derived_cause(ImplDerivedObligation);
            self.vtable_impl(
                impl_def_id,
                substs,
                cause,
                obligation.recursion_depth + 1,
                obligation.param_env,
            )
        })
    }

    fn vtable_impl(
        &mut self,
        impl_def_id: DefId,
        mut substs: Normalized<'tcx, SubstsRef<'tcx>>,
        cause: ObligationCause<'tcx>,
        recursion_depth: usize,
        param_env: ty::ParamEnv<'tcx>,
    ) -> VtableImplData<'tcx, PredicateObligation<'tcx>> {
        debug!(
            "vtable_impl(impl_def_id={:?}, substs={:?}, recursion_depth={})",
            impl_def_id, substs, recursion_depth,
        );

        let mut impl_obligations = self.impl_or_trait_obligations(
            cause,
            recursion_depth,
            param_env,
            impl_def_id,
            &substs.value,
        );

        debug!(
            "vtable_impl: impl_def_id={:?} impl_obligations={:?}",
            impl_def_id, impl_obligations
        );

        // Because of RFC447, the impl-trait-ref and obligations
        // are sufficient to determine the impl substs, without
        // relying on projections in the impl-trait-ref.
        //
        // e.g., `impl<U: Tr, V: Iterator<Item=U>> Foo<<U as Tr>::T> for V`
        impl_obligations.append(&mut substs.obligations);

        VtableImplData { impl_def_id, substs: substs.value, nested: impl_obligations }
    }

    fn confirm_object_candidate(
        &mut self,
        obligation: &TraitObligation<'tcx>,
    ) -> VtableObjectData<'tcx, PredicateObligation<'tcx>> {
        debug!("confirm_object_candidate({:?})", obligation);

        // FIXME(nmatsakis) skipping binder here seems wrong -- we should
        // probably flatten the binder from the obligation and the binder
        // from the object. Have to try to make a broken test case that
        // results.
        let self_ty = self.infcx.shallow_resolve(*obligation.self_ty().skip_binder());
        let poly_trait_ref = match self_ty.kind {
            ty::Dynamic(ref data, ..) => data
                .principal()
                .unwrap_or_else(|| {
                    span_bug!(obligation.cause.span, "object candidate with no principal")
                })
                .with_self_ty(self.tcx(), self_ty),
            _ => span_bug!(obligation.cause.span, "object candidate with non-object"),
        };

        let mut upcast_trait_ref = None;
        let mut nested = vec![];
        let vtable_base;

        {
            let tcx = self.tcx();

            // We want to find the first supertrait in the list of
            // supertraits that we can unify with, and do that
            // unification. We know that there is exactly one in the list
            // where we can unify, because otherwise select would have
            // reported an ambiguity. (When we do find a match, also
            // record it for later.)
            let nonmatching = util::supertraits(tcx, poly_trait_ref).take_while(|&t| {
                match self.infcx.commit_if_ok(|_| self.match_poly_trait_ref(obligation, t)) {
                    Ok(obligations) => {
                        upcast_trait_ref = Some(t);
                        nested.extend(obligations);
                        false
                    }
                    Err(_) => true,
                }
            });

            // Additionally, for each of the non-matching predicates that
            // we pass over, we sum up the set of number of vtable
            // entries, so that we can compute the offset for the selected
            // trait.
            vtable_base = nonmatching.map(|t| super::util::count_own_vtable_entries(tcx, t)).sum();
        }

        VtableObjectData { upcast_trait_ref: upcast_trait_ref.unwrap(), vtable_base, nested }
    }

    fn confirm_fn_pointer_candidate(
        &mut self,
        obligation: &TraitObligation<'tcx>,
    ) -> Result<VtableFnPointerData<'tcx, PredicateObligation<'tcx>>, SelectionError<'tcx>> {
        debug!("confirm_fn_pointer_candidate({:?})", obligation);

        // Okay to skip binder; it is reintroduced below.
        let self_ty = self.infcx.shallow_resolve(*obligation.self_ty().skip_binder());
        let sig = self_ty.fn_sig(self.tcx());
        let trait_ref = closure_trait_ref_and_return_type(
            self.tcx(),
            obligation.predicate.def_id(),
            self_ty,
            sig,
            util::TupleArgumentsFlag::Yes,
        )
        .map_bound(|(trait_ref, _)| trait_ref);

        let Normalized { value: trait_ref, obligations } = project::normalize_with_depth(
            self,
            obligation.param_env,
            obligation.cause.clone(),
            obligation.recursion_depth + 1,
            &trait_ref,
        );

        self.confirm_poly_trait_refs(
            obligation.cause.clone(),
            obligation.param_env,
            obligation.predicate.to_poly_trait_ref(),
            trait_ref,
        )?;
        Ok(VtableFnPointerData { fn_ty: self_ty, nested: obligations })
    }

    fn confirm_trait_alias_candidate(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        alias_def_id: DefId,
    ) -> VtableTraitAliasData<'tcx, PredicateObligation<'tcx>> {
        debug!("confirm_trait_alias_candidate({:?}, {:?})", obligation, alias_def_id);

        self.infcx.commit_unconditionally(|_| {
            let (predicate, _) =
                self.infcx().replace_bound_vars_with_placeholders(&obligation.predicate);
            let trait_ref = predicate.trait_ref;
            let trait_def_id = trait_ref.def_id;
            let substs = trait_ref.substs;

            let trait_obligations = self.impl_or_trait_obligations(
                obligation.cause.clone(),
                obligation.recursion_depth,
                obligation.param_env,
                trait_def_id,
                &substs,
            );

            debug!(
                "confirm_trait_alias_candidate: trait_def_id={:?} trait_obligations={:?}",
                trait_def_id, trait_obligations
            );

            VtableTraitAliasData { alias_def_id, substs: substs, nested: trait_obligations }
        })
    }

    fn confirm_generator_candidate(
        &mut self,
        obligation: &TraitObligation<'tcx>,
    ) -> Result<VtableGeneratorData<'tcx, PredicateObligation<'tcx>>, SelectionError<'tcx>> {
        // Okay to skip binder because the substs on generator types never
        // touch bound regions, they just capture the in-scope
        // type/region parameters.
        let self_ty = self.infcx.shallow_resolve(*obligation.self_ty().skip_binder());
        let (generator_def_id, substs) = match self_ty.kind {
            ty::Generator(id, substs, _) => (id, substs),
            _ => bug!("closure candidate for non-closure {:?}", obligation),
        };

        debug!("confirm_generator_candidate({:?},{:?},{:?})", obligation, generator_def_id, substs);

        let trait_ref = self.generator_trait_ref_unnormalized(obligation, generator_def_id, substs);
        let Normalized { value: trait_ref, mut obligations } = normalize_with_depth(
            self,
            obligation.param_env,
            obligation.cause.clone(),
            obligation.recursion_depth + 1,
            &trait_ref,
        );

        debug!(
            "confirm_generator_candidate(generator_def_id={:?}, \
             trait_ref={:?}, obligations={:?})",
            generator_def_id, trait_ref, obligations
        );

        obligations.extend(self.confirm_poly_trait_refs(
            obligation.cause.clone(),
            obligation.param_env,
            obligation.predicate.to_poly_trait_ref(),
            trait_ref,
        )?);

        Ok(VtableGeneratorData { generator_def_id, substs, nested: obligations })
    }

    fn confirm_closure_candidate(
        &mut self,
        obligation: &TraitObligation<'tcx>,
    ) -> Result<VtableClosureData<'tcx, PredicateObligation<'tcx>>, SelectionError<'tcx>> {
        debug!("confirm_closure_candidate({:?})", obligation);

        let kind = self
            .tcx()
            .fn_trait_kind_from_lang_item(obligation.predicate.def_id())
            .unwrap_or_else(|| bug!("closure candidate for non-fn trait {:?}", obligation));

        // Okay to skip binder because the substs on closure types never
        // touch bound regions, they just capture the in-scope
        // type/region parameters.
        let self_ty = self.infcx.shallow_resolve(*obligation.self_ty().skip_binder());
        let (closure_def_id, substs) = match self_ty.kind {
            ty::Closure(id, substs) => (id, substs),
            _ => bug!("closure candidate for non-closure {:?}", obligation),
        };

        let trait_ref = self.closure_trait_ref_unnormalized(obligation, closure_def_id, substs);
        let Normalized { value: trait_ref, mut obligations } = normalize_with_depth(
            self,
            obligation.param_env,
            obligation.cause.clone(),
            obligation.recursion_depth + 1,
            &trait_ref,
        );

        debug!(
            "confirm_closure_candidate(closure_def_id={:?}, trait_ref={:?}, obligations={:?})",
            closure_def_id, trait_ref, obligations
        );

        obligations.extend(self.confirm_poly_trait_refs(
            obligation.cause.clone(),
            obligation.param_env,
            obligation.predicate.to_poly_trait_ref(),
            trait_ref,
        )?);

        obligations.push(Obligation::new(
            obligation.cause.clone(),
            obligation.param_env,
            ty::Predicate::ClosureKind(closure_def_id, substs, kind),
        ));

        Ok(VtableClosureData { closure_def_id, substs, nested: obligations })
    }

    /// In the case of closure types and fn pointers,
    /// we currently treat the input type parameters on the trait as
    /// outputs. This means that when we have a match we have only
    /// considered the self type, so we have to go back and make sure
    /// to relate the argument types too. This is kind of wrong, but
    /// since we control the full set of impls, also not that wrong,
    /// and it DOES yield better error messages (since we don't report
    /// errors as if there is no applicable impl, but rather report
    /// errors are about mismatched argument types.
    ///
    /// Here is an example. Imagine we have a closure expression
    /// and we desugared it so that the type of the expression is
    /// `Closure`, and `Closure` expects an int as argument. Then it
    /// is "as if" the compiler generated this impl:
    ///
    ///     impl Fn(int) for Closure { ... }
    ///
    /// Now imagine our obligation is `Fn(usize) for Closure`. So far
    /// we have matched the self type `Closure`. At this point we'll
    /// compare the `int` to `usize` and generate an error.
    ///
    /// Note that this checking occurs *after* the impl has selected,
    /// because these output type parameters should not affect the
    /// selection of the impl. Therefore, if there is a mismatch, we
    /// report an error to the user.
    fn confirm_poly_trait_refs(
        &mut self,
        obligation_cause: ObligationCause<'tcx>,
        obligation_param_env: ty::ParamEnv<'tcx>,
        obligation_trait_ref: ty::PolyTraitRef<'tcx>,
        expected_trait_ref: ty::PolyTraitRef<'tcx>,
    ) -> Result<Vec<PredicateObligation<'tcx>>, SelectionError<'tcx>> {
        self.infcx
            .at(&obligation_cause, obligation_param_env)
            .sup(obligation_trait_ref, expected_trait_ref)
            .map(|InferOk { obligations, .. }| obligations)
            .map_err(|e| OutputTypeParameterMismatch(expected_trait_ref, obligation_trait_ref, e))
    }

    fn confirm_builtin_unsize_candidate(
        &mut self,
        obligation: &TraitObligation<'tcx>,
    ) -> Result<VtableBuiltinData<PredicateObligation<'tcx>>, SelectionError<'tcx>> {
        let tcx = self.tcx();

        // `assemble_candidates_for_unsizing` should ensure there are no late-bound
        // regions here. See the comment there for more details.
        let source = self.infcx.shallow_resolve(obligation.self_ty().no_bound_vars().unwrap());
        let target = obligation.predicate.skip_binder().trait_ref.substs.type_at(1);
        let target = self.infcx.shallow_resolve(target);

        debug!("confirm_builtin_unsize_candidate(source={:?}, target={:?})", source, target);

        let mut nested = vec![];
        match (&source.kind, &target.kind) {
            // Trait+Kx+'a -> Trait+Ky+'b (upcasts).
            (&ty::Dynamic(ref data_a, r_a), &ty::Dynamic(ref data_b, r_b)) => {
                // See `assemble_candidates_for_unsizing` for more info.
                let existential_predicates = data_a.map_bound(|data_a| {
                    let iter = data_a
                        .principal()
                        .map(|x| ty::ExistentialPredicate::Trait(x))
                        .into_iter()
                        .chain(
                            data_a
                                .projection_bounds()
                                .map(|x| ty::ExistentialPredicate::Projection(x)),
                        )
                        .chain(data_b.auto_traits().map(ty::ExistentialPredicate::AutoTrait));
                    tcx.mk_existential_predicates(iter)
                });
                let source_trait = tcx.mk_dynamic(existential_predicates, r_b);

                // Require that the traits involved in this upcast are **equal**;
                // only the **lifetime bound** is changed.
                //
                // FIXME: This condition is arguably too strong -- it would
                // suffice for the source trait to be a *subtype* of the target
                // trait. In particular, changing from something like
                // `for<'a, 'b> Foo<'a, 'b>` to `for<'a> Foo<'a, 'a>` should be
                // permitted. And, indeed, in the in commit
                // 904a0bde93f0348f69914ee90b1f8b6e4e0d7cbc, this
                // condition was loosened. However, when the leak check was
                // added back, using subtype here actually guides the coercion
                // code in such a way that it accepts `old-lub-glb-object.rs`.
                // This is probably a good thing, but I've modified this to `.eq`
                // because I want to continue rejecting that test (as we have
                // done for quite some time) before we are firmly comfortable
                // with what our behavior should be there. -nikomatsakis
                let InferOk { obligations, .. } = self
                    .infcx
                    .at(&obligation.cause, obligation.param_env)
                    .eq(target, source_trait) // FIXME -- see below
                    .map_err(|_| Unimplemented)?;
                nested.extend(obligations);

                // Register one obligation for 'a: 'b.
                let cause = ObligationCause::new(
                    obligation.cause.span,
                    obligation.cause.body_id,
                    ObjectCastObligation(target),
                );
                let outlives = ty::OutlivesPredicate(r_a, r_b);
                nested.push(Obligation::with_depth(
                    cause,
                    obligation.recursion_depth + 1,
                    obligation.param_env,
                    ty::Binder::bind(outlives).to_predicate(),
                ));
            }

            // `T` -> `Trait`
            (_, &ty::Dynamic(ref data, r)) => {
                let mut object_dids = data.auto_traits().chain(data.principal_def_id());
                if let Some(did) = object_dids.find(|did| !tcx.is_object_safe(*did)) {
                    return Err(TraitNotObjectSafe(did));
                }

                let cause = ObligationCause::new(
                    obligation.cause.span,
                    obligation.cause.body_id,
                    ObjectCastObligation(target),
                );

                let predicate_to_obligation = |predicate| {
                    Obligation::with_depth(
                        cause.clone(),
                        obligation.recursion_depth + 1,
                        obligation.param_env,
                        predicate,
                    )
                };

                // Create obligations:
                //  - Casting `T` to `Trait`
                //  - For all the various builtin bounds attached to the object cast. (In other
                //  words, if the object type is `Foo + Send`, this would create an obligation for
                //  the `Send` check.)
                //  - Projection predicates
                nested.extend(
                    data.iter().map(|predicate| {
                        predicate_to_obligation(predicate.with_self_ty(tcx, source))
                    }),
                );

                // We can only make objects from sized types.
                let tr = ty::TraitRef::new(
                    tcx.require_lang_item(lang_items::SizedTraitLangItem, None),
                    tcx.mk_substs_trait(source, &[]),
                );
                nested.push(predicate_to_obligation(tr.without_const().to_predicate()));

                // If the type is `Foo + 'a`, ensure that the type
                // being cast to `Foo + 'a` outlives `'a`:
                let outlives = ty::OutlivesPredicate(source, r);
                nested.push(predicate_to_obligation(ty::Binder::dummy(outlives).to_predicate()));
            }

            // `[T; n]` -> `[T]`
            (&ty::Array(a, _), &ty::Slice(b)) => {
                let InferOk { obligations, .. } = self
                    .infcx
                    .at(&obligation.cause, obligation.param_env)
                    .eq(b, a)
                    .map_err(|_| Unimplemented)?;
                nested.extend(obligations);
            }

            // `Struct<T>` -> `Struct<U>`
            (&ty::Adt(def, substs_a), &ty::Adt(_, substs_b)) => {
                let fields =
                    def.all_fields().map(|field| tcx.type_of(field.did)).collect::<Vec<_>>();

                // The last field of the structure has to exist and contain type parameters.
                let field = if let Some(&field) = fields.last() {
                    field
                } else {
                    return Err(Unimplemented);
                };
                let mut ty_params = GrowableBitSet::new_empty();
                let mut found = false;
                for ty in field.walk() {
                    if let ty::Param(p) = ty.kind {
                        ty_params.insert(p.index as usize);
                        found = true;
                    }
                }
                if !found {
                    return Err(Unimplemented);
                }

                // Replace type parameters used in unsizing with
                // Error and ensure they do not affect any other fields.
                // This could be checked after type collection for any struct
                // with a potentially unsized trailing field.
                let params = substs_a
                    .iter()
                    .enumerate()
                    .map(|(i, &k)| if ty_params.contains(i) { tcx.types.err.into() } else { k });
                let substs = tcx.mk_substs(params);
                for &ty in fields.split_last().unwrap().1 {
                    if ty.subst(tcx, substs).references_error() {
                        return Err(Unimplemented);
                    }
                }

                // Extract `Field<T>` and `Field<U>` from `Struct<T>` and `Struct<U>`.
                let inner_source = field.subst(tcx, substs_a);
                let inner_target = field.subst(tcx, substs_b);

                // Check that the source struct with the target's
                // unsized parameters is equal to the target.
                let params = substs_a.iter().enumerate().map(|(i, &k)| {
                    if ty_params.contains(i) { substs_b.type_at(i).into() } else { k }
                });
                let new_struct = tcx.mk_adt(def, tcx.mk_substs(params));
                let InferOk { obligations, .. } = self
                    .infcx
                    .at(&obligation.cause, obligation.param_env)
                    .eq(target, new_struct)
                    .map_err(|_| Unimplemented)?;
                nested.extend(obligations);

                // Construct the nested `Field<T>: Unsize<Field<U>>` predicate.
                nested.push(predicate_for_trait_def(
                    tcx,
                    obligation.param_env,
                    obligation.cause.clone(),
                    obligation.predicate.def_id(),
                    obligation.recursion_depth + 1,
                    inner_source,
                    &[inner_target.into()],
                ));
            }

            // `(.., T)` -> `(.., U)`
            (&ty::Tuple(tys_a), &ty::Tuple(tys_b)) => {
                assert_eq!(tys_a.len(), tys_b.len());

                // The last field of the tuple has to exist.
                let (&a_last, a_mid) = if let Some(x) = tys_a.split_last() {
                    x
                } else {
                    return Err(Unimplemented);
                };
                let &b_last = tys_b.last().unwrap();

                // Check that the source tuple with the target's
                // last element is equal to the target.
                let new_tuple = tcx.mk_tup(
                    a_mid.iter().map(|k| k.expect_ty()).chain(iter::once(b_last.expect_ty())),
                );
                let InferOk { obligations, .. } = self
                    .infcx
                    .at(&obligation.cause, obligation.param_env)
                    .eq(target, new_tuple)
                    .map_err(|_| Unimplemented)?;
                nested.extend(obligations);

                // Construct the nested `T: Unsize<U>` predicate.
                nested.push(predicate_for_trait_def(
                    tcx,
                    obligation.param_env,
                    obligation.cause.clone(),
                    obligation.predicate.def_id(),
                    obligation.recursion_depth + 1,
                    a_last.expect_ty(),
                    &[b_last],
                ));
            }

            _ => bug!(),
        };

        Ok(VtableBuiltinData { nested })
    }

    ///////////////////////////////////////////////////////////////////////////
    // Matching
    //
    // Matching is a common path used for both evaluation and
    // confirmation.  It basically unifies types that appear in impls
    // and traits. This does affect the surrounding environment;
    // therefore, when used during evaluation, match routines must be
    // run inside of a `probe()` so that their side-effects are
    // contained.

    fn rematch_impl(
        &mut self,
        impl_def_id: DefId,
        obligation: &TraitObligation<'tcx>,
        snapshot: &CombinedSnapshot<'_, 'tcx>,
    ) -> Normalized<'tcx, SubstsRef<'tcx>> {
        match self.match_impl(impl_def_id, obligation, snapshot) {
            Ok(substs) => substs,
            Err(()) => {
                bug!(
                    "Impl {:?} was matchable against {:?} but now is not",
                    impl_def_id,
                    obligation
                );
            }
        }
    }

    fn match_impl(
        &mut self,
        impl_def_id: DefId,
        obligation: &TraitObligation<'tcx>,
        snapshot: &CombinedSnapshot<'_, 'tcx>,
    ) -> Result<Normalized<'tcx, SubstsRef<'tcx>>, ()> {
        let impl_trait_ref = self.tcx().impl_trait_ref(impl_def_id).unwrap();

        // Before we create the substitutions and everything, first
        // consider a "quick reject". This avoids creating more types
        // and so forth that we need to.
        if self.fast_reject_trait_refs(obligation, &impl_trait_ref) {
            return Err(());
        }

        let (skol_obligation, placeholder_map) =
            self.infcx().replace_bound_vars_with_placeholders(&obligation.predicate);
        let skol_obligation_trait_ref = skol_obligation.trait_ref;

        let impl_substs = self.infcx.fresh_substs_for_item(obligation.cause.span, impl_def_id);

        let impl_trait_ref = impl_trait_ref.subst(self.tcx(), impl_substs);

        let Normalized { value: impl_trait_ref, obligations: mut nested_obligations } =
            project::normalize_with_depth(
                self,
                obligation.param_env,
                obligation.cause.clone(),
                obligation.recursion_depth + 1,
                &impl_trait_ref,
            );

        debug!(
            "match_impl(impl_def_id={:?}, obligation={:?}, \
             impl_trait_ref={:?}, skol_obligation_trait_ref={:?})",
            impl_def_id, obligation, impl_trait_ref, skol_obligation_trait_ref
        );

        let InferOk { obligations, .. } = self
            .infcx
            .at(&obligation.cause, obligation.param_env)
            .eq(skol_obligation_trait_ref, impl_trait_ref)
            .map_err(|e| debug!("match_impl: failed eq_trait_refs due to `{}`", e))?;
        nested_obligations.extend(obligations);

        if let Err(e) = self.infcx.leak_check(false, &placeholder_map, snapshot) {
            debug!("match_impl: failed leak check due to `{}`", e);
            return Err(());
        }

        if !self.intercrate
            && self.tcx().impl_polarity(impl_def_id) == ty::ImplPolarity::Reservation
        {
            debug!("match_impl: reservation impls only apply in intercrate mode");
            return Err(());
        }

        debug!("match_impl: success impl_substs={:?}", impl_substs);
        Ok(Normalized { value: impl_substs, obligations: nested_obligations })
    }

    fn fast_reject_trait_refs(
        &mut self,
        obligation: &TraitObligation<'_>,
        impl_trait_ref: &ty::TraitRef<'_>,
    ) -> bool {
        // We can avoid creating type variables and doing the full
        // substitution if we find that any of the input types, when
        // simplified, do not match.

        obligation.predicate.skip_binder().input_types().zip(impl_trait_ref.input_types()).any(
            |(obligation_ty, impl_ty)| {
                let simplified_obligation_ty =
                    fast_reject::simplify_type(self.tcx(), obligation_ty, true);
                let simplified_impl_ty = fast_reject::simplify_type(self.tcx(), impl_ty, false);

                simplified_obligation_ty.is_some()
                    && simplified_impl_ty.is_some()
                    && simplified_obligation_ty != simplified_impl_ty
            },
        )
    }

    /// Normalize `where_clause_trait_ref` and try to match it against
    /// `obligation`. If successful, return any predicates that
    /// result from the normalization. Normalization is necessary
    /// because where-clauses are stored in the parameter environment
    /// unnormalized.
    fn match_where_clause_trait_ref(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        where_clause_trait_ref: ty::PolyTraitRef<'tcx>,
    ) -> Result<Vec<PredicateObligation<'tcx>>, ()> {
        self.match_poly_trait_ref(obligation, where_clause_trait_ref)
    }

    /// Returns `Ok` if `poly_trait_ref` being true implies that the
    /// obligation is satisfied.
    fn match_poly_trait_ref(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        poly_trait_ref: ty::PolyTraitRef<'tcx>,
    ) -> Result<Vec<PredicateObligation<'tcx>>, ()> {
        debug!(
            "match_poly_trait_ref: obligation={:?} poly_trait_ref={:?}",
            obligation, poly_trait_ref
        );

        self.infcx
            .at(&obligation.cause, obligation.param_env)
            .sup(obligation.predicate.to_poly_trait_ref(), poly_trait_ref)
            .map(|InferOk { obligations, .. }| obligations)
            .map_err(|_| ())
    }

    ///////////////////////////////////////////////////////////////////////////
    // Miscellany

    fn match_fresh_trait_refs(
        &self,
        previous: &ty::PolyTraitRef<'tcx>,
        current: &ty::PolyTraitRef<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> bool {
        let mut matcher = ty::_match::Match::new(self.tcx(), param_env);
        matcher.relate(previous, current).is_ok()
    }

    fn push_stack<'o>(
        &mut self,
        previous_stack: TraitObligationStackList<'o, 'tcx>,
        obligation: &'o TraitObligation<'tcx>,
    ) -> TraitObligationStack<'o, 'tcx> {
        let fresh_trait_ref =
            obligation.predicate.to_poly_trait_ref().fold_with(&mut self.freshener);

        let dfn = previous_stack.cache.next_dfn();
        let depth = previous_stack.depth() + 1;
        TraitObligationStack {
            obligation,
            fresh_trait_ref,
            reached_depth: Cell::new(depth),
            previous: previous_stack,
            dfn,
            depth,
        }
    }

    fn closure_trait_ref_unnormalized(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        closure_def_id: DefId,
        substs: SubstsRef<'tcx>,
    ) -> ty::PolyTraitRef<'tcx> {
        debug!(
            "closure_trait_ref_unnormalized(obligation={:?}, closure_def_id={:?}, substs={:?})",
            obligation, closure_def_id, substs,
        );
        let closure_type = self.infcx.closure_sig(closure_def_id, substs);

        debug!("closure_trait_ref_unnormalized: closure_type = {:?}", closure_type);

        // (1) Feels icky to skip the binder here, but OTOH we know
        // that the self-type is an unboxed closure type and hence is
        // in fact unparameterized (or at least does not reference any
        // regions bound in the obligation). Still probably some
        // refactoring could make this nicer.
        closure_trait_ref_and_return_type(
            self.tcx(),
            obligation.predicate.def_id(),
            obligation.predicate.skip_binder().self_ty(), // (1)
            closure_type,
            util::TupleArgumentsFlag::No,
        )
        .map_bound(|(trait_ref, _)| trait_ref)
    }

    fn generator_trait_ref_unnormalized(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        closure_def_id: DefId,
        substs: SubstsRef<'tcx>,
    ) -> ty::PolyTraitRef<'tcx> {
        let gen_sig = substs.as_generator().poly_sig(closure_def_id, self.tcx());

        // (1) Feels icky to skip the binder here, but OTOH we know
        // that the self-type is an generator type and hence is
        // in fact unparameterized (or at least does not reference any
        // regions bound in the obligation). Still probably some
        // refactoring could make this nicer.

        super::util::generator_trait_ref_and_outputs(
            self.tcx(),
            obligation.predicate.def_id(),
            obligation.predicate.skip_binder().self_ty(), // (1)
            gen_sig,
        )
        .map_bound(|(trait_ref, ..)| trait_ref)
    }

    /// Returns the obligations that are implied by instantiating an
    /// impl or trait. The obligations are substituted and fully
    /// normalized. This is used when confirming an impl or default
    /// impl.
    fn impl_or_trait_obligations(
        &mut self,
        cause: ObligationCause<'tcx>,
        recursion_depth: usize,
        param_env: ty::ParamEnv<'tcx>,
        def_id: DefId,           // of impl or trait
        substs: SubstsRef<'tcx>, // for impl or trait
    ) -> Vec<PredicateObligation<'tcx>> {
        debug!("impl_or_trait_obligations(def_id={:?})", def_id);
        let tcx = self.tcx();

        // To allow for one-pass evaluation of the nested obligation,
        // each predicate must be preceded by the obligations required
        // to normalize it.
        // for example, if we have:
        //    impl<U: Iterator<Item: Copy>, V: Iterator<Item = U>> Foo for V
        // the impl will have the following predicates:
        //    <V as Iterator>::Item = U,
        //    U: Iterator, U: Sized,
        //    V: Iterator, V: Sized,
        //    <U as Iterator>::Item: Copy
        // When we substitute, say, `V => IntoIter<u32>, U => $0`, the last
        // obligation will normalize to `<$0 as Iterator>::Item = $1` and
        // `$1: Copy`, so we must ensure the obligations are emitted in
        // that order.
        let predicates = tcx.predicates_of(def_id);
        assert_eq!(predicates.parent, None);
        let mut obligations = Vec::with_capacity(predicates.predicates.len());
        for (predicate, _) in predicates.predicates {
            let predicate = normalize_with_depth_to(
                self,
                param_env,
                cause.clone(),
                recursion_depth,
                &predicate.subst(tcx, substs),
                &mut obligations,
            );
            obligations.push(Obligation {
                cause: cause.clone(),
                recursion_depth,
                param_env,
                predicate,
            });
        }

        // We are performing deduplication here to avoid exponential blowups
        // (#38528) from happening, but the real cause of the duplication is
        // unknown. What we know is that the deduplication avoids exponential
        // amount of predicates being propagated when processing deeply nested
        // types.
        //
        // This code is hot enough that it's worth avoiding the allocation
        // required for the FxHashSet when possible. Special-casing lengths 0,
        // 1 and 2 covers roughly 75-80% of the cases.
        if obligations.len() <= 1 {
            // No possibility of duplicates.
        } else if obligations.len() == 2 {
            // Only two elements. Drop the second if they are equal.
            if obligations[0] == obligations[1] {
                obligations.truncate(1);
            }
        } else {
            // Three or more elements. Use a general deduplication process.
            let mut seen = FxHashSet::default();
            obligations.retain(|i| seen.insert(i.clone()));
        }

        obligations
    }
}

impl<'tcx> TraitObligation<'tcx> {
    #[allow(unused_comparisons)]
    pub fn derived_cause(
        &self,
        variant: fn(DerivedObligationCause<'tcx>) -> ObligationCauseCode<'tcx>,
    ) -> ObligationCause<'tcx> {
        /*!
         * Creates a cause for obligations that are derived from
         * `obligation` by a recursive search (e.g., for a builtin
         * bound, or eventually a `auto trait Foo`). If `obligation`
         * is itself a derived obligation, this is just a clone, but
         * otherwise we create a "derived obligation" cause so as to
         * keep track of the original root obligation for error
         * reporting.
         */

        let obligation = self;

        // NOTE(flaper87): As of now, it keeps track of the whole error
        // chain. Ideally, we should have a way to configure this either
        // by using -Z verbose or just a CLI argument.
        let derived_cause = DerivedObligationCause {
            parent_trait_ref: obligation.predicate.to_poly_trait_ref(),
            parent_code: Rc::new(obligation.cause.code.clone()),
        };
        let derived_code = variant(derived_cause);
        ObligationCause::new(obligation.cause.span, obligation.cause.body_id, derived_code)
    }
}

impl<'o, 'tcx> TraitObligationStack<'o, 'tcx> {
    fn list(&'o self) -> TraitObligationStackList<'o, 'tcx> {
        TraitObligationStackList::with(self)
    }

    fn cache(&self) -> &'o ProvisionalEvaluationCache<'tcx> {
        self.previous.cache
    }

    fn iter(&'o self) -> TraitObligationStackList<'o, 'tcx> {
        self.list()
    }

    /// Indicates that attempting to evaluate this stack entry
    /// required accessing something from the stack at depth `reached_depth`.
    fn update_reached_depth(&self, reached_depth: usize) {
        assert!(
            self.depth > reached_depth,
            "invoked `update_reached_depth` with something under this stack: \
             self.depth={} reached_depth={}",
            self.depth,
            reached_depth,
        );
        debug!("update_reached_depth(reached_depth={})", reached_depth);
        let mut p = self;
        while reached_depth < p.depth {
            debug!("update_reached_depth: marking {:?} as cycle participant", p.fresh_trait_ref);
            p.reached_depth.set(p.reached_depth.get().min(reached_depth));
            p = p.previous.head.unwrap();
        }
    }
}

/// The "provisional evaluation cache" is used to store intermediate cache results
/// when solving auto traits. Auto traits are unusual in that they can support
/// cycles. So, for example, a "proof tree" like this would be ok:
///
/// - `Foo<T>: Send` :-
///   - `Bar<T>: Send` :-
///     - `Foo<T>: Send` -- cycle, but ok
///   - `Baz<T>: Send`
///
/// Here, to prove `Foo<T>: Send`, we have to prove `Bar<T>: Send` and
/// `Baz<T>: Send`. Proving `Bar<T>: Send` in turn required `Foo<T>: Send`.
/// For non-auto traits, this cycle would be an error, but for auto traits (because
/// they are coinductive) it is considered ok.
///
/// However, there is a complication: at the point where we have
/// "proven" `Bar<T>: Send`, we have in fact only proven it
/// *provisionally*. In particular, we proved that `Bar<T>: Send`
/// *under the assumption* that `Foo<T>: Send`. But what if we later
/// find out this assumption is wrong?  Specifically, we could
/// encounter some kind of error proving `Baz<T>: Send`. In that case,
/// `Bar<T>: Send` didn't turn out to be true.
///
/// In Issue #60010, we found a bug in rustc where it would cache
/// these intermediate results. This was fixed in #60444 by disabling
/// *all* caching for things involved in a cycle -- in our example,
/// that would mean we don't cache that `Bar<T>: Send`.  But this led
/// to large slowdowns.
///
/// Specifically, imagine this scenario, where proving `Baz<T>: Send`
/// first requires proving `Bar<T>: Send` (which is true:
///
/// - `Foo<T>: Send` :-
///   - `Bar<T>: Send` :-
///     - `Foo<T>: Send` -- cycle, but ok
///   - `Baz<T>: Send`
///     - `Bar<T>: Send` -- would be nice for this to be a cache hit!
///     - `*const T: Send` -- but what if we later encounter an error?
///
/// The *provisional evaluation cache* resolves this issue. It stores
/// cache results that we've proven but which were involved in a cycle
/// in some way. We track the minimal stack depth (i.e., the
/// farthest from the top of the stack) that we are dependent on.
/// The idea is that the cache results within are all valid -- so long as
/// none of the nodes in between the current node and the node at that minimum
/// depth result in an error (in which case the cached results are just thrown away).
///
/// During evaluation, we consult this provisional cache and rely on
/// it. Accessing a cached value is considered equivalent to accessing
/// a result at `reached_depth`, so it marks the *current* solution as
/// provisional as well. If an error is encountered, we toss out any
/// provisional results added from the subtree that encountered the
/// error.  When we pop the node at `reached_depth` from the stack, we
/// can commit all the things that remain in the provisional cache.
struct ProvisionalEvaluationCache<'tcx> {
    /// next "depth first number" to issue -- just a counter
    dfn: Cell<usize>,

    /// Stores the "coldest" depth (bottom of stack) reached by any of
    /// the evaluation entries. The idea here is that all things in the provisional
    /// cache are always dependent on *something* that is colder in the stack:
    /// therefore, if we add a new entry that is dependent on something *colder still*,
    /// we have to modify the depth for all entries at once.
    ///
    /// Example:
    ///
    /// Imagine we have a stack `A B C D E` (with `E` being the top of
    /// the stack).  We cache something with depth 2, which means that
    /// it was dependent on C.  Then we pop E but go on and process a
    /// new node F: A B C D F.  Now F adds something to the cache with
    /// depth 1, meaning it is dependent on B.  Our original cache
    /// entry is also dependent on B, because there is a path from E
    /// to C and then from C to F and from F to B.
    reached_depth: Cell<usize>,

    /// Map from cache key to the provisionally evaluated thing.
    /// The cache entries contain the result but also the DFN in which they
    /// were added. The DFN is used to clear out values on failure.
    ///
    /// Imagine we have a stack like:
    ///
    /// - `A B C` and we add a cache for the result of C (DFN 2)
    /// - Then we have a stack `A B D` where `D` has DFN 3
    /// - We try to solve D by evaluating E: `A B D E` (DFN 4)
    /// - `E` generates various cache entries which have cyclic dependices on `B`
    ///   - `A B D E F` and so forth
    ///   - the DFN of `F` for example would be 5
    /// - then we determine that `E` is in error -- we will then clear
    ///   all cache values whose DFN is >= 4 -- in this case, that
    ///   means the cached value for `F`.
    map: RefCell<FxHashMap<ty::PolyTraitRef<'tcx>, ProvisionalEvaluation>>,
}

/// A cache value for the provisional cache: contains the depth-first
/// number (DFN) and result.
#[derive(Copy, Clone, Debug)]
struct ProvisionalEvaluation {
    from_dfn: usize,
    result: EvaluationResult,
}

impl<'tcx> Default for ProvisionalEvaluationCache<'tcx> {
    fn default() -> Self {
        Self {
            dfn: Cell::new(0),
            reached_depth: Cell::new(std::usize::MAX),
            map: Default::default(),
        }
    }
}

impl<'tcx> ProvisionalEvaluationCache<'tcx> {
    /// Get the next DFN in sequence (basically a counter).
    fn next_dfn(&self) -> usize {
        let result = self.dfn.get();
        self.dfn.set(result + 1);
        result
    }

    /// Check the provisional cache for any result for
    /// `fresh_trait_ref`. If there is a hit, then you must consider
    /// it an access to the stack slots at depth
    /// `self.current_reached_depth()` and above.
    fn get_provisional(&self, fresh_trait_ref: ty::PolyTraitRef<'tcx>) -> Option<EvaluationResult> {
        debug!(
            "get_provisional(fresh_trait_ref={:?}) = {:#?} with reached-depth {}",
            fresh_trait_ref,
            self.map.borrow().get(&fresh_trait_ref),
            self.reached_depth.get(),
        );
        Some(self.map.borrow().get(&fresh_trait_ref)?.result)
    }

    /// Current value of the `reached_depth` counter -- all the
    /// provisional cache entries are dependent on the item at this
    /// depth.
    fn current_reached_depth(&self) -> usize {
        self.reached_depth.get()
    }

    /// Insert a provisional result into the cache. The result came
    /// from the node with the given DFN. It accessed a minimum depth
    /// of `reached_depth` to compute. It evaluated `fresh_trait_ref`
    /// and resulted in `result`.
    fn insert_provisional(
        &self,
        from_dfn: usize,
        reached_depth: usize,
        fresh_trait_ref: ty::PolyTraitRef<'tcx>,
        result: EvaluationResult,
    ) {
        debug!(
            "insert_provisional(from_dfn={}, reached_depth={}, fresh_trait_ref={:?}, result={:?})",
            from_dfn, reached_depth, fresh_trait_ref, result,
        );
        let r_d = self.reached_depth.get();
        self.reached_depth.set(r_d.min(reached_depth));

        debug!("insert_provisional: reached_depth={:?}", self.reached_depth.get());

        self.map.borrow_mut().insert(fresh_trait_ref, ProvisionalEvaluation { from_dfn, result });
    }

    /// Invoked when the node with dfn `dfn` does not get a successful
    /// result.  This will clear out any provisional cache entries
    /// that were added since `dfn` was created. This is because the
    /// provisional entries are things which must assume that the
    /// things on the stack at the time of their creation succeeded --
    /// since the failing node is presently at the top of the stack,
    /// these provisional entries must either depend on it or some
    /// ancestor of it.
    fn on_failure(&self, dfn: usize) {
        debug!("on_failure(dfn={:?})", dfn,);
        self.map.borrow_mut().retain(|key, eval| {
            if !eval.from_dfn >= dfn {
                debug!("on_failure: removing {:?}", key);
                false
            } else {
                true
            }
        });
    }

    /// Invoked when the node at depth `depth` completed without
    /// depending on anything higher in the stack (if that completion
    /// was a failure, then `on_failure` should have been invoked
    /// already). The callback `op` will be invoked for each
    /// provisional entry that we can now confirm.
    fn on_completion(
        &self,
        depth: usize,
        mut op: impl FnMut(ty::PolyTraitRef<'tcx>, EvaluationResult),
    ) {
        debug!("on_completion(depth={}, reached_depth={})", depth, self.reached_depth.get(),);

        if self.reached_depth.get() < depth {
            debug!("on_completion: did not yet reach depth to complete");
            return;
        }

        for (fresh_trait_ref, eval) in self.map.borrow_mut().drain() {
            debug!("on_completion: fresh_trait_ref={:?} eval={:?}", fresh_trait_ref, eval,);

            op(fresh_trait_ref, eval.result);
        }

        self.reached_depth.set(std::usize::MAX);
    }
}

#[derive(Copy, Clone)]
struct TraitObligationStackList<'o, 'tcx> {
    cache: &'o ProvisionalEvaluationCache<'tcx>,
    head: Option<&'o TraitObligationStack<'o, 'tcx>>,
}

impl<'o, 'tcx> TraitObligationStackList<'o, 'tcx> {
    fn empty(cache: &'o ProvisionalEvaluationCache<'tcx>) -> TraitObligationStackList<'o, 'tcx> {
        TraitObligationStackList { cache, head: None }
    }

    fn with(r: &'o TraitObligationStack<'o, 'tcx>) -> TraitObligationStackList<'o, 'tcx> {
        TraitObligationStackList { cache: r.cache(), head: Some(r) }
    }

    fn head(&self) -> Option<&'o TraitObligationStack<'o, 'tcx>> {
        self.head
    }

    fn depth(&self) -> usize {
        if let Some(head) = self.head { head.depth } else { 0 }
    }
}

impl<'o, 'tcx> Iterator for TraitObligationStackList<'o, 'tcx> {
    type Item = &'o TraitObligationStack<'o, 'tcx>;

    fn next(&mut self) -> Option<&'o TraitObligationStack<'o, 'tcx>> {
        match self.head {
            Some(o) => {
                *self = o.previous;
                Some(o)
            }
            None => None,
        }
    }
}

impl<'o, 'tcx> fmt::Debug for TraitObligationStack<'o, 'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TraitObligationStack({:?})", self.obligation)
    }
}
