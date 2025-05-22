//! Candidate selection. See the [rustc dev guide] for more information on how this works.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/traits/resolution.html#selection

use std::assert_matches::assert_matches;
use std::cell::{Cell, RefCell};
use std::fmt::{self, Display};
use std::ops::ControlFlow;
use std::{cmp, iter};

use hir::def::DefKind;
use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_errors::{Diag, EmissionGuarantee};
use rustc_hir as hir;
use rustc_hir::LangItem;
use rustc_hir::def_id::DefId;
use rustc_infer::infer::BoundRegionConversionTime::{self, HigherRankedType};
use rustc_infer::infer::DefineOpaqueTypes;
use rustc_infer::infer::at::ToTrace;
use rustc_infer::infer::relate::TypeRelation;
use rustc_infer::traits::{PredicateObligations, TraitObligation};
use rustc_middle::bug;
use rustc_middle::dep_graph::{DepNodeIndex, dep_kinds};
pub use rustc_middle::traits::select::*;
use rustc_middle::ty::abstract_const::NotConstEvaluatable;
use rustc_middle::ty::error::TypeErrorToStringExt;
use rustc_middle::ty::print::{PrintTraitRefExt as _, with_no_trimmed_paths};
use rustc_middle::ty::{
    self, DeepRejectCtxt, GenericArgsRef, PolyProjectionPredicate, Ty, TyCtxt, TypeFoldable,
    TypeVisitableExt, TypingMode, Upcast, elaborate,
};
use rustc_span::{Symbol, sym};
use tracing::{debug, instrument, trace};

use self::EvaluationResult::*;
use self::SelectionCandidate::*;
use super::coherence::{self, Conflict};
use super::project::ProjectionTermObligation;
use super::util::closure_trait_ref_and_return_type;
use super::{
    ImplDerivedCause, Normalized, Obligation, ObligationCause, ObligationCauseCode, Overflow,
    PolyTraitObligation, PredicateObligation, Selection, SelectionError, SelectionResult,
    TraitQueryMode, const_evaluatable, project, util, wf,
};
use crate::error_reporting::InferCtxtErrorExt;
use crate::infer::{InferCtxt, InferOk, TypeFreshener};
use crate::solve::InferCtxtSelectExt as _;
use crate::traits::normalize::{normalize_with_depth, normalize_with_depth_to};
use crate::traits::project::{ProjectAndUnifyResult, ProjectionCacheKeyExt};
use crate::traits::{
    EvaluateConstErr, ProjectionCacheKey, Unimplemented, effects, sizedness_fast_path,
};

mod _match;
mod candidate_assembly;
mod confirmation;

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum IntercrateAmbiguityCause<'tcx> {
    DownstreamCrate { trait_ref: ty::TraitRef<'tcx>, self_ty: Option<Ty<'tcx>> },
    UpstreamCrateUpdate { trait_ref: ty::TraitRef<'tcx>, self_ty: Option<Ty<'tcx>> },
    ReservationImpl { message: Symbol },
}

impl<'tcx> IntercrateAmbiguityCause<'tcx> {
    /// Emits notes when the overlap is caused by complex intercrate ambiguities.
    /// See #23980 for details.
    pub fn add_intercrate_ambiguity_hint<G: EmissionGuarantee>(&self, err: &mut Diag<'_, G>) {
        err.note(self.intercrate_ambiguity_hint());
    }

    pub fn intercrate_ambiguity_hint(&self) -> String {
        with_no_trimmed_paths!(match self {
            IntercrateAmbiguityCause::DownstreamCrate { trait_ref, self_ty } => {
                format!(
                    "downstream crates may implement trait `{trait_desc}`{self_desc}",
                    trait_desc = trait_ref.print_trait_sugared(),
                    self_desc = if let Some(self_ty) = self_ty {
                        format!(" for type `{self_ty}`")
                    } else {
                        String::new()
                    }
                )
            }
            IntercrateAmbiguityCause::UpstreamCrateUpdate { trait_ref, self_ty } => {
                format!(
                    "upstream crates may add a new impl of trait `{trait_desc}`{self_desc} \
                in future versions",
                    trait_desc = trait_ref.print_trait_sugared(),
                    self_desc = if let Some(self_ty) = self_ty {
                        format!(" for type `{self_ty}`")
                    } else {
                        String::new()
                    }
                )
            }
            IntercrateAmbiguityCause::ReservationImpl { message } => message.to_string(),
        })
    }
}

pub struct SelectionContext<'cx, 'tcx> {
    pub infcx: &'cx InferCtxt<'tcx>,

    /// Freshener used specifically for entries on the obligation
    /// stack. This ensures that all entries on the stack at one time
    /// will have the same set of placeholder entries, which is
    /// important for checking for trait bounds that recursively
    /// require themselves.
    freshener: TypeFreshener<'cx, 'tcx>,

    /// If `intercrate` is set, we remember predicates which were
    /// considered ambiguous because of impls potentially added in other crates.
    /// This is used in coherence to give improved diagnostics.
    /// We don't do his until we detect a coherence error because it can
    /// lead to false overflow results (#47139) and because always
    /// computing it may negatively impact performance.
    intercrate_ambiguity_causes: Option<FxIndexSet<IntercrateAmbiguityCause<'tcx>>>,

    /// The mode that trait queries run in, which informs our error handling
    /// policy. In essence, canonicalized queries need their errors propagated
    /// rather than immediately reported because we do not have accurate spans.
    query_mode: TraitQueryMode,
}

// A stack that walks back up the stack frame.
struct TraitObligationStack<'prev, 'tcx> {
    obligation: &'prev PolyTraitObligation<'tcx>,

    /// The trait predicate from `obligation` but "freshened" with the
    /// selection-context's freshener. Used to check for recursion.
    fresh_trait_pred: ty::PolyTraitPredicate<'tcx>,

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
    /// encountered a problem (later on) with `A: AutoTrait`. So we
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
    /// A list of candidates that definitely apply to the current
    /// obligation (meaning: types unify).
    vec: Vec<SelectionCandidate<'tcx>>,

    /// If `true`, then there were candidates that might or might
    /// not have applied, but we couldn't tell. This occurs when some
    /// of the input types are type variables, in which case there are
    /// various "builtin" rules that might or might not trigger.
    ambiguous: bool,
}

#[derive(PartialEq, Eq, Debug, Clone)]
struct EvaluatedCandidate<'tcx> {
    candidate: SelectionCandidate<'tcx>,
    evaluation: EvaluationResult,
}

/// When does the builtin impl for `T: Trait` apply?
#[derive(Debug)]
enum BuiltinImplConditions<'tcx> {
    /// The impl is conditional on `T1, T2, ...: Trait`.
    Where(ty::Binder<'tcx, Vec<Ty<'tcx>>>),
    /// There is no built-in impl. There may be some other
    /// candidate (a where-clause or user-defined impl).
    None,
    /// It is unknown whether there is an impl.
    Ambiguous,
}

impl<'cx, 'tcx> SelectionContext<'cx, 'tcx> {
    pub fn new(infcx: &'cx InferCtxt<'tcx>) -> SelectionContext<'cx, 'tcx> {
        SelectionContext {
            infcx,
            freshener: infcx.freshener(),
            intercrate_ambiguity_causes: None,
            query_mode: TraitQueryMode::Standard,
        }
    }

    pub fn with_query_mode(
        infcx: &'cx InferCtxt<'tcx>,
        query_mode: TraitQueryMode,
    ) -> SelectionContext<'cx, 'tcx> {
        debug!(?query_mode, "with_query_mode");
        SelectionContext { query_mode, ..SelectionContext::new(infcx) }
    }

    /// Enables tracking of intercrate ambiguity causes. See
    /// the documentation of [`Self::intercrate_ambiguity_causes`] for more.
    pub fn enable_tracking_intercrate_ambiguity_causes(&mut self) {
        assert_matches!(self.infcx.typing_mode(), TypingMode::Coherence);
        assert!(self.intercrate_ambiguity_causes.is_none());
        self.intercrate_ambiguity_causes = Some(FxIndexSet::default());
        debug!("selcx: enable_tracking_intercrate_ambiguity_causes");
    }

    /// Gets the intercrate ambiguity causes collected since tracking
    /// was enabled and disables tracking at the same time. If
    /// tracking is not enabled, just returns an empty vector.
    pub fn take_intercrate_ambiguity_causes(
        &mut self,
    ) -> FxIndexSet<IntercrateAmbiguityCause<'tcx>> {
        assert_matches!(self.infcx.typing_mode(), TypingMode::Coherence);
        self.intercrate_ambiguity_causes.take().unwrap_or_default()
    }

    pub fn tcx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
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
    #[instrument(level = "debug", skip(self), ret)]
    pub fn poly_select(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
    ) -> SelectionResult<'tcx, Selection<'tcx>> {
        assert!(!self.infcx.next_trait_solver());

        let candidate = match self.select_from_obligation(obligation) {
            Err(SelectionError::Overflow(OverflowError::Canonical)) => {
                // In standard mode, overflow must have been caught and reported
                // earlier.
                assert!(self.query_mode == TraitQueryMode::Canonical);
                return Err(SelectionError::Overflow(OverflowError::Canonical));
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
            Err(SelectionError::Overflow(OverflowError::Canonical)) => {
                assert!(self.query_mode == TraitQueryMode::Canonical);
                Err(SelectionError::Overflow(OverflowError::Canonical))
            }
            Err(e) => Err(e),
            Ok(candidate) => Ok(Some(candidate)),
        }
    }

    pub fn select(
        &mut self,
        obligation: &TraitObligation<'tcx>,
    ) -> SelectionResult<'tcx, Selection<'tcx>> {
        if self.infcx.next_trait_solver() {
            return self.infcx.select_in_new_trait_solver(obligation);
        }

        self.poly_select(&Obligation {
            cause: obligation.cause.clone(),
            param_env: obligation.param_env,
            predicate: ty::Binder::dummy(obligation.predicate),
            recursion_depth: obligation.recursion_depth,
        })
    }

    fn select_from_obligation(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
    ) -> SelectionResult<'tcx, SelectionCandidate<'tcx>> {
        debug_assert!(!obligation.predicate.has_escaping_bound_vars());

        let pec = &ProvisionalEvaluationCache::default();
        let stack = self.push_stack(TraitObligationStackList::empty(pec), obligation);

        self.candidate_from_obligation(&stack)
    }

    #[instrument(level = "debug", skip(self), ret)]
    fn candidate_from_obligation<'o>(
        &mut self,
        stack: &TraitObligationStack<'o, 'tcx>,
    ) -> SelectionResult<'tcx, SelectionCandidate<'tcx>> {
        debug_assert!(!self.infcx.next_trait_solver());
        // Watch out for overflow. This intentionally bypasses (and does
        // not update) the cache.
        self.check_recursion_limit(stack.obligation, stack.obligation)?;

        // Check the cache. Note that we freshen the trait-ref
        // separately rather than using `stack.fresh_trait_ref` --
        // this is because we want the unbound variables to be
        // replaced with fresh types starting from index 0.
        let cache_fresh_trait_pred = self.infcx.freshen(stack.obligation.predicate);
        debug!(?cache_fresh_trait_pred);
        debug_assert!(!stack.obligation.predicate.has_escaping_bound_vars());

        if let Some(c) =
            self.check_candidate_cache(stack.obligation.param_env, cache_fresh_trait_pred)
        {
            debug!("CACHE HIT");
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

        debug!("CACHE MISS");
        self.insert_candidate_cache(
            stack.obligation.param_env,
            cache_fresh_trait_pred,
            dep_node,
            candidate.clone(),
        );
        candidate
    }

    fn candidate_from_obligation_no_cache<'o>(
        &mut self,
        stack: &TraitObligationStack<'o, 'tcx>,
    ) -> SelectionResult<'tcx, SelectionCandidate<'tcx>> {
        if let Err(conflict) = self.is_knowable(stack) {
            debug!("coherence stage: not knowable");
            if self.intercrate_ambiguity_causes.is_some() {
                debug!("evaluate_stack: intercrate_ambiguity_causes is some");
                // Heuristics: show the diagnostics when there are no candidates in crate.
                if let Ok(candidate_set) = self.assemble_candidates(stack) {
                    let mut no_candidates_apply = true;

                    for c in candidate_set.vec.iter() {
                        if self.evaluate_candidate(stack, c)?.may_apply() {
                            no_candidates_apply = false;
                            break;
                        }
                    }

                    if !candidate_set.ambiguous && no_candidates_apply {
                        let trait_ref = self.infcx.resolve_vars_if_possible(
                            stack.obligation.predicate.skip_binder().trait_ref,
                        );
                        if !trait_ref.references_error() {
                            let self_ty = trait_ref.self_ty();
                            let self_ty = self_ty.has_concrete_skeleton().then(|| self_ty);
                            let cause = if let Conflict::Upstream = conflict {
                                IntercrateAmbiguityCause::UpstreamCrateUpdate { trait_ref, self_ty }
                            } else {
                                IntercrateAmbiguityCause::DownstreamCrate { trait_ref, self_ty }
                            };
                            debug!(?cause, "evaluate_stack: pushing cause");
                            self.intercrate_ambiguity_causes.as_mut().unwrap().insert(cause);
                        }
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

        let candidates = candidate_set.vec;

        debug!(?stack, ?candidates, "assembled {} candidates", candidates.len());

        // At this point, we know that each of the entries in the
        // candidate set is *individually* applicable. Now we have to
        // figure out if they contain mutual incompatibilities. This
        // frequently arises if we have an unconstrained input type --
        // for example, we are looking for `$0: Eq` where `$0` is some
        // unconstrained type variable. In that case, we'll get a
        // candidate which assumes $0 == int, one that assumes `$0 ==
        // usize`, etc. This spells an ambiguity.

        let mut candidates = self.filter_impls(candidates, stack.obligation);

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
        // is a `Vec<Bar>` and `Bar` does not implement `Clone`. If
        // we were to winnow, we'd wind up with zero candidates.
        // Instead, we select the right impl now but report "`Bar` does
        // not implement `Clone`".
        if candidates.len() == 1 {
            return self.filter_reservation_impls(candidates.pop().unwrap());
        }

        // Winnow, but record the exact outcome of evaluation, which
        // is needed for specialization. Propagate overflow if it occurs.
        let candidates = candidates
            .into_iter()
            .map(|c| match self.evaluate_candidate(stack, &c) {
                Ok(eval) if eval.may_apply() => {
                    Ok(Some(EvaluatedCandidate { candidate: c, evaluation: eval }))
                }
                Ok(_) => Ok(None),
                Err(OverflowError::Canonical) => Err(Overflow(OverflowError::Canonical)),
                Err(OverflowError::Error(e)) => Err(Overflow(OverflowError::Error(e))),
            })
            .flat_map(Result::transpose)
            .collect::<Result<Vec<_>, _>>()?;

        debug!(?stack, ?candidates, "{} potentially applicable candidates", candidates.len());
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
            // If there's an error type, 'downgrade' our result from
            // `Err(Unimplemented)` to `Ok(None)`. This helps us avoid
            // emitting additional spurious errors, since we're guaranteed
            // to have emitted at least one.
            if stack.obligation.predicate.references_error() {
                debug!(?stack.obligation.predicate, "found error type in predicate, treating as ambiguous");
                Ok(None)
            } else {
                Err(Unimplemented)
            }
        } else {
            let has_non_region_infer = stack.obligation.predicate.has_non_region_infer();
            if let Some(candidate) = self.winnow_candidates(has_non_region_infer, candidates) {
                self.filter_reservation_impls(candidate)
            } else {
                Ok(None)
            }
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

    /// Evaluates whether the obligation `obligation` can be satisfied
    /// and returns an `EvaluationResult`. This is meant for the
    /// *initial* call.
    ///
    /// Do not use this directly, use `infcx.evaluate_obligation` instead.
    pub fn evaluate_root_obligation(
        &mut self,
        obligation: &PredicateObligation<'tcx>,
    ) -> Result<EvaluationResult, OverflowError> {
        debug_assert!(!self.infcx.next_trait_solver());
        self.evaluation_probe(|this| {
            let goal =
                this.infcx.resolve_vars_if_possible((obligation.predicate, obligation.param_env));
            let mut result = this.evaluate_predicate_recursively(
                TraitObligationStackList::empty(&ProvisionalEvaluationCache::default()),
                obligation.clone(),
            )?;
            // If the predicate has done any inference, then downgrade the
            // result to ambiguous.
            if this.infcx.resolve_vars_if_possible(goal) != goal {
                result = result.max(EvaluatedToAmbig);
            }
            Ok(result)
        })
    }

    /// Computes the evaluation result of `op`, discarding any constraints.
    ///
    /// This also runs for leak check to allow higher ranked region errors to impact
    /// selection. By default it checks for leaks from all universes created inside of
    /// `op`, but this can be overwritten if necessary.
    fn evaluation_probe(
        &mut self,
        op: impl FnOnce(&mut Self) -> Result<EvaluationResult, OverflowError>,
    ) -> Result<EvaluationResult, OverflowError> {
        self.infcx.probe(|snapshot| -> Result<EvaluationResult, OverflowError> {
            let outer_universe = self.infcx.universe();
            let result = op(self)?;

            match self.infcx.leak_check(outer_universe, Some(snapshot)) {
                Ok(()) => {}
                Err(_) => return Ok(EvaluatedToErr),
            }

            if self.infcx.opaque_types_added_in_snapshot(snapshot) {
                return Ok(result.max(EvaluatedToOkModuloOpaqueTypes));
            }

            if self.infcx.region_constraints_added_in_snapshot(snapshot) {
                Ok(result.max(EvaluatedToOkModuloRegions))
            } else {
                Ok(result)
            }
        })
    }

    /// Evaluates the predicates in `predicates` recursively. This may
    /// guide inference. If this is not desired, run it inside of a
    /// is run within an inference probe.
    /// `probe`.
    #[instrument(skip(self, stack), level = "debug")]
    fn evaluate_predicates_recursively<'o, I>(
        &mut self,
        stack: TraitObligationStackList<'o, 'tcx>,
        predicates: I,
    ) -> Result<EvaluationResult, OverflowError>
    where
        I: IntoIterator<Item = PredicateObligation<'tcx>> + std::fmt::Debug,
    {
        let mut result = EvaluatedToOk;
        for mut obligation in predicates {
            obligation.set_depth_from_parent(stack.depth());
            let eval = self.evaluate_predicate_recursively(stack, obligation.clone())?;
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

    #[instrument(
        level = "debug",
        skip(self, previous_stack),
        fields(previous_stack = ?previous_stack.head())
        ret,
    )]
    fn evaluate_predicate_recursively<'o>(
        &mut self,
        previous_stack: TraitObligationStackList<'o, 'tcx>,
        obligation: PredicateObligation<'tcx>,
    ) -> Result<EvaluationResult, OverflowError> {
        debug_assert!(!self.infcx.next_trait_solver());
        // `previous_stack` stores a `PolyTraitObligation`, while `obligation` is
        // a `PredicateObligation`. These are distinct types, so we can't
        // use any `Option` combinator method that would force them to be
        // the same.
        match previous_stack.head() {
            Some(h) => self.check_recursion_limit(&obligation, h.obligation)?,
            None => self.check_recursion_limit(&obligation, &obligation)?,
        }

        if sizedness_fast_path(self.tcx(), obligation.predicate) {
            return Ok(EvaluatedToOk);
        }

        ensure_sufficient_stack(|| {
            let bound_predicate = obligation.predicate.kind();
            match bound_predicate.skip_binder() {
                ty::PredicateKind::Clause(ty::ClauseKind::Trait(t)) => {
                    let t = bound_predicate.rebind(t);
                    debug_assert!(!t.has_escaping_bound_vars());
                    let obligation = obligation.with(self.tcx(), t);
                    self.evaluate_trait_predicate_recursively(previous_stack, obligation)
                }

                ty::PredicateKind::Clause(ty::ClauseKind::HostEffect(data)) => {
                    self.infcx.enter_forall(bound_predicate.rebind(data), |data| {
                        match effects::evaluate_host_effect_obligation(
                            self,
                            &obligation.with(self.tcx(), data),
                        ) {
                            Ok(nested) => {
                                self.evaluate_predicates_recursively(previous_stack, nested)
                            }
                            Err(effects::EvaluationFailure::Ambiguous) => Ok(EvaluatedToAmbig),
                            Err(effects::EvaluationFailure::NoSolution) => Ok(EvaluatedToErr),
                        }
                    })
                }

                ty::PredicateKind::Subtype(p) => {
                    let p = bound_predicate.rebind(p);
                    // Does this code ever run?
                    match self.infcx.subtype_predicate(&obligation.cause, obligation.param_env, p) {
                        Ok(Ok(InferOk { obligations, .. })) => {
                            self.evaluate_predicates_recursively(previous_stack, obligations)
                        }
                        Ok(Err(_)) => Ok(EvaluatedToErr),
                        Err(..) => Ok(EvaluatedToAmbig),
                    }
                }

                ty::PredicateKind::Coerce(p) => {
                    let p = bound_predicate.rebind(p);
                    // Does this code ever run?
                    match self.infcx.coerce_predicate(&obligation.cause, obligation.param_env, p) {
                        Ok(Ok(InferOk { obligations, .. })) => {
                            self.evaluate_predicates_recursively(previous_stack, obligations)
                        }
                        Ok(Err(_)) => Ok(EvaluatedToErr),
                        Err(..) => Ok(EvaluatedToAmbig),
                    }
                }

                ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(term)) => {
                    // So, there is a bit going on here. First, `WellFormed` predicates
                    // are coinductive, like trait predicates with auto traits.
                    // This means that we need to detect if we have recursively
                    // evaluated `WellFormed(X)`. Otherwise, we would run into
                    // a "natural" overflow error.
                    //
                    // Now, the next question is whether we need to do anything
                    // special with caching. Considering the following tree:
                    // - `WF(Foo<T>)`
                    //   - `Bar<T>: Send`
                    //     - `WF(Foo<T>)`
                    //   - `Foo<T>: Trait`
                    // In this case, the innermost `WF(Foo<T>)` should return
                    // `EvaluatedToOk`, since it's coinductive. Then if
                    // `Bar<T>: Send` is resolved to `EvaluatedToOk`, it can be
                    // inserted into a cache (because without thinking about `WF`
                    // goals, it isn't in a cycle). If `Foo<T>: Trait` later doesn't
                    // hold, then `Bar<T>: Send` shouldn't hold. Therefore, we
                    // *do* need to keep track of coinductive cycles.

                    let cache = previous_stack.cache;
                    let dfn = cache.next_dfn();

                    for stack_term in previous_stack.cache.wf_args.borrow().iter().rev() {
                        if stack_term.0 != term {
                            continue;
                        }
                        debug!("WellFormed({:?}) on stack", term);
                        if let Some(stack) = previous_stack.head {
                            // Okay, let's imagine we have two different stacks:
                            //   `T: NonAutoTrait -> WF(T) -> T: NonAutoTrait`
                            //   `WF(T) -> T: NonAutoTrait -> WF(T)`
                            // Because of this, we need to check that all
                            // predicates between the WF goals are coinductive.
                            // Otherwise, we can say that `T: NonAutoTrait` is
                            // true.
                            // Let's imagine we have a predicate stack like
                            //         `Foo: Bar -> WF(T) -> T: NonAutoTrait -> T: Auto`
                            // depth   ^1                    ^2                 ^3
                            // and the current predicate is `WF(T)`. `wf_args`
                            // would contain `(T, 1)`. We want to check all
                            // trait predicates greater than `1`. The previous
                            // stack would be `T: Auto`.
                            let cycle = stack.iter().take_while(|s| s.depth > stack_term.1);
                            let tcx = self.tcx();
                            let cycle = cycle.map(|stack| stack.obligation.predicate.upcast(tcx));
                            if self.coinductive_match(cycle) {
                                stack.update_reached_depth(stack_term.1);
                                return Ok(EvaluatedToOk);
                            } else {
                                return Ok(EvaluatedToAmbigStackDependent);
                            }
                        }
                        return Ok(EvaluatedToOk);
                    }

                    match wf::obligations(
                        self.infcx,
                        obligation.param_env,
                        obligation.cause.body_id,
                        obligation.recursion_depth + 1,
                        term,
                        obligation.cause.span,
                    ) {
                        Some(obligations) => {
                            cache.wf_args.borrow_mut().push((term, previous_stack.depth()));
                            let result =
                                self.evaluate_predicates_recursively(previous_stack, obligations);
                            cache.wf_args.borrow_mut().pop();

                            let result = result?;

                            if !result.must_apply_modulo_regions() {
                                cache.on_failure(dfn);
                            }

                            cache.on_completion(dfn);

                            Ok(result)
                        }
                        None => Ok(EvaluatedToAmbig),
                    }
                }

                ty::PredicateKind::Clause(ty::ClauseKind::TypeOutlives(pred)) => {
                    // A global type with no free lifetimes or generic parameters
                    // outlives anything.
                    if pred.0.has_free_regions()
                        || pred.0.has_bound_regions()
                        || pred.0.has_non_region_infer()
                        || pred.0.has_non_region_infer()
                    {
                        Ok(EvaluatedToOkModuloRegions)
                    } else {
                        Ok(EvaluatedToOk)
                    }
                }

                ty::PredicateKind::Clause(ty::ClauseKind::RegionOutlives(..)) => {
                    // We do not consider region relationships when evaluating trait matches.
                    Ok(EvaluatedToOkModuloRegions)
                }

                ty::PredicateKind::DynCompatible(trait_def_id) => {
                    if self.tcx().is_dyn_compatible(trait_def_id) {
                        Ok(EvaluatedToOk)
                    } else {
                        Ok(EvaluatedToErr)
                    }
                }

                ty::PredicateKind::Clause(ty::ClauseKind::Projection(data)) => {
                    let data = bound_predicate.rebind(data);
                    let project_obligation = obligation.with(self.tcx(), data);
                    match project::poly_project_and_unify_term(self, &project_obligation) {
                        ProjectAndUnifyResult::Holds(mut subobligations) => {
                            'compute_res: {
                                // If we've previously marked this projection as 'complete', then
                                // use the final cached result (either `EvaluatedToOk` or
                                // `EvaluatedToOkModuloRegions`), and skip re-evaluating the
                                // sub-obligations.
                                if let Some(key) =
                                    ProjectionCacheKey::from_poly_projection_obligation(
                                        self,
                                        &project_obligation,
                                    )
                                {
                                    if let Some(cached_res) = self
                                        .infcx
                                        .inner
                                        .borrow_mut()
                                        .projection_cache()
                                        .is_complete(key)
                                    {
                                        break 'compute_res Ok(cached_res);
                                    }
                                }

                                // Need to explicitly set the depth of nested goals here as
                                // projection obligations can cycle by themselves and in
                                // `evaluate_predicates_recursively` we only add the depth
                                // for parent trait goals because only these get added to the
                                // `TraitObligationStackList`.
                                for subobligation in subobligations.iter_mut() {
                                    subobligation.set_depth_from_parent(obligation.recursion_depth);
                                }
                                let res = self.evaluate_predicates_recursively(
                                    previous_stack,
                                    subobligations,
                                );
                                if let Ok(eval_rslt) = res
                                    && (eval_rslt == EvaluatedToOk
                                        || eval_rslt == EvaluatedToOkModuloRegions)
                                    && let Some(key) =
                                        ProjectionCacheKey::from_poly_projection_obligation(
                                            self,
                                            &project_obligation,
                                        )
                                {
                                    // If the result is something that we can cache, then mark this
                                    // entry as 'complete'. This will allow us to skip evaluating the
                                    // subobligations at all the next time we evaluate the projection
                                    // predicate.
                                    self.infcx
                                        .inner
                                        .borrow_mut()
                                        .projection_cache()
                                        .complete(key, eval_rslt);
                                }
                                res
                            }
                        }
                        ProjectAndUnifyResult::FailedNormalization => Ok(EvaluatedToAmbig),
                        ProjectAndUnifyResult::Recursive => Ok(EvaluatedToAmbigStackDependent),
                        ProjectAndUnifyResult::MismatchedProjectionTypes(_) => Ok(EvaluatedToErr),
                    }
                }

                ty::PredicateKind::Clause(ty::ClauseKind::ConstEvaluatable(uv)) => {
                    match const_evaluatable::is_const_evaluatable(
                        self.infcx,
                        uv,
                        obligation.param_env,
                        obligation.cause.span,
                    ) {
                        Ok(()) => Ok(EvaluatedToOk),
                        Err(NotConstEvaluatable::MentionsInfer) => Ok(EvaluatedToAmbig),
                        Err(NotConstEvaluatable::MentionsParam) => Ok(EvaluatedToErr),
                        Err(_) => Ok(EvaluatedToErr),
                    }
                }

                ty::PredicateKind::ConstEquate(c1, c2) => {
                    let tcx = self.tcx();
                    assert!(
                        tcx.features().generic_const_exprs(),
                        "`ConstEquate` without a feature gate: {c1:?} {c2:?}",
                    );

                    {
                        let c1 = tcx.expand_abstract_consts(c1);
                        let c2 = tcx.expand_abstract_consts(c2);
                        debug!(
                            "evaluate_predicate_recursively: equating consts:\nc1= {:?}\nc2= {:?}",
                            c1, c2
                        );

                        use rustc_hir::def::DefKind;
                        match (c1.kind(), c2.kind()) {
                            (ty::ConstKind::Unevaluated(a), ty::ConstKind::Unevaluated(b))
                                if a.def == b.def && tcx.def_kind(a.def) == DefKind::AssocConst =>
                            {
                                if let Ok(InferOk { obligations, value: () }) = self
                                    .infcx
                                    .at(&obligation.cause, obligation.param_env)
                                    // Can define opaque types as this is only reachable with
                                    // `generic_const_exprs`
                                    .eq(
                                        DefineOpaqueTypes::Yes,
                                        ty::AliasTerm::from(a),
                                        ty::AliasTerm::from(b),
                                    )
                                {
                                    return self.evaluate_predicates_recursively(
                                        previous_stack,
                                        obligations,
                                    );
                                }
                            }
                            (_, ty::ConstKind::Unevaluated(_))
                            | (ty::ConstKind::Unevaluated(_), _) => (),
                            (_, _) => {
                                if let Ok(InferOk { obligations, value: () }) = self
                                    .infcx
                                    .at(&obligation.cause, obligation.param_env)
                                    // Can define opaque types as this is only reachable with
                                    // `generic_const_exprs`
                                    .eq(DefineOpaqueTypes::Yes, c1, c2)
                                {
                                    return self.evaluate_predicates_recursively(
                                        previous_stack,
                                        obligations,
                                    );
                                }
                            }
                        }
                    }

                    let evaluate = |c: ty::Const<'tcx>| {
                        if let ty::ConstKind::Unevaluated(_) = c.kind() {
                            match crate::traits::try_evaluate_const(
                                self.infcx,
                                c,
                                obligation.param_env,
                            ) {
                                Ok(val) => Ok(val),
                                Err(e) => Err(e),
                            }
                        } else {
                            Ok(c)
                        }
                    };

                    match (evaluate(c1), evaluate(c2)) {
                        (Ok(c1), Ok(c2)) => {
                            match self.infcx.at(&obligation.cause, obligation.param_env).eq(
                                // Can define opaque types as this is only reachable with
                                // `generic_const_exprs`
                                DefineOpaqueTypes::Yes,
                                c1,
                                c2,
                            ) {
                                Ok(inf_ok) => self.evaluate_predicates_recursively(
                                    previous_stack,
                                    inf_ok.into_obligations(),
                                ),
                                Err(_) => Ok(EvaluatedToErr),
                            }
                        }
                        (Err(EvaluateConstErr::InvalidConstParamTy(..)), _)
                        | (_, Err(EvaluateConstErr::InvalidConstParamTy(..))) => Ok(EvaluatedToErr),
                        (Err(EvaluateConstErr::EvaluationFailure(..)), _)
                        | (_, Err(EvaluateConstErr::EvaluationFailure(..))) => Ok(EvaluatedToErr),
                        (Err(EvaluateConstErr::HasGenericsOrInfers), _)
                        | (_, Err(EvaluateConstErr::HasGenericsOrInfers)) => {
                            if c1.has_non_region_infer() || c2.has_non_region_infer() {
                                Ok(EvaluatedToAmbig)
                            } else {
                                // Two different constants using generic parameters ~> error.
                                Ok(EvaluatedToErr)
                            }
                        }
                    }
                }
                ty::PredicateKind::NormalizesTo(..) => {
                    bug!("NormalizesTo is only used by the new solver")
                }
                ty::PredicateKind::AliasRelate(..) => {
                    bug!("AliasRelate is only used by the new solver")
                }
                ty::PredicateKind::Ambiguous => Ok(EvaluatedToAmbig),
                ty::PredicateKind::Clause(ty::ClauseKind::ConstArgHasType(ct, ty)) => {
                    let ct = self.infcx.shallow_resolve_const(ct);
                    let ct_ty = match ct.kind() {
                        ty::ConstKind::Infer(_) => {
                            return Ok(EvaluatedToAmbig);
                        }
                        ty::ConstKind::Error(_) => return Ok(EvaluatedToOk),
                        ty::ConstKind::Value(cv) => cv.ty,
                        ty::ConstKind::Unevaluated(uv) => {
                            self.tcx().type_of(uv.def).instantiate(self.tcx(), uv.args)
                        }
                        // FIXME(generic_const_exprs): See comment in `fulfill.rs`
                        ty::ConstKind::Expr(_) => return Ok(EvaluatedToOk),
                        ty::ConstKind::Placeholder(_) => {
                            bug!("placeholder const {:?} in old solver", ct)
                        }
                        ty::ConstKind::Bound(_, _) => bug!("escaping bound vars in {:?}", ct),
                        ty::ConstKind::Param(param_ct) => {
                            param_ct.find_ty_from_env(obligation.param_env)
                        }
                    };

                    match self.infcx.at(&obligation.cause, obligation.param_env).eq(
                        // Only really exercised by generic_const_exprs
                        DefineOpaqueTypes::Yes,
                        ct_ty,
                        ty,
                    ) {
                        Ok(inf_ok) => self.evaluate_predicates_recursively(
                            previous_stack,
                            inf_ok.into_obligations(),
                        ),
                        Err(_) => Ok(EvaluatedToErr),
                    }
                }
            }
        })
    }

    #[instrument(skip(self, previous_stack), level = "debug", ret)]
    fn evaluate_trait_predicate_recursively<'o>(
        &mut self,
        previous_stack: TraitObligationStackList<'o, 'tcx>,
        mut obligation: PolyTraitObligation<'tcx>,
    ) -> Result<EvaluationResult, OverflowError> {
        if !matches!(self.infcx.typing_mode(), TypingMode::Coherence)
            && obligation.is_global()
            && obligation.param_env.caller_bounds().iter().all(|bound| bound.has_param())
        {
            // If a param env has no global bounds, global obligations do not
            // depend on its particular value in order to work, so we can clear
            // out the param env and get better caching.
            debug!("in global");
            obligation.param_env = ty::ParamEnv::empty();
        }

        let stack = self.push_stack(previous_stack, &obligation);
        let fresh_trait_pred = stack.fresh_trait_pred;
        let param_env = obligation.param_env;

        debug!(?fresh_trait_pred);

        // If a trait predicate is in the (local or global) evaluation cache,
        // then we know it holds without cycles.
        if let Some(result) = self.check_evaluation_cache(param_env, fresh_trait_pred) {
            debug!("CACHE HIT");
            return Ok(result);
        }

        if let Some(result) = stack.cache().get_provisional(fresh_trait_pred) {
            debug!("PROVISIONAL CACHE HIT");
            stack.update_reached_depth(result.reached_depth);
            return Ok(result.result);
        }

        // Check if this is a match for something already on the
        // stack. If so, we don't want to insert the result into the
        // main cache (it is cycle dependent) nor the provisional
        // cache (which is meant for things that have completed but
        // for a "backedge" -- this result *is* the backedge).
        if let Some(cycle_result) = self.check_evaluation_cycle(&stack) {
            return Ok(cycle_result);
        }

        let (result, dep_node) = self.in_task(|this| {
            let mut result = this.evaluate_stack(&stack)?;

            // fix issue #103563, we don't normalize
            // nested obligations which produced by `TraitDef` candidate
            // (i.e. using bounds on assoc items as assumptions).
            // because we don't have enough information to
            // normalize these obligations before evaluating.
            // so we will try to normalize the obligation and evaluate again.
            // we will replace it with new solver in the future.
            if EvaluationResult::EvaluatedToErr == result
                && fresh_trait_pred.has_aliases()
                && fresh_trait_pred.is_global()
            {
                let mut nested_obligations = PredicateObligations::new();
                let predicate = normalize_with_depth_to(
                    this,
                    param_env,
                    obligation.cause.clone(),
                    obligation.recursion_depth + 1,
                    obligation.predicate,
                    &mut nested_obligations,
                );
                if predicate != obligation.predicate {
                    let mut nested_result = EvaluationResult::EvaluatedToOk;
                    for obligation in nested_obligations {
                        nested_result = cmp::max(
                            this.evaluate_predicate_recursively(previous_stack, obligation)?,
                            nested_result,
                        );
                    }

                    if nested_result.must_apply_modulo_regions() {
                        let obligation = obligation.with(this.tcx(), predicate);
                        result = cmp::max(
                            nested_result,
                            this.evaluate_trait_predicate_recursively(previous_stack, obligation)?,
                        );
                    }
                }
            }

            Ok::<_, OverflowError>(result)
        });

        let result = result?;

        if !result.must_apply_modulo_regions() {
            stack.cache().on_failure(stack.dfn);
        }

        let reached_depth = stack.reached_depth.get();
        if reached_depth >= stack.depth {
            debug!("CACHE MISS");
            self.insert_evaluation_cache(param_env, fresh_trait_pred, dep_node, result);
            stack.cache().on_completion(stack.dfn);
        } else {
            debug!("PROVISIONAL");
            debug!(
                "caching provisionally because {:?} \
                 is a cycle participant (at depth {}, reached depth {})",
                fresh_trait_pred, stack.depth, reached_depth,
            );

            stack.cache().insert_provisional(stack.dfn, reached_depth, fresh_trait_pred, result);
        }

        Ok(result)
    }

    /// If there is any previous entry on the stack that precisely
    /// matches this obligation, then we can assume that the
    /// obligation is satisfied for now (still all other conditions
    /// must be met of course). One obvious case this comes up is
    /// marker traits like `Send`. Think of a linked list:
    ///
    ///     struct List<T> { data: T, next: Option<Box<List<T>>> }
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
                    && stack.fresh_trait_pred == prev.fresh_trait_pred
            })
            .map(|stack| stack.depth)
        {
            debug!("evaluate_stack --> recursive at depth {}", cycle_depth);

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
            let tcx = self.tcx();
            let cycle = cycle.map(|stack| stack.obligation.predicate.upcast(tcx));
            if self.coinductive_match(cycle) {
                debug!("evaluate_stack --> recursive, coinductive");
                Some(EvaluatedToOk)
            } else {
                debug!("evaluate_stack --> recursive, inductive");
                Some(EvaluatedToAmbigStackDependent)
            }
        } else {
            None
        }
    }

    fn evaluate_stack<'o>(
        &mut self,
        stack: &TraitObligationStack<'o, 'tcx>,
    ) -> Result<EvaluationResult, OverflowError> {
        debug_assert!(!self.infcx.next_trait_solver());
        // In intercrate mode, whenever any of the generics are unbound,
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
            stack.fresh_trait_pred.skip_binder().trait_ref.args.types().any(|ty| ty.is_fresh());

        if unbound_input_types
            && stack.iter().skip(1).any(|prev| {
                stack.obligation.param_env == prev.obligation.param_env
                    && self.match_fresh_trait_refs(stack.fresh_trait_pred, prev.fresh_trait_pred)
            })
        {
            debug!("evaluate_stack --> unbound argument, recursive --> giving up",);
            return Ok(EvaluatedToAmbigStackDependent);
        }

        match self.candidate_from_obligation(stack) {
            Ok(Some(c)) => self.evaluate_candidate(stack, &c),
            Ok(None) => Ok(EvaluatedToAmbig),
            Err(Overflow(OverflowError::Canonical)) => Err(OverflowError::Canonical),
            Err(..) => Ok(EvaluatedToErr),
        }
    }

    /// For defaulted traits, we use a co-inductive strategy to solve, so
    /// that recursion is ok. This routine returns `true` if the top of the
    /// stack (`cycle[0]`):
    ///
    /// - is a coinductive trait: an auto-trait or `Sized`,
    /// - it also appears in the backtrace at some position `X`,
    /// - all the predicates at positions `X..` between `X` and the top are
    ///   also coinductive traits.
    pub(crate) fn coinductive_match<I>(&mut self, mut cycle: I) -> bool
    where
        I: Iterator<Item = ty::Predicate<'tcx>>,
    {
        cycle.all(|p| match p.kind().skip_binder() {
            ty::PredicateKind::Clause(ty::ClauseKind::Trait(data)) => {
                self.infcx.tcx.trait_is_coinductive(data.def_id())
            }
            ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(_)) => {
                // FIXME(generic_const_exprs): GCE needs well-formedness predicates to be
                // coinductive, but GCE is on the way out anyways, so this should eventually
                // be replaced with `false`.
                self.infcx.tcx.features().generic_const_exprs()
            }
            _ => false,
        })
    }

    /// Further evaluates `candidate` to decide whether all type parameters match and whether nested
    /// obligations are met. Returns whether `candidate` remains viable after this further
    /// scrutiny.
    #[instrument(
        level = "debug",
        skip(self, stack),
        fields(depth = stack.obligation.recursion_depth),
        ret
    )]
    fn evaluate_candidate<'o>(
        &mut self,
        stack: &TraitObligationStack<'o, 'tcx>,
        candidate: &SelectionCandidate<'tcx>,
    ) -> Result<EvaluationResult, OverflowError> {
        let mut result = self.evaluation_probe(|this| {
            match this.confirm_candidate(stack.obligation, candidate.clone()) {
                Ok(selection) => {
                    debug!(?selection);
                    this.evaluate_predicates_recursively(
                        stack.list(),
                        selection.nested_obligations().into_iter(),
                    )
                }
                Err(..) => Ok(EvaluatedToErr),
            }
        })?;

        // If we erased any lifetimes, then we want to use
        // `EvaluatedToOkModuloRegions` instead of `EvaluatedToOk`
        // as your final result. The result will be cached using
        // the freshened trait predicate as a key, so we need
        // our result to be correct by *any* choice of original lifetimes,
        // not just the lifetime choice for this particular (non-erased)
        // predicate.
        // See issue #80691
        if stack.fresh_trait_pred.has_erased_regions() {
            result = result.max(EvaluatedToOkModuloRegions);
        }

        Ok(result)
    }

    fn check_evaluation_cache(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
    ) -> Option<EvaluationResult> {
        let infcx = self.infcx;
        let tcx = infcx.tcx;
        if self.can_use_global_caches(param_env, trait_pred) {
            let key = (infcx.typing_env(param_env), trait_pred);
            if let Some(res) = tcx.evaluation_cache.get(&key, tcx) {
                Some(res)
            } else {
                debug_assert_eq!(infcx.evaluation_cache.get(&(param_env, trait_pred), tcx), None);
                None
            }
        } else {
            self.infcx.evaluation_cache.get(&(param_env, trait_pred), tcx)
        }
    }

    fn insert_evaluation_cache(
        &mut self,
        param_env: ty::ParamEnv<'tcx>,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
        dep_node: DepNodeIndex,
        result: EvaluationResult,
    ) {
        // Avoid caching results that depend on more than just the trait-ref
        // - the stack can create recursion.
        if result.is_stack_dependent() {
            return;
        }

        let infcx = self.infcx;
        let tcx = infcx.tcx;
        if self.can_use_global_caches(param_env, trait_pred) {
            debug!(?trait_pred, ?result, "insert_evaluation_cache global");
            // This may overwrite the cache with the same value
            tcx.evaluation_cache.insert(
                (infcx.typing_env(param_env), trait_pred),
                dep_node,
                result,
            );
            return;
        } else {
            debug!(?trait_pred, ?result, "insert_evaluation_cache local");
            self.infcx.evaluation_cache.insert((param_env, trait_pred), dep_node, result);
        }
    }

    fn check_recursion_depth<T>(
        &self,
        depth: usize,
        error_obligation: &Obligation<'tcx, T>,
    ) -> Result<(), OverflowError>
    where
        T: Upcast<TyCtxt<'tcx>, ty::Predicate<'tcx>> + Clone,
    {
        if !self.infcx.tcx.recursion_limit().value_within_limit(depth) {
            match self.query_mode {
                TraitQueryMode::Standard => {
                    if let Some(e) = self.infcx.tainted_by_errors() {
                        return Err(OverflowError::Error(e));
                    }
                    self.infcx.err_ctxt().report_overflow_obligation(error_obligation, true);
                }
                TraitQueryMode::Canonical => {
                    return Err(OverflowError::Canonical);
                }
            }
        }
        Ok(())
    }

    /// Checks that the recursion limit has not been exceeded.
    ///
    /// The weird return type of this function allows it to be used with the `try` (`?`)
    /// operator within certain functions.
    #[inline(always)]
    fn check_recursion_limit<T: Display + TypeFoldable<TyCtxt<'tcx>>, V>(
        &self,
        obligation: &Obligation<'tcx, T>,
        error_obligation: &Obligation<'tcx, V>,
    ) -> Result<(), OverflowError>
    where
        V: Upcast<TyCtxt<'tcx>, ty::Predicate<'tcx>> + Clone,
    {
        self.check_recursion_depth(obligation.recursion_depth, error_obligation)
    }

    fn in_task<OP, R>(&mut self, op: OP) -> (R, DepNodeIndex)
    where
        OP: FnOnce(&mut Self) -> R,
    {
        self.tcx().dep_graph.with_anon_task(self.tcx(), dep_kinds::TraitSelect, || op(self))
    }

    /// filter_impls filters candidates that have a positive impl for a negative
    /// goal and a negative impl for a positive goal
    #[instrument(level = "debug", skip(self, candidates))]
    fn filter_impls(
        &mut self,
        candidates: Vec<SelectionCandidate<'tcx>>,
        obligation: &PolyTraitObligation<'tcx>,
    ) -> Vec<SelectionCandidate<'tcx>> {
        trace!("{candidates:#?}");
        let tcx = self.tcx();
        let mut result = Vec::with_capacity(candidates.len());

        for candidate in candidates {
            if let ImplCandidate(def_id) = candidate {
                match (tcx.impl_polarity(def_id), obligation.polarity()) {
                    (ty::ImplPolarity::Reservation, _)
                    | (ty::ImplPolarity::Positive, ty::PredicatePolarity::Positive)
                    | (ty::ImplPolarity::Negative, ty::PredicatePolarity::Negative) => {
                        result.push(candidate);
                    }
                    _ => {}
                }
            } else {
                result.push(candidate);
            }
        }

        trace!("{result:#?}");
        result
    }

    /// filter_reservation_impls filter reservation impl for any goal as ambiguous
    #[instrument(level = "debug", skip(self))]
    fn filter_reservation_impls(
        &mut self,
        candidate: SelectionCandidate<'tcx>,
    ) -> SelectionResult<'tcx, SelectionCandidate<'tcx>> {
        let tcx = self.tcx();
        // Treat reservation impls as ambiguity.
        if let ImplCandidate(def_id) = candidate {
            if let ty::ImplPolarity::Reservation = tcx.impl_polarity(def_id) {
                if let Some(intercrate_ambiguity_clauses) = &mut self.intercrate_ambiguity_causes {
                    let message = tcx
                        .get_attr(def_id, sym::rustc_reservation_impl)
                        .and_then(|a| a.value_str());
                    if let Some(message) = message {
                        debug!(
                            "filter_reservation_impls: \
                                 reservation impl ambiguity on {:?}",
                            def_id
                        );
                        intercrate_ambiguity_clauses
                            .insert(IntercrateAmbiguityCause::ReservationImpl { message });
                    }
                }
                return Ok(None);
            }
        }
        Ok(Some(candidate))
    }

    fn is_knowable<'o>(&mut self, stack: &TraitObligationStack<'o, 'tcx>) -> Result<(), Conflict> {
        let obligation = &stack.obligation;
        match self.infcx.typing_mode() {
            TypingMode::Coherence => {}
            TypingMode::Analysis { .. }
            | TypingMode::Borrowck { .. }
            | TypingMode::PostBorrowckAnalysis { .. }
            | TypingMode::PostAnalysis => return Ok(()),
        }

        debug!("is_knowable()");

        let predicate = self.infcx.resolve_vars_if_possible(obligation.predicate);

        // Okay to skip binder because of the nature of the
        // trait-ref-is-knowable check, which does not care about
        // bound regions.
        let trait_ref = predicate.skip_binder().trait_ref;

        coherence::trait_ref_is_knowable(self.infcx, trait_ref, |ty| Ok::<_, !>(ty)).into_ok()
    }

    /// Returns `true` if the global caches can be used.
    fn can_use_global_caches(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        pred: ty::PolyTraitPredicate<'tcx>,
    ) -> bool {
        // If there are any inference variables in the `ParamEnv`, then we
        // always use a cache local to this particular scope. Otherwise, we
        // switch to a global cache.
        if param_env.has_infer() || pred.has_infer() {
            return false;
        }

        match self.infcx.typing_mode() {
            // Avoid using the global cache during coherence and just rely
            // on the local cache. It is really just a simplification to
            // avoid us having to fear that coherence results "pollute"
            // the master cache. Since coherence executes pretty quickly,
            // it's not worth going to more trouble to increase the
            // hit-rate, I don't think.
            TypingMode::Coherence => false,
            // Avoid using the global cache when we're defining opaque types
            // as their hidden type may impact the result of candidate selection.
            //
            // HACK: This is still theoretically unsound. Goals can indirectly rely
            // on opaques in the defining scope, and it's easier to do so with TAIT.
            // However, if we disqualify *all* goals from being cached, perf suffers.
            // This is likely fixed by better caching in general in the new solver.
            // See: <https://github.com/rust-lang/rust/issues/132064>.
            TypingMode::Analysis {
                defining_opaque_types_and_generators: defining_opaque_types,
            }
            | TypingMode::Borrowck { defining_opaque_types } => {
                defining_opaque_types.is_empty() || !pred.has_opaque_types()
            }
            // The hidden types of `defined_opaque_types` is not local to the current
            // inference context, so we can freely move this to the global cache.
            TypingMode::PostBorrowckAnalysis { .. } => true,
            // The global cache is only used if there are no opaque types in
            // the defining scope or we're outside of analysis.
            //
            // FIXME(#132279): This is still incorrect as we treat opaque types
            // and default associated items differently between these two modes.
            TypingMode::PostAnalysis => true,
        }
    }

    fn check_candidate_cache(
        &mut self,
        param_env: ty::ParamEnv<'tcx>,
        cache_fresh_trait_pred: ty::PolyTraitPredicate<'tcx>,
    ) -> Option<SelectionResult<'tcx, SelectionCandidate<'tcx>>> {
        let infcx = self.infcx;
        let tcx = infcx.tcx;
        let pred = cache_fresh_trait_pred.skip_binder();

        if self.can_use_global_caches(param_env, cache_fresh_trait_pred) {
            if let Some(res) = tcx.selection_cache.get(&(infcx.typing_env(param_env), pred), tcx) {
                return Some(res);
            } else if cfg!(debug_assertions) {
                match infcx.selection_cache.get(&(param_env, pred), tcx) {
                    None | Some(Err(Overflow(OverflowError::Canonical))) => {}
                    res => bug!("unexpected local cache result: {res:?}"),
                }
            }
        }

        // Subtle: we need to check the local cache even if we're able to use the
        // global cache as we don't cache overflow in the global cache but need to
        // cache it as otherwise rustdoc hangs when compiling diesel.
        infcx.selection_cache.get(&(param_env, pred), tcx)
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
            Ok(Some(SelectionCandidate::ParamCandidate(trait_ref))) => !trait_ref.has_infer(),
            _ => true,
        }
    }

    #[instrument(skip(self, param_env, cache_fresh_trait_pred, dep_node), level = "debug")]
    fn insert_candidate_cache(
        &mut self,
        param_env: ty::ParamEnv<'tcx>,
        cache_fresh_trait_pred: ty::PolyTraitPredicate<'tcx>,
        dep_node: DepNodeIndex,
        candidate: SelectionResult<'tcx, SelectionCandidate<'tcx>>,
    ) {
        let infcx = self.infcx;
        let tcx = infcx.tcx;
        let pred = cache_fresh_trait_pred.skip_binder();

        if !self.can_cache_candidate(&candidate) {
            debug!(?pred, ?candidate, "insert_candidate_cache - candidate is not cacheable");
            return;
        }

        if self.can_use_global_caches(param_env, cache_fresh_trait_pred) {
            if let Err(Overflow(OverflowError::Canonical)) = candidate {
                // Don't cache overflow globally; we only produce this in certain modes.
            } else {
                debug!(?pred, ?candidate, "insert_candidate_cache global");
                debug_assert!(!candidate.has_infer());

                // This may overwrite the cache with the same value.
                tcx.selection_cache.insert(
                    (infcx.typing_env(param_env), pred),
                    dep_node,
                    candidate,
                );
                return;
            }
        }

        debug!(?pred, ?candidate, "insert_candidate_cache local");
        self.infcx.selection_cache.insert((param_env, pred), dep_node, candidate);
    }

    /// Looks at the item bounds of the projection or opaque type.
    /// If this is a nested rigid projection, such as
    /// `<<T as Tr1>::Assoc as Tr2>::Assoc`, consider the item bounds
    /// on both `Tr1::Assoc` and `Tr2::Assoc`, since we may encounter
    /// relative bounds on both via the `associated_type_bounds` feature.
    pub(super) fn for_each_item_bound<T>(
        &mut self,
        mut self_ty: Ty<'tcx>,
        mut for_each: impl FnMut(&mut Self, ty::Clause<'tcx>, usize) -> ControlFlow<T, ()>,
        on_ambiguity: impl FnOnce(),
    ) -> ControlFlow<T, ()> {
        let mut idx = 0;
        let mut in_parent_alias_type = false;

        loop {
            let (kind, alias_ty) = match *self_ty.kind() {
                ty::Alias(kind @ (ty::Projection | ty::Opaque), alias_ty) => (kind, alias_ty),
                ty::Infer(ty::TyVar(_)) => {
                    on_ambiguity();
                    return ControlFlow::Continue(());
                }
                _ => return ControlFlow::Continue(()),
            };

            // HACK: On subsequent recursions, we only care about bounds that don't
            // share the same type as `self_ty`. This is because for truly rigid
            // projections, we will never be able to equate, e.g. `<T as Tr>::A`
            // with `<<T as Tr>::A as Tr>::A`.
            let relevant_bounds = if in_parent_alias_type {
                self.tcx().item_non_self_bounds(alias_ty.def_id)
            } else {
                self.tcx().item_self_bounds(alias_ty.def_id)
            };

            for bound in relevant_bounds.instantiate(self.tcx(), alias_ty.args) {
                for_each(self, bound, idx)?;
                idx += 1;
            }

            if kind == ty::Projection {
                self_ty = alias_ty.self_ty();
            } else {
                return ControlFlow::Continue(());
            }

            in_parent_alias_type = true;
        }
    }

    /// Equates the trait in `obligation` with trait bound. If the two traits
    /// can be equated and the normalized trait bound doesn't contain inference
    /// variables or placeholders, the normalized bound is returned.
    fn match_normalize_trait_ref(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
        placeholder_trait_ref: ty::TraitRef<'tcx>,
        trait_bound: ty::PolyTraitRef<'tcx>,
    ) -> Result<Option<ty::TraitRef<'tcx>>, ()> {
        debug_assert!(!placeholder_trait_ref.has_escaping_bound_vars());
        if placeholder_trait_ref.def_id != trait_bound.def_id() {
            // Avoid unnecessary normalization
            return Err(());
        }

        let drcx = DeepRejectCtxt::relate_rigid_rigid(self.infcx.tcx);
        let obligation_args = obligation.predicate.skip_binder().trait_ref.args;
        if !drcx.args_may_unify(obligation_args, trait_bound.skip_binder().args) {
            return Err(());
        }

        let trait_bound = self.infcx.instantiate_binder_with_fresh_vars(
            obligation.cause.span,
            HigherRankedType,
            trait_bound,
        );
        let Normalized { value: trait_bound, obligations: _ } = ensure_sufficient_stack(|| {
            normalize_with_depth(
                self,
                obligation.param_env,
                obligation.cause.clone(),
                obligation.recursion_depth + 1,
                trait_bound,
            )
        });
        self.infcx
            .at(&obligation.cause, obligation.param_env)
            .eq(DefineOpaqueTypes::No, placeholder_trait_ref, trait_bound)
            .map(|InferOk { obligations: _, value: () }| {
                // This method is called within a probe, so we can't have
                // inference variables and placeholders escape.
                if !trait_bound.has_infer() && !trait_bound.has_placeholders() {
                    Some(trait_bound)
                } else {
                    None
                }
            })
            .map_err(|_| ())
    }

    fn where_clause_may_apply<'o>(
        &mut self,
        stack: &TraitObligationStack<'o, 'tcx>,
        where_clause_trait_ref: ty::PolyTraitRef<'tcx>,
    ) -> Result<EvaluationResult, OverflowError> {
        self.evaluation_probe(|this| {
            match this.match_where_clause_trait_ref(stack.obligation, where_clause_trait_ref) {
                Ok(obligations) => this.evaluate_predicates_recursively(stack.list(), obligations),
                Err(()) => Ok(EvaluatedToErr),
            }
        })
    }

    /// Return `Yes` if the obligation's predicate type applies to the env_predicate, and
    /// `No` if it does not. Return `Ambiguous` in the case that the projection type is a GAT,
    /// and applying this env_predicate constrains any of the obligation's GAT parameters.
    ///
    /// This behavior is a somewhat of a hack to prevent over-constraining inference variables
    /// in cases like #91762.
    pub(super) fn match_projection_projections(
        &mut self,
        obligation: &ProjectionTermObligation<'tcx>,
        env_predicate: PolyProjectionPredicate<'tcx>,
        potentially_unnormalized_candidates: bool,
    ) -> ProjectionMatchesProjection {
        debug_assert_eq!(obligation.predicate.def_id, env_predicate.item_def_id());

        let mut nested_obligations = PredicateObligations::new();
        let infer_predicate = self.infcx.instantiate_binder_with_fresh_vars(
            obligation.cause.span,
            BoundRegionConversionTime::HigherRankedType,
            env_predicate,
        );
        let infer_projection = if potentially_unnormalized_candidates {
            ensure_sufficient_stack(|| {
                normalize_with_depth_to(
                    self,
                    obligation.param_env,
                    obligation.cause.clone(),
                    obligation.recursion_depth + 1,
                    infer_predicate.projection_term,
                    &mut nested_obligations,
                )
            })
        } else {
            infer_predicate.projection_term
        };

        let is_match = self
            .infcx
            .at(&obligation.cause, obligation.param_env)
            .eq(DefineOpaqueTypes::No, obligation.predicate, infer_projection)
            .is_ok_and(|InferOk { obligations, value: () }| {
                self.evaluate_predicates_recursively(
                    TraitObligationStackList::empty(&ProvisionalEvaluationCache::default()),
                    nested_obligations.into_iter().chain(obligations),
                )
                .is_ok_and(|res| res.may_apply())
            });

        if is_match {
            let generics = self.tcx().generics_of(obligation.predicate.def_id);
            // FIXME(generic_associated_types): Addresses aggressive inference in #92917.
            // If this type is a GAT, and of the GAT args resolve to something new,
            // that means that we must have newly inferred something about the GAT.
            // We should give up in that case.
            //
            // This only detects one layer of inference, which is probably not what we actually
            // want, but fixing it causes some ambiguity:
            // <https://github.com/rust-lang/rust/issues/125196>.
            if !generics.is_own_empty()
                && obligation.predicate.args[generics.parent_count..].iter().any(|&p| {
                    p.has_non_region_infer()
                        && match p.unpack() {
                            ty::GenericArgKind::Const(ct) => {
                                self.infcx.shallow_resolve_const(ct) != ct
                            }
                            ty::GenericArgKind::Type(ty) => self.infcx.shallow_resolve(ty) != ty,
                            ty::GenericArgKind::Lifetime(_) => false,
                        }
                })
            {
                ProjectionMatchesProjection::Ambiguous
            } else {
                ProjectionMatchesProjection::Yes
            }
        } else {
            ProjectionMatchesProjection::No
        }
    }
}

/// ## Winnowing
///
/// Winnowing is the process of attempting to resolve ambiguity by
/// probing further. During the winnowing process, we unify all
/// type variables and then we also attempt to evaluate recursive
/// bounds to see if they are satisfied.
impl<'tcx> SelectionContext<'_, 'tcx> {
    /// If there are multiple ways to prove a trait goal, we make some
    /// *fairly arbitrary* choices about which candidate is actually used.
    ///
    /// For more details, look at the implementation of this method :)
    #[instrument(level = "debug", skip(self), ret)]
    fn winnow_candidates(
        &mut self,
        has_non_region_infer: bool,
        mut candidates: Vec<EvaluatedCandidate<'tcx>>,
    ) -> Option<SelectionCandidate<'tcx>> {
        if candidates.len() == 1 {
            return Some(candidates.pop().unwrap().candidate);
        }

        // We prefer `Sized` candidates over everything.
        let mut sized_candidates =
            candidates.iter().filter(|c| matches!(c.candidate, SizedCandidate { has_nested: _ }));
        if let Some(sized_candidate) = sized_candidates.next() {
            // There should only ever be a single sized candidate
            // as they would otherwise overlap.
            debug_assert_eq!(sized_candidates.next(), None);
            // Only prefer the built-in `Sized` candidate if its nested goals are certain.
            // Otherwise, we may encounter failure later on if inference causes this candidate
            // to not hold, but a where clause would've applied instead.
            if sized_candidate.evaluation.must_apply_modulo_regions() {
                return Some(sized_candidate.candidate.clone());
            } else {
                return None;
            }
        }

        // Before we consider where-bounds, we have to deduplicate them here and also
        // drop where-bounds in case the same where-bound exists without bound vars.
        // This is necessary as elaborating super-trait bounds may result in duplicates.
        'search_victim: loop {
            for (i, this) in candidates.iter().enumerate() {
                let ParamCandidate(this) = this.candidate else { continue };
                for (j, other) in candidates.iter().enumerate() {
                    if i == j {
                        continue;
                    }

                    let ParamCandidate(other) = other.candidate else { continue };
                    if this == other {
                        candidates.remove(j);
                        continue 'search_victim;
                    }

                    if this.skip_binder().trait_ref == other.skip_binder().trait_ref
                        && this.skip_binder().polarity == other.skip_binder().polarity
                        && !this.skip_binder().trait_ref.has_escaping_bound_vars()
                    {
                        candidates.remove(j);
                        continue 'search_victim;
                    }
                }
            }

            break;
        }

        // The next highest priority is for non-global where-bounds. However, while we don't
        // prefer global where-clauses here, we do bail with ambiguity when encountering both
        // a global and a non-global where-clause.
        //
        // Our handling of where-bounds is generally fairly messy but necessary for backwards
        // compatibility, see #50825 for why we need to handle global where-bounds like this.
        let is_global = |c: ty::PolyTraitPredicate<'tcx>| c.is_global() && !c.has_bound_vars();
        let param_candidates = candidates
            .iter()
            .filter_map(|c| if let ParamCandidate(p) = c.candidate { Some(p) } else { None });
        let mut has_global_bounds = false;
        let mut param_candidate = None;
        for c in param_candidates {
            if is_global(c) {
                has_global_bounds = true;
            } else if param_candidate.replace(c).is_some() {
                // Ambiguity, two potentially different where-clauses
                return None;
            }
        }
        if let Some(predicate) = param_candidate {
            // Ambiguity, a global and a non-global where-bound.
            if has_global_bounds {
                return None;
            } else {
                return Some(ParamCandidate(predicate));
            }
        }

        // Prefer alias-bounds over blanket impls for rigid associated types. This is
        // fairly arbitrary but once again necessary for backwards compatibility.
        // If there are multiple applicable candidates which don't affect type inference,
        // choose the one with the lowest index.
        let alias_bound = candidates
            .iter()
            .filter_map(|c| if let ProjectionCandidate(i) = c.candidate { Some(i) } else { None })
            .try_reduce(|c1, c2| if has_non_region_infer { None } else { Some(c1.min(c2)) });
        match alias_bound {
            Some(Some(index)) => return Some(ProjectionCandidate(index)),
            Some(None) => {}
            None => return None,
        }

        // Need to prioritize builtin trait object impls as `<dyn Any as Any>::type_id`
        // should use the vtable method and not the method provided by the user-defined
        // impl `impl<T: ?Sized> Any for T { .. }`. This really shouldn't exist but is
        // necessary due to #57893. We again arbitrarily prefer the applicable candidate
        // with the lowest index.
        let object_bound = candidates
            .iter()
            .filter_map(|c| if let ObjectCandidate(i) = c.candidate { Some(i) } else { None })
            .try_reduce(|c1, c2| if has_non_region_infer { None } else { Some(c1.min(c2)) });
        match object_bound {
            Some(Some(index)) => return Some(ObjectCandidate(index)),
            Some(None) => {}
            None => return None,
        }
        // Same for upcasting.
        let upcast_bound = candidates
            .iter()
            .filter_map(|c| {
                if let TraitUpcastingUnsizeCandidate(i) = c.candidate { Some(i) } else { None }
            })
            .try_reduce(|c1, c2| if has_non_region_infer { None } else { Some(c1.min(c2)) });
        match upcast_bound {
            Some(Some(index)) => return Some(TraitUpcastingUnsizeCandidate(index)),
            Some(None) => {}
            None => return None,
        }

        // Finally, handle overlapping user-written impls.
        let impls = candidates.iter().filter_map(|c| {
            if let ImplCandidate(def_id) = c.candidate {
                Some((def_id, c.evaluation))
            } else {
                None
            }
        });
        let mut impl_candidate = None;
        for c in impls {
            if let Some(prev) = impl_candidate.replace(c) {
                if self.prefer_lhs_over_victim(has_non_region_infer, c, prev.0) {
                    // Ok, prefer `c` over the previous entry
                } else if self.prefer_lhs_over_victim(has_non_region_infer, prev, c.0) {
                    // Ok, keep `prev` instead of the new entry
                    impl_candidate = Some(prev);
                } else {
                    // Ambiguity, two potentially different where-clauses
                    return None;
                }
            }
        }
        if let Some((def_id, _evaluation)) = impl_candidate {
            // Don't use impl candidates which overlap with other candidates.
            // This should pretty much only ever happen with malformed impls.
            if candidates.iter().all(|c| match c.candidate {
                SizedCandidate { has_nested: _ }
                | BuiltinCandidate { has_nested: _ }
                | TransmutabilityCandidate
                | AutoImplCandidate
                | ClosureCandidate { .. }
                | AsyncClosureCandidate
                | AsyncFnKindHelperCandidate
                | CoroutineCandidate
                | FutureCandidate
                | IteratorCandidate
                | AsyncIteratorCandidate
                | FnPointerCandidate
                | TraitAliasCandidate
                | TraitUpcastingUnsizeCandidate(_)
                | BuiltinObjectCandidate
                | BuiltinUnsizeCandidate
                | BikeshedGuaranteedNoDropCandidate => false,
                // Non-global param candidates have already been handled, global
                // where-bounds get ignored.
                ParamCandidate(_) | ImplCandidate(_) => true,
                ProjectionCandidate(_) | ObjectCandidate(_) => unreachable!(),
            }) {
                return Some(ImplCandidate(def_id));
            } else {
                return None;
            }
        }

        if candidates.len() == 1 {
            Some(candidates.pop().unwrap().candidate)
        } else {
            // Also try ignoring all global where-bounds and check whether we end
            // with a unique candidate in this case.
            let mut not_a_global_where_bound = candidates
                .into_iter()
                .filter(|c| !matches!(c.candidate, ParamCandidate(p) if is_global(p)));
            not_a_global_where_bound
                .next()
                .map(|c| c.candidate)
                .filter(|_| not_a_global_where_bound.next().is_none())
        }
    }

    fn prefer_lhs_over_victim(
        &self,
        has_non_region_infer: bool,
        (lhs, lhs_evaluation): (DefId, EvaluationResult),
        victim: DefId,
    ) -> bool {
        let tcx = self.tcx();
        // See if we can toss out `victim` based on specialization.
        //
        // While this requires us to know *for sure* that the `lhs` impl applies
        // we still use modulo regions here. This is fine as specialization currently
        // assumes that specializing impls have to be always applicable, meaning that
        // the only allowed region constraints may be constraints also present on the default impl.
        if lhs_evaluation.must_apply_modulo_regions() {
            if tcx.specializes((lhs, victim)) {
                return true;
            }
        }

        match tcx.impls_are_allowed_to_overlap(lhs, victim) {
            // For candidates which already reference errors it doesn't really
            // matter what we do 🤷
            Some(ty::ImplOverlapKind::Permitted { marker: false }) => {
                lhs_evaluation.must_apply_considering_regions()
            }
            Some(ty::ImplOverlapKind::Permitted { marker: true }) => {
                // Subtle: If the predicate we are evaluating has inference
                // variables, do *not* allow discarding candidates due to
                // marker trait impls.
                //
                // Without this restriction, we could end up accidentally
                // constraining inference variables based on an arbitrarily
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
                // none left to constrain)
                // 2) Be left with some unconstrained inference variables. We
                // will then correctly report an inference error, since the
                // existence of multiple marker trait impls tells us nothing
                // about which one should actually apply.
                !has_non_region_infer && lhs_evaluation.must_apply_considering_regions()
            }
            None => false,
        }
    }
}

impl<'tcx> SelectionContext<'_, 'tcx> {
    fn sized_conditions(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
    ) -> BuiltinImplConditions<'tcx> {
        use self::BuiltinImplConditions::{Ambiguous, None, Where};

        // NOTE: binder moved to (*)
        let self_ty = self.infcx.shallow_resolve(obligation.predicate.skip_binder().self_ty());

        match self_ty.kind() {
            ty::Infer(ty::IntVar(_) | ty::FloatVar(_))
            | ty::Uint(_)
            | ty::Int(_)
            | ty::Bool
            | ty::Float(_)
            | ty::FnDef(..)
            | ty::FnPtr(..)
            | ty::RawPtr(..)
            | ty::Char
            | ty::Ref(..)
            | ty::Coroutine(..)
            | ty::CoroutineWitness(..)
            | ty::Array(..)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Never
            | ty::Dynamic(_, _, ty::DynStar)
            | ty::Error(_) => {
                // safe for everything
                Where(ty::Binder::dummy(Vec::new()))
            }

            ty::Str | ty::Slice(_) | ty::Dynamic(..) | ty::Foreign(..) => None,

            ty::Tuple(tys) => Where(
                obligation.predicate.rebind(tys.last().map_or_else(Vec::new, |&last| vec![last])),
            ),

            ty::Pat(ty, _) => Where(obligation.predicate.rebind(vec![*ty])),

            ty::Adt(def, args) => {
                if let Some(sized_crit) = def.sized_constraint(self.tcx()) {
                    // (*) binder moved here
                    Where(
                        obligation.predicate.rebind(vec![sized_crit.instantiate(self.tcx(), args)]),
                    )
                } else {
                    Where(ty::Binder::dummy(Vec::new()))
                }
            }

            // FIXME(unsafe_binders): This binder needs to be squashed
            ty::UnsafeBinder(binder_ty) => Where(binder_ty.map_bound(|ty| vec![ty])),

            ty::Alias(..) | ty::Param(_) | ty::Placeholder(..) => None,
            ty::Infer(ty::TyVar(_)) => Ambiguous,

            // We can make this an ICE if/once we actually instantiate the trait obligation eagerly.
            ty::Bound(..) => None,

            ty::Infer(ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) => {
                bug!("asked to assemble builtin bounds of unexpected type: {:?}", self_ty);
            }
        }
    }

    fn copy_clone_conditions(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
    ) -> BuiltinImplConditions<'tcx> {
        // NOTE: binder moved to (*)
        let self_ty = self.infcx.shallow_resolve(obligation.predicate.skip_binder().self_ty());

        use self::BuiltinImplConditions::{Ambiguous, None, Where};

        match *self_ty.kind() {
            ty::FnDef(..) | ty::FnPtr(..) | ty::Error(_) => Where(ty::Binder::dummy(Vec::new())),

            ty::Uint(_)
            | ty::Int(_)
            | ty::Infer(ty::IntVar(_) | ty::FloatVar(_))
            | ty::Bool
            | ty::Float(_)
            | ty::Char
            | ty::RawPtr(..)
            | ty::Never
            | ty::Ref(_, _, hir::Mutability::Not)
            | ty::Array(..) => {
                // Implementations provided in libcore
                None
            }

            // FIXME(unsafe_binder): Should we conditionally
            // (i.e. universally) implement copy/clone?
            ty::UnsafeBinder(_) => None,

            ty::Dynamic(..)
            | ty::Str
            | ty::Slice(..)
            | ty::Foreign(..)
            | ty::Ref(_, _, hir::Mutability::Mut) => None,

            ty::Tuple(tys) => {
                // (*) binder moved here
                Where(obligation.predicate.rebind(tys.iter().collect()))
            }

            ty::Pat(ty, _) => {
                // (*) binder moved here
                Where(obligation.predicate.rebind(vec![ty]))
            }

            ty::Coroutine(coroutine_def_id, args) => {
                match self.tcx().coroutine_movability(coroutine_def_id) {
                    hir::Movability::Static => None,
                    hir::Movability::Movable => {
                        if self.tcx().features().coroutine_clone() {
                            let resolved_upvars =
                                self.infcx.shallow_resolve(args.as_coroutine().tupled_upvars_ty());
                            let resolved_witness =
                                self.infcx.shallow_resolve(args.as_coroutine().witness());
                            if resolved_upvars.is_ty_var() || resolved_witness.is_ty_var() {
                                // Not yet resolved.
                                Ambiguous
                            } else {
                                let all = args
                                    .as_coroutine()
                                    .upvar_tys()
                                    .iter()
                                    .chain([args.as_coroutine().witness()])
                                    .collect::<Vec<_>>();
                                Where(obligation.predicate.rebind(all))
                            }
                        } else {
                            None
                        }
                    }
                }
            }

            ty::CoroutineWitness(def_id, args) => {
                let hidden_types = rebind_coroutine_witness_types(
                    self.infcx.tcx,
                    def_id,
                    args,
                    obligation.predicate.bound_vars(),
                );
                Where(hidden_types)
            }

            ty::Closure(_, args) => {
                // (*) binder moved here
                let ty = self.infcx.shallow_resolve(args.as_closure().tupled_upvars_ty());
                if let ty::Infer(ty::TyVar(_)) = ty.kind() {
                    // Not yet resolved.
                    Ambiguous
                } else {
                    Where(obligation.predicate.rebind(args.as_closure().upvar_tys().to_vec()))
                }
            }

            ty::CoroutineClosure(_, args) => {
                // (*) binder moved here
                let ty = self.infcx.shallow_resolve(args.as_coroutine_closure().tupled_upvars_ty());
                if let ty::Infer(ty::TyVar(_)) = ty.kind() {
                    // Not yet resolved.
                    Ambiguous
                } else {
                    Where(
                        obligation
                            .predicate
                            .rebind(args.as_coroutine_closure().upvar_tys().to_vec()),
                    )
                }
            }

            ty::Adt(..) | ty::Alias(..) | ty::Param(..) | ty::Placeholder(..) => {
                // Fallback to whatever user-defined impls exist in this case.
                None
            }

            ty::Infer(ty::TyVar(_)) => {
                // Unbound type variable. Might or might not have
                // applicable impls and so forth, depending on what
                // those type variables wind up being bound to.
                Ambiguous
            }

            // We can make this an ICE if/once we actually instantiate the trait obligation eagerly.
            ty::Bound(..) => None,

            ty::Infer(ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) => {
                bug!("asked to assemble builtin bounds of unexpected type: {:?}", self_ty);
            }
        }
    }

    fn fused_iterator_conditions(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
    ) -> BuiltinImplConditions<'tcx> {
        let self_ty = self.infcx.shallow_resolve(obligation.self_ty().skip_binder());
        if let ty::Coroutine(did, ..) = *self_ty.kind()
            && self.tcx().coroutine_is_gen(did)
        {
            BuiltinImplConditions::Where(ty::Binder::dummy(Vec::new()))
        } else {
            BuiltinImplConditions::None
        }
    }

    /// For default impls, we need to break apart a type into its
    /// "constituent types" -- meaning, the types that it contains.
    ///
    /// Here are some (simple) examples:
    ///
    /// ```ignore (illustrative)
    /// (i32, u32) -> [i32, u32]
    /// Foo where struct Foo { x: i32, y: u32 } -> [i32, u32]
    /// Bar<i32> where struct Bar<T> { x: T, y: u32 } -> [i32, u32]
    /// Zed<i32> where enum Zed { A(T), B(u32) } -> [i32, u32]
    /// ```
    #[instrument(level = "debug", skip(self), ret)]
    fn constituent_types_for_ty(
        &self,
        t: ty::Binder<'tcx, Ty<'tcx>>,
    ) -> Result<ty::Binder<'tcx, Vec<Ty<'tcx>>>, SelectionError<'tcx>> {
        Ok(match *t.skip_binder().kind() {
            ty::Uint(_)
            | ty::Int(_)
            | ty::Bool
            | ty::Float(_)
            | ty::FnDef(..)
            | ty::FnPtr(..)
            | ty::Error(_)
            | ty::Infer(ty::IntVar(_) | ty::FloatVar(_))
            | ty::Never
            | ty::Char => ty::Binder::dummy(Vec::new()),

            // This branch is only for `experimental_default_bounds`.
            // Other foreign types were rejected earlier in
            // `assemble_candidates_from_auto_impls`.
            ty::Foreign(..) => ty::Binder::dummy(Vec::new()),

            // FIXME(unsafe_binders): Squash the double binder for now, I guess.
            ty::UnsafeBinder(_) => return Err(SelectionError::Unimplemented),

            // Treat this like `struct str([u8]);`
            ty::Str => ty::Binder::dummy(vec![Ty::new_slice(self.tcx(), self.tcx().types.u8)]),

            ty::Placeholder(..)
            | ty::Dynamic(..)
            | ty::Param(..)
            | ty::Alias(ty::Projection | ty::Inherent | ty::Free, ..)
            | ty::Bound(..)
            | ty::Infer(ty::TyVar(_) | ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) => {
                bug!("asked to assemble constituent types of unexpected type: {:?}", t);
            }

            ty::RawPtr(element_ty, _) | ty::Ref(_, element_ty, _) => t.rebind(vec![element_ty]),

            ty::Pat(ty, _) | ty::Array(ty, _) | ty::Slice(ty) => t.rebind(vec![ty]),

            ty::Tuple(tys) => {
                // (T1, ..., Tn) -- meets any bound that all of T1...Tn meet
                t.rebind(tys.iter().collect())
            }

            ty::Closure(_, args) => {
                let ty = self.infcx.shallow_resolve(args.as_closure().tupled_upvars_ty());
                t.rebind(vec![ty])
            }

            ty::CoroutineClosure(_, args) => {
                let ty = self.infcx.shallow_resolve(args.as_coroutine_closure().tupled_upvars_ty());
                t.rebind(vec![ty])
            }

            ty::Coroutine(_, args) => {
                let ty = self.infcx.shallow_resolve(args.as_coroutine().tupled_upvars_ty());
                let witness = args.as_coroutine().witness();
                t.rebind([ty].into_iter().chain(iter::once(witness)).collect())
            }

            ty::CoroutineWitness(def_id, args) => {
                rebind_coroutine_witness_types(self.infcx.tcx, def_id, args, t.bound_vars())
            }

            // For `PhantomData<T>`, we pass `T`.
            ty::Adt(def, args) if def.is_phantom_data() => t.rebind(args.types().collect()),

            ty::Adt(def, args) => {
                t.rebind(def.all_fields().map(|f| f.ty(self.tcx(), args)).collect())
            }

            ty::Alias(ty::Opaque, ty::AliasTy { def_id, args, .. }) => {
                if self.infcx.can_define_opaque_ty(def_id) {
                    unreachable!()
                } else {
                    // We can resolve the `impl Trait` to its concrete type,
                    // which enforces a DAG between the functions requiring
                    // the auto trait bounds in question.
                    match self.tcx().type_of_opaque(def_id) {
                        Ok(ty) => t.rebind(vec![ty.instantiate(self.tcx(), args)]),
                        Err(_) => {
                            return Err(SelectionError::OpaqueTypeAutoTraitLeakageUnknown(def_id));
                        }
                    }
                }
            }
        })
    }

    fn collect_predicates_for_types(
        &mut self,
        param_env: ty::ParamEnv<'tcx>,
        cause: ObligationCause<'tcx>,
        recursion_depth: usize,
        trait_def_id: DefId,
        types: Vec<Ty<'tcx>>,
    ) -> PredicateObligations<'tcx> {
        // Because the types were potentially derived from
        // higher-ranked obligations they may reference late-bound
        // regions. For example, `for<'a> Foo<&'a i32> : Copy` would
        // yield a type like `for<'a> &'a i32`. In general, we
        // maintain the invariant that we never manipulate bound
        // regions, so we have to process these bound regions somehow.
        //
        // The strategy is to:
        //
        // 1. Instantiate those regions to placeholder regions (e.g.,
        //    `for<'a> &'a i32` becomes `&0 i32`.
        // 2. Produce something like `&'0 i32 : Copy`
        // 3. Re-bind the regions back to `for<'a> &'a i32 : Copy`

        types
            .into_iter()
            .flat_map(|placeholder_ty| {
                let Normalized { value: normalized_ty, mut obligations } =
                    ensure_sufficient_stack(|| {
                        normalize_with_depth(
                            self,
                            param_env,
                            cause.clone(),
                            recursion_depth,
                            placeholder_ty,
                        )
                    });

                let tcx = self.tcx();
                let trait_ref = if tcx.generics_of(trait_def_id).own_params.len() == 1 {
                    ty::TraitRef::new(tcx, trait_def_id, [normalized_ty])
                } else {
                    // If this is an ill-formed auto/built-in trait, then synthesize
                    // new error args for the missing generics.
                    let err_args = ty::GenericArgs::extend_with_error(
                        tcx,
                        trait_def_id,
                        &[normalized_ty.into()],
                    );
                    ty::TraitRef::new_from_args(tcx, trait_def_id, err_args)
                };

                let obligation = Obligation::new(self.tcx(), cause.clone(), param_env, trait_ref);
                obligations.push(obligation);
                obligations
            })
            .collect()
    }

    ///////////////////////////////////////////////////////////////////////////
    // Matching
    //
    // Matching is a common path used for both evaluation and
    // confirmation. It basically unifies types that appear in impls
    // and traits. This does affect the surrounding environment;
    // therefore, when used during evaluation, match routines must be
    // run inside of a `probe()` so that their side-effects are
    // contained.

    fn rematch_impl(
        &mut self,
        impl_def_id: DefId,
        obligation: &PolyTraitObligation<'tcx>,
    ) -> Normalized<'tcx, GenericArgsRef<'tcx>> {
        let impl_trait_header = self.tcx().impl_trait_header(impl_def_id).unwrap();
        match self.match_impl(impl_def_id, impl_trait_header, obligation) {
            Ok(args) => args,
            Err(()) => {
                let predicate = self.infcx.resolve_vars_if_possible(obligation.predicate);
                bug!("impl {impl_def_id:?} was matchable against {predicate:?} but now is not")
            }
        }
    }

    #[instrument(level = "debug", skip(self), ret)]
    fn match_impl(
        &mut self,
        impl_def_id: DefId,
        impl_trait_header: ty::ImplTraitHeader<'tcx>,
        obligation: &PolyTraitObligation<'tcx>,
    ) -> Result<Normalized<'tcx, GenericArgsRef<'tcx>>, ()> {
        let placeholder_obligation =
            self.infcx.enter_forall_and_leak_universe(obligation.predicate);
        let placeholder_obligation_trait_ref = placeholder_obligation.trait_ref;

        let impl_args = self.infcx.fresh_args_for_item(obligation.cause.span, impl_def_id);

        let trait_ref = impl_trait_header.trait_ref.instantiate(self.tcx(), impl_args);
        debug!(?impl_trait_header);

        let Normalized { value: impl_trait_ref, obligations: mut nested_obligations } =
            ensure_sufficient_stack(|| {
                normalize_with_depth(
                    self,
                    obligation.param_env,
                    obligation.cause.clone(),
                    obligation.recursion_depth + 1,
                    trait_ref,
                )
            });

        debug!(?impl_trait_ref, ?placeholder_obligation_trait_ref);

        let cause = ObligationCause::new(
            obligation.cause.span,
            obligation.cause.body_id,
            ObligationCauseCode::MatchImpl(obligation.cause.clone(), impl_def_id),
        );

        let InferOk { obligations, .. } = self
            .infcx
            .at(&cause, obligation.param_env)
            .eq(DefineOpaqueTypes::No, placeholder_obligation_trait_ref, impl_trait_ref)
            .map_err(|e| {
                debug!("match_impl: failed eq_trait_refs due to `{}`", e.to_string(self.tcx()))
            })?;
        nested_obligations.extend(obligations);

        if impl_trait_header.polarity == ty::ImplPolarity::Reservation
            && !matches!(self.infcx.typing_mode(), TypingMode::Coherence)
        {
            debug!("reservation impls only apply in intercrate mode");
            return Err(());
        }

        Ok(Normalized { value: impl_args, obligations: nested_obligations })
    }

    fn match_upcast_principal(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
        unnormalized_upcast_principal: ty::PolyTraitRef<'tcx>,
        a_data: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
        b_data: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
        a_region: ty::Region<'tcx>,
        b_region: ty::Region<'tcx>,
    ) -> SelectionResult<'tcx, PredicateObligations<'tcx>> {
        let tcx = self.tcx();
        let mut nested = PredicateObligations::new();

        // We may upcast to auto traits that are either explicitly listed in
        // the object type's bounds, or implied by the principal trait ref's
        // supertraits.
        let a_auto_traits: FxIndexSet<DefId> = a_data
            .auto_traits()
            .chain(a_data.principal_def_id().into_iter().flat_map(|principal_def_id| {
                elaborate::supertrait_def_ids(tcx, principal_def_id)
                    .filter(|def_id| tcx.trait_is_auto(*def_id))
            }))
            .collect();

        let upcast_principal = normalize_with_depth_to(
            self,
            obligation.param_env,
            obligation.cause.clone(),
            obligation.recursion_depth + 1,
            unnormalized_upcast_principal,
            &mut nested,
        );

        for bound in b_data {
            match bound.skip_binder() {
                // Check that a_ty's supertrait (upcast_principal) is compatible
                // with the target (b_ty).
                ty::ExistentialPredicate::Trait(target_principal) => {
                    let hr_source_principal = upcast_principal.map_bound(|trait_ref| {
                        ty::ExistentialTraitRef::erase_self_ty(tcx, trait_ref)
                    });
                    let hr_target_principal = bound.rebind(target_principal);

                    nested.extend(
                        self.infcx
                            .enter_forall(hr_target_principal, |target_principal| {
                                let source_principal =
                                    self.infcx.instantiate_binder_with_fresh_vars(
                                        obligation.cause.span,
                                        HigherRankedType,
                                        hr_source_principal,
                                    );
                                self.infcx.at(&obligation.cause, obligation.param_env).eq_trace(
                                    DefineOpaqueTypes::Yes,
                                    ToTrace::to_trace(
                                        &obligation.cause,
                                        hr_target_principal,
                                        hr_source_principal,
                                    ),
                                    target_principal,
                                    source_principal,
                                )
                            })
                            .map_err(|_| SelectionError::Unimplemented)?
                            .into_obligations(),
                    );
                }
                // Check that b_ty's projection is satisfied by exactly one of
                // a_ty's projections. First, we look through the list to see if
                // any match. If not, error. Then, if *more* than one matches, we
                // return ambiguity. Otherwise, if exactly one matches, equate
                // it with b_ty's projection.
                ty::ExistentialPredicate::Projection(target_projection) => {
                    let hr_target_projection = bound.rebind(target_projection);

                    let mut matching_projections =
                        a_data.projection_bounds().filter(|&hr_source_projection| {
                            // Eager normalization means that we can just use can_eq
                            // here instead of equating and processing obligations.
                            hr_source_projection.item_def_id() == hr_target_projection.item_def_id()
                                && self.infcx.probe(|_| {
                                    self.infcx
                                        .enter_forall(hr_target_projection, |target_projection| {
                                            let source_projection =
                                                self.infcx.instantiate_binder_with_fresh_vars(
                                                    obligation.cause.span,
                                                    HigherRankedType,
                                                    hr_source_projection,
                                                );
                                            self.infcx
                                                .at(&obligation.cause, obligation.param_env)
                                                .eq_trace(
                                                    DefineOpaqueTypes::Yes,
                                                    ToTrace::to_trace(
                                                        &obligation.cause,
                                                        hr_target_projection,
                                                        hr_source_projection,
                                                    ),
                                                    target_projection,
                                                    source_projection,
                                                )
                                        })
                                        .is_ok()
                                })
                        });

                    let Some(hr_source_projection) = matching_projections.next() else {
                        return Err(SelectionError::Unimplemented);
                    };
                    if matching_projections.next().is_some() {
                        return Ok(None);
                    }
                    nested.extend(
                        self.infcx
                            .enter_forall(hr_target_projection, |target_projection| {
                                let source_projection =
                                    self.infcx.instantiate_binder_with_fresh_vars(
                                        obligation.cause.span,
                                        HigherRankedType,
                                        hr_source_projection,
                                    );
                                self.infcx.at(&obligation.cause, obligation.param_env).eq_trace(
                                    DefineOpaqueTypes::Yes,
                                    ToTrace::to_trace(
                                        &obligation.cause,
                                        hr_target_projection,
                                        hr_source_projection,
                                    ),
                                    target_projection,
                                    source_projection,
                                )
                            })
                            .map_err(|_| SelectionError::Unimplemented)?
                            .into_obligations(),
                    );
                }
                // Check that b_ty's auto traits are present in a_ty's bounds.
                ty::ExistentialPredicate::AutoTrait(def_id) => {
                    if !a_auto_traits.contains(&def_id) {
                        return Err(SelectionError::Unimplemented);
                    }
                }
            }
        }

        nested.push(Obligation::with_depth(
            tcx,
            obligation.cause.clone(),
            obligation.recursion_depth + 1,
            obligation.param_env,
            ty::Binder::dummy(ty::OutlivesPredicate(a_region, b_region)),
        ));

        Ok(Some(nested))
    }

    /// Normalize `where_clause_trait_ref` and try to match it against
    /// `obligation`. If successful, return any predicates that
    /// result from the normalization.
    fn match_where_clause_trait_ref(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
        where_clause_trait_ref: ty::PolyTraitRef<'tcx>,
    ) -> Result<PredicateObligations<'tcx>, ()> {
        self.match_poly_trait_ref(obligation, where_clause_trait_ref)
    }

    /// Returns `Ok` if `poly_trait_ref` being true implies that the
    /// obligation is satisfied.
    #[instrument(skip(self), level = "debug")]
    fn match_poly_trait_ref(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
        poly_trait_ref: ty::PolyTraitRef<'tcx>,
    ) -> Result<PredicateObligations<'tcx>, ()> {
        let predicate = self.infcx.enter_forall_and_leak_universe(obligation.predicate);
        let trait_ref = self.infcx.instantiate_binder_with_fresh_vars(
            obligation.cause.span,
            HigherRankedType,
            poly_trait_ref,
        );
        self.infcx
            .at(&obligation.cause, obligation.param_env)
            .eq(DefineOpaqueTypes::No, predicate.trait_ref, trait_ref)
            .map(|InferOk { obligations, .. }| obligations)
            .map_err(|_| ())
    }

    ///////////////////////////////////////////////////////////////////////////
    // Miscellany

    fn match_fresh_trait_refs(
        &self,
        previous: ty::PolyTraitPredicate<'tcx>,
        current: ty::PolyTraitPredicate<'tcx>,
    ) -> bool {
        let mut matcher = _match::MatchAgainstFreshVars::new(self.tcx());
        matcher.relate(previous, current).is_ok()
    }

    fn push_stack<'o>(
        &mut self,
        previous_stack: TraitObligationStackList<'o, 'tcx>,
        obligation: &'o PolyTraitObligation<'tcx>,
    ) -> TraitObligationStack<'o, 'tcx> {
        let fresh_trait_pred = obligation.predicate.fold_with(&mut self.freshener);

        let dfn = previous_stack.cache.next_dfn();
        let depth = previous_stack.depth() + 1;
        TraitObligationStack {
            obligation,
            fresh_trait_pred,
            reached_depth: Cell::new(depth),
            previous: previous_stack,
            dfn,
            depth,
        }
    }

    #[instrument(skip(self), level = "debug")]
    fn closure_trait_ref_unnormalized(
        &mut self,
        self_ty: Ty<'tcx>,
        fn_trait_def_id: DefId,
    ) -> ty::PolyTraitRef<'tcx> {
        let ty::Closure(_, args) = *self_ty.kind() else {
            bug!("expected closure, found {self_ty}");
        };
        let closure_sig = args.as_closure().sig();

        closure_trait_ref_and_return_type(
            self.tcx(),
            fn_trait_def_id,
            self_ty,
            closure_sig,
            util::TupleArgumentsFlag::No,
        )
        .map_bound(|(trait_ref, _)| trait_ref)
    }

    /// Returns the obligations that are implied by instantiating an
    /// impl or trait. The obligations are instantiated and fully
    /// normalized. This is used when confirming an impl or default
    /// impl.
    #[instrument(level = "debug", skip(self, cause, param_env))]
    fn impl_or_trait_obligations(
        &mut self,
        cause: &ObligationCause<'tcx>,
        recursion_depth: usize,
        param_env: ty::ParamEnv<'tcx>,
        def_id: DefId,              // of impl or trait
        args: GenericArgsRef<'tcx>, // for impl or trait
        parent_trait_pred: ty::Binder<'tcx, ty::TraitPredicate<'tcx>>,
    ) -> PredicateObligations<'tcx> {
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
        // When we instantiate, say, `V => IntoIter<u32>, U => $0`, the last
        // obligation will normalize to `<$0 as Iterator>::Item = $1` and
        // `$1: Copy`, so we must ensure the obligations are emitted in
        // that order.
        let predicates = tcx.predicates_of(def_id);
        assert_eq!(predicates.parent, None);
        let predicates = predicates.instantiate_own(tcx, args);
        let mut obligations = PredicateObligations::with_capacity(predicates.len());
        for (index, (predicate, span)) in predicates.into_iter().enumerate() {
            let cause = if tcx.is_lang_item(parent_trait_pred.def_id(), LangItem::CoerceUnsized) {
                cause.clone()
            } else {
                cause.clone().derived_cause(parent_trait_pred, |derived| {
                    ObligationCauseCode::ImplDerived(Box::new(ImplDerivedCause {
                        derived,
                        impl_or_alias_def_id: def_id,
                        impl_def_predicate_index: Some(index),
                        span,
                    }))
                })
            };
            let clause = normalize_with_depth_to(
                self,
                param_env,
                cause.clone(),
                recursion_depth,
                predicate,
                &mut obligations,
            );
            obligations.push(Obligation {
                cause,
                recursion_depth,
                param_env,
                predicate: clause.as_predicate(),
            });
        }

        // Register any outlives obligations from the trait here, cc #124336.
        if matches!(tcx.def_kind(def_id), DefKind::Impl { of_trait: true }) {
            for clause in tcx.impl_super_outlives(def_id).iter_instantiated(tcx, args) {
                let clause = normalize_with_depth_to(
                    self,
                    param_env,
                    cause.clone(),
                    recursion_depth,
                    clause,
                    &mut obligations,
                );
                obligations.push(Obligation {
                    cause: cause.clone(),
                    recursion_depth,
                    param_env,
                    predicate: clause.as_predicate(),
                });
            }
        }

        obligations
    }
}

fn rebind_coroutine_witness_types<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    args: ty::GenericArgsRef<'tcx>,
    bound_vars: &'tcx ty::List<ty::BoundVariableKind>,
) -> ty::Binder<'tcx, Vec<Ty<'tcx>>> {
    let bound_coroutine_types = tcx.coroutine_hidden_types(def_id).skip_binder();
    let shifted_coroutine_types =
        tcx.shift_bound_var_indices(bound_vars.len(), bound_coroutine_types.skip_binder());
    ty::Binder::bind_with_vars(
        ty::EarlyBinder::bind(shifted_coroutine_types.types.to_vec()).instantiate(tcx, args),
        tcx.mk_bound_variable_kinds_from_iter(
            bound_vars.iter().chain(bound_coroutine_types.bound_vars()),
        ),
    )
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
            self.depth >= reached_depth,
            "invoked `update_reached_depth` with something under this stack: \
             self.depth={} reached_depth={}",
            self.depth,
            reached_depth,
        );
        debug!(reached_depth, "update_reached_depth");
        let mut p = self;
        while reached_depth < p.depth {
            debug!(?p.fresh_trait_pred, "update_reached_depth: marking as cycle participant");
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
/// that would mean we don't cache that `Bar<T>: Send`. But this led
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
/// error. When we pop the node at `reached_depth` from the stack, we
/// can commit all the things that remain in the provisional cache.
struct ProvisionalEvaluationCache<'tcx> {
    /// next "depth first number" to issue -- just a counter
    dfn: Cell<usize>,

    /// Map from cache key to the provisionally evaluated thing.
    /// The cache entries contain the result but also the DFN in which they
    /// were added. The DFN is used to clear out values on failure.
    ///
    /// Imagine we have a stack like:
    ///
    /// - `A B C` and we add a cache for the result of C (DFN 2)
    /// - Then we have a stack `A B D` where `D` has DFN 3
    /// - We try to solve D by evaluating E: `A B D E` (DFN 4)
    /// - `E` generates various cache entries which have cyclic dependencies on `B`
    ///   - `A B D E F` and so forth
    ///   - the DFN of `F` for example would be 5
    /// - then we determine that `E` is in error -- we will then clear
    ///   all cache values whose DFN is >= 4 -- in this case, that
    ///   means the cached value for `F`.
    map: RefCell<FxIndexMap<ty::PolyTraitPredicate<'tcx>, ProvisionalEvaluation>>,

    /// The stack of terms that we assume to be well-formed because a `WF(term)` predicate
    /// is on the stack above (and because of wellformedness is coinductive).
    /// In an "ideal" world, this would share a stack with trait predicates in
    /// `TraitObligationStack`. However, trait predicates are *much* hotter than
    /// `WellFormed` predicates, and it's very likely that the additional matches
    /// will have a perf effect. The value here is the well-formed `GenericArg`
    /// and the depth of the trait predicate *above* that well-formed predicate.
    wf_args: RefCell<Vec<(ty::Term<'tcx>, usize)>>,
}

/// A cache value for the provisional cache: contains the depth-first
/// number (DFN) and result.
#[derive(Copy, Clone, Debug)]
struct ProvisionalEvaluation {
    from_dfn: usize,
    reached_depth: usize,
    result: EvaluationResult,
}

impl<'tcx> Default for ProvisionalEvaluationCache<'tcx> {
    fn default() -> Self {
        Self { dfn: Cell::new(0), map: Default::default(), wf_args: Default::default() }
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
    /// `reached_depth` (from the returned value).
    fn get_provisional(
        &self,
        fresh_trait_pred: ty::PolyTraitPredicate<'tcx>,
    ) -> Option<ProvisionalEvaluation> {
        debug!(
            ?fresh_trait_pred,
            "get_provisional = {:#?}",
            self.map.borrow().get(&fresh_trait_pred),
        );
        Some(*self.map.borrow().get(&fresh_trait_pred)?)
    }

    /// Insert a provisional result into the cache. The result came
    /// from the node with the given DFN. It accessed a minimum depth
    /// of `reached_depth` to compute. It evaluated `fresh_trait_pred`
    /// and resulted in `result`.
    fn insert_provisional(
        &self,
        from_dfn: usize,
        reached_depth: usize,
        fresh_trait_pred: ty::PolyTraitPredicate<'tcx>,
        result: EvaluationResult,
    ) {
        debug!(?from_dfn, ?fresh_trait_pred, ?result, "insert_provisional");

        let mut map = self.map.borrow_mut();

        // Subtle: when we complete working on the DFN `from_dfn`, anything
        // that remains in the provisional cache must be dependent on some older
        // stack entry than `from_dfn`. We have to update their depth with our transitive
        // depth in that case or else it would be referring to some popped note.
        //
        // Example:
        // A (reached depth 0)
        //   ...
        //      B // depth 1 -- reached depth = 0
        //          C // depth 2 -- reached depth = 1 (should be 0)
        //              B
        //          A // depth 0
        //   D (reached depth 1)
        //      C (cache -- reached depth = 2)
        for (_k, v) in &mut *map {
            if v.from_dfn >= from_dfn {
                v.reached_depth = reached_depth.min(v.reached_depth);
            }
        }

        map.insert(fresh_trait_pred, ProvisionalEvaluation { from_dfn, reached_depth, result });
    }

    /// Invoked when the node with dfn `dfn` does not get a successful
    /// result. This will clear out any provisional cache entries
    /// that were added since `dfn` was created. This is because the
    /// provisional entries are things which must assume that the
    /// things on the stack at the time of their creation succeeded --
    /// since the failing node is presently at the top of the stack,
    /// these provisional entries must either depend on it or some
    /// ancestor of it.
    fn on_failure(&self, dfn: usize) {
        debug!(?dfn, "on_failure");
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
    /// already).
    ///
    /// Note that we may still have provisional cache items remaining
    /// in the cache when this is done. For example, if there is a
    /// cycle:
    ///
    /// * A depends on...
    ///     * B depends on A
    ///     * C depends on...
    ///         * D depends on C
    ///     * ...
    ///
    /// Then as we complete the C node we will have a provisional cache
    /// with results for A, B, C, and D. This method would clear out
    /// the C and D results, but leave A and B provisional.
    ///
    /// This is determined based on the DFN: we remove any provisional
    /// results created since `dfn` started (e.g., in our example, dfn
    /// would be 2, representing the C node, and hence we would
    /// remove the result for D, which has DFN 3, but not the results for
    /// A and B, which have DFNs 0 and 1 respectively).
    ///
    /// Note that we *do not* attempt to cache these cycle participants
    /// in the evaluation cache. Doing so would require carefully computing
    /// the correct `DepNode` to store in the cache entry:
    /// cycle participants may implicitly depend on query results
    /// related to other participants in the cycle, due to our logic
    /// which examines the evaluation stack.
    ///
    /// We used to try to perform this caching,
    /// but it lead to multiple incremental compilation ICEs
    /// (see #92987 and #96319), and was very hard to understand.
    /// Fortunately, removing the caching didn't seem to
    /// have a performance impact in practice.
    fn on_completion(&self, dfn: usize) {
        debug!(?dfn, "on_completion");
        self.map.borrow_mut().retain(|fresh_trait_pred, eval| {
            if eval.from_dfn >= dfn {
                debug!(?fresh_trait_pred, ?eval, "on_completion");
                return false;
            }
            true
        });
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
        let o = self.head?;
        *self = o.previous;
        Some(o)
    }
}

impl<'o, 'tcx> fmt::Debug for TraitObligationStack<'o, 'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TraitObligationStack({:?})", self.obligation)
    }
}

pub(crate) enum ProjectionMatchesProjection {
    Yes,
    Ambiguous,
    No,
}
