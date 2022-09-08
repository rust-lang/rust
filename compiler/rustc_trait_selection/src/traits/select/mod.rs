//! Candidate selection. See the [rustc dev guide] for more information on how this works.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/traits/resolution.html#selection

use self::EvaluationResult::*;
use self::SelectionCandidate::*;

use super::coherence::{self, Conflict};
use super::const_evaluatable;
use super::project;
use super::project::normalize_with_depth_to;
use super::project::ProjectionTyObligation;
use super::util;
use super::util::{closure_trait_ref_and_return_type, predicate_for_trait_def};
use super::wf;
use super::{
    ErrorReporting, ImplDerivedObligation, ImplDerivedObligationCause, Normalized, Obligation,
    ObligationCause, ObligationCauseCode, Overflow, PredicateObligation, Selection, SelectionError,
    SelectionResult, TraitObligation, TraitQueryMode,
};

use crate::infer::{InferCtxt, InferOk, TypeFreshener};
use crate::traits::error_reporting::InferCtxtExt;
use crate::traits::project::ProjectAndUnifyResult;
use crate::traits::project::ProjectionCacheKeyExt;
use crate::traits::ProjectionCacheKey;
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexSet};
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_errors::{Diagnostic, ErrorGuaranteed};
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_infer::infer::LateBoundRegionConversionTime;
use rustc_middle::dep_graph::{DepKind, DepNodeIndex};
use rustc_middle::mir::interpret::ErrorHandled;
use rustc_middle::ty::abstract_const::NotConstEvaluatable;
use rustc_middle::ty::fast_reject::{DeepRejectCtxt, TreatParams};
use rustc_middle::ty::fold::BottomUpFolder;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::relate::TypeRelation;
use rustc_middle::ty::subst::{Subst, SubstsRef};
use rustc_middle::ty::{self, EarlyBinder, PolyProjectionPredicate, ToPolyTraitRef, ToPredicate};
use rustc_middle::ty::{Ty, TyCtxt, TypeFoldable, TypeVisitable};
use rustc_span::symbol::sym;

use std::cell::{Cell, RefCell};
use std::cmp;
use std::fmt::{self, Display};
use std::iter;

pub use rustc_middle::traits::select::*;

mod candidate_assembly;
mod confirmation;

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum IntercrateAmbiguityCause {
    DownstreamCrate { trait_desc: String, self_desc: Option<String> },
    UpstreamCrateUpdate { trait_desc: String, self_desc: Option<String> },
    ReservationImpl { message: String },
}

impl IntercrateAmbiguityCause {
    /// Emits notes when the overlap is caused by complex intercrate ambiguities.
    /// See #23980 for details.
    pub fn add_intercrate_ambiguity_hint(&self, err: &mut Diagnostic) {
        err.note(&self.intercrate_ambiguity_hint());
    }

    pub fn intercrate_ambiguity_hint(&self) -> String {
        match self {
            IntercrateAmbiguityCause::DownstreamCrate { trait_desc, self_desc } => {
                let self_desc = if let Some(ty) = self_desc {
                    format!(" for type `{}`", ty)
                } else {
                    String::new()
                };
                format!("downstream crates may implement trait `{}`{}", trait_desc, self_desc)
            }
            IntercrateAmbiguityCause::UpstreamCrateUpdate { trait_desc, self_desc } => {
                let self_desc = if let Some(ty) = self_desc {
                    format!(" for type `{}`", ty)
                } else {
                    String::new()
                };
                format!(
                    "upstream crates may add a new impl of trait `{}`{} \
                     in future versions",
                    trait_desc, self_desc
                )
            }
            IntercrateAmbiguityCause::ReservationImpl { message } => message.clone(),
        }
    }
}

pub struct SelectionContext<'cx, 'tcx> {
    infcx: &'cx InferCtxt<'cx, 'tcx>,

    /// Freshener used specifically for entries on the obligation
    /// stack. This ensures that all entries on the stack at one time
    /// will have the same set of placeholder entries, which is
    /// important for checking for trait bounds that recursively
    /// require themselves.
    freshener: TypeFreshener<'cx, 'tcx>,

    /// During coherence we have to assume that other crates may add
    /// additional impls which we currently don't know about.
    ///
    /// To deal with this evaluation should be conservative
    /// and consider the possibility of impls from outside this crate.
    /// This comes up primarily when resolving ambiguity. Imagine
    /// there is some trait reference `$0: Bar` where `$0` is an
    /// inference variable. If `intercrate` is true, then we can never
    /// say for sure that this reference is not implemented, even if
    /// there are *no impls at all for `Bar`*, because `$0` could be
    /// bound to some type that in a downstream crate that implements
    /// `Bar`.
    ///
    /// Outside of coherence we set this to false because we are only
    /// interested in types that the user could actually have written.
    /// In other words, we consider `$0: Bar` to be unimplemented if
    /// there is no type that the user could *actually name* that
    /// would satisfy it. This avoids crippling inference, basically.
    intercrate: bool,
    /// If `intercrate` is set, we remember predicates which were
    /// considered ambiguous because of impls potentially added in other crates.
    /// This is used in coherence to give improved diagnostics.
    /// We don't do his until we detect a coherence error because it can
    /// lead to false overflow results (#47139) and because always
    /// computing it may negatively impact performance.
    intercrate_ambiguity_causes: Option<FxIndexSet<IntercrateAmbiguityCause>>,

    /// The mode that trait queries run in, which informs our error handling
    /// policy. In essence, canonicalized queries need their errors propagated
    /// rather than immediately reported because we do not have accurate spans.
    query_mode: TraitQueryMode,
}

// A stack that walks back up the stack frame.
struct TraitObligationStack<'prev, 'tcx> {
    obligation: &'prev TraitObligation<'tcx>,

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
    pub fn new(infcx: &'cx InferCtxt<'cx, 'tcx>) -> SelectionContext<'cx, 'tcx> {
        SelectionContext {
            infcx,
            freshener: infcx.freshener_keep_static(),
            intercrate: false,
            intercrate_ambiguity_causes: None,
            query_mode: TraitQueryMode::Standard,
        }
    }

    pub fn intercrate(infcx: &'cx InferCtxt<'cx, 'tcx>) -> SelectionContext<'cx, 'tcx> {
        SelectionContext {
            infcx,
            freshener: infcx.freshener_keep_static(),
            intercrate: true,
            intercrate_ambiguity_causes: None,
            query_mode: TraitQueryMode::Standard,
        }
    }

    pub fn with_query_mode(
        infcx: &'cx InferCtxt<'cx, 'tcx>,
        query_mode: TraitQueryMode,
    ) -> SelectionContext<'cx, 'tcx> {
        debug!(?query_mode, "with_query_mode");
        SelectionContext {
            infcx,
            freshener: infcx.freshener_keep_static(),
            intercrate: false,
            intercrate_ambiguity_causes: None,
            query_mode,
        }
    }

    /// Enables tracking of intercrate ambiguity causes. See
    /// the documentation of [`Self::intercrate_ambiguity_causes`] for more.
    pub fn enable_tracking_intercrate_ambiguity_causes(&mut self) {
        assert!(self.intercrate);
        assert!(self.intercrate_ambiguity_causes.is_none());
        self.intercrate_ambiguity_causes = Some(FxIndexSet::default());
        debug!("selcx: enable_tracking_intercrate_ambiguity_causes");
    }

    /// Gets the intercrate ambiguity causes collected since tracking
    /// was enabled and disables tracking at the same time. If
    /// tracking is not enabled, just returns an empty vector.
    pub fn take_intercrate_ambiguity_causes(&mut self) -> FxIndexSet<IntercrateAmbiguityCause> {
        assert!(self.intercrate);
        self.intercrate_ambiguity_causes.take().unwrap_or_default()
    }

    pub fn infcx(&self) -> &'cx InferCtxt<'cx, 'tcx> {
        self.infcx
    }

    pub fn tcx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    pub fn is_intercrate(&self) -> bool {
        self.intercrate
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
    pub fn select(
        &mut self,
        obligation: &TraitObligation<'tcx>,
    ) -> SelectionResult<'tcx, Selection<'tcx>> {
        let candidate = match self.select_from_obligation(obligation) {
            Err(SelectionError::Overflow(OverflowError::Canonical)) => {
                // In standard mode, overflow must have been caught and reported
                // earlier.
                assert!(self.query_mode == TraitQueryMode::Canonical);
                return Err(SelectionError::Overflow(OverflowError::Canonical));
            }
            Err(SelectionError::Ambiguous(_)) => {
                return Ok(None);
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

    pub(crate) fn select_from_obligation(
        &mut self,
        obligation: &TraitObligation<'tcx>,
    ) -> SelectionResult<'tcx, SelectionCandidate<'tcx>> {
        debug_assert!(!obligation.predicate.has_escaping_bound_vars());

        let pec = &ProvisionalEvaluationCache::default();
        let stack = self.push_stack(TraitObligationStackList::empty(pec), obligation);

        self.candidate_from_obligation(&stack)
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
        debug!(?obligation, "predicate_may_hold_fatal");

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

            match self.infcx.leak_check(true, snapshot) {
                Ok(()) => {}
                Err(_) => return Ok(EvaluatedToErr),
            }

            if self.infcx.opaque_types_added_in_snapshot(snapshot) {
                return Ok(result.max(EvaluatedToOkModuloOpaqueTypes));
            }

            match self.infcx.region_constraints_added_in_snapshot(snapshot) {
                None => Ok(result),
                Some(_) => Ok(result.max(EvaluatedToOkModuloRegions)),
            }
        })
    }

    /// Evaluates the predicates in `predicates` recursively. Note that
    /// this applies projections in the predicates, and therefore
    /// is run within an inference probe.
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
        for obligation in predicates {
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
        // `previous_stack` stores a `TraitObligation`, while `obligation` is
        // a `PredicateObligation`. These are distinct types, so we can't
        // use any `Option` combinator method that would force them to be
        // the same.
        match previous_stack.head() {
            Some(h) => self.check_recursion_limit(&obligation, h.obligation)?,
            None => self.check_recursion_limit(&obligation, &obligation)?,
        }

        ensure_sufficient_stack(|| {
            let bound_predicate = obligation.predicate.kind();
            match bound_predicate.skip_binder() {
                ty::PredicateKind::Trait(t) => {
                    let t = bound_predicate.rebind(t);
                    debug_assert!(!t.has_escaping_bound_vars());
                    let obligation = obligation.with(t);
                    self.evaluate_trait_predicate_recursively(previous_stack, obligation)
                }

                ty::PredicateKind::Subtype(p) => {
                    let p = bound_predicate.rebind(p);
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

                ty::PredicateKind::Coerce(p) => {
                    let p = bound_predicate.rebind(p);
                    // Does this code ever run?
                    match self.infcx.coerce_predicate(&obligation.cause, obligation.param_env, p) {
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

                ty::PredicateKind::WellFormed(arg) => {
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

                    for stack_arg in previous_stack.cache.wf_args.borrow().iter().rev() {
                        if stack_arg.0 != arg {
                            continue;
                        }
                        debug!("WellFormed({:?}) on stack", arg);
                        if let Some(stack) = previous_stack.head {
                            // Okay, let's imagine we have two different stacks:
                            //   `T: NonAutoTrait -> WF(T) -> T: NonAutoTrait`
                            //   `WF(T) -> T: NonAutoTrait -> WF(T)`
                            // Because of this, we need to check that all
                            // predicates between the WF goals are coinductive.
                            // Otherwise, we can say that `T: NonAutoTrait` is
                            // true.
                            // Let's imagine we have a predicate stack like
                            //         `Foo: Bar -> WF(T) -> T: NonAutoTrait -> T: Auto
                            // depth   ^1                    ^2                 ^3
                            // and the current predicate is `WF(T)`. `wf_args`
                            // would contain `(T, 1)`. We want to check all
                            // trait predicates greater than `1`. The previous
                            // stack would be `T: Auto`.
                            let cycle = stack.iter().take_while(|s| s.depth > stack_arg.1);
                            let tcx = self.tcx();
                            let cycle =
                                cycle.map(|stack| stack.obligation.predicate.to_predicate(tcx));
                            if self.coinductive_match(cycle) {
                                stack.update_reached_depth(stack_arg.1);
                                return Ok(EvaluatedToOk);
                            } else {
                                return Ok(EvaluatedToRecur);
                            }
                        }
                        return Ok(EvaluatedToOk);
                    }

                    match wf::obligations(
                        self.infcx,
                        obligation.param_env,
                        obligation.cause.body_id,
                        obligation.recursion_depth + 1,
                        arg,
                        obligation.cause.span,
                    ) {
                        Some(mut obligations) => {
                            self.add_depth(obligations.iter_mut(), obligation.recursion_depth);

                            cache.wf_args.borrow_mut().push((arg, previous_stack.depth()));
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

                ty::PredicateKind::TypeOutlives(pred) => {
                    // A global type with no late-bound regions can only
                    // contain the "'static" lifetime (any other lifetime
                    // would either be late-bound or local), so it is guaranteed
                    // to outlive any other lifetime
                    if pred.0.is_global() && !pred.0.has_late_bound_regions() {
                        Ok(EvaluatedToOk)
                    } else {
                        Ok(EvaluatedToOkModuloRegions)
                    }
                }

                ty::PredicateKind::RegionOutlives(..) => {
                    // We do not consider region relationships when evaluating trait matches.
                    Ok(EvaluatedToOkModuloRegions)
                }

                ty::PredicateKind::ObjectSafe(trait_def_id) => {
                    if self.tcx().is_object_safe(trait_def_id) {
                        Ok(EvaluatedToOk)
                    } else {
                        Ok(EvaluatedToErr)
                    }
                }

                ty::PredicateKind::Projection(data) => {
                    let data = bound_predicate.rebind(data);
                    let project_obligation = obligation.with(data);
                    match project::poly_project_and_unify_type(self, &project_obligation) {
                        ProjectAndUnifyResult::Holds(mut subobligations) => {
                            'compute_res: {
                                // If we've previously marked this projection as 'complete', then
                                // use the final cached result (either `EvaluatedToOk` or
                                // `EvaluatedToOkModuloRegions`), and skip re-evaluating the
                                // sub-obligations.
                                if let Some(key) =
                                    ProjectionCacheKey::from_poly_projection_predicate(self, data)
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

                                self.add_depth(
                                    subobligations.iter_mut(),
                                    obligation.recursion_depth,
                                );
                                let res = self.evaluate_predicates_recursively(
                                    previous_stack,
                                    subobligations,
                                );
                                if let Ok(eval_rslt) = res
                                    && (eval_rslt == EvaluatedToOk || eval_rslt == EvaluatedToOkModuloRegions)
                                    && let Some(key) =
                                        ProjectionCacheKey::from_poly_projection_predicate(
                                            self, data,
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
                        ProjectAndUnifyResult::Recursive => Ok(EvaluatedToRecur),
                        ProjectAndUnifyResult::MismatchedProjectionTypes(_) => Ok(EvaluatedToErr),
                    }
                }

                ty::PredicateKind::ClosureKind(_, closure_substs, kind) => {
                    match self.infcx.closure_kind(closure_substs) {
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

                ty::PredicateKind::ConstEvaluatable(uv) => {
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
                    debug!(?c1, ?c2, "evaluate_predicate_recursively: equating consts");

                    if self.tcx().features().generic_const_exprs {
                        // FIXME: we probably should only try to unify abstract constants
                        // if the constants depend on generic parameters.
                        //
                        // Let's just see where this breaks :shrug:
                        if let (ty::ConstKind::Unevaluated(a), ty::ConstKind::Unevaluated(b)) =
                            (c1.kind(), c2.kind())
                        {
                            if self.infcx.try_unify_abstract_consts(
                                a.shrink(),
                                b.shrink(),
                                obligation.param_env,
                            ) {
                                return Ok(EvaluatedToOk);
                            }
                        }
                    }

                    let evaluate = |c: ty::Const<'tcx>| {
                        if let ty::ConstKind::Unevaluated(unevaluated) = c.kind() {
                            match self.infcx.try_const_eval_resolve(
                                obligation.param_env,
                                unevaluated,
                                c.ty(),
                                Some(obligation.cause.span),
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
                            match self
                                .infcx()
                                .at(&obligation.cause, obligation.param_env)
                                .eq(c1, c2)
                            {
                                Ok(_) => Ok(EvaluatedToOk),
                                Err(_) => Ok(EvaluatedToErr),
                            }
                        }
                        (Err(ErrorHandled::Reported(_)), _)
                        | (_, Err(ErrorHandled::Reported(_))) => Ok(EvaluatedToErr),
                        (Err(ErrorHandled::Linted), _) | (_, Err(ErrorHandled::Linted)) => {
                            span_bug!(
                                obligation.cause.span(),
                                "ConstEquate: const_eval_resolve returned an unexpected error"
                            )
                        }
                        (Err(ErrorHandled::TooGeneric), _) | (_, Err(ErrorHandled::TooGeneric)) => {
                            if c1.has_infer_types_or_consts() || c2.has_infer_types_or_consts() {
                                Ok(EvaluatedToAmbig)
                            } else {
                                // Two different constants using generic parameters ~> error.
                                Ok(EvaluatedToErr)
                            }
                        }
                    }
                }
                ty::PredicateKind::TypeWellFormedFromEnv(..) => {
                    bug!("TypeWellFormedFromEnv is only used for chalk")
                }
            }
        })
    }

    #[instrument(skip(self, previous_stack), level = "debug", ret)]
    fn evaluate_trait_predicate_recursively<'o>(
        &mut self,
        previous_stack: TraitObligationStackList<'o, 'tcx>,
        mut obligation: TraitObligation<'tcx>,
    ) -> Result<EvaluationResult, OverflowError> {
        if !self.intercrate
            && obligation.is_global()
            && obligation.param_env.caller_bounds().iter().all(|bound| bound.needs_subst())
        {
            // If a param env has no global bounds, global obligations do not
            // depend on its particular value in order to work, so we can clear
            // out the param env and get better caching.
            debug!("in global");
            obligation.param_env = obligation.param_env.without_caller_bounds();
        }

        let stack = self.push_stack(previous_stack, &obligation);
        let mut fresh_trait_pred = stack.fresh_trait_pred;
        let mut param_env = obligation.param_env;

        fresh_trait_pred = fresh_trait_pred.map_bound(|mut pred| {
            pred.remap_constness(&mut param_env);
            pred
        });

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

        let (result, dep_node) = self.in_task(|this| this.evaluate_stack(&stack));
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
            let cycle = cycle.map(|stack| stack.obligation.predicate.to_predicate(tcx));
            if self.coinductive_match(cycle) {
                debug!("evaluate_stack --> recursive, coinductive");
                Some(EvaluatedToOk)
            } else {
                debug!("evaluate_stack --> recursive, inductive");
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
            stack.fresh_trait_pred.skip_binder().trait_ref.substs.types().any(|ty| ty.is_fresh());

        if stack.obligation.polarity() != ty::ImplPolarity::Negative {
            // This check was an imperfect workaround for a bug in the old
            // intercrate mode; it should be removed when that goes away.
            if unbound_input_types && self.intercrate {
                debug!("evaluate_stack --> unbound argument, intercrate -->  ambiguous",);
                // Heuristics: show the diagnostics when there are no candidates in crate.
                if self.intercrate_ambiguity_causes.is_some() {
                    debug!("evaluate_stack: intercrate_ambiguity_causes is some");
                    if let Ok(candidate_set) = self.assemble_candidates(stack) {
                        if !candidate_set.ambiguous && candidate_set.vec.is_empty() {
                            let trait_ref = stack.obligation.predicate.skip_binder().trait_ref;
                            let self_ty = trait_ref.self_ty();
                            let cause = with_no_trimmed_paths!({
                                IntercrateAmbiguityCause::DownstreamCrate {
                                    trait_desc: trait_ref.print_only_trait_path().to_string(),
                                    self_desc: if self_ty.has_concrete_skeleton() {
                                        Some(self_ty.to_string())
                                    } else {
                                        None
                                    },
                                }
                            });

                            debug!(?cause, "evaluate_stack: pushing cause");
                            self.intercrate_ambiguity_causes.as_mut().unwrap().insert(cause);
                        }
                    }
                }
                return Ok(EvaluatedToAmbig);
            }
        }

        if unbound_input_types
            && stack.iter().skip(1).any(|prev| {
                stack.obligation.param_env == prev.obligation.param_env
                    && self.match_fresh_trait_refs(
                        stack.fresh_trait_pred,
                        prev.fresh_trait_pred,
                        prev.obligation.param_env,
                    )
            })
        {
            debug!("evaluate_stack --> unbound argument, recursive --> giving up",);
            return Ok(EvaluatedToUnknown);
        }

        match self.candidate_from_obligation(stack) {
            Ok(Some(c)) => self.evaluate_candidate(stack, &c),
            Err(SelectionError::Ambiguous(_)) => Ok(EvaluatedToAmbig),
            Ok(None) => Ok(EvaluatedToAmbig),
            Err(Overflow(OverflowError::Canonical)) => Err(OverflowError::Canonical),
            Err(ErrorReporting) => Err(OverflowError::ErrorReporting),
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
    pub(crate) fn coinductive_match<I>(&mut self, mut cycle: I) -> bool
    where
        I: Iterator<Item = ty::Predicate<'tcx>>,
    {
        cycle.all(|predicate| self.coinductive_predicate(predicate))
    }

    fn coinductive_predicate(&self, predicate: ty::Predicate<'tcx>) -> bool {
        let result = match predicate.kind().skip_binder() {
            ty::PredicateKind::Trait(ref data) => self.tcx().trait_is_auto(data.def_id()),
            ty::PredicateKind::WellFormed(_) => true,
            _ => false,
        };
        debug!(?predicate, ?result, "coinductive_predicate");
        result
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
            let candidate = (*candidate).clone();
            match this.confirm_candidate(stack.obligation, candidate) {
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
        // Neither the global nor local cache is aware of intercrate
        // mode, so don't do any caching. In particular, we might
        // re-use the same `InferCtxt` with both an intercrate
        // and non-intercrate `SelectionContext`
        if self.intercrate {
            return None;
        }

        let tcx = self.tcx();
        if self.can_use_global_caches(param_env) {
            if let Some(res) = tcx.evaluation_cache.get(&(param_env, trait_pred), tcx) {
                return Some(res);
            }
        }
        self.infcx.evaluation_cache.get(&(param_env, trait_pred), tcx)
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

        // Neither the global nor local cache is aware of intercrate
        // mode, so don't do any caching. In particular, we might
        // re-use the same `InferCtxt` with both an intercrate
        // and non-intercrate `SelectionContext`
        if self.intercrate {
            return;
        }

        if self.can_use_global_caches(param_env) {
            if !trait_pred.needs_infer() {
                debug!(?trait_pred, ?result, "insert_evaluation_cache global");
                // This may overwrite the cache with the same value
                // FIXME: Due to #50507 this overwrites the different values
                // This should be changed to use HashMapExt::insert_same
                // when that is fixed
                self.tcx().evaluation_cache.insert((param_env, trait_pred), dep_node, result);
                return;
            }
        }

        debug!(?trait_pred, ?result, "insert_evaluation_cache");
        self.infcx.evaluation_cache.insert((param_env, trait_pred), dep_node, result);
    }

    /// For various reasons, it's possible for a subobligation
    /// to have a *lower* recursion_depth than the obligation used to create it.
    /// Projection sub-obligations may be returned from the projection cache,
    /// which results in obligations with an 'old' `recursion_depth`.
    /// Additionally, methods like `InferCtxt.subtype_predicate` produce
    /// subobligations without taking in a 'parent' depth, causing the
    /// generated subobligations to have a `recursion_depth` of `0`.
    ///
    /// To ensure that obligation_depth never decreases, we force all subobligations
    /// to have at least the depth of the original obligation.
    fn add_depth<T: 'cx, I: Iterator<Item = &'cx mut Obligation<'tcx, T>>>(
        &self,
        it: I,
        min_depth: usize,
    ) {
        it.for_each(|o| o.recursion_depth = cmp::max(min_depth, o.recursion_depth) + 1);
    }

    fn check_recursion_depth<T: Display + TypeFoldable<'tcx>>(
        &self,
        depth: usize,
        error_obligation: &Obligation<'tcx, T>,
    ) -> Result<(), OverflowError> {
        if !self.infcx.tcx.recursion_limit().value_within_limit(depth) {
            match self.query_mode {
                TraitQueryMode::Standard => {
                    if self.infcx.is_tainted_by_errors() {
                        return Err(OverflowError::Error(
                            ErrorGuaranteed::unchecked_claim_error_was_emitted(),
                        ));
                    }
                    self.infcx.report_overflow_error(error_obligation, true);
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
    fn check_recursion_limit<T: Display + TypeFoldable<'tcx>, V: Display + TypeFoldable<'tcx>>(
        &self,
        obligation: &Obligation<'tcx, T>,
        error_obligation: &Obligation<'tcx, V>,
    ) -> Result<(), OverflowError> {
        self.check_recursion_depth(obligation.recursion_depth, error_obligation)
    }

    fn in_task<OP, R>(&mut self, op: OP) -> (R, DepNodeIndex)
    where
        OP: FnOnce(&mut Self) -> R,
    {
        let (result, dep_node) =
            self.tcx().dep_graph.with_anon_task(self.tcx(), DepKind::TraitSelect, || op(self));
        self.tcx().dep_graph.read_index(dep_node);
        (result, dep_node)
    }

    /// filter_impls filters constant trait obligations and candidates that have a positive impl
    /// for a negative goal and a negative impl for a positive goal
    #[instrument(level = "debug", skip(self))]
    fn filter_impls(
        &mut self,
        candidates: Vec<SelectionCandidate<'tcx>>,
        obligation: &TraitObligation<'tcx>,
    ) -> Vec<SelectionCandidate<'tcx>> {
        let tcx = self.tcx();
        let mut result = Vec::with_capacity(candidates.len());

        for candidate in candidates {
            // Respect const trait obligations
            if obligation.is_const() {
                match candidate {
                    // const impl
                    ImplCandidate(def_id) if tcx.constness(def_id) == hir::Constness::Const => {}
                    // const param
                    ParamCandidate(trait_pred) if trait_pred.is_const_if_const() => {}
                    // auto trait impl
                    AutoImplCandidate(..) => {}
                    // generator, this will raise error in other places
                    // or ignore error with const_async_blocks feature
                    GeneratorCandidate => {}
                    // FnDef where the function is const
                    FnPointerCandidate { is_const: true } => {}
                    ConstDestructCandidate(_) => {}
                    _ => {
                        // reject all other types of candidates
                        continue;
                    }
                }
            }

            if let ImplCandidate(def_id) = candidate {
                if ty::ImplPolarity::Reservation == tcx.impl_polarity(def_id)
                    || obligation.polarity() == tcx.impl_polarity(def_id)
                {
                    result.push(candidate);
                }
            } else {
                result.push(candidate);
            }
        }

        result
    }

    /// filter_reservation_impls filter reservation impl for any goal as ambiguous
    #[instrument(level = "debug", skip(self))]
    fn filter_reservation_impls(
        &mut self,
        candidate: SelectionCandidate<'tcx>,
        obligation: &TraitObligation<'tcx>,
    ) -> SelectionResult<'tcx, SelectionCandidate<'tcx>> {
        let tcx = self.tcx();
        // Treat reservation impls as ambiguity.
        if let ImplCandidate(def_id) = candidate {
            if let ty::ImplPolarity::Reservation = tcx.impl_polarity(def_id) {
                if let Some(intercrate_ambiguity_clauses) = &mut self.intercrate_ambiguity_causes {
                    let value = tcx
                        .get_attr(def_id, sym::rustc_reservation_impl)
                        .and_then(|a| a.value_str());
                    if let Some(value) = value {
                        debug!(
                            "filter_reservation_impls: \
                                 reservation impl ambiguity on {:?}",
                            def_id
                        );
                        intercrate_ambiguity_clauses.insert(
                            IntercrateAmbiguityCause::ReservationImpl {
                                message: value.to_string(),
                            },
                        );
                    }
                }
                return Ok(None);
            }
        }
        Ok(Some(candidate))
    }

    fn is_knowable<'o>(&mut self, stack: &TraitObligationStack<'o, 'tcx>) -> Result<(), Conflict> {
        debug!("is_knowable(intercrate={:?})", self.intercrate);

        if !self.intercrate || stack.obligation.polarity() == ty::ImplPolarity::Negative {
            return Ok(());
        }

        let obligation = &stack.obligation;
        let predicate = self.infcx().resolve_vars_if_possible(obligation.predicate);

        // Okay to skip binder because of the nature of the
        // trait-ref-is-knowable check, which does not care about
        // bound regions.
        let trait_ref = predicate.skip_binder().trait_ref;

        coherence::trait_ref_is_knowable(self.tcx(), trait_ref)
    }

    /// Returns `true` if the global caches can be used.
    fn can_use_global_caches(&self, param_env: ty::ParamEnv<'tcx>) -> bool {
        // If there are any inference variables in the `ParamEnv`, then we
        // always use a cache local to this particular scope. Otherwise, we
        // switch to a global cache.
        if param_env.needs_infer() {
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
        mut param_env: ty::ParamEnv<'tcx>,
        cache_fresh_trait_pred: ty::PolyTraitPredicate<'tcx>,
    ) -> Option<SelectionResult<'tcx, SelectionCandidate<'tcx>>> {
        // Neither the global nor local cache is aware of intercrate
        // mode, so don't do any caching. In particular, we might
        // re-use the same `InferCtxt` with both an intercrate
        // and non-intercrate `SelectionContext`
        if self.intercrate {
            return None;
        }
        let tcx = self.tcx();
        let mut pred = cache_fresh_trait_pred.skip_binder();
        pred.remap_constness(&mut param_env);

        if self.can_use_global_caches(param_env) {
            if let Some(res) = tcx.selection_cache.get(&(param_env, pred), tcx) {
                return Some(res);
            }
        }
        self.infcx.selection_cache.get(&(param_env, pred), tcx)
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
        // Neither the global nor local cache is aware of intercrate
        // mode, so don't do any caching. In particular, we might
        // re-use the same `InferCtxt` with both an intercrate
        // and non-intercrate `SelectionContext`
        if self.intercrate {
            return false;
        }
        match result {
            Ok(Some(SelectionCandidate::ParamCandidate(trait_ref))) => !trait_ref.needs_infer(),
            _ => true,
        }
    }

    #[instrument(skip(self, param_env, cache_fresh_trait_pred, dep_node), level = "debug")]
    fn insert_candidate_cache(
        &mut self,
        mut param_env: ty::ParamEnv<'tcx>,
        cache_fresh_trait_pred: ty::PolyTraitPredicate<'tcx>,
        dep_node: DepNodeIndex,
        candidate: SelectionResult<'tcx, SelectionCandidate<'tcx>>,
    ) {
        let tcx = self.tcx();
        let mut pred = cache_fresh_trait_pred.skip_binder();

        pred.remap_constness(&mut param_env);

        if !self.can_cache_candidate(&candidate) {
            debug!(?pred, ?candidate, "insert_candidate_cache - candidate is not cacheable");
            return;
        }

        if self.can_use_global_caches(param_env) {
            if let Err(Overflow(OverflowError::Canonical)) = candidate {
                // Don't cache overflow globally; we only produce this in certain modes.
            } else if !pred.needs_infer() {
                if !candidate.needs_infer() {
                    debug!(?pred, ?candidate, "insert_candidate_cache global");
                    // This may overwrite the cache with the same value.
                    tcx.selection_cache.insert((param_env, pred), dep_node, candidate);
                    return;
                }
            }
        }

        debug!(?pred, ?candidate, "insert_candidate_cache local");
        self.infcx.selection_cache.insert((param_env, pred), dep_node, candidate);
    }

    /// Matches a predicate against the bounds of its self type.
    ///
    /// Given an obligation like `<T as Foo>::Bar: Baz` where the self type is
    /// a projection, look at the bounds of `T::Bar`, see if we can find a
    /// `Baz` bound. We return indexes into the list returned by
    /// `tcx.item_bounds` for any applicable bounds.
    #[instrument(level = "debug", skip(self), ret)]
    fn match_projection_obligation_against_definition_bounds(
        &mut self,
        obligation: &TraitObligation<'tcx>,
    ) -> smallvec::SmallVec<[usize; 2]> {
        let poly_trait_predicate = self.infcx().resolve_vars_if_possible(obligation.predicate);
        let placeholder_trait_predicate =
            self.infcx().replace_bound_vars_with_placeholders(poly_trait_predicate);
        debug!(?placeholder_trait_predicate);

        let tcx = self.infcx.tcx;
        let (def_id, substs) = match *placeholder_trait_predicate.trait_ref.self_ty().kind() {
            ty::Projection(ref data) => (data.item_def_id, data.substs),
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
        let bounds = tcx.bound_item_bounds(def_id).subst(tcx, substs);

        // The bounds returned by `item_bounds` may contain duplicates after
        // normalization, so try to deduplicate when possible to avoid
        // unnecessary ambiguity.
        let mut distinct_normalized_bounds = FxHashSet::default();

        bounds
            .iter()
            .enumerate()
            .filter_map(|(idx, bound)| {
                let bound_predicate = bound.kind();
                if let ty::PredicateKind::Trait(pred) = bound_predicate.skip_binder() {
                    let bound = bound_predicate.rebind(pred.trait_ref);
                    if self.infcx.probe(|_| {
                        match self.match_normalize_trait_ref(
                            obligation,
                            bound,
                            placeholder_trait_predicate.trait_ref,
                        ) {
                            Ok(None) => true,
                            Ok(Some(normalized_trait))
                                if distinct_normalized_bounds.insert(normalized_trait) =>
                            {
                                true
                            }
                            _ => false,
                        }
                    }) {
                        return Some(idx);
                    }
                }
                None
            })
            .collect()
    }

    /// Equates the trait in `obligation` with trait bound. If the two traits
    /// can be equated and the normalized trait bound doesn't contain inference
    /// variables or placeholders, the normalized bound is returned.
    fn match_normalize_trait_ref(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        trait_bound: ty::PolyTraitRef<'tcx>,
        placeholder_trait_ref: ty::TraitRef<'tcx>,
    ) -> Result<Option<ty::PolyTraitRef<'tcx>>, ()> {
        debug_assert!(!placeholder_trait_ref.has_escaping_bound_vars());
        if placeholder_trait_ref.def_id != trait_bound.def_id() {
            // Avoid unnecessary normalization
            return Err(());
        }

        let Normalized { value: trait_bound, obligations: _ } = ensure_sufficient_stack(|| {
            project::normalize_with_depth(
                self,
                obligation.param_env,
                obligation.cause.clone(),
                obligation.recursion_depth + 1,
                trait_bound,
            )
        });
        self.infcx
            .at(&obligation.cause, obligation.param_env)
            .define_opaque_types(false)
            .sup(ty::Binder::dummy(placeholder_trait_ref), trait_bound)
            .map(|InferOk { obligations: _, value: () }| {
                // This method is called within a probe, so we can't have
                // inference variables and placeholders escape.
                if !trait_bound.needs_infer() && !trait_bound.has_placeholders() {
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
    /// and applying this env_predicate constrains any of the obligation's GAT substitutions.
    ///
    /// This behavior is a somewhat of a hack to prevent over-constraining inference variables
    /// in cases like #91762.
    pub(super) fn match_projection_projections(
        &mut self,
        obligation: &ProjectionTyObligation<'tcx>,
        env_predicate: PolyProjectionPredicate<'tcx>,
        potentially_unnormalized_candidates: bool,
    ) -> ProjectionMatchesProjection {
        let mut nested_obligations = Vec::new();
        let infer_predicate = self.infcx.replace_bound_vars_with_fresh_vars(
            obligation.cause.span,
            LateBoundRegionConversionTime::HigherRankedType,
            env_predicate,
        );
        let infer_projection = if potentially_unnormalized_candidates {
            ensure_sufficient_stack(|| {
                project::normalize_with_depth_to(
                    self,
                    obligation.param_env,
                    obligation.cause.clone(),
                    obligation.recursion_depth + 1,
                    infer_predicate.projection_ty,
                    &mut nested_obligations,
                )
            })
        } else {
            infer_predicate.projection_ty
        };

        let is_match = self
            .infcx
            .at(&obligation.cause, obligation.param_env)
            .define_opaque_types(false)
            .sup(obligation.predicate, infer_projection)
            .map_or(false, |InferOk { obligations, value: () }| {
                self.evaluate_predicates_recursively(
                    TraitObligationStackList::empty(&ProvisionalEvaluationCache::default()),
                    nested_obligations.into_iter().chain(obligations),
                )
                .map_or(false, |res| res.may_apply())
            });

        if is_match {
            let generics = self.tcx().generics_of(obligation.predicate.item_def_id);
            // FIXME(generic-associated-types): Addresses aggressive inference in #92917.
            // If this type is a GAT, and of the GAT substs resolve to something new,
            // that means that we must have newly inferred something about the GAT.
            // We should give up in that case.
            if !generics.params.is_empty()
                && obligation.predicate.substs[generics.parent_count..]
                    .iter()
                    .any(|&p| p.has_infer_types_or_consts() && self.infcx.shallow_resolve(p) != p)
            {
                ProjectionMatchesProjection::Ambiguous
            } else {
                ProjectionMatchesProjection::Yes
            }
        } else {
            ProjectionMatchesProjection::No
        }
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
        let is_global = |cand: &ty::PolyTraitPredicate<'tcx>| {
            cand.is_global() && !cand.has_late_bound_regions()
        };

        // (*) Prefer `BuiltinCandidate { has_nested: false }`, `PointeeCandidate`,
        // `DiscriminantKindCandidate`, and `ConstDestructCandidate` to anything else.
        //
        // This is a fix for #53123 and prevents winnowing from accidentally extending the
        // lifetime of a variable.
        match (&other.candidate, &victim.candidate) {
            (_, AutoImplCandidate(..)) | (AutoImplCandidate(..), _) => {
                bug!(
                    "default implementations shouldn't be recorded \
                    when there are other valid candidates"
                );
            }

            // FIXME(@jswrenn): this should probably be more sophisticated
            (TransmutabilityCandidate, _) | (_, TransmutabilityCandidate) => false,

            // (*)
            (
                BuiltinCandidate { has_nested: false }
                | DiscriminantKindCandidate
                | PointeeCandidate
                | ConstDestructCandidate(_),
                _,
            ) => true,
            (
                _,
                BuiltinCandidate { has_nested: false }
                | DiscriminantKindCandidate
                | PointeeCandidate
                | ConstDestructCandidate(_),
            ) => false,

            (ParamCandidate(other), ParamCandidate(victim)) => {
                let same_except_bound_vars = other.skip_binder().trait_ref
                    == victim.skip_binder().trait_ref
                    && other.skip_binder().constness == victim.skip_binder().constness
                    && other.skip_binder().polarity == victim.skip_binder().polarity
                    && !other.skip_binder().trait_ref.has_escaping_bound_vars();
                if same_except_bound_vars {
                    // See issue #84398. In short, we can generate multiple ParamCandidates which are
                    // the same except for unused bound vars. Just pick the one with the fewest bound vars
                    // or the current one if tied (they should both evaluate to the same answer). This is
                    // probably best characterized as a "hack", since we might prefer to just do our
                    // best to *not* create essentially duplicate candidates in the first place.
                    other.bound_vars().len() <= victim.bound_vars().len()
                } else if other.skip_binder().trait_ref == victim.skip_binder().trait_ref
                    && victim.skip_binder().constness == ty::BoundConstness::NotConst
                    && other.skip_binder().polarity == victim.skip_binder().polarity
                {
                    // Drop otherwise equivalent non-const candidates in favor of const candidates.
                    true
                } else {
                    false
                }
            }

            // Drop otherwise equivalent non-const fn pointer candidates
            (FnPointerCandidate { .. }, FnPointerCandidate { is_const: false }) => true,

            // Global bounds from the where clause should be ignored
            // here (see issue #50825). Otherwise, we have a where
            // clause so don't go around looking for impls.
            // Arbitrarily give param candidates priority
            // over projection and object candidates.
            (
                ParamCandidate(ref cand),
                ImplCandidate(..)
                | ClosureCandidate
                | GeneratorCandidate
                | FnPointerCandidate { .. }
                | BuiltinObjectCandidate
                | BuiltinUnsizeCandidate
                | TraitUpcastingUnsizeCandidate(_)
                | BuiltinCandidate { .. }
                | TraitAliasCandidate(..)
                | ObjectCandidate(_)
                | ProjectionCandidate(_),
            ) => !is_global(cand),
            (ObjectCandidate(_) | ProjectionCandidate(_), ParamCandidate(ref cand)) => {
                // Prefer these to a global where-clause bound
                // (see issue #50825).
                is_global(cand)
            }
            (
                ImplCandidate(_)
                | ClosureCandidate
                | GeneratorCandidate
                | FnPointerCandidate { .. }
                | BuiltinObjectCandidate
                | BuiltinUnsizeCandidate
                | TraitUpcastingUnsizeCandidate(_)
                | BuiltinCandidate { has_nested: true }
                | TraitAliasCandidate(..),
                ParamCandidate(ref cand),
            ) => {
                // Prefer these to a global where-clause bound
                // (see issue #50825).
                is_global(cand) && other.evaluation.must_apply_modulo_regions()
            }

            (ProjectionCandidate(i), ProjectionCandidate(j))
            | (ObjectCandidate(i), ObjectCandidate(j)) => {
                // Arbitrarily pick the lower numbered candidate for backwards
                // compatibility reasons. Don't let this affect inference.
                i < j && !needs_infer
            }
            (ObjectCandidate(_), ProjectionCandidate(_))
            | (ProjectionCandidate(_), ObjectCandidate(_)) => {
                bug!("Have both object and projection candidate")
            }

            // Arbitrarily give projection and object candidates priority.
            (
                ObjectCandidate(_) | ProjectionCandidate(_),
                ImplCandidate(..)
                | ClosureCandidate
                | GeneratorCandidate
                | FnPointerCandidate { .. }
                | BuiltinObjectCandidate
                | BuiltinUnsizeCandidate
                | TraitUpcastingUnsizeCandidate(_)
                | BuiltinCandidate { .. }
                | TraitAliasCandidate(..),
            ) => true,

            (
                ImplCandidate(..)
                | ClosureCandidate
                | GeneratorCandidate
                | FnPointerCandidate { .. }
                | BuiltinObjectCandidate
                | BuiltinUnsizeCandidate
                | TraitUpcastingUnsizeCandidate(_)
                | BuiltinCandidate { .. }
                | TraitAliasCandidate(..),
                ObjectCandidate(_) | ProjectionCandidate(_),
            ) => false,

            (&ImplCandidate(other_def), &ImplCandidate(victim_def)) => {
                // See if we can toss out `victim` based on specialization.
                // This requires us to know *for sure* that the `other` impl applies
                // i.e., `EvaluatedToOk`.
                //
                // FIXME(@lcnr): Using `modulo_regions` here seems kind of scary
                // to me but is required for `std` to compile, so I didn't change it
                // for now.
                let tcx = self.tcx();
                if other.evaluation.must_apply_modulo_regions() {
                    if tcx.specializes((other_def, victim_def)) {
                        return true;
                    }
                }

                if other.evaluation.must_apply_considering_regions() {
                    match tcx.impls_are_allowed_to_overlap(other_def, victim_def) {
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
                            !needs_infer
                        }
                        Some(_) => true,
                        None => false,
                    }
                } else {
                    false
                }
            }

            // Everything else is ambiguous
            (
                ImplCandidate(_)
                | ClosureCandidate
                | GeneratorCandidate
                | FnPointerCandidate { .. }
                | BuiltinObjectCandidate
                | BuiltinUnsizeCandidate
                | TraitUpcastingUnsizeCandidate(_)
                | BuiltinCandidate { has_nested: true }
                | TraitAliasCandidate(..),
                ImplCandidate(_)
                | ClosureCandidate
                | GeneratorCandidate
                | FnPointerCandidate { .. }
                | BuiltinObjectCandidate
                | BuiltinUnsizeCandidate
                | TraitUpcastingUnsizeCandidate(_)
                | BuiltinCandidate { has_nested: true }
                | TraitAliasCandidate(..),
            ) => false,
        }
    }

    fn sized_conditions(
        &mut self,
        obligation: &TraitObligation<'tcx>,
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
            | ty::FnPtr(_)
            | ty::RawPtr(..)
            | ty::Char
            | ty::Ref(..)
            | ty::Generator(..)
            | ty::GeneratorWitness(..)
            | ty::Array(..)
            | ty::Closure(..)
            | ty::Never
            | ty::Error(_) => {
                // safe for everything
                Where(ty::Binder::dummy(Vec::new()))
            }

            ty::Str | ty::Slice(_) | ty::Dynamic(..) | ty::Foreign(..) => None,

            ty::Tuple(tys) => Where(
                obligation.predicate.rebind(tys.last().map_or_else(Vec::new, |&last| vec![last])),
            ),

            ty::Adt(def, substs) => {
                let sized_crit = def.sized_constraint(self.tcx());
                // (*) binder moved here
                Where(obligation.predicate.rebind({
                    sized_crit
                        .0
                        .iter()
                        .map(|ty| sized_crit.rebind(*ty).subst(self.tcx(), substs))
                        .collect()
                }))
            }

            ty::Projection(_) | ty::Param(_) | ty::Opaque(..) => None,
            ty::Infer(ty::TyVar(_)) => Ambiguous,

            ty::Placeholder(..)
            | ty::Bound(..)
            | ty::Infer(ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) => {
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

        match *self_ty.kind() {
            ty::Infer(ty::IntVar(_))
            | ty::Infer(ty::FloatVar(_))
            | ty::FnDef(..)
            | ty::FnPtr(_)
            | ty::Error(_) => Where(ty::Binder::dummy(Vec::new())),

            ty::Uint(_)
            | ty::Int(_)
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

            ty::Dynamic(..)
            | ty::Str
            | ty::Slice(..)
            | ty::Generator(..)
            | ty::GeneratorWitness(..)
            | ty::Foreign(..)
            | ty::Ref(_, _, hir::Mutability::Mut) => None,

            ty::Tuple(tys) => {
                // (*) binder moved here
                Where(obligation.predicate.rebind(tys.iter().collect()))
            }

            ty::Closure(_, substs) => {
                // (*) binder moved here
                let ty = self.infcx.shallow_resolve(substs.as_closure().tupled_upvars_ty());
                if let ty::Infer(ty::TyVar(_)) = ty.kind() {
                    // Not yet resolved.
                    Ambiguous
                } else {
                    Where(obligation.predicate.rebind(substs.as_closure().upvar_tys().collect()))
                }
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

            ty::Placeholder(..)
            | ty::Bound(..)
            | ty::Infer(ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) => {
                bug!("asked to assemble builtin bounds of unexpected type: {:?}", self_ty);
            }
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
    fn constituent_types_for_ty(
        &self,
        t: ty::Binder<'tcx, Ty<'tcx>>,
    ) -> ty::Binder<'tcx, Vec<Ty<'tcx>>> {
        match *t.skip_binder().kind() {
            ty::Uint(_)
            | ty::Int(_)
            | ty::Bool
            | ty::Float(_)
            | ty::FnDef(..)
            | ty::FnPtr(_)
            | ty::Str
            | ty::Error(_)
            | ty::Infer(ty::IntVar(_) | ty::FloatVar(_))
            | ty::Never
            | ty::Char => ty::Binder::dummy(Vec::new()),

            ty::Placeholder(..)
            | ty::Dynamic(..)
            | ty::Param(..)
            | ty::Foreign(..)
            | ty::Projection(..)
            | ty::Bound(..)
            | ty::Infer(ty::TyVar(_) | ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) => {
                bug!("asked to assemble constituent types of unexpected type: {:?}", t);
            }

            ty::RawPtr(ty::TypeAndMut { ty: element_ty, .. }) | ty::Ref(_, element_ty, _) => {
                t.rebind(vec![element_ty])
            }

            ty::Array(element_ty, _) | ty::Slice(element_ty) => t.rebind(vec![element_ty]),

            ty::Tuple(ref tys) => {
                // (T1, ..., Tn) -- meets any bound that all of T1...Tn meet
                t.rebind(tys.iter().collect())
            }

            ty::Closure(_, ref substs) => {
                let ty = self.infcx.shallow_resolve(substs.as_closure().tupled_upvars_ty());
                t.rebind(vec![ty])
            }

            ty::Generator(_, ref substs, _) => {
                let ty = self.infcx.shallow_resolve(substs.as_generator().tupled_upvars_ty());
                let witness = substs.as_generator().witness();
                t.rebind([ty].into_iter().chain(iter::once(witness)).collect())
            }

            ty::GeneratorWitness(types) => {
                debug_assert!(!types.has_escaping_bound_vars());
                types.map_bound(|types| types.to_vec())
            }

            // For `PhantomData<T>`, we pass `T`.
            ty::Adt(def, substs) if def.is_phantom_data() => t.rebind(substs.types().collect()),

            ty::Adt(def, substs) => {
                t.rebind(def.all_fields().map(|f| f.ty(self.tcx(), substs)).collect())
            }

            ty::Opaque(def_id, substs) => {
                // We can resolve the `impl Trait` to its concrete type,
                // which enforces a DAG between the functions requiring
                // the auto trait bounds in question.
                t.rebind(vec![self.tcx().bound_type_of(def_id).subst(self.tcx(), substs)])
            }
        }
    }

    fn collect_predicates_for_types(
        &mut self,
        param_env: ty::ParamEnv<'tcx>,
        cause: ObligationCause<'tcx>,
        recursion_depth: usize,
        trait_def_id: DefId,
        types: ty::Binder<'tcx, Vec<Ty<'tcx>>>,
    ) -> Vec<PredicateObligation<'tcx>> {
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
            .as_ref()
            .skip_binder() // binder moved -\
            .iter()
            .flat_map(|ty| {
                let ty: ty::Binder<'tcx, Ty<'tcx>> = types.rebind(*ty); // <----/

                let placeholder_ty = self.infcx.replace_bound_vars_with_placeholders(ty);
                let Normalized { value: normalized_ty, mut obligations } =
                    ensure_sufficient_stack(|| {
                        project::normalize_with_depth(
                            self,
                            param_env,
                            cause.clone(),
                            recursion_depth,
                            placeholder_ty,
                        )
                    });
                let placeholder_obligation = predicate_for_trait_def(
                    self.tcx(),
                    param_env,
                    cause.clone(),
                    trait_def_id,
                    recursion_depth,
                    normalized_ty,
                    &[],
                );
                obligations.push(placeholder_obligation);
                obligations
            })
            .collect()
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
    ) -> Normalized<'tcx, SubstsRef<'tcx>> {
        let impl_trait_ref = self.tcx().bound_impl_trait_ref(impl_def_id).unwrap();
        match self.match_impl(impl_def_id, impl_trait_ref, obligation) {
            Ok(substs) => substs,
            Err(()) => {
                self.infcx.tcx.sess.delay_span_bug(
                    obligation.cause.span,
                    &format!(
                        "Impl {:?} was matchable against {:?} but now is not",
                        impl_def_id, obligation
                    ),
                );
                let value = self.infcx.fresh_substs_for_item(obligation.cause.span, impl_def_id);
                let err = self.tcx().ty_error();
                let value = value.fold_with(&mut BottomUpFolder {
                    tcx: self.tcx(),
                    ty_op: |_| err,
                    lt_op: |l| l,
                    ct_op: |c| c,
                });
                Normalized { value, obligations: vec![] }
            }
        }
    }

    #[instrument(level = "debug", skip(self), ret)]
    fn match_impl(
        &mut self,
        impl_def_id: DefId,
        impl_trait_ref: EarlyBinder<ty::TraitRef<'tcx>>,
        obligation: &TraitObligation<'tcx>,
    ) -> Result<Normalized<'tcx, SubstsRef<'tcx>>, ()> {
        let placeholder_obligation =
            self.infcx().replace_bound_vars_with_placeholders(obligation.predicate);
        let placeholder_obligation_trait_ref = placeholder_obligation.trait_ref;

        let impl_substs = self.infcx.fresh_substs_for_item(obligation.cause.span, impl_def_id);

        let impl_trait_ref = impl_trait_ref.subst(self.tcx(), impl_substs);

        debug!(?impl_trait_ref);

        let Normalized { value: impl_trait_ref, obligations: mut nested_obligations } =
            ensure_sufficient_stack(|| {
                project::normalize_with_depth(
                    self,
                    obligation.param_env,
                    obligation.cause.clone(),
                    obligation.recursion_depth + 1,
                    impl_trait_ref,
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
            .define_opaque_types(false)
            .eq(placeholder_obligation_trait_ref, impl_trait_ref)
            .map_err(|e| debug!("match_impl: failed eq_trait_refs due to `{e}`"))?;
        nested_obligations.extend(obligations);

        if !self.intercrate
            && self.tcx().impl_polarity(impl_def_id) == ty::ImplPolarity::Reservation
        {
            debug!("reservation impls only apply in intercrate mode");
            return Err(());
        }

        Ok(Normalized { value: impl_substs, obligations: nested_obligations })
    }

    fn fast_reject_trait_refs(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        impl_trait_ref: &ty::TraitRef<'tcx>,
    ) -> bool {
        // We can avoid creating type variables and doing the full
        // substitution if we find that any of the input types, when
        // simplified, do not match.
        let drcx = DeepRejectCtxt { treat_obligation_params: TreatParams::AsPlaceholder };
        iter::zip(obligation.predicate.skip_binder().trait_ref.substs, impl_trait_ref.substs)
            .any(|(obl, imp)| !drcx.generic_args_may_unify(obl, imp))
    }

    /// Normalize `where_clause_trait_ref` and try to match it against
    /// `obligation`. If successful, return any predicates that
    /// result from the normalization.
    fn match_where_clause_trait_ref(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        where_clause_trait_ref: ty::PolyTraitRef<'tcx>,
    ) -> Result<Vec<PredicateObligation<'tcx>>, ()> {
        self.match_poly_trait_ref(obligation, where_clause_trait_ref)
    }

    /// Returns `Ok` if `poly_trait_ref` being true implies that the
    /// obligation is satisfied.
    #[instrument(skip(self), level = "debug")]
    fn match_poly_trait_ref(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        poly_trait_ref: ty::PolyTraitRef<'tcx>,
    ) -> Result<Vec<PredicateObligation<'tcx>>, ()> {
        self.infcx
            .at(&obligation.cause, obligation.param_env)
            // We don't want predicates for opaque types to just match all other types,
            // if there is an obligation on the opaque type, then that obligation must be met
            // opaquely. Otherwise we'd match any obligation to the opaque type and then error
            // out later.
            .define_opaque_types(false)
            .sup(obligation.predicate.to_poly_trait_ref(), poly_trait_ref)
            .map(|InferOk { obligations, .. }| obligations)
            .map_err(|_| ())
    }

    ///////////////////////////////////////////////////////////////////////////
    // Miscellany

    fn match_fresh_trait_refs(
        &self,
        previous: ty::PolyTraitPredicate<'tcx>,
        current: ty::PolyTraitPredicate<'tcx>,
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
        obligation: &TraitObligation<'tcx>,
        substs: SubstsRef<'tcx>,
    ) -> ty::PolyTraitRef<'tcx> {
        let closure_sig = substs.as_closure().sig();

        debug!(?closure_sig);

        // (1) Feels icky to skip the binder here, but OTOH we know
        // that the self-type is an unboxed closure type and hence is
        // in fact unparameterized (or at least does not reference any
        // regions bound in the obligation). Still probably some
        // refactoring could make this nicer.
        closure_trait_ref_and_return_type(
            self.tcx(),
            obligation.predicate.def_id(),
            obligation.predicate.skip_binder().self_ty(), // (1)
            closure_sig,
            util::TupleArgumentsFlag::No,
        )
        .map_bound(|(trait_ref, _)| trait_ref)
    }

    fn generator_trait_ref_unnormalized(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        substs: SubstsRef<'tcx>,
    ) -> ty::PolyTraitRef<'tcx> {
        let gen_sig = substs.as_generator().poly_sig();

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
    #[instrument(level = "debug", skip(self, cause, param_env))]
    fn impl_or_trait_obligations(
        &mut self,
        cause: &ObligationCause<'tcx>,
        recursion_depth: usize,
        param_env: ty::ParamEnv<'tcx>,
        def_id: DefId,           // of impl or trait
        substs: SubstsRef<'tcx>, // for impl or trait
        parent_trait_pred: ty::Binder<'tcx, ty::TraitPredicate<'tcx>>,
    ) -> Vec<PredicateObligation<'tcx>> {
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
        let predicates = tcx.bound_predicates_of(def_id);
        debug!(?predicates);
        assert_eq!(predicates.0.parent, None);
        let mut obligations = Vec::with_capacity(predicates.0.predicates.len());
        for (predicate, span) in predicates.0.predicates {
            let span = *span;
            let cause = cause.clone().derived_cause(parent_trait_pred, |derived| {
                ImplDerivedObligation(Box::new(ImplDerivedObligationCause {
                    derived,
                    impl_def_id: def_id,
                    span,
                }))
            });
            let predicate = normalize_with_depth_to(
                self,
                param_env,
                cause.clone(),
                recursion_depth,
                predicates.rebind(*predicate).subst(tcx, substs),
                &mut obligations,
            );
            obligations.push(Obligation { cause, recursion_depth, param_env, predicate });
        }

        obligations
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
    map: RefCell<FxHashMap<ty::PolyTraitPredicate<'tcx>, ProvisionalEvaluation>>,

    /// The stack of args that we assume to be true because a `WF(arg)` predicate
    /// is on the stack above (and because of wellformedness is coinductive).
    /// In an "ideal" world, this would share a stack with trait predicates in
    /// `TraitObligationStack`. However, trait predicates are *much* hotter than
    /// `WellFormed` predicates, and it's very likely that the additional matches
    /// will have a perf effect. The value here is the well-formed `GenericArg`
    /// and the depth of the trait predicate *above* that well-formed predicate.
    wf_args: RefCell<Vec<(ty::GenericArg<'tcx>, usize)>>,
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
    /// result.  This will clear out any provisional cache entries
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

        for (fresh_trait_pred, eval) in
            self.map.borrow_mut().drain_filter(|_k, eval| eval.from_dfn >= dfn)
        {
            debug!(?fresh_trait_pred, ?eval, "on_completion");
        }
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

pub enum ProjectionMatchesProjection {
    Yes,
    Ambiguous,
    No,
}
