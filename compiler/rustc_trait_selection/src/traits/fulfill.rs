use crate::infer::{InferCtxt, TyOrConstInferVar};
use rustc_data_structures::obligation_forest::ProcessResult;
use rustc_data_structures::obligation_forest::{Error, ForestObligation, Outcome};
use rustc_data_structures::obligation_forest::{ObligationForest, ObligationProcessor};
use rustc_infer::traits::ProjectionCacheKey;
use rustc_infer::traits::{SelectionError, TraitEngine, TraitObligation};
use rustc_middle::mir::interpret::ErrorHandled;
use rustc_middle::ty::abstract_const::NotConstEvaluatable;
use rustc_middle::ty::error::{ExpectedFound, TypeError};
use rustc_middle::ty::subst::SubstsRef;
use rustc_middle::ty::{self, Binder, Const, TypeVisitableExt};
use std::marker::PhantomData;

use super::const_evaluatable;
use super::project::{self, ProjectAndUnifyResult};
use super::select::SelectionContext;
use super::wf;
use super::CodeAmbiguity;
use super::CodeProjectionError;
use super::CodeSelectionError;
use super::EvaluationResult;
use super::PredicateObligation;
use super::Unimplemented;
use super::{FulfillmentError, FulfillmentErrorCode};

use crate::traits::project::PolyProjectionObligation;
use crate::traits::project::ProjectionCacheKeyExt as _;
use crate::traits::query::evaluate_obligation::InferCtxtExt;

impl<'tcx> ForestObligation for PendingPredicateObligation<'tcx> {
    /// Note that we include both the `ParamEnv` and the `Predicate`,
    /// as the `ParamEnv` can influence whether fulfillment succeeds
    /// or fails.
    type CacheKey = ty::ParamEnvAnd<'tcx, ty::Predicate<'tcx>>;

    fn as_cache_key(&self) -> Self::CacheKey {
        self.obligation.param_env.and(self.obligation.predicate)
    }
}

/// The fulfillment context is used to drive trait resolution. It
/// consists of a list of obligations that must be (eventually)
/// satisfied. The job is to track which are satisfied, which yielded
/// errors, and which are still pending. At any point, users can call
/// `select_where_possible`, and the fulfillment context will try to do
/// selection, retaining only those obligations that remain
/// ambiguous. This may be helpful in pushing type inference
/// along. Once all type inference constraints have been generated, the
/// method `select_all_or_error` can be used to report any remaining
/// ambiguous cases as errors.
pub struct FulfillmentContext<'tcx> {
    // A list of all obligations that have been registered with this
    // fulfillment context.
    predicates: ObligationForest<PendingPredicateObligation<'tcx>>,

    // Is it OK to register obligations into this infcx inside
    // an infcx snapshot?
    //
    // The "primary fulfillment" in many cases in typeck lives
    // outside of any snapshot, so any use of it inside a snapshot
    // will lead to trouble and therefore is checked against, but
    // other fulfillment contexts sometimes do live inside of
    // a snapshot (they don't *straddle* a snapshot, so there
    // is no trouble there).
    usable_in_snapshot: bool,
}

#[derive(Clone, Debug)]
pub struct PendingPredicateObligation<'tcx> {
    pub obligation: PredicateObligation<'tcx>,
    // This is far more often read than modified, meaning that we
    // should mostly optimize for reading speed, while modifying is not as relevant.
    //
    // For whatever reason using a boxed slice is slower than using a `Vec` here.
    pub stalled_on: Vec<TyOrConstInferVar<'tcx>>,
}

// `PendingPredicateObligation` is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
static_assert_size!(PendingPredicateObligation<'_>, 72);

impl<'a, 'tcx> FulfillmentContext<'tcx> {
    /// Creates a new fulfillment context.
    pub(super) fn new() -> FulfillmentContext<'tcx> {
        FulfillmentContext { predicates: ObligationForest::new(), usable_in_snapshot: false }
    }

    pub(super) fn new_in_snapshot() -> FulfillmentContext<'tcx> {
        FulfillmentContext { predicates: ObligationForest::new(), usable_in_snapshot: true }
    }

    /// Attempts to select obligations using `selcx`.
    fn select(&mut self, selcx: SelectionContext<'a, 'tcx>) -> Vec<FulfillmentError<'tcx>> {
        let span = debug_span!("select", obligation_forest_size = ?self.predicates.len());
        let _enter = span.enter();

        // Process pending obligations.
        let outcome: Outcome<_, _> =
            self.predicates.process_obligations(&mut FulfillProcessor { selcx });

        // FIXME: if we kept the original cache key, we could mark projection
        // obligations as complete for the projection cache here.

        let errors: Vec<FulfillmentError<'tcx>> =
            outcome.errors.into_iter().map(to_fulfillment_error).collect();

        debug!(
            "select({} predicates remaining, {} errors) done",
            self.predicates.len(),
            errors.len()
        );

        errors
    }
}

impl<'tcx> TraitEngine<'tcx> for FulfillmentContext<'tcx> {
    fn register_predicate_obligation(
        &mut self,
        infcx: &InferCtxt<'tcx>,
        obligation: PredicateObligation<'tcx>,
    ) {
        // this helps to reduce duplicate errors, as well as making
        // debug output much nicer to read and so on.
        let obligation = infcx.resolve_vars_if_possible(obligation);

        debug!(?obligation, "register_predicate_obligation");

        assert!(!infcx.is_in_snapshot() || self.usable_in_snapshot);

        self.predicates
            .register_obligation(PendingPredicateObligation { obligation, stalled_on: vec![] });
    }

    fn collect_remaining_errors(&mut self) -> Vec<FulfillmentError<'tcx>> {
        self.predicates.to_errors(CodeAmbiguity).into_iter().map(to_fulfillment_error).collect()
    }

    fn select_where_possible(&mut self, infcx: &InferCtxt<'tcx>) -> Vec<FulfillmentError<'tcx>> {
        let selcx = SelectionContext::new(infcx);
        self.select(selcx)
    }

    fn drain_unstalled_obligations(
        &mut self,
        infcx: &InferCtxt<'tcx>,
    ) -> Vec<PredicateObligation<'tcx>> {
        let mut processor = DrainProcessor { removed_predicates: Vec::new(), infcx };
        let outcome: Outcome<_, _> = self.predicates.process_obligations(&mut processor);
        assert!(outcome.errors.is_empty());
        return processor.removed_predicates;

        struct DrainProcessor<'a, 'tcx> {
            infcx: &'a InferCtxt<'tcx>,
            removed_predicates: Vec<PredicateObligation<'tcx>>,
        }

        impl<'tcx> ObligationProcessor for DrainProcessor<'_, 'tcx> {
            type Obligation = PendingPredicateObligation<'tcx>;
            type Error = !;
            type OUT = Outcome<Self::Obligation, Self::Error>;

            fn needs_process_obligation(&self, pending_obligation: &Self::Obligation) -> bool {
                pending_obligation
                    .stalled_on
                    .iter()
                    .any(|&var| self.infcx.ty_or_const_infer_var_changed(var))
            }

            fn process_obligation(
                &mut self,
                pending_obligation: &mut PendingPredicateObligation<'tcx>,
            ) -> ProcessResult<PendingPredicateObligation<'tcx>, !> {
                assert!(self.needs_process_obligation(pending_obligation));
                self.removed_predicates.push(pending_obligation.obligation.clone());
                ProcessResult::Changed(vec![])
            }

            fn process_backedge<'c, I>(
                &mut self,
                cycle: I,
                _marker: PhantomData<&'c PendingPredicateObligation<'tcx>>,
            ) -> Result<(), !>
            where
                I: Clone + Iterator<Item = &'c PendingPredicateObligation<'tcx>>,
            {
                self.removed_predicates.extend(cycle.map(|c| c.obligation.clone()));
                Ok(())
            }
        }
    }

    fn pending_obligations(&self) -> Vec<PredicateObligation<'tcx>> {
        self.predicates.map_pending_obligations(|o| o.obligation.clone())
    }
}

struct FulfillProcessor<'a, 'tcx> {
    selcx: SelectionContext<'a, 'tcx>,
}

fn mk_pending(os: Vec<PredicateObligation<'_>>) -> Vec<PendingPredicateObligation<'_>> {
    os.into_iter()
        .map(|o| PendingPredicateObligation { obligation: o, stalled_on: vec![] })
        .collect()
}

impl<'a, 'tcx> ObligationProcessor for FulfillProcessor<'a, 'tcx> {
    type Obligation = PendingPredicateObligation<'tcx>;
    type Error = FulfillmentErrorCode<'tcx>;
    type OUT = Outcome<Self::Obligation, Self::Error>;

    /// Identifies whether a predicate obligation needs processing.
    ///
    /// This is always inlined, despite its size, because it has a single
    /// callsite and it is called *very* frequently.
    #[inline(always)]
    fn needs_process_obligation(&self, pending_obligation: &Self::Obligation) -> bool {
        // If we were stalled on some unresolved variables, first check whether
        // any of them have been resolved; if not, don't bother doing more work
        // yet.
        match pending_obligation.stalled_on.len() {
            // Match arms are in order of frequency, which matters because this
            // code is so hot. 1 and 0 dominate; 2+ is fairly rare.
            1 => {
                let infer_var = pending_obligation.stalled_on[0];
                self.selcx.infcx.ty_or_const_infer_var_changed(infer_var)
            }
            0 => {
                // In this case we haven't changed, but wish to make a change.
                true
            }
            _ => {
                // This `for` loop was once a call to `all()`, but this lower-level
                // form was a perf win. See #64545 for details.
                (|| {
                    for &infer_var in &pending_obligation.stalled_on {
                        if self.selcx.infcx.ty_or_const_infer_var_changed(infer_var) {
                            return true;
                        }
                    }
                    false
                })()
            }
        }
    }

    /// Processes a predicate obligation and returns either:
    /// - `Changed(v)` if the predicate is true, presuming that `v` are also true
    /// - `Unchanged` if we don't have enough info to be sure
    /// - `Error(e)` if the predicate does not hold
    ///
    /// This is called much less often than `needs_process_obligation`, so we
    /// never inline it.
    #[inline(never)]
    #[instrument(level = "debug", skip(self, pending_obligation))]
    fn process_obligation(
        &mut self,
        pending_obligation: &mut PendingPredicateObligation<'tcx>,
    ) -> ProcessResult<PendingPredicateObligation<'tcx>, FulfillmentErrorCode<'tcx>> {
        pending_obligation.stalled_on.truncate(0);

        let obligation = &mut pending_obligation.obligation;

        debug!(?obligation, "pre-resolve");

        if obligation.predicate.has_non_region_infer() {
            obligation.predicate = self.selcx.infcx.resolve_vars_if_possible(obligation.predicate);
        }

        let obligation = &pending_obligation.obligation;

        let infcx = self.selcx.infcx;

        if obligation.predicate.has_projections() {
            let mut obligations = Vec::new();
            let predicate = crate::traits::project::try_normalize_with_depth_to(
                &mut self.selcx,
                obligation.param_env,
                obligation.cause.clone(),
                obligation.recursion_depth + 1,
                obligation.predicate,
                &mut obligations,
            );
            if predicate != obligation.predicate {
                obligations.push(obligation.with(infcx.tcx, predicate));
                return ProcessResult::Changed(mk_pending(obligations));
            }
        }
        let binder = obligation.predicate.kind();
        match binder.no_bound_vars() {
            None => match binder.skip_binder() {
                // Evaluation will discard candidates using the leak check.
                // This means we need to pass it the bound version of our
                // predicate.
                ty::PredicateKind::Clause(ty::Clause::Trait(trait_ref)) => {
                    let trait_obligation = obligation.with(infcx.tcx, binder.rebind(trait_ref));

                    self.process_trait_obligation(
                        obligation,
                        trait_obligation,
                        &mut pending_obligation.stalled_on,
                    )
                }
                ty::PredicateKind::Clause(ty::Clause::Projection(data)) => {
                    let project_obligation = obligation.with(infcx.tcx, binder.rebind(data));

                    self.process_projection_obligation(
                        obligation,
                        project_obligation,
                        &mut pending_obligation.stalled_on,
                    )
                }
                ty::PredicateKind::Clause(ty::Clause::RegionOutlives(_))
                | ty::PredicateKind::Clause(ty::Clause::TypeOutlives(_))
                | ty::PredicateKind::Clause(ty::Clause::ConstArgHasType(..))
                | ty::PredicateKind::WellFormed(_)
                | ty::PredicateKind::ObjectSafe(_)
                | ty::PredicateKind::ClosureKind(..)
                | ty::PredicateKind::Subtype(_)
                | ty::PredicateKind::Coerce(_)
                | ty::PredicateKind::ConstEvaluatable(..)
                | ty::PredicateKind::ConstEquate(..) => {
                    let pred =
                        ty::Binder::dummy(infcx.instantiate_binder_with_placeholders(binder));
                    ProcessResult::Changed(mk_pending(vec![obligation.with(infcx.tcx, pred)]))
                }
                ty::PredicateKind::Ambiguous => ProcessResult::Unchanged,
                ty::PredicateKind::TypeWellFormedFromEnv(..) => {
                    bug!("TypeWellFormedFromEnv is only used for Chalk")
                }
                ty::PredicateKind::AliasEq(..) => {
                    bug!("AliasEq is only used for new solver")
                }
            },
            Some(pred) => match pred {
                ty::PredicateKind::Clause(ty::Clause::Trait(data)) => {
                    let trait_obligation = obligation.with(infcx.tcx, Binder::dummy(data));

                    self.process_trait_obligation(
                        obligation,
                        trait_obligation,
                        &mut pending_obligation.stalled_on,
                    )
                }

                ty::PredicateKind::Clause(ty::Clause::RegionOutlives(data)) => {
                    if infcx.considering_regions {
                        infcx.region_outlives_predicate(&obligation.cause, Binder::dummy(data));
                    }

                    ProcessResult::Changed(vec![])
                }

                ty::PredicateKind::Clause(ty::Clause::TypeOutlives(ty::OutlivesPredicate(
                    t_a,
                    r_b,
                ))) => {
                    if infcx.considering_regions {
                        infcx.register_region_obligation_with_cause(t_a, r_b, &obligation.cause);
                    }
                    ProcessResult::Changed(vec![])
                }

                ty::PredicateKind::Clause(ty::Clause::Projection(ref data)) => {
                    let project_obligation = obligation.with(infcx.tcx, Binder::dummy(*data));

                    self.process_projection_obligation(
                        obligation,
                        project_obligation,
                        &mut pending_obligation.stalled_on,
                    )
                }

                ty::PredicateKind::ObjectSafe(trait_def_id) => {
                    if !self.selcx.tcx().check_is_object_safe(trait_def_id) {
                        ProcessResult::Error(CodeSelectionError(Unimplemented))
                    } else {
                        ProcessResult::Changed(vec![])
                    }
                }

                ty::PredicateKind::ClosureKind(_, closure_substs, kind) => {
                    match self.selcx.infcx.closure_kind(closure_substs) {
                        Some(closure_kind) => {
                            if closure_kind.extends(kind) {
                                ProcessResult::Changed(vec![])
                            } else {
                                ProcessResult::Error(CodeSelectionError(Unimplemented))
                            }
                        }
                        None => ProcessResult::Unchanged,
                    }
                }

                ty::PredicateKind::WellFormed(arg) => {
                    match wf::obligations(
                        self.selcx.infcx,
                        obligation.param_env,
                        obligation.cause.body_id,
                        obligation.recursion_depth + 1,
                        arg,
                        obligation.cause.span,
                    ) {
                        None => {
                            pending_obligation.stalled_on =
                                vec![TyOrConstInferVar::maybe_from_generic_arg(arg).unwrap()];
                            ProcessResult::Unchanged
                        }
                        Some(os) => ProcessResult::Changed(mk_pending(os)),
                    }
                }

                ty::PredicateKind::Subtype(subtype) => {
                    match self.selcx.infcx.subtype_predicate(
                        &obligation.cause,
                        obligation.param_env,
                        Binder::dummy(subtype),
                    ) {
                        Err((a, b)) => {
                            // None means that both are unresolved.
                            pending_obligation.stalled_on =
                                vec![TyOrConstInferVar::Ty(a), TyOrConstInferVar::Ty(b)];
                            ProcessResult::Unchanged
                        }
                        Ok(Ok(ok)) => ProcessResult::Changed(mk_pending(ok.obligations)),
                        Ok(Err(err)) => {
                            let expected_found =
                                ExpectedFound::new(subtype.a_is_expected, subtype.a, subtype.b);
                            ProcessResult::Error(FulfillmentErrorCode::CodeSubtypeError(
                                expected_found,
                                err,
                            ))
                        }
                    }
                }

                ty::PredicateKind::Coerce(coerce) => {
                    match self.selcx.infcx.coerce_predicate(
                        &obligation.cause,
                        obligation.param_env,
                        Binder::dummy(coerce),
                    ) {
                        Err((a, b)) => {
                            // None means that both are unresolved.
                            pending_obligation.stalled_on =
                                vec![TyOrConstInferVar::Ty(a), TyOrConstInferVar::Ty(b)];
                            ProcessResult::Unchanged
                        }
                        Ok(Ok(ok)) => ProcessResult::Changed(mk_pending(ok.obligations)),
                        Ok(Err(err)) => {
                            let expected_found = ExpectedFound::new(false, coerce.a, coerce.b);
                            ProcessResult::Error(FulfillmentErrorCode::CodeSubtypeError(
                                expected_found,
                                err,
                            ))
                        }
                    }
                }

                ty::PredicateKind::ConstEvaluatable(uv) => {
                    match const_evaluatable::is_const_evaluatable(
                        self.selcx.infcx,
                        uv,
                        obligation.param_env,
                        obligation.cause.span,
                    ) {
                        Ok(()) => ProcessResult::Changed(vec![]),
                        Err(NotConstEvaluatable::MentionsInfer) => {
                            pending_obligation.stalled_on.clear();
                            pending_obligation.stalled_on.extend(
                                uv.walk().filter_map(TyOrConstInferVar::maybe_from_generic_arg),
                            );
                            ProcessResult::Unchanged
                        }
                        Err(
                            e @ NotConstEvaluatable::MentionsParam
                            | e @ NotConstEvaluatable::Error(_),
                        ) => ProcessResult::Error(CodeSelectionError(
                            SelectionError::NotConstEvaluatable(e),
                        )),
                    }
                }

                ty::PredicateKind::ConstEquate(c1, c2) => {
                    let tcx = self.selcx.tcx();
                    assert!(
                        tcx.features().generic_const_exprs,
                        "`ConstEquate` without a feature gate: {c1:?} {c2:?}",
                    );
                    // FIXME: we probably should only try to unify abstract constants
                    // if the constants depend on generic parameters.
                    //
                    // Let's just see where this breaks :shrug:
                    {
                        let c1 = tcx.expand_abstract_consts(c1);
                        let c2 = tcx.expand_abstract_consts(c2);
                        debug!("equating consts:\nc1= {:?}\nc2= {:?}", c1, c2);

                        use rustc_hir::def::DefKind;
                        use ty::ConstKind::Unevaluated;
                        match (c1.kind(), c2.kind()) {
                            (Unevaluated(a), Unevaluated(b))
                                if a.def.did == b.def.did
                                    && tcx.def_kind(a.def.did) == DefKind::AssocConst =>
                            {
                                if let Ok(new_obligations) = infcx
                                    .at(&obligation.cause, obligation.param_env)
                                    .trace(c1, c2)
                                    .eq(a.substs, b.substs)
                                {
                                    return ProcessResult::Changed(mk_pending(
                                        new_obligations.into_obligations(),
                                    ));
                                }
                            }
                            (_, Unevaluated(_)) | (Unevaluated(_), _) => (),
                            (_, _) => {
                                if let Ok(new_obligations) =
                                    infcx.at(&obligation.cause, obligation.param_env).eq(c1, c2)
                                {
                                    return ProcessResult::Changed(mk_pending(
                                        new_obligations.into_obligations(),
                                    ));
                                }
                            }
                        }
                    }

                    let stalled_on = &mut pending_obligation.stalled_on;

                    let mut evaluate = |c: Const<'tcx>| {
                        if let ty::ConstKind::Unevaluated(unevaluated) = c.kind() {
                            match self.selcx.infcx.try_const_eval_resolve(
                                obligation.param_env,
                                unevaluated,
                                c.ty(),
                                Some(obligation.cause.span),
                            ) {
                                Ok(val) => Ok(val),
                                Err(e) => match e {
                                    ErrorHandled::TooGeneric => {
                                        stalled_on.extend(
                                            unevaluated.substs.iter().filter_map(
                                                TyOrConstInferVar::maybe_from_generic_arg,
                                            ),
                                        );
                                        Err(ErrorHandled::TooGeneric)
                                    }
                                    _ => Err(e),
                                },
                            }
                        } else {
                            Ok(c)
                        }
                    };

                    match (evaluate(c1), evaluate(c2)) {
                        (Ok(c1), Ok(c2)) => {
                            match self
                                .selcx
                                .infcx
                                .at(&obligation.cause, obligation.param_env)
                                .eq(c1, c2)
                            {
                                Ok(inf_ok) => {
                                    ProcessResult::Changed(mk_pending(inf_ok.into_obligations()))
                                }
                                Err(err) => ProcessResult::Error(
                                    FulfillmentErrorCode::CodeConstEquateError(
                                        ExpectedFound::new(true, c1, c2),
                                        err,
                                    ),
                                ),
                            }
                        }
                        (Err(ErrorHandled::Reported(reported)), _)
                        | (_, Err(ErrorHandled::Reported(reported))) => ProcessResult::Error(
                            CodeSelectionError(SelectionError::NotConstEvaluatable(
                                NotConstEvaluatable::Error(reported),
                            )),
                        ),
                        (Err(ErrorHandled::TooGeneric), _) | (_, Err(ErrorHandled::TooGeneric)) => {
                            if c1.has_non_region_infer() || c2.has_non_region_infer() {
                                ProcessResult::Unchanged
                            } else {
                                // Two different constants using generic parameters ~> error.
                                let expected_found = ExpectedFound::new(true, c1, c2);
                                ProcessResult::Error(FulfillmentErrorCode::CodeConstEquateError(
                                    expected_found,
                                    TypeError::ConstMismatch(expected_found),
                                ))
                            }
                        }
                    }
                }
                ty::PredicateKind::Ambiguous => ProcessResult::Unchanged,
                ty::PredicateKind::TypeWellFormedFromEnv(..) => {
                    bug!("TypeWellFormedFromEnv is only used for Chalk")
                }
                ty::PredicateKind::AliasEq(..) => {
                    bug!("AliasEq is only used for new solver")
                }
                ty::PredicateKind::Clause(ty::Clause::ConstArgHasType(ct, ty)) => {
                    match self
                        .selcx
                        .infcx
                        .at(&obligation.cause, obligation.param_env)
                        .eq(ct.ty(), ty)
                    {
                        Ok(inf_ok) => ProcessResult::Changed(mk_pending(inf_ok.into_obligations())),
                        Err(_) => ProcessResult::Error(FulfillmentErrorCode::CodeSelectionError(
                            SelectionError::Unimplemented,
                        )),
                    }
                }
            },
        }
    }

    #[inline(never)]
    fn process_backedge<'c, I>(
        &mut self,
        cycle: I,
        _marker: PhantomData<&'c PendingPredicateObligation<'tcx>>,
    ) -> Result<(), FulfillmentErrorCode<'tcx>>
    where
        I: Clone + Iterator<Item = &'c PendingPredicateObligation<'tcx>>,
    {
        if self.selcx.coinductive_match(cycle.clone().map(|s| s.obligation.predicate)) {
            debug!("process_child_obligations: coinductive match");
            Ok(())
        } else {
            let cycle: Vec<_> = cycle.map(|c| c.obligation.clone()).collect();
            Err(FulfillmentErrorCode::CodeCycle(cycle))
        }
    }
}

impl<'a, 'tcx> FulfillProcessor<'a, 'tcx> {
    #[instrument(level = "debug", skip(self, obligation, stalled_on))]
    fn process_trait_obligation(
        &mut self,
        obligation: &PredicateObligation<'tcx>,
        trait_obligation: TraitObligation<'tcx>,
        stalled_on: &mut Vec<TyOrConstInferVar<'tcx>>,
    ) -> ProcessResult<PendingPredicateObligation<'tcx>, FulfillmentErrorCode<'tcx>> {
        let infcx = self.selcx.infcx;
        if obligation.predicate.is_global() {
            // no type variables present, can use evaluation for better caching.
            // FIXME: consider caching errors too.
            if infcx.predicate_must_hold_considering_regions(obligation) {
                debug!(
                    "selecting trait at depth {} evaluated to holds",
                    obligation.recursion_depth
                );
                return ProcessResult::Changed(vec![]);
            }
        }

        match self.selcx.select(&trait_obligation) {
            Ok(Some(impl_source)) => {
                debug!("selecting trait at depth {} yielded Ok(Some)", obligation.recursion_depth);
                ProcessResult::Changed(mk_pending(impl_source.nested_obligations()))
            }
            Ok(None) => {
                debug!("selecting trait at depth {} yielded Ok(None)", obligation.recursion_depth);

                // This is a bit subtle: for the most part, the
                // only reason we can fail to make progress on
                // trait selection is because we don't have enough
                // information about the types in the trait.
                stalled_on.clear();
                stalled_on.extend(substs_infer_vars(
                    &self.selcx,
                    trait_obligation.predicate.map_bound(|pred| pred.trait_ref.substs),
                ));

                debug!(
                    "process_predicate: pending obligation {:?} now stalled on {:?}",
                    infcx.resolve_vars_if_possible(obligation.clone()),
                    stalled_on
                );

                ProcessResult::Unchanged
            }
            Err(selection_err) => {
                debug!("selecting trait at depth {} yielded Err", obligation.recursion_depth);

                ProcessResult::Error(CodeSelectionError(selection_err))
            }
        }
    }

    fn process_projection_obligation(
        &mut self,
        obligation: &PredicateObligation<'tcx>,
        project_obligation: PolyProjectionObligation<'tcx>,
        stalled_on: &mut Vec<TyOrConstInferVar<'tcx>>,
    ) -> ProcessResult<PendingPredicateObligation<'tcx>, FulfillmentErrorCode<'tcx>> {
        let tcx = self.selcx.tcx();

        if obligation.predicate.is_global() {
            // no type variables present, can use evaluation for better caching.
            // FIXME: consider caching errors too.
            if self.selcx.infcx.predicate_must_hold_considering_regions(obligation) {
                if let Some(key) = ProjectionCacheKey::from_poly_projection_predicate(
                    &mut self.selcx,
                    project_obligation.predicate,
                ) {
                    // If `predicate_must_hold_considering_regions` succeeds, then we've
                    // evaluated all sub-obligations. We can therefore mark the 'root'
                    // obligation as complete, and skip evaluating sub-obligations.
                    self.selcx
                        .infcx
                        .inner
                        .borrow_mut()
                        .projection_cache()
                        .complete(key, EvaluationResult::EvaluatedToOk);
                }
                return ProcessResult::Changed(vec![]);
            } else {
                debug!("Does NOT hold: {:?}", obligation);
            }
        }

        match project::poly_project_and_unify_type(&mut self.selcx, &project_obligation) {
            ProjectAndUnifyResult::Holds(os) => ProcessResult::Changed(mk_pending(os)),
            ProjectAndUnifyResult::FailedNormalization => {
                stalled_on.clear();
                stalled_on.extend(substs_infer_vars(
                    &self.selcx,
                    project_obligation.predicate.map_bound(|pred| pred.projection_ty.substs),
                ));
                ProcessResult::Unchanged
            }
            // Let the caller handle the recursion
            ProjectAndUnifyResult::Recursive => ProcessResult::Changed(mk_pending(vec![
                project_obligation.with(tcx, project_obligation.predicate),
            ])),
            ProjectAndUnifyResult::MismatchedProjectionTypes(e) => {
                ProcessResult::Error(CodeProjectionError(e))
            }
        }
    }
}

/// Returns the set of inference variables contained in `substs`.
fn substs_infer_vars<'a, 'tcx>(
    selcx: &SelectionContext<'a, 'tcx>,
    substs: ty::Binder<'tcx, SubstsRef<'tcx>>,
) -> impl Iterator<Item = TyOrConstInferVar<'tcx>> {
    selcx
        .infcx
        .resolve_vars_if_possible(substs)
        .skip_binder() // ok because this check doesn't care about regions
        .iter()
        .filter(|arg| arg.has_non_region_infer())
        .flat_map(|arg| {
            let mut walker = arg.walk();
            while let Some(c) = walker.next() {
                if !c.has_non_region_infer() {
                    walker.visited.remove(&c);
                    walker.skip_current_subtree();
                }
            }
            walker.visited.into_iter()
        })
        .filter_map(TyOrConstInferVar::maybe_from_generic_arg)
}

fn to_fulfillment_error<'tcx>(
    error: Error<PendingPredicateObligation<'tcx>, FulfillmentErrorCode<'tcx>>,
) -> FulfillmentError<'tcx> {
    let mut iter = error.backtrace.into_iter();
    let obligation = iter.next().unwrap().obligation;
    // The root obligation is the last item in the backtrace - if there's only
    // one item, then it's the same as the main obligation
    let root_obligation = iter.next_back().map_or_else(|| obligation.clone(), |e| e.obligation);
    FulfillmentError::new(obligation, error.error, root_obligation)
}
