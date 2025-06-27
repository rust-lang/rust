use std::marker::PhantomData;

use rustc_data_structures::obligation_forest::{
    Error, ForestObligation, ObligationForest, ObligationProcessor, Outcome, ProcessResult,
};
use rustc_infer::infer::DefineOpaqueTypes;
use rustc_infer::traits::{
    FromSolverError, PolyTraitObligation, PredicateObligations, ProjectionCacheKey, SelectionError,
    TraitEngine,
};
use rustc_middle::bug;
use rustc_middle::ty::abstract_const::NotConstEvaluatable;
use rustc_middle::ty::error::{ExpectedFound, TypeError};
use rustc_middle::ty::{self, Binder, Const, GenericArgsRef, TypeVisitableExt, TypingMode};
use thin_vec::{ThinVec, thin_vec};
use tracing::{debug, debug_span, instrument};

use super::effects::{self, HostEffectObligation};
use super::project::{self, ProjectAndUnifyResult};
use super::select::SelectionContext;
use super::{
    EvaluationResult, FulfillmentError, FulfillmentErrorCode, PredicateObligation,
    ScrubbedTraitError, const_evaluatable, wf,
};
use crate::error_reporting::InferCtxtErrorExt;
use crate::infer::{InferCtxt, TyOrConstInferVar};
use crate::traits::normalize::normalize_with_depth_to;
use crate::traits::project::{PolyProjectionObligation, ProjectionCacheKeyExt as _};
use crate::traits::query::evaluate_obligation::InferCtxtExt;
use crate::traits::{EvaluateConstErr, sizedness_fast_path};

pub(crate) type PendingPredicateObligations<'tcx> = ThinVec<PendingPredicateObligation<'tcx>>;

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
pub struct FulfillmentContext<'tcx, E: 'tcx> {
    /// A list of all obligations that have been registered with this
    /// fulfillment context.
    predicates: ObligationForest<PendingPredicateObligation<'tcx>>,

    /// The snapshot in which this context was created. Using the context
    /// outside of this snapshot leads to subtle bugs if the snapshot
    /// gets rolled back. Because of this we explicitly check that we only
    /// use the context in exactly this snapshot.
    usable_in_snapshot: usize,

    _errors: PhantomData<E>,
}

#[derive(Clone, Debug)]
pub struct PendingPredicateObligation<'tcx> {
    pub obligation: PredicateObligation<'tcx>,
    // This is far more often read than modified, meaning that we
    // should mostly optimize for reading speed, while modifying is not as relevant.
    //
    // For whatever reason using a boxed slice is slower than using a `Vec` here.
    pub stalled_on: Vec<TyOrConstInferVar>,
}

// `PendingPredicateObligation` is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(target_pointer_width = "64")]
rustc_data_structures::static_assert_size!(PendingPredicateObligation<'_>, 72);

impl<'tcx, E> FulfillmentContext<'tcx, E>
where
    E: FromSolverError<'tcx, OldSolverError<'tcx>>,
{
    /// Creates a new fulfillment context.
    pub(super) fn new(infcx: &InferCtxt<'tcx>) -> FulfillmentContext<'tcx, E> {
        assert!(
            !infcx.next_trait_solver(),
            "old trait solver fulfillment context created when \
            infcx is set up for new trait solver"
        );
        FulfillmentContext {
            predicates: ObligationForest::new(),
            usable_in_snapshot: infcx.num_open_snapshots(),
            _errors: PhantomData,
        }
    }

    /// Attempts to select obligations using `selcx`.
    fn select(&mut self, selcx: SelectionContext<'_, 'tcx>) -> Vec<E> {
        let span = debug_span!("select", obligation_forest_size = ?self.predicates.len());
        let _enter = span.enter();
        let infcx = selcx.infcx;

        // Process pending obligations.
        let outcome: Outcome<_, _> =
            self.predicates.process_obligations(&mut FulfillProcessor { selcx });

        // FIXME: if we kept the original cache key, we could mark projection
        // obligations as complete for the projection cache here.

        let errors: Vec<E> = outcome
            .errors
            .into_iter()
            .map(|err| E::from_solver_error(infcx, OldSolverError(err)))
            .collect();

        debug!(
            "select({} predicates remaining, {} errors) done",
            self.predicates.len(),
            errors.len()
        );

        errors
    }
}

impl<'tcx, E> TraitEngine<'tcx, E> for FulfillmentContext<'tcx, E>
where
    E: FromSolverError<'tcx, OldSolverError<'tcx>>,
{
    #[inline]
    fn register_predicate_obligation(
        &mut self,
        infcx: &InferCtxt<'tcx>,
        mut obligation: PredicateObligation<'tcx>,
    ) {
        assert_eq!(self.usable_in_snapshot, infcx.num_open_snapshots());
        // this helps to reduce duplicate errors, as well as making
        // debug output much nicer to read and so on.
        debug_assert!(!obligation.param_env.has_non_region_infer());
        obligation.predicate = infcx.resolve_vars_if_possible(obligation.predicate);

        debug!(?obligation, "register_predicate_obligation");

        self.predicates
            .register_obligation(PendingPredicateObligation { obligation, stalled_on: vec![] });
    }

    fn collect_remaining_errors(&mut self, infcx: &InferCtxt<'tcx>) -> Vec<E> {
        self.predicates
            .to_errors(FulfillmentErrorCode::Ambiguity { overflow: None })
            .into_iter()
            .map(|err| E::from_solver_error(infcx, OldSolverError(err)))
            .collect()
    }

    fn select_where_possible(&mut self, infcx: &InferCtxt<'tcx>) -> Vec<E> {
        let selcx = SelectionContext::new(infcx);
        self.select(selcx)
    }

    fn drain_stalled_obligations_for_coroutines(
        &mut self,
        infcx: &InferCtxt<'tcx>,
    ) -> PredicateObligations<'tcx> {
        let mut processor =
            DrainProcessor { removed_predicates: PredicateObligations::new(), infcx };
        let outcome: Outcome<_, _> = self.predicates.process_obligations(&mut processor);
        assert!(outcome.errors.is_empty());
        return processor.removed_predicates;

        struct DrainProcessor<'a, 'tcx> {
            infcx: &'a InferCtxt<'tcx>,
            removed_predicates: PredicateObligations<'tcx>,
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
                ProcessResult::Changed(Default::default())
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

    fn has_pending_obligations(&self) -> bool {
        self.predicates.has_pending_obligations()
    }

    fn pending_obligations(&self) -> PredicateObligations<'tcx> {
        self.predicates.map_pending_obligations(|o| o.obligation.clone())
    }
}

struct FulfillProcessor<'a, 'tcx> {
    selcx: SelectionContext<'a, 'tcx>,
}

fn mk_pending<'tcx>(
    parent: &PredicateObligation<'tcx>,
    os: PredicateObligations<'tcx>,
) -> PendingPredicateObligations<'tcx> {
    os.into_iter()
        .map(|mut o| {
            o.set_depth_from_parent(parent.recursion_depth);
            PendingPredicateObligation { obligation: o, stalled_on: vec![] }
        })
        .collect()
}

impl<'a, 'tcx> ObligationProcessor for FulfillProcessor<'a, 'tcx> {
    type Obligation = PendingPredicateObligation<'tcx>;
    type Error = FulfillmentErrorCode<'tcx>;
    type OUT = Outcome<Self::Obligation, Self::Error>;

    /// Compared to `needs_process_obligation` this and its callees
    /// contain some optimizations that come at the price of false negatives.
    ///
    /// They
    /// - reduce branching by covering only the most common case
    /// - take a read-only view of the unification tables which allows skipping undo_log
    ///   construction.
    /// - bail out on value-cache misses in ena to avoid pointer chasing
    /// - hoist RefCell locking out of the loop
    #[inline]
    fn skippable_obligations<'b>(
        &'b self,
        it: impl Iterator<Item = &'b Self::Obligation>,
    ) -> usize {
        let is_unchanged = self.selcx.infcx.is_ty_infer_var_definitely_unchanged();

        it.take_while(|o| match o.stalled_on.as_slice() {
            [o] => is_unchanged(*o),
            _ => false,
        })
        .count()
    }

    /// Identifies whether a predicate obligation needs processing.
    ///
    /// This is always inlined because it has a single callsite and it is
    /// called *very* frequently. Be careful modifying this code! Several
    /// compile-time benchmarks are very sensitive to even small changes.
    #[inline(always)]
    fn needs_process_obligation(&self, pending_obligation: &Self::Obligation) -> bool {
        // If we were stalled on some unresolved variables, first check whether
        // any of them have been resolved; if not, don't bother doing more work
        // yet.
        let stalled_on = &pending_obligation.stalled_on;
        match stalled_on.len() {
            // This case is the hottest most of the time, being hit up to 99%
            // of the time. `keccak` and `cranelift-codegen-0.82.1` are
            // benchmarks that particularly stress this path.
            1 => self.selcx.infcx.ty_or_const_infer_var_changed(stalled_on[0]),

            // In this case we haven't changed, but wish to make a change. Note
            // that this is a special case, and is not equivalent to the `_`
            // case below, which would return `false` for an empty `stalled_on`
            // vector.
            //
            // This case is usually hit only 1% of the time or less, though it
            // reaches 20% in `wasmparser-0.101.0`.
            0 => true,

            // This case is usually hit only 1% of the time or less, though it
            // reaches 95% in `mime-0.3.16`, 64% in `wast-54.0.0`, and 12% in
            // `inflate-0.4.5`.
            //
            // The obvious way of writing this, with a call to `any()` and no
            // closure, is currently slower than this version.
            _ => (|| {
                for &infer_var in stalled_on {
                    if self.selcx.infcx.ty_or_const_infer_var_changed(infer_var) {
                        return true;
                    }
                }
                false
            })(),
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

        if sizedness_fast_path(infcx.tcx, obligation.predicate) {
            return ProcessResult::Changed(thin_vec![]);
        }

        if obligation.predicate.has_aliases() {
            let mut obligations = PredicateObligations::new();
            let predicate = normalize_with_depth_to(
                &mut self.selcx,
                obligation.param_env,
                obligation.cause.clone(),
                obligation.recursion_depth + 1,
                obligation.predicate,
                &mut obligations,
            );
            if predicate != obligation.predicate {
                obligations.push(obligation.with(infcx.tcx, predicate));
                return ProcessResult::Changed(mk_pending(obligation, obligations));
            }
        }
        let binder = obligation.predicate.kind();
        match binder.no_bound_vars() {
            None => match binder.skip_binder() {
                // Evaluation will discard candidates using the leak check.
                // This means we need to pass it the bound version of our
                // predicate.
                ty::PredicateKind::Clause(ty::ClauseKind::Trait(trait_ref)) => {
                    let trait_obligation = obligation.with(infcx.tcx, binder.rebind(trait_ref));

                    self.process_trait_obligation(
                        obligation,
                        trait_obligation,
                        &mut pending_obligation.stalled_on,
                    )
                }
                ty::PredicateKind::Clause(ty::ClauseKind::Projection(data)) => {
                    let project_obligation = obligation.with(infcx.tcx, binder.rebind(data));

                    self.process_projection_obligation(
                        obligation,
                        project_obligation,
                        &mut pending_obligation.stalled_on,
                    )
                }
                ty::PredicateKind::Clause(ty::ClauseKind::RegionOutlives(_))
                | ty::PredicateKind::Clause(ty::ClauseKind::TypeOutlives(_))
                | ty::PredicateKind::Clause(ty::ClauseKind::ConstArgHasType(..))
                | ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(_))
                | ty::PredicateKind::DynCompatible(_)
                | ty::PredicateKind::Subtype(_)
                | ty::PredicateKind::Coerce(_)
                | ty::PredicateKind::Clause(ty::ClauseKind::ConstEvaluatable(..))
                | ty::PredicateKind::ConstEquate(..)
                // FIXME(const_trait_impl): We may need to do this using the higher-ranked
                // pred instead of just instantiating it with placeholders b/c of
                // higher-ranked implied bound issues in the old solver.
                | ty::PredicateKind::Clause(ty::ClauseKind::HostEffect(..)) => {
                    let pred = ty::Binder::dummy(infcx.enter_forall_and_leak_universe(binder));
                    let mut obligations = PredicateObligations::with_capacity(1);
                    obligations.push(obligation.with(infcx.tcx, pred));

                    ProcessResult::Changed(mk_pending(obligation, obligations))
                }
                ty::PredicateKind::Ambiguous => ProcessResult::Unchanged,
                ty::PredicateKind::NormalizesTo(..) => {
                    bug!("NormalizesTo is only used by the new solver")
                }
                ty::PredicateKind::AliasRelate(..) => {
                    bug!("AliasRelate is only used by the new solver")
                }
            },
            Some(pred) => match pred {
                ty::PredicateKind::Clause(ty::ClauseKind::Trait(data)) => {
                    let trait_obligation = obligation.with(infcx.tcx, Binder::dummy(data));

                    self.process_trait_obligation(
                        obligation,
                        trait_obligation,
                        &mut pending_obligation.stalled_on,
                    )
                }

                ty::PredicateKind::Clause(ty::ClauseKind::HostEffect(data)) => {
                    let host_obligation = obligation.with(infcx.tcx, data);

                    self.process_host_obligation(
                        obligation,
                        host_obligation,
                        &mut pending_obligation.stalled_on,
                    )
                }

                ty::PredicateKind::Clause(ty::ClauseKind::RegionOutlives(data)) => {
                    if infcx.considering_regions {
                        infcx.register_region_outlives_constraint(data, &obligation.cause);
                    }

                    ProcessResult::Changed(Default::default())
                }

                ty::PredicateKind::Clause(ty::ClauseKind::TypeOutlives(ty::OutlivesPredicate(
                    t_a,
                    r_b,
                ))) => {
                    if infcx.considering_regions {
                        infcx.register_type_outlives_constraint(t_a, r_b, &obligation.cause);
                    }
                    ProcessResult::Changed(Default::default())
                }

                ty::PredicateKind::Clause(ty::ClauseKind::Projection(ref data)) => {
                    let project_obligation = obligation.with(infcx.tcx, Binder::dummy(*data));

                    self.process_projection_obligation(
                        obligation,
                        project_obligation,
                        &mut pending_obligation.stalled_on,
                    )
                }

                ty::PredicateKind::DynCompatible(trait_def_id) => {
                    if !self.selcx.tcx().is_dyn_compatible(trait_def_id) {
                        ProcessResult::Error(FulfillmentErrorCode::Select(
                            SelectionError::Unimplemented,
                        ))
                    } else {
                        ProcessResult::Changed(Default::default())
                    }
                }

                ty::PredicateKind::Ambiguous => ProcessResult::Unchanged,
                ty::PredicateKind::NormalizesTo(..) => {
                    bug!("NormalizesTo is only used by the new solver")
                }
                ty::PredicateKind::AliasRelate(..) => {
                    bug!("AliasRelate is only used by the new solver")
                }
                // Compute `ConstArgHasType` above the overflow check below.
                // This is because this is not ever a useful obligation to report
                // as the cause of an overflow.
                ty::PredicateKind::Clause(ty::ClauseKind::ConstArgHasType(ct, ty)) => {
                    let ct = infcx.shallow_resolve_const(ct);
                    let ct_ty = match ct.kind() {
                        ty::ConstKind::Infer(var) => {
                            let var = match var {
                                ty::InferConst::Var(vid) => TyOrConstInferVar::Const(vid),
                                ty::InferConst::Fresh(_) => {
                                    bug!("encountered fresh const in fulfill")
                                }
                            };
                            pending_obligation.stalled_on.clear();
                            pending_obligation.stalled_on.extend([var]);
                            return ProcessResult::Unchanged;
                        }
                        ty::ConstKind::Error(_) => {
                            return ProcessResult::Changed(PendingPredicateObligations::new());
                        }
                        ty::ConstKind::Value(cv) => cv.ty,
                        ty::ConstKind::Unevaluated(uv) => {
                            infcx.tcx.type_of(uv.def).instantiate(infcx.tcx, uv.args)
                        }
                        // FIXME(generic_const_exprs): we should construct an alias like
                        // `<lhs_ty as Add<rhs_ty>>::Output` when this is an `Expr` representing
                        // `lhs + rhs`.
                        ty::ConstKind::Expr(_) => {
                            return ProcessResult::Changed(mk_pending(
                                obligation,
                                PredicateObligations::new(),
                            ));
                        }
                        ty::ConstKind::Placeholder(_) => {
                            bug!("placeholder const {:?} in old solver", ct)
                        }
                        ty::ConstKind::Bound(_, _) => bug!("escaping bound vars in {:?}", ct),
                        ty::ConstKind::Param(param_ct) => {
                            param_ct.find_const_ty_from_env(obligation.param_env)
                        }
                    };

                    match infcx.at(&obligation.cause, obligation.param_env).eq(
                        // Only really exercised by generic_const_exprs
                        DefineOpaqueTypes::Yes,
                        ct_ty,
                        ty,
                    ) {
                        Ok(inf_ok) => ProcessResult::Changed(mk_pending(
                            obligation,
                            inf_ok.into_obligations(),
                        )),
                        Err(_) => ProcessResult::Error(FulfillmentErrorCode::Select(
                            SelectionError::ConstArgHasWrongType { ct, ct_ty, expected_ty: ty },
                        )),
                    }
                }

                // General case overflow check. Allow `process_trait_obligation`
                // and `process_projection_obligation` to handle checking for
                // the recursion limit themselves. Also don't check some
                // predicate kinds that don't give further obligations.
                _ if !self
                    .selcx
                    .tcx()
                    .recursion_limit()
                    .value_within_limit(obligation.recursion_depth) =>
                {
                    self.selcx.infcx.err_ctxt().report_overflow_obligation(&obligation, false);
                }

                ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(term)) => {
                    if term.is_trivially_wf(self.selcx.tcx()) {
                        return ProcessResult::Changed(thin_vec![]);
                    }

                    match wf::obligations(
                        self.selcx.infcx,
                        obligation.param_env,
                        obligation.cause.body_id,
                        obligation.recursion_depth + 1,
                        term,
                        obligation.cause.span,
                    ) {
                        None => {
                            pending_obligation.stalled_on =
                                vec![TyOrConstInferVar::maybe_from_term(term).unwrap()];
                            ProcessResult::Unchanged
                        }
                        Some(os) => ProcessResult::Changed(mk_pending(obligation, os)),
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
                        Ok(Ok(ok)) => {
                            ProcessResult::Changed(mk_pending(obligation, ok.obligations))
                        }
                        Ok(Err(err)) => {
                            let expected_found = if subtype.a_is_expected {
                                ExpectedFound::new(subtype.a, subtype.b)
                            } else {
                                ExpectedFound::new(subtype.b, subtype.a)
                            };
                            ProcessResult::Error(FulfillmentErrorCode::Subtype(expected_found, err))
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
                        Ok(Ok(ok)) => {
                            ProcessResult::Changed(mk_pending(obligation, ok.obligations))
                        }
                        Ok(Err(err)) => {
                            let expected_found = ExpectedFound::new(coerce.b, coerce.a);
                            ProcessResult::Error(FulfillmentErrorCode::Subtype(expected_found, err))
                        }
                    }
                }

                ty::PredicateKind::Clause(ty::ClauseKind::ConstEvaluatable(uv)) => {
                    match const_evaluatable::is_const_evaluatable(
                        self.selcx.infcx,
                        uv,
                        obligation.param_env,
                        obligation.cause.span,
                    ) {
                        Ok(()) => ProcessResult::Changed(Default::default()),
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
                        ) => ProcessResult::Error(FulfillmentErrorCode::Select(
                            SelectionError::NotConstEvaluatable(e),
                        )),
                    }
                }

                ty::PredicateKind::ConstEquate(c1, c2) => {
                    let tcx = self.selcx.tcx();
                    assert!(
                        tcx.features().generic_const_exprs(),
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
                        match (c1.kind(), c2.kind()) {
                            (ty::ConstKind::Unevaluated(a), ty::ConstKind::Unevaluated(b))
                                if a.def == b.def && tcx.def_kind(a.def) == DefKind::AssocConst =>
                            {
                                if let Ok(new_obligations) = infcx
                                    .at(&obligation.cause, obligation.param_env)
                                    // Can define opaque types as this is only reachable with
                                    // `generic_const_exprs`
                                    .eq(
                                        DefineOpaqueTypes::Yes,
                                        ty::AliasTerm::from(a),
                                        ty::AliasTerm::from(b),
                                    )
                                {
                                    return ProcessResult::Changed(mk_pending(
                                        obligation,
                                        new_obligations.into_obligations(),
                                    ));
                                }
                            }
                            (_, ty::ConstKind::Unevaluated(_))
                            | (ty::ConstKind::Unevaluated(_), _) => (),
                            (_, _) => {
                                if let Ok(new_obligations) = infcx
                                    .at(&obligation.cause, obligation.param_env)
                                    // Can define opaque types as this is only reachable with
                                    // `generic_const_exprs`
                                    .eq(DefineOpaqueTypes::Yes, c1, c2)
                                {
                                    return ProcessResult::Changed(mk_pending(
                                        obligation,
                                        new_obligations.into_obligations(),
                                    ));
                                }
                            }
                        }
                    }

                    let stalled_on = &mut pending_obligation.stalled_on;

                    let mut evaluate = |c: Const<'tcx>| {
                        if let ty::ConstKind::Unevaluated(unevaluated) = c.kind() {
                            match super::try_evaluate_const(
                                self.selcx.infcx,
                                c,
                                obligation.param_env,
                            ) {
                                Ok(val) => Ok(val),
                                e @ Err(EvaluateConstErr::HasGenericsOrInfers) => {
                                    stalled_on.extend(
                                        unevaluated
                                            .args
                                            .iter()
                                            .filter_map(TyOrConstInferVar::maybe_from_generic_arg),
                                    );
                                    e
                                }
                                e @ Err(
                                    EvaluateConstErr::EvaluationFailure(_)
                                    | EvaluateConstErr::InvalidConstParamTy(_),
                                ) => e,
                            }
                        } else {
                            Ok(c)
                        }
                    };

                    match (evaluate(c1), evaluate(c2)) {
                        (Ok(c1), Ok(c2)) => {
                            match self.selcx.infcx.at(&obligation.cause, obligation.param_env).eq(
                                // Can define opaque types as this is only reachable with
                                // `generic_const_exprs`
                                DefineOpaqueTypes::Yes,
                                c1,
                                c2,
                            ) {
                                Ok(inf_ok) => ProcessResult::Changed(mk_pending(
                                    obligation,
                                    inf_ok.into_obligations(),
                                )),
                                Err(err) => {
                                    ProcessResult::Error(FulfillmentErrorCode::ConstEquate(
                                        ExpectedFound::new(c1, c2),
                                        err,
                                    ))
                                }
                            }
                        }
                        (Err(EvaluateConstErr::InvalidConstParamTy(e)), _)
                        | (_, Err(EvaluateConstErr::InvalidConstParamTy(e))) => {
                            ProcessResult::Error(FulfillmentErrorCode::Select(
                                SelectionError::NotConstEvaluatable(NotConstEvaluatable::Error(e)),
                            ))
                        }
                        (Err(EvaluateConstErr::EvaluationFailure(e)), _)
                        | (_, Err(EvaluateConstErr::EvaluationFailure(e))) => {
                            ProcessResult::Error(FulfillmentErrorCode::Select(
                                SelectionError::NotConstEvaluatable(NotConstEvaluatable::Error(e)),
                            ))
                        }
                        (Err(EvaluateConstErr::HasGenericsOrInfers), _)
                        | (_, Err(EvaluateConstErr::HasGenericsOrInfers)) => {
                            if c1.has_non_region_infer() || c2.has_non_region_infer() {
                                ProcessResult::Unchanged
                            } else {
                                // Two different constants using generic parameters ~> error.
                                let expected_found = ExpectedFound::new(c1, c2);
                                ProcessResult::Error(FulfillmentErrorCode::ConstEquate(
                                    expected_found,
                                    TypeError::ConstMismatch(expected_found),
                                ))
                            }
                        }
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
            let cycle = cycle.map(|c| c.obligation.clone()).collect();
            Err(FulfillmentErrorCode::Cycle(cycle))
        }
    }
}

impl<'a, 'tcx> FulfillProcessor<'a, 'tcx> {
    #[instrument(level = "debug", skip(self, obligation, stalled_on))]
    fn process_trait_obligation(
        &mut self,
        obligation: &PredicateObligation<'tcx>,
        trait_obligation: PolyTraitObligation<'tcx>,
        stalled_on: &mut Vec<TyOrConstInferVar>,
    ) -> ProcessResult<PendingPredicateObligation<'tcx>, FulfillmentErrorCode<'tcx>> {
        let infcx = self.selcx.infcx;
        if obligation.predicate.is_global() && !matches!(infcx.typing_mode(), TypingMode::Coherence)
        {
            // no type variables present, can use evaluation for better caching.
            // FIXME: consider caching errors too.
            if infcx.predicate_must_hold_considering_regions(obligation) {
                debug!(
                    "selecting trait at depth {} evaluated to holds",
                    obligation.recursion_depth
                );
                return ProcessResult::Changed(Default::default());
            }
        }

        match self.selcx.poly_select(&trait_obligation) {
            Ok(Some(impl_source)) => {
                debug!("selecting trait at depth {} yielded Ok(Some)", obligation.recursion_depth);
                ProcessResult::Changed(mk_pending(obligation, impl_source.nested_obligations()))
            }
            Ok(None) => {
                debug!("selecting trait at depth {} yielded Ok(None)", obligation.recursion_depth);

                // This is a bit subtle: for the most part, the
                // only reason we can fail to make progress on
                // trait selection is because we don't have enough
                // information about the types in the trait.
                stalled_on.clear();
                stalled_on.extend(args_infer_vars(
                    &self.selcx,
                    trait_obligation.predicate.map_bound(|pred| pred.trait_ref.args),
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

                ProcessResult::Error(FulfillmentErrorCode::Select(selection_err))
            }
        }
    }

    fn process_projection_obligation(
        &mut self,
        obligation: &PredicateObligation<'tcx>,
        project_obligation: PolyProjectionObligation<'tcx>,
        stalled_on: &mut Vec<TyOrConstInferVar>,
    ) -> ProcessResult<PendingPredicateObligation<'tcx>, FulfillmentErrorCode<'tcx>> {
        let tcx = self.selcx.tcx();
        let infcx = self.selcx.infcx;
        if obligation.predicate.is_global() && !matches!(infcx.typing_mode(), TypingMode::Coherence)
        {
            // no type variables present, can use evaluation for better caching.
            // FIXME: consider caching errors too.
            if infcx.predicate_must_hold_considering_regions(obligation) {
                if let Some(key) = ProjectionCacheKey::from_poly_projection_obligation(
                    &mut self.selcx,
                    &project_obligation,
                ) {
                    // If `predicate_must_hold_considering_regions` succeeds, then we've
                    // evaluated all sub-obligations. We can therefore mark the 'root'
                    // obligation as complete, and skip evaluating sub-obligations.
                    infcx
                        .inner
                        .borrow_mut()
                        .projection_cache()
                        .complete(key, EvaluationResult::EvaluatedToOk);
                }
                return ProcessResult::Changed(Default::default());
            } else {
                debug!("Does NOT hold: {:?}", obligation);
            }
        }

        match project::poly_project_and_unify_term(&mut self.selcx, &project_obligation) {
            ProjectAndUnifyResult::Holds(os) => ProcessResult::Changed(mk_pending(obligation, os)),
            ProjectAndUnifyResult::FailedNormalization => {
                stalled_on.clear();
                stalled_on.extend(args_infer_vars(
                    &self.selcx,
                    project_obligation.predicate.map_bound(|pred| pred.projection_term.args),
                ));
                ProcessResult::Unchanged
            }
            // Let the caller handle the recursion
            ProjectAndUnifyResult::Recursive => {
                let mut obligations = PredicateObligations::with_capacity(1);
                obligations.push(project_obligation.with(tcx, project_obligation.predicate));

                ProcessResult::Changed(mk_pending(obligation, obligations))
            }
            ProjectAndUnifyResult::MismatchedProjectionTypes(e) => {
                ProcessResult::Error(FulfillmentErrorCode::Project(e))
            }
        }
    }

    fn process_host_obligation(
        &mut self,
        obligation: &PredicateObligation<'tcx>,
        host_obligation: HostEffectObligation<'tcx>,
        stalled_on: &mut Vec<TyOrConstInferVar>,
    ) -> ProcessResult<PendingPredicateObligation<'tcx>, FulfillmentErrorCode<'tcx>> {
        match effects::evaluate_host_effect_obligation(&mut self.selcx, &host_obligation) {
            Ok(nested) => ProcessResult::Changed(mk_pending(obligation, nested)),
            Err(effects::EvaluationFailure::Ambiguous) => {
                stalled_on.clear();
                stalled_on.extend(args_infer_vars(
                    &self.selcx,
                    ty::Binder::dummy(host_obligation.predicate.trait_ref.args),
                ));
                ProcessResult::Unchanged
            }
            Err(effects::EvaluationFailure::NoSolution) => {
                ProcessResult::Error(FulfillmentErrorCode::Select(SelectionError::Unimplemented))
            }
        }
    }
}

/// Returns the set of inference variables contained in `args`.
fn args_infer_vars<'tcx>(
    selcx: &SelectionContext<'_, 'tcx>,
    args: ty::Binder<'tcx, GenericArgsRef<'tcx>>,
) -> impl Iterator<Item = TyOrConstInferVar> {
    selcx
        .infcx
        .resolve_vars_if_possible(args)
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

#[derive(Debug)]
pub struct OldSolverError<'tcx>(
    Error<PendingPredicateObligation<'tcx>, FulfillmentErrorCode<'tcx>>,
);

impl<'tcx> FromSolverError<'tcx, OldSolverError<'tcx>> for FulfillmentError<'tcx> {
    fn from_solver_error(_infcx: &InferCtxt<'tcx>, error: OldSolverError<'tcx>) -> Self {
        let mut iter = error.0.backtrace.into_iter();
        let obligation = iter.next().unwrap().obligation;
        // The root obligation is the last item in the backtrace - if there's only
        // one item, then it's the same as the main obligation
        let root_obligation = iter.next_back().map_or_else(|| obligation.clone(), |e| e.obligation);
        FulfillmentError::new(obligation, error.0.error, root_obligation)
    }
}

impl<'tcx> FromSolverError<'tcx, OldSolverError<'tcx>> for ScrubbedTraitError<'tcx> {
    fn from_solver_error(_infcx: &InferCtxt<'tcx>, error: OldSolverError<'tcx>) -> Self {
        match error.0.error {
            FulfillmentErrorCode::Select(_)
            | FulfillmentErrorCode::Project(_)
            | FulfillmentErrorCode::Subtype(_, _)
            | FulfillmentErrorCode::ConstEquate(_, _) => ScrubbedTraitError::TrueError,
            FulfillmentErrorCode::Ambiguity { overflow: _ } => ScrubbedTraitError::Ambiguity,
            FulfillmentErrorCode::Cycle(cycle) => ScrubbedTraitError::Cycle(cycle),
        }
    }
}
