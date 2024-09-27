use std::marker::PhantomData;
use std::mem;
use std::ops::ControlFlow;

use rustc_infer::infer::InferCtxt;
use rustc_infer::traits::query::NoSolution;
use rustc_infer::traits::solve::{CandidateSource, GoalSource, MaybeCause};
use rustc_infer::traits::{
    self, FromSolverError, MismatchedProjectionTypes, Obligation, ObligationCause,
    ObligationCauseCode, PredicateObligation, SelectionError, TraitEngine,
};
use rustc_middle::bug;
use rustc_middle::ty::error::{ExpectedFound, TypeError};
use rustc_middle::ty::{self, TyCtxt};
use rustc_next_trait_solver::solve::{GenerateProofTree, SolverDelegateEvalExt as _};
use tracing::instrument;

use super::Certainty;
use super::delegate::SolverDelegate;
use super::inspect::{self, ProofTreeInferCtxtExt, ProofTreeVisitor};
use crate::traits::{FulfillmentError, FulfillmentErrorCode, ScrubbedTraitError};

/// A trait engine using the new trait solver.
///
/// This is mostly identical to how `evaluate_all` works inside of the
/// solver, except that the requirements are slightly different.
///
/// Unlike `evaluate_all` it is possible to add new obligations later on
/// and we also have to track diagnostics information by using `Obligation`
/// instead of `Goal`.
///
/// It is also likely that we want to use slightly different datastructures
/// here as this will have to deal with far more root goals than `evaluate_all`.
pub struct FulfillmentCtxt<'tcx, E: 'tcx> {
    obligations: ObligationStorage<'tcx>,

    /// The snapshot in which this context was created. Using the context
    /// outside of this snapshot leads to subtle bugs if the snapshot
    /// gets rolled back. Because of this we explicitly check that we only
    /// use the context in exactly this snapshot.
    usable_in_snapshot: usize,
    _errors: PhantomData<E>,
}

#[derive(Default)]
struct ObligationStorage<'tcx> {
    /// Obligations which resulted in an overflow in fulfillment itself.
    ///
    /// We cannot eagerly return these as error so we instead store them here
    /// to avoid recomputing them each time `select_where_possible` is called.
    /// This also allows us to return the correct `FulfillmentError` for them.
    overflowed: Vec<PredicateObligation<'tcx>>,
    pending: Vec<PredicateObligation<'tcx>>,
}

impl<'tcx> ObligationStorage<'tcx> {
    fn register(&mut self, obligation: PredicateObligation<'tcx>) {
        self.pending.push(obligation);
    }

    fn clone_pending(&self) -> Vec<PredicateObligation<'tcx>> {
        let mut obligations = self.pending.clone();
        obligations.extend(self.overflowed.iter().cloned());
        obligations
    }

    fn take_pending(&mut self) -> Vec<PredicateObligation<'tcx>> {
        let mut obligations = mem::take(&mut self.pending);
        obligations.append(&mut self.overflowed);
        obligations
    }

    fn unstalled_for_select(&mut self) -> impl Iterator<Item = PredicateObligation<'tcx>> {
        mem::take(&mut self.pending).into_iter()
    }

    fn on_fulfillment_overflow(&mut self, infcx: &InferCtxt<'tcx>) {
        infcx.probe(|_| {
            // IMPORTANT: we must not use solve any inference variables in the obligations
            // as this is all happening inside of a probe. We use a probe to make sure
            // we get all obligations involved in the overflow. We pretty much check: if
            // we were to do another step of `select_where_possible`, which goals would
            // change.
            self.overflowed.extend(self.pending.extract_if(|o| {
                let goal = o.clone().into();
                let result = <&SolverDelegate<'tcx>>::from(infcx)
                    .evaluate_root_goal(goal, GenerateProofTree::No)
                    .0;
                match result {
                    Ok((has_changed, _)) => has_changed,
                    _ => false,
                }
            }));
        })
    }
}

impl<'tcx, E: 'tcx> FulfillmentCtxt<'tcx, E> {
    pub fn new(infcx: &InferCtxt<'tcx>) -> FulfillmentCtxt<'tcx, E> {
        assert!(
            infcx.next_trait_solver(),
            "new trait solver fulfillment context created when \
            infcx is set up for old trait solver"
        );
        FulfillmentCtxt {
            obligations: Default::default(),
            usable_in_snapshot: infcx.num_open_snapshots(),
            _errors: PhantomData,
        }
    }

    fn inspect_evaluated_obligation(
        &self,
        infcx: &InferCtxt<'tcx>,
        obligation: &PredicateObligation<'tcx>,
        result: &Result<(bool, Certainty), NoSolution>,
    ) {
        if let Some(inspector) = infcx.obligation_inspector.get() {
            let result = match result {
                Ok((_, c)) => Ok(*c),
                Err(NoSolution) => Err(NoSolution),
            };
            (inspector)(infcx, &obligation, result);
        }
    }
}

impl<'tcx, E> TraitEngine<'tcx, E> for FulfillmentCtxt<'tcx, E>
where
    E: FromSolverError<'tcx, NextSolverError<'tcx>>,
{
    #[instrument(level = "trace", skip(self, infcx))]
    fn register_predicate_obligation(
        &mut self,
        infcx: &InferCtxt<'tcx>,
        obligation: PredicateObligation<'tcx>,
    ) {
        assert_eq!(self.usable_in_snapshot, infcx.num_open_snapshots());
        self.obligations.register(obligation);
    }

    fn collect_remaining_errors(&mut self, infcx: &InferCtxt<'tcx>) -> Vec<E> {
        self.obligations
            .pending
            .drain(..)
            .map(|obligation| NextSolverError::Ambiguity(obligation))
            .chain(
                self.obligations
                    .overflowed
                    .drain(..)
                    .map(|obligation| NextSolverError::Overflow(obligation)),
            )
            .map(|e| E::from_solver_error(infcx, e))
            .collect()
    }

    fn select_where_possible(&mut self, infcx: &InferCtxt<'tcx>) -> Vec<E> {
        assert_eq!(self.usable_in_snapshot, infcx.num_open_snapshots());
        let mut errors = Vec::new();
        for i in 0.. {
            if !infcx.tcx.recursion_limit().value_within_limit(i) {
                self.obligations.on_fulfillment_overflow(infcx);
                // Only return true errors that we have accumulated while processing.
                return errors;
            }

            let mut has_changed = false;
            for obligation in self.obligations.unstalled_for_select() {
                let goal = obligation.clone().into();
                let result = <&SolverDelegate<'tcx>>::from(infcx)
                    .evaluate_root_goal(goal, GenerateProofTree::No)
                    .0;
                self.inspect_evaluated_obligation(infcx, &obligation, &result);
                let (changed, certainty) = match result {
                    Ok(result) => result,
                    Err(NoSolution) => {
                        errors.push(E::from_solver_error(
                            infcx,
                            NextSolverError::TrueError(obligation),
                        ));
                        continue;
                    }
                };
                has_changed |= changed;
                match certainty {
                    Certainty::Yes => {}
                    Certainty::Maybe(_) => self.obligations.register(obligation),
                }
            }

            if !has_changed {
                break;
            }
        }

        errors
    }

    fn pending_obligations(&self) -> Vec<PredicateObligation<'tcx>> {
        self.obligations.clone_pending()
    }

    fn drain_unstalled_obligations(
        &mut self,
        _: &InferCtxt<'tcx>,
    ) -> Vec<PredicateObligation<'tcx>> {
        self.obligations.take_pending()
    }
}

pub enum NextSolverError<'tcx> {
    TrueError(PredicateObligation<'tcx>),
    Ambiguity(PredicateObligation<'tcx>),
    Overflow(PredicateObligation<'tcx>),
}

impl<'tcx> FromSolverError<'tcx, NextSolverError<'tcx>> for FulfillmentError<'tcx> {
    fn from_solver_error(infcx: &InferCtxt<'tcx>, error: NextSolverError<'tcx>) -> Self {
        match error {
            NextSolverError::TrueError(obligation) => {
                fulfillment_error_for_no_solution(infcx, obligation)
            }
            NextSolverError::Ambiguity(obligation) => {
                fulfillment_error_for_stalled(infcx, obligation)
            }
            NextSolverError::Overflow(obligation) => {
                fulfillment_error_for_overflow(infcx, obligation)
            }
        }
    }
}

impl<'tcx> FromSolverError<'tcx, NextSolverError<'tcx>> for ScrubbedTraitError<'tcx> {
    fn from_solver_error(_infcx: &InferCtxt<'tcx>, error: NextSolverError<'tcx>) -> Self {
        match error {
            NextSolverError::TrueError(_) => ScrubbedTraitError::TrueError,
            NextSolverError::Ambiguity(_) | NextSolverError::Overflow(_) => {
                ScrubbedTraitError::Ambiguity
            }
        }
    }
}

fn fulfillment_error_for_no_solution<'tcx>(
    infcx: &InferCtxt<'tcx>,
    root_obligation: PredicateObligation<'tcx>,
) -> FulfillmentError<'tcx> {
    let obligation = find_best_leaf_obligation(infcx, &root_obligation, false);

    let code = match obligation.predicate.kind().skip_binder() {
        ty::PredicateKind::Clause(ty::ClauseKind::Projection(_)) => {
            FulfillmentErrorCode::Project(
                // FIXME: This could be a `Sorts` if the term is a type
                MismatchedProjectionTypes { err: TypeError::Mismatch },
            )
        }
        ty::PredicateKind::NormalizesTo(..) => {
            FulfillmentErrorCode::Project(MismatchedProjectionTypes { err: TypeError::Mismatch })
        }
        ty::PredicateKind::AliasRelate(_, _, _) => {
            FulfillmentErrorCode::Project(MismatchedProjectionTypes { err: TypeError::Mismatch })
        }
        ty::PredicateKind::Subtype(pred) => {
            let (a, b) = infcx.enter_forall_and_leak_universe(
                obligation.predicate.kind().rebind((pred.a, pred.b)),
            );
            let expected_found = ExpectedFound::new(true, a, b);
            FulfillmentErrorCode::Subtype(expected_found, TypeError::Sorts(expected_found))
        }
        ty::PredicateKind::Coerce(pred) => {
            let (a, b) = infcx.enter_forall_and_leak_universe(
                obligation.predicate.kind().rebind((pred.a, pred.b)),
            );
            let expected_found = ExpectedFound::new(false, a, b);
            FulfillmentErrorCode::Subtype(expected_found, TypeError::Sorts(expected_found))
        }
        ty::PredicateKind::Clause(_)
        | ty::PredicateKind::DynCompatible(_)
        | ty::PredicateKind::Ambiguous => {
            FulfillmentErrorCode::Select(SelectionError::Unimplemented)
        }
        ty::PredicateKind::ConstEquate(..) => {
            bug!("unexpected goal: {obligation:?}")
        }
    };

    FulfillmentError { obligation, code, root_obligation }
}

fn fulfillment_error_for_stalled<'tcx>(
    infcx: &InferCtxt<'tcx>,
    root_obligation: PredicateObligation<'tcx>,
) -> FulfillmentError<'tcx> {
    let (code, refine_obligation) = infcx.probe(|_| {
        match <&SolverDelegate<'tcx>>::from(infcx)
            .evaluate_root_goal(root_obligation.clone().into(), GenerateProofTree::No)
            .0
        {
            Ok((_, Certainty::Maybe(MaybeCause::Ambiguity))) => {
                (FulfillmentErrorCode::Ambiguity { overflow: None }, true)
            }
            Ok((_, Certainty::Maybe(MaybeCause::Overflow { suggest_increasing_limit }))) => (
                FulfillmentErrorCode::Ambiguity { overflow: Some(suggest_increasing_limit) },
                // Don't look into overflows because we treat overflows weirdly anyways.
                // We discard the inference constraints from overflowing goals, so
                // recomputing the goal again during `find_best_leaf_obligation` may apply
                // inference guidance that makes other goals go from ambig -> pass, for example.
                //
                // FIXME: We should probably just look into overflows here.
                false,
            ),
            Ok((_, Certainty::Yes)) => {
                bug!("did not expect successful goal when collecting ambiguity errors")
            }
            Err(_) => {
                bug!("did not expect selection error when collecting ambiguity errors")
            }
        }
    });

    FulfillmentError {
        obligation: if refine_obligation {
            find_best_leaf_obligation(infcx, &root_obligation, true)
        } else {
            root_obligation.clone()
        },
        code,
        root_obligation,
    }
}

fn fulfillment_error_for_overflow<'tcx>(
    infcx: &InferCtxt<'tcx>,
    root_obligation: PredicateObligation<'tcx>,
) -> FulfillmentError<'tcx> {
    FulfillmentError {
        obligation: find_best_leaf_obligation(infcx, &root_obligation, true),
        code: FulfillmentErrorCode::Ambiguity { overflow: Some(true) },
        root_obligation,
    }
}

fn find_best_leaf_obligation<'tcx>(
    infcx: &InferCtxt<'tcx>,
    obligation: &PredicateObligation<'tcx>,
    consider_ambiguities: bool,
) -> PredicateObligation<'tcx> {
    let obligation = infcx.resolve_vars_if_possible(obligation.clone());
    infcx
        .visit_proof_tree(obligation.clone().into(), &mut BestObligation {
            obligation: obligation.clone(),
            consider_ambiguities,
        })
        .break_value()
        .unwrap_or(obligation)
}

struct BestObligation<'tcx> {
    obligation: PredicateObligation<'tcx>,
    consider_ambiguities: bool,
}

impl<'tcx> BestObligation<'tcx> {
    fn with_derived_obligation(
        &mut self,
        derived_obligation: PredicateObligation<'tcx>,
        and_then: impl FnOnce(&mut Self) -> <Self as ProofTreeVisitor<'tcx>>::Result,
    ) -> <Self as ProofTreeVisitor<'tcx>>::Result {
        let old_obligation = std::mem::replace(&mut self.obligation, derived_obligation);
        let res = and_then(self);
        self.obligation = old_obligation;
        res
    }

    /// Filter out the candidates that aren't interesting to visit for the
    /// purposes of reporting errors. For ambiguities, we only consider
    /// candidates that may hold. For errors, we only consider candidates that
    /// *don't* hold and which have impl-where clauses that also don't hold.
    fn non_trivial_candidates<'a>(
        &self,
        goal: &'a inspect::InspectGoal<'a, 'tcx>,
    ) -> Vec<inspect::InspectCandidate<'a, 'tcx>> {
        let mut candidates = goal.candidates();
        match self.consider_ambiguities {
            true => {
                // If we have an ambiguous obligation, we must consider *all* candidates
                // that hold, or else we may guide inference causing other goals to go
                // from ambig -> pass/fail.
                candidates.retain(|candidate| candidate.result().is_ok());
            }
            false => {
                // If we have >1 candidate, one may still be due to "boring" reasons, like
                // an alias-relate that failed to hold when deeply evaluated. We really
                // don't care about reasons like this.
                if candidates.len() > 1 {
                    candidates.retain(|candidate| {
                        goal.infcx().probe(|_| {
                            candidate.instantiate_nested_goals(self.span()).iter().any(
                                |nested_goal| {
                                    matches!(
                                        nested_goal.source(),
                                        GoalSource::ImplWhereBound
                                            | GoalSource::InstantiateHigherRanked
                                    ) && match self.consider_ambiguities {
                                        true => {
                                            matches!(
                                                nested_goal.result(),
                                                Ok(Certainty::Maybe(MaybeCause::Ambiguity))
                                            )
                                        }
                                        false => matches!(nested_goal.result(), Err(_)),
                                    }
                                },
                            )
                        })
                    });
                }
            }
        }

        candidates
    }
}

impl<'tcx> ProofTreeVisitor<'tcx> for BestObligation<'tcx> {
    type Result = ControlFlow<PredicateObligation<'tcx>>;

    fn span(&self) -> rustc_span::Span {
        self.obligation.cause.span
    }

    fn visit_goal(&mut self, goal: &inspect::InspectGoal<'_, 'tcx>) -> Self::Result {
        let candidates = self.non_trivial_candidates(goal);
        let [candidate] = candidates.as_slice() else {
            return ControlFlow::Break(self.obligation.clone());
        };

        // Don't walk into impls that have `do_not_recommend`.
        if let inspect::ProbeKind::TraitCandidate {
            source: CandidateSource::Impl(impl_def_id),
            result: _,
        } = candidate.kind()
            && goal.infcx().tcx.do_not_recommend_impl(impl_def_id)
        {
            return ControlFlow::Break(self.obligation.clone());
        }

        let tcx = goal.infcx().tcx;
        // FIXME: Also, what about considering >1 layer up the stack? May be necessary
        // for normalizes-to.
        let pred_kind = goal.goal().predicate.kind();
        let child_mode = match pred_kind.skip_binder() {
            ty::PredicateKind::Clause(ty::ClauseKind::Trait(parent_trait_pred)) => {
                ChildMode::Trait(pred_kind.rebind(parent_trait_pred))
            }
            ty::PredicateKind::NormalizesTo(normalizes_to)
                if matches!(
                    normalizes_to.alias.kind(tcx),
                    ty::AliasTermKind::ProjectionTy | ty::AliasTermKind::ProjectionConst
                ) =>
            {
                ChildMode::Trait(pred_kind.rebind(ty::TraitPredicate {
                    trait_ref: normalizes_to.alias.trait_ref(tcx),
                    polarity: ty::PredicatePolarity::Positive,
                }))
            }
            ty::PredicateKind::Clause(
                ty::ClauseKind::WellFormed(_) | ty::ClauseKind::Projection(..),
            )
            | ty::PredicateKind::AliasRelate(..) => ChildMode::PassThrough,
            _ => {
                return ControlFlow::Break(self.obligation.clone());
            }
        };

        let mut impl_where_bound_count = 0;
        for nested_goal in candidate.instantiate_nested_goals(self.span()) {
            let make_obligation = |cause| Obligation {
                cause,
                param_env: nested_goal.goal().param_env,
                predicate: nested_goal.goal().predicate,
                recursion_depth: self.obligation.recursion_depth + 1,
            };

            let obligation;
            match (child_mode, nested_goal.source()) {
                (ChildMode::Trait(_), GoalSource::Misc) => {
                    continue;
                }
                (ChildMode::Trait(parent_trait_pred), GoalSource::ImplWhereBound) => {
                    obligation = make_obligation(derive_cause(
                        tcx,
                        candidate.kind(),
                        self.obligation.cause.clone(),
                        impl_where_bound_count,
                        parent_trait_pred,
                    ));
                    impl_where_bound_count += 1;
                }
                // Skip over a higher-ranked predicate.
                (_, GoalSource::InstantiateHigherRanked) => {
                    obligation = self.obligation.clone();
                }
                (ChildMode::PassThrough, _) => {
                    obligation = make_obligation(self.obligation.cause.clone());
                }
            }

            // Skip nested goals that aren't the *reason* for our goal's failure.
            match self.consider_ambiguities {
                true if matches!(
                    nested_goal.result(),
                    Ok(Certainty::Maybe(MaybeCause::Ambiguity))
                ) => {}
                false if matches!(nested_goal.result(), Err(_)) => {}
                _ => continue,
            }

            self.with_derived_obligation(obligation, |this| nested_goal.visit_with(this))?;
        }

        // alias-relate may fail because the lhs or rhs can't be normalized,
        // and therefore is treated as rigid.
        if let Some(ty::PredicateKind::AliasRelate(lhs, rhs, _)) = pred_kind.no_bound_vars() {
            if let Some(obligation) = goal
                .infcx()
                .visit_proof_tree_at_depth(
                    goal.goal().with(goal.infcx().tcx, ty::ClauseKind::WellFormed(lhs.into())),
                    goal.depth() + 1,
                    self,
                )
                .break_value()
            {
                return ControlFlow::Break(obligation);
            } else if let Some(obligation) = goal
                .infcx()
                .visit_proof_tree_at_depth(
                    goal.goal().with(goal.infcx().tcx, ty::ClauseKind::WellFormed(rhs.into())),
                    goal.depth() + 1,
                    self,
                )
                .break_value()
            {
                return ControlFlow::Break(obligation);
            }
        }

        ControlFlow::Break(self.obligation.clone())
    }
}

#[derive(Copy, Clone)]
enum ChildMode<'tcx> {
    // Try to derive an `ObligationCause::{ImplDerived,BuiltinDerived}`,
    // and skip all `GoalSource::Misc`, which represent useless obligations
    // such as alias-eq which may not hold.
    Trait(ty::PolyTraitPredicate<'tcx>),
    // Skip trying to derive an `ObligationCause` from this obligation, and
    // report *all* sub-obligations as if they came directly from the parent
    // obligation.
    PassThrough,
}

fn derive_cause<'tcx>(
    tcx: TyCtxt<'tcx>,
    candidate_kind: inspect::ProbeKind<TyCtxt<'tcx>>,
    mut cause: ObligationCause<'tcx>,
    idx: usize,
    parent_trait_pred: ty::PolyTraitPredicate<'tcx>,
) -> ObligationCause<'tcx> {
    match candidate_kind {
        inspect::ProbeKind::TraitCandidate {
            source: CandidateSource::Impl(impl_def_id),
            result: _,
        } => {
            if let Some((_, span)) =
                tcx.predicates_of(impl_def_id).instantiate_identity(tcx).iter().nth(idx)
            {
                cause = cause.derived_cause(parent_trait_pred, |derived| {
                    ObligationCauseCode::ImplDerived(Box::new(traits::ImplDerivedCause {
                        derived,
                        impl_or_alias_def_id: impl_def_id,
                        impl_def_predicate_index: Some(idx),
                        span,
                    }))
                })
            }
        }
        inspect::ProbeKind::TraitCandidate {
            source: CandidateSource::BuiltinImpl(..),
            result: _,
        } => {
            cause = cause.derived_cause(parent_trait_pred, ObligationCauseCode::BuiltinDerived);
        }
        _ => {}
    };
    cause
}
