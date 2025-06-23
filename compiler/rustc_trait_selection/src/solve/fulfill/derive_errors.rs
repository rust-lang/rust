use std::ops::ControlFlow;

use rustc_hir::LangItem;
use rustc_infer::infer::InferCtxt;
use rustc_infer::traits::solve::{CandidateSource, GoalSource, MaybeCause};
use rustc_infer::traits::{
    self, MismatchedProjectionTypes, Obligation, ObligationCause, ObligationCauseCode,
    PredicateObligation, SelectionError,
};
use rustc_middle::traits::query::NoSolution;
use rustc_middle::ty::error::{ExpectedFound, TypeError};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_middle::{bug, span_bug};
use rustc_next_trait_solver::solve::{GoalEvaluation, SolverDelegateEvalExt as _};
use tracing::{instrument, trace};

use crate::solve::delegate::SolverDelegate;
use crate::solve::inspect::{self, ProofTreeInferCtxtExt, ProofTreeVisitor};
use crate::solve::{Certainty, deeply_normalize_for_diagnostics};
use crate::traits::{FulfillmentError, FulfillmentErrorCode, wf};

pub(super) fn fulfillment_error_for_no_solution<'tcx>(
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
        ty::PredicateKind::Clause(ty::ClauseKind::ConstArgHasType(ct, expected_ty)) => {
            let ct_ty = match ct.kind() {
                ty::ConstKind::Unevaluated(uv) => {
                    infcx.tcx.type_of(uv.def).instantiate(infcx.tcx, uv.args)
                }
                ty::ConstKind::Param(param_ct) => param_ct.find_ty_from_env(obligation.param_env),
                ty::ConstKind::Value(cv) => cv.ty,
                kind => span_bug!(
                    obligation.cause.span,
                    "ConstArgHasWrongType failed but we don't know how to compute type for {kind:?}"
                ),
            };
            FulfillmentErrorCode::Select(SelectionError::ConstArgHasWrongType {
                ct,
                ct_ty,
                expected_ty,
            })
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
            let expected_found = ExpectedFound::new(a, b);
            FulfillmentErrorCode::Subtype(expected_found, TypeError::Sorts(expected_found))
        }
        ty::PredicateKind::Coerce(pred) => {
            let (a, b) = infcx.enter_forall_and_leak_universe(
                obligation.predicate.kind().rebind((pred.a, pred.b)),
            );
            let expected_found = ExpectedFound::new(b, a);
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

pub(super) fn fulfillment_error_for_stalled<'tcx>(
    infcx: &InferCtxt<'tcx>,
    root_obligation: PredicateObligation<'tcx>,
) -> FulfillmentError<'tcx> {
    let (code, refine_obligation) = infcx.probe(|_| {
        match <&SolverDelegate<'tcx>>::from(infcx).evaluate_root_goal(
            root_obligation.as_goal(),
            root_obligation.cause.span,
            None,
        ) {
            Ok(GoalEvaluation { certainty: Certainty::Maybe(MaybeCause::Ambiguity), .. }) => {
                (FulfillmentErrorCode::Ambiguity { overflow: None }, true)
            }
            Ok(GoalEvaluation {
                certainty:
                    Certainty::Maybe(MaybeCause::Overflow {
                        suggest_increasing_limit,
                        keep_constraints: _,
                    }),
                ..
            }) => (
                FulfillmentErrorCode::Ambiguity { overflow: Some(suggest_increasing_limit) },
                // Don't look into overflows because we treat overflows weirdly anyways.
                // We discard the inference constraints from overflowing goals, so
                // recomputing the goal again during `find_best_leaf_obligation` may apply
                // inference guidance that makes other goals go from ambig -> pass, for example.
                //
                // FIXME: We should probably just look into overflows here.
                false,
            ),
            Ok(GoalEvaluation { certainty: Certainty::Yes, .. }) => {
                span_bug!(
                    root_obligation.cause.span,
                    "did not expect successful goal when collecting ambiguity errors for `{:?}`",
                    infcx.resolve_vars_if_possible(root_obligation.predicate),
                )
            }
            Err(_) => {
                span_bug!(
                    root_obligation.cause.span,
                    "did not expect selection error when collecting ambiguity errors for `{:?}`",
                    infcx.resolve_vars_if_possible(root_obligation.predicate),
                )
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

pub(super) fn fulfillment_error_for_overflow<'tcx>(
    infcx: &InferCtxt<'tcx>,
    root_obligation: PredicateObligation<'tcx>,
) -> FulfillmentError<'tcx> {
    FulfillmentError {
        obligation: find_best_leaf_obligation(infcx, &root_obligation, true),
        code: FulfillmentErrorCode::Ambiguity { overflow: Some(true) },
        root_obligation,
    }
}

#[instrument(level = "debug", skip(infcx), ret)]
fn find_best_leaf_obligation<'tcx>(
    infcx: &InferCtxt<'tcx>,
    obligation: &PredicateObligation<'tcx>,
    consider_ambiguities: bool,
) -> PredicateObligation<'tcx> {
    let obligation = infcx.resolve_vars_if_possible(obligation.clone());
    // FIXME: we use a probe here as the `BestObligation` visitor does not
    // check whether it uses candidates which get shadowed by where-bounds.
    //
    // We should probably fix the visitor to not do so instead, as this also
    // means the leaf obligation may be incorrect.
    let obligation = infcx
        .fudge_inference_if_ok(|| {
            infcx
                .visit_proof_tree(
                    obligation.as_goal(),
                    &mut BestObligation { obligation: obligation.clone(), consider_ambiguities },
                )
                .break_value()
                .ok_or(())
        })
        .unwrap_or(obligation);
    deeply_normalize_for_diagnostics(infcx, obligation.param_env, obligation)
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
                // We always handle rigid alias candidates separately as we may not add them for
                // aliases whose trait bound doesn't hold.
                candidates.retain(|c| !matches!(c.kind(), inspect::ProbeKind::RigidAlias { .. }));
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
                                            | GoalSource::AliasBoundConstCondition
                                            | GoalSource::InstantiateHigherRanked
                                            | GoalSource::AliasWellFormed
                                    ) && nested_goal.result().is_err()
                                },
                            )
                        })
                    });
                }
            }
        }

        candidates
    }

    /// HACK: We walk the nested obligations for a well-formed arg manually,
    /// since there's nontrivial logic in `wf.rs` to set up an obligation cause.
    /// Ideally we'd be able to track this better.
    fn visit_well_formed_goal(
        &mut self,
        candidate: &inspect::InspectCandidate<'_, 'tcx>,
        term: ty::Term<'tcx>,
    ) -> ControlFlow<PredicateObligation<'tcx>> {
        let infcx = candidate.goal().infcx();
        let param_env = candidate.goal().goal().param_env;
        let body_id = self.obligation.cause.body_id;

        for obligation in wf::unnormalized_obligations(infcx, param_env, term, self.span(), body_id)
            .into_iter()
            .flatten()
        {
            let nested_goal = candidate.instantiate_proof_tree_for_nested_goal(
                GoalSource::Misc,
                obligation.as_goal(),
                self.span(),
            );
            // Skip nested goals that aren't the *reason* for our goal's failure.
            match (self.consider_ambiguities, nested_goal.result()) {
                (true, Ok(Certainty::Maybe(MaybeCause::Ambiguity))) | (false, Err(_)) => {}
                _ => continue,
            }

            self.with_derived_obligation(obligation, |this| nested_goal.visit_with(this))?;
        }

        ControlFlow::Break(self.obligation.clone())
    }

    /// If a normalization of an associated item or a trait goal fails without trying any
    /// candidates it's likely that normalizing its self type failed. We manually detect
    /// such cases here.
    fn detect_error_in_self_ty_normalization(
        &mut self,
        goal: &inspect::InspectGoal<'_, 'tcx>,
        self_ty: Ty<'tcx>,
    ) -> ControlFlow<PredicateObligation<'tcx>> {
        assert!(!self.consider_ambiguities);
        let tcx = goal.infcx().tcx;
        if let ty::Alias(..) = self_ty.kind() {
            let infer_term = goal.infcx().next_ty_var(self.obligation.cause.span);
            let pred = ty::PredicateKind::AliasRelate(
                self_ty.into(),
                infer_term.into(),
                ty::AliasRelationDirection::Equate,
            );
            let obligation =
                Obligation::new(tcx, self.obligation.cause.clone(), goal.goal().param_env, pred);
            self.with_derived_obligation(obligation, |this| {
                goal.infcx().visit_proof_tree_at_depth(
                    goal.goal().with(tcx, pred),
                    goal.depth() + 1,
                    this,
                )
            })
        } else {
            ControlFlow::Continue(())
        }
    }

    /// When a higher-ranked projection goal fails, check that the corresponding
    /// higher-ranked trait goal holds or not. This is because the process of
    /// instantiating and then re-canonicalizing the binder of the projection goal
    /// forces us to be unable to see that the leak check failed in the nested
    /// `NormalizesTo` goal, so we don't fall back to the rigid projection check
    /// that should catch when a projection goal fails due to an unsatisfied trait
    /// goal.
    fn detect_trait_error_in_higher_ranked_projection(
        &mut self,
        goal: &inspect::InspectGoal<'_, 'tcx>,
    ) -> ControlFlow<PredicateObligation<'tcx>> {
        let tcx = goal.infcx().tcx;
        if let Some(projection_clause) = goal.goal().predicate.as_projection_clause()
            && !projection_clause.bound_vars().is_empty()
        {
            let pred = projection_clause.map_bound(|proj| proj.projection_term.trait_ref(tcx));
            let obligation = Obligation::new(
                tcx,
                self.obligation.cause.clone(),
                goal.goal().param_env,
                deeply_normalize_for_diagnostics(goal.infcx(), goal.goal().param_env, pred),
            );
            self.with_derived_obligation(obligation, |this| {
                goal.infcx().visit_proof_tree_at_depth(
                    goal.goal().with(tcx, pred),
                    goal.depth() + 1,
                    this,
                )
            })
        } else {
            ControlFlow::Continue(())
        }
    }

    /// It is likely that `NormalizesTo` failed without any applicable candidates
    /// because the alias is not well-formed.
    ///
    /// As we only enter `RigidAlias` candidates if the trait bound of the associated type
    /// holds, we discard these candidates in `non_trivial_candidates` and always manually
    /// check this here.
    fn detect_non_well_formed_assoc_item(
        &mut self,
        goal: &inspect::InspectGoal<'_, 'tcx>,
        alias: ty::AliasTerm<'tcx>,
    ) -> ControlFlow<PredicateObligation<'tcx>> {
        let tcx = goal.infcx().tcx;
        let obligation = Obligation::new(
            tcx,
            self.obligation.cause.clone(),
            goal.goal().param_env,
            alias.trait_ref(tcx),
        );
        self.with_derived_obligation(obligation, |this| {
            goal.infcx().visit_proof_tree_at_depth(
                goal.goal().with(tcx, alias.trait_ref(tcx)),
                goal.depth() + 1,
                this,
            )
        })
    }

    /// If we have no candidates, then it's likely that there is a
    /// non-well-formed alias in the goal.
    fn detect_error_from_empty_candidates(
        &mut self,
        goal: &inspect::InspectGoal<'_, 'tcx>,
    ) -> ControlFlow<PredicateObligation<'tcx>> {
        let tcx = goal.infcx().tcx;
        let pred_kind = goal.goal().predicate.kind();

        match pred_kind.no_bound_vars() {
            Some(ty::PredicateKind::Clause(ty::ClauseKind::Trait(pred))) => {
                self.detect_error_in_self_ty_normalization(goal, pred.self_ty())?;
            }
            Some(ty::PredicateKind::NormalizesTo(pred))
                if let ty::AliasTermKind::ProjectionTy | ty::AliasTermKind::ProjectionConst =
                    pred.alias.kind(tcx) =>
            {
                self.detect_error_in_self_ty_normalization(goal, pred.alias.self_ty())?;
                self.detect_non_well_formed_assoc_item(goal, pred.alias)?;
            }
            Some(_) | None => {}
        }

        ControlFlow::Break(self.obligation.clone())
    }
}

impl<'tcx> ProofTreeVisitor<'tcx> for BestObligation<'tcx> {
    type Result = ControlFlow<PredicateObligation<'tcx>>;

    fn span(&self) -> rustc_span::Span {
        self.obligation.cause.span
    }

    #[instrument(level = "trace", skip(self, goal), fields(goal = ?goal.goal()))]
    fn visit_goal(&mut self, goal: &inspect::InspectGoal<'_, 'tcx>) -> Self::Result {
        let tcx = goal.infcx().tcx;
        // Skip goals that aren't the *reason* for our goal's failure.
        match (self.consider_ambiguities, goal.result()) {
            (true, Ok(Certainty::Maybe(MaybeCause::Ambiguity))) | (false, Err(_)) => {}
            _ => return ControlFlow::Continue(()),
        }

        let pred = goal.goal().predicate;

        let candidates = self.non_trivial_candidates(goal);
        let candidate = match candidates.as_slice() {
            [candidate] => candidate,
            [] => return self.detect_error_from_empty_candidates(goal),
            _ => return ControlFlow::Break(self.obligation.clone()),
        };

        // Don't walk into impls that have `do_not_recommend`.
        if let inspect::ProbeKind::TraitCandidate {
            source: CandidateSource::Impl(impl_def_id),
            result: _,
        } = candidate.kind()
            && tcx.do_not_recommend_impl(impl_def_id)
        {
            trace!("#[do_not_recommend] -> exit");
            return ControlFlow::Break(self.obligation.clone());
        }

        // FIXME: Also, what about considering >1 layer up the stack? May be necessary
        // for normalizes-to.
        let child_mode = match pred.kind().skip_binder() {
            ty::PredicateKind::Clause(ty::ClauseKind::Trait(trait_pred)) => {
                ChildMode::Trait(pred.kind().rebind(trait_pred))
            }
            ty::PredicateKind::Clause(ty::ClauseKind::HostEffect(host_pred)) => {
                ChildMode::Host(pred.kind().rebind(host_pred))
            }
            ty::PredicateKind::NormalizesTo(normalizes_to)
                if matches!(
                    normalizes_to.alias.kind(tcx),
                    ty::AliasTermKind::ProjectionTy | ty::AliasTermKind::ProjectionConst
                ) =>
            {
                ChildMode::Trait(pred.kind().rebind(ty::TraitPredicate {
                    trait_ref: normalizes_to.alias.trait_ref(tcx),
                    polarity: ty::PredicatePolarity::Positive,
                }))
            }
            ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(term)) => {
                return self.visit_well_formed_goal(candidate, term);
            }
            _ => ChildMode::PassThrough,
        };

        let nested_goals = candidate.instantiate_nested_goals(self.span());

        // If the candidate requires some `T: FnPtr` bound which does not hold should not be treated as
        // an actual candidate, instead we should treat them as if the impl was never considered to
        // have potentially applied. As if `impl<A, R> Trait for for<..> fn(..A) -> R` was written
        // instead of `impl<T: FnPtr> Trait for T`.
        //
        // We do this as a separate loop so that we do not choose to tell the user about some nested
        // goal before we encounter a `T: FnPtr` nested goal.
        for nested_goal in &nested_goals {
            if let Some(poly_trait_pred) = nested_goal.goal().predicate.as_trait_clause()
                && tcx.is_lang_item(poly_trait_pred.def_id(), LangItem::FnPtrTrait)
                && let Err(NoSolution) = nested_goal.result()
            {
                return ControlFlow::Break(self.obligation.clone());
            }
        }

        let mut impl_where_bound_count = 0;
        for nested_goal in nested_goals {
            trace!(nested_goal = ?(nested_goal.goal(), nested_goal.source(), nested_goal.result()));

            let nested_pred = nested_goal.goal().predicate;

            let make_obligation = |cause| Obligation {
                cause,
                param_env: nested_goal.goal().param_env,
                predicate: nested_pred,
                recursion_depth: self.obligation.recursion_depth + 1,
            };

            let obligation;
            match (child_mode, nested_goal.source()) {
                (
                    ChildMode::Trait(_) | ChildMode::Host(_),
                    GoalSource::Misc | GoalSource::TypeRelating | GoalSource::NormalizeGoal(_),
                ) => {
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
                (
                    ChildMode::Host(parent_host_pred),
                    GoalSource::ImplWhereBound | GoalSource::AliasBoundConstCondition,
                ) => {
                    obligation = make_obligation(derive_host_cause(
                        tcx,
                        candidate.kind(),
                        self.obligation.cause.clone(),
                        impl_where_bound_count,
                        parent_host_pred,
                    ));
                    impl_where_bound_count += 1;
                }
                // Skip over a higher-ranked predicate.
                (_, GoalSource::InstantiateHigherRanked) => {
                    obligation = self.obligation.clone();
                }
                (ChildMode::PassThrough, _)
                | (_, GoalSource::AliasWellFormed | GoalSource::AliasBoundConstCondition) => {
                    obligation = make_obligation(self.obligation.cause.clone());
                }
            }

            self.with_derived_obligation(obligation, |this| nested_goal.visit_with(this))?;
        }

        // alias-relate may fail because the lhs or rhs can't be normalized,
        // and therefore is treated as rigid.
        if let Some(ty::PredicateKind::AliasRelate(lhs, rhs, _)) = pred.kind().no_bound_vars() {
            goal.infcx().visit_proof_tree_at_depth(
                goal.goal().with(tcx, ty::ClauseKind::WellFormed(lhs.into())),
                goal.depth() + 1,
                self,
            )?;
            goal.infcx().visit_proof_tree_at_depth(
                goal.goal().with(tcx, ty::ClauseKind::WellFormed(rhs.into())),
                goal.depth() + 1,
                self,
            )?;
        }

        self.detect_trait_error_in_higher_ranked_projection(goal)?;

        ControlFlow::Break(self.obligation.clone())
    }
}

#[derive(Debug, Copy, Clone)]
enum ChildMode<'tcx> {
    // Try to derive an `ObligationCause::{ImplDerived,BuiltinDerived}`,
    // and skip all `GoalSource::Misc`, which represent useless obligations
    // such as alias-eq which may not hold.
    Trait(ty::PolyTraitPredicate<'tcx>),
    // Try to derive an `ObligationCause::{ImplDerived,BuiltinDerived}`,
    // and skip all `GoalSource::Misc`, which represent useless obligations
    // such as alias-eq which may not hold.
    Host(ty::Binder<'tcx, ty::HostEffectPredicate<'tcx>>),
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

fn derive_host_cause<'tcx>(
    tcx: TyCtxt<'tcx>,
    candidate_kind: inspect::ProbeKind<TyCtxt<'tcx>>,
    mut cause: ObligationCause<'tcx>,
    idx: usize,
    parent_host_pred: ty::Binder<'tcx, ty::HostEffectPredicate<'tcx>>,
) -> ObligationCause<'tcx> {
    match candidate_kind {
        inspect::ProbeKind::TraitCandidate {
            source: CandidateSource::Impl(impl_def_id),
            result: _,
        } => {
            if let Some((_, span)) = tcx
                .predicates_of(impl_def_id)
                .instantiate_identity(tcx)
                .into_iter()
                .chain(tcx.const_conditions(impl_def_id).instantiate_identity(tcx).into_iter().map(
                    |(trait_ref, span)| {
                        (
                            trait_ref.to_host_effect_clause(
                                tcx,
                                parent_host_pred.skip_binder().constness,
                            ),
                            span,
                        )
                    },
                ))
                .nth(idx)
            {
                cause =
                    cause.derived_host_cause(parent_host_pred, |derived| {
                        ObligationCauseCode::ImplDerivedHost(Box::new(
                            traits::ImplDerivedHostCause { derived, impl_def_id, span },
                        ))
                    })
            }
        }
        inspect::ProbeKind::TraitCandidate {
            source: CandidateSource::BuiltinImpl(..),
            result: _,
        } => {
            cause =
                cause.derived_host_cause(parent_host_pred, ObligationCauseCode::BuiltinDerivedHost);
        }
        _ => {}
    };
    cause
}
