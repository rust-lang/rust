//! Trait solving error diagnosis and reporting.
//!
//! This code isn't used by rust-analyzer (it should, but then it'll probably be better to re-port it from rustc).
//! It's only there because without it, debugging trait solver errors is a nightmare.

use std::{fmt::Debug, ops::ControlFlow};

use rustc_next_trait_solver::solve::{GoalEvaluation, SolverDelegateEvalExt};
use rustc_type_ir::{
    AliasRelationDirection, AliasTermKind, HostEffectPredicate, Interner, PredicatePolarity,
    error::ExpectedFound,
    inherent::{IntoKind, SliceLike, Span as _},
    lang_items::SolverTraitLangItem,
    solve::{Certainty, GoalSource, MaybeCause, NoSolution},
};
use tracing::{instrument, trace};

use crate::next_solver::{
    AliasTerm, Binder, ClauseKind, Const, ConstKind, DbInterner, PolyTraitPredicate, PredicateKind,
    SolverContext, Span, Term, TraitPredicate, Ty, TyKind, TypeError,
    fulfill::NextSolverError,
    infer::{
        InferCtxt,
        select::SelectionError,
        traits::{Obligation, ObligationCause, PredicateObligation, PredicateObligations},
    },
    inspect::{self, ProofTreeVisitor},
    normalize::deeply_normalize_for_diagnostics,
};

#[derive(Debug)]
pub struct FulfillmentError<'db> {
    pub obligation: PredicateObligation<'db>,
    pub code: FulfillmentErrorCode<'db>,
    /// Diagnostics only: the 'root' obligation which resulted in
    /// the failure to process `obligation`. This is the obligation
    /// that was initially passed to `register_predicate_obligation`
    pub root_obligation: PredicateObligation<'db>,
}

impl<'db> FulfillmentError<'db> {
    pub fn new(
        obligation: PredicateObligation<'db>,
        code: FulfillmentErrorCode<'db>,
        root_obligation: PredicateObligation<'db>,
    ) -> FulfillmentError<'db> {
        FulfillmentError { obligation, code, root_obligation }
    }

    pub fn is_true_error(&self) -> bool {
        match self.code {
            FulfillmentErrorCode::Select(_)
            | FulfillmentErrorCode::Project(_)
            | FulfillmentErrorCode::Subtype(_, _)
            | FulfillmentErrorCode::ConstEquate(_, _) => true,
            FulfillmentErrorCode::Cycle(_) | FulfillmentErrorCode::Ambiguity { overflow: _ } => {
                false
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum FulfillmentErrorCode<'db> {
    /// Inherently impossible to fulfill; this trait is implemented if and only
    /// if it is already implemented.
    Cycle(PredicateObligations<'db>),
    Select(SelectionError<'db>),
    Project(MismatchedProjectionTypes<'db>),
    Subtype(ExpectedFound<Ty<'db>>, TypeError<'db>), // always comes from a SubtypePredicate
    ConstEquate(ExpectedFound<Const<'db>>, TypeError<'db>),
    Ambiguity {
        /// Overflow is only `Some(suggest_recursion_limit)` when using the next generation
        /// trait solver `-Znext-solver`. With the old solver overflow is eagerly handled by
        /// emitting a fatal error instead.
        overflow: Option<bool>,
    },
}

#[derive(Debug, Clone)]
pub struct MismatchedProjectionTypes<'db> {
    pub err: TypeError<'db>,
}

pub(super) fn fulfillment_error_for_no_solution<'db>(
    infcx: &InferCtxt<'db>,
    root_obligation: PredicateObligation<'db>,
) -> FulfillmentError<'db> {
    let obligation = find_best_leaf_obligation(infcx, &root_obligation, false);

    let code = match obligation.predicate.kind().skip_binder() {
        PredicateKind::Clause(ClauseKind::Projection(_)) => {
            FulfillmentErrorCode::Project(
                // FIXME: This could be a `Sorts` if the term is a type
                MismatchedProjectionTypes { err: TypeError::Mismatch },
            )
        }
        PredicateKind::Clause(ClauseKind::ConstArgHasType(ct, expected_ty)) => {
            let ct_ty = match ct.kind() {
                ConstKind::Unevaluated(uv) => {
                    infcx.interner.type_of(uv.def).instantiate(infcx.interner, uv.args)
                }
                ConstKind::Param(param_ct) => param_ct.find_const_ty_from_env(obligation.param_env),
                ConstKind::Value(cv) => cv.ty,
                kind => panic!(
                    "ConstArgHasWrongType failed but we don't know how to compute type for {kind:?}"
                ),
            };
            FulfillmentErrorCode::Select(SelectionError::ConstArgHasWrongType {
                ct,
                ct_ty,
                expected_ty,
            })
        }
        PredicateKind::NormalizesTo(..) => {
            FulfillmentErrorCode::Project(MismatchedProjectionTypes { err: TypeError::Mismatch })
        }
        PredicateKind::AliasRelate(_, _, _) => {
            FulfillmentErrorCode::Project(MismatchedProjectionTypes { err: TypeError::Mismatch })
        }
        PredicateKind::Subtype(pred) => {
            let (a, b) = infcx.enter_forall_and_leak_universe(
                obligation.predicate.kind().rebind((pred.a, pred.b)),
            );
            let expected_found = ExpectedFound::new(a, b);
            FulfillmentErrorCode::Subtype(expected_found, TypeError::Sorts(expected_found))
        }
        PredicateKind::Coerce(pred) => {
            let (a, b) = infcx.enter_forall_and_leak_universe(
                obligation.predicate.kind().rebind((pred.a, pred.b)),
            );
            let expected_found = ExpectedFound::new(b, a);
            FulfillmentErrorCode::Subtype(expected_found, TypeError::Sorts(expected_found))
        }
        PredicateKind::Clause(_) | PredicateKind::DynCompatible(_) | PredicateKind::Ambiguous => {
            FulfillmentErrorCode::Select(SelectionError::Unimplemented)
        }
        PredicateKind::ConstEquate(..) => {
            panic!("unexpected goal: {obligation:?}")
        }
    };

    FulfillmentError { obligation, code, root_obligation }
}

pub(super) fn fulfillment_error_for_stalled<'db>(
    infcx: &InferCtxt<'db>,
    root_obligation: PredicateObligation<'db>,
) -> FulfillmentError<'db> {
    let (code, refine_obligation) = infcx.probe(|_| {
        match <&SolverContext<'db>>::from(infcx).evaluate_root_goal(
            root_obligation.as_goal(),
            Span::dummy(),
            None,
        ) {
            Ok(GoalEvaluation {
                certainty: Certainty::Maybe { cause: MaybeCause::Ambiguity, .. },
                ..
            }) => (FulfillmentErrorCode::Ambiguity { overflow: None }, true),
            Ok(GoalEvaluation {
                certainty:
                    Certainty::Maybe {
                        cause:
                            MaybeCause::Overflow { suggest_increasing_limit, keep_constraints: _ },
                        ..
                    },
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
                panic!(
                    "did not expect successful goal when collecting ambiguity errors for `{:?}`",
                    infcx.resolve_vars_if_possible(root_obligation.predicate),
                )
            }
            Err(_) => {
                panic!(
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

pub(super) fn fulfillment_error_for_overflow<'db>(
    infcx: &InferCtxt<'db>,
    root_obligation: PredicateObligation<'db>,
) -> FulfillmentError<'db> {
    FulfillmentError {
        obligation: find_best_leaf_obligation(infcx, &root_obligation, true),
        code: FulfillmentErrorCode::Ambiguity { overflow: Some(true) },
        root_obligation,
    }
}

#[instrument(level = "debug", skip(infcx), ret)]
fn find_best_leaf_obligation<'db>(
    infcx: &InferCtxt<'db>,
    obligation: &PredicateObligation<'db>,
    consider_ambiguities: bool,
) -> PredicateObligation<'db> {
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

struct BestObligation<'db> {
    obligation: PredicateObligation<'db>,
    consider_ambiguities: bool,
}

impl<'db> BestObligation<'db> {
    fn with_derived_obligation(
        &mut self,
        derived_obligation: PredicateObligation<'db>,
        and_then: impl FnOnce(&mut Self) -> <Self as ProofTreeVisitor<'db>>::Result,
    ) -> <Self as ProofTreeVisitor<'db>>::Result {
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
        goal: &'a inspect::InspectGoal<'a, 'db>,
    ) -> Vec<inspect::InspectCandidate<'a, 'db>> {
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
                            candidate.instantiate_nested_goals().iter().any(|nested_goal| {
                                matches!(
                                    nested_goal.source(),
                                    GoalSource::ImplWhereBound
                                        | GoalSource::AliasBoundConstCondition
                                        | GoalSource::InstantiateHigherRanked
                                        | GoalSource::AliasWellFormed
                                ) && nested_goal.result().is_err()
                            })
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
        candidate: &inspect::InspectCandidate<'_, 'db>,
        term: Term<'db>,
    ) -> ControlFlow<PredicateObligation<'db>> {
        let infcx = candidate.goal().infcx();
        let param_env = candidate.goal().goal().param_env;

        for obligation in wf::unnormalized_obligations(infcx, param_env, term).into_iter().flatten()
        {
            let nested_goal = candidate
                .instantiate_proof_tree_for_nested_goal(GoalSource::Misc, obligation.as_goal());
            // Skip nested goals that aren't the *reason* for our goal's failure.
            match (self.consider_ambiguities, nested_goal.result()) {
                (true, Ok(Certainty::Maybe { cause: MaybeCause::Ambiguity, .. }))
                | (false, Err(_)) => {}
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
        goal: &inspect::InspectGoal<'_, 'db>,
        self_ty: Ty<'db>,
    ) -> ControlFlow<PredicateObligation<'db>> {
        assert!(!self.consider_ambiguities);
        let interner = goal.infcx().interner;
        if let TyKind::Alias(..) = self_ty.kind() {
            let infer_term = goal.infcx().next_ty_var();
            let pred = PredicateKind::AliasRelate(
                self_ty.into(),
                infer_term.into(),
                AliasRelationDirection::Equate,
            );
            let obligation = Obligation::new(
                interner,
                self.obligation.cause.clone(),
                goal.goal().param_env,
                pred,
            );
            self.with_derived_obligation(obligation, |this| {
                goal.infcx().visit_proof_tree_at_depth(
                    goal.goal().with(interner, pred),
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
        goal: &inspect::InspectGoal<'_, 'db>,
    ) -> ControlFlow<PredicateObligation<'db>> {
        let interner = goal.infcx().interner;
        if let Some(projection_clause) = goal.goal().predicate.as_projection_clause()
            && !projection_clause.bound_vars().is_empty()
        {
            let pred = projection_clause.map_bound(|proj| proj.projection_term.trait_ref(interner));
            let obligation = Obligation::new(
                interner,
                self.obligation.cause.clone(),
                goal.goal().param_env,
                deeply_normalize_for_diagnostics(goal.infcx(), goal.goal().param_env, pred),
            );
            self.with_derived_obligation(obligation, |this| {
                goal.infcx().visit_proof_tree_at_depth(
                    goal.goal().with(interner, pred),
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
        goal: &inspect::InspectGoal<'_, 'db>,
        alias: AliasTerm<'db>,
    ) -> ControlFlow<PredicateObligation<'db>> {
        let interner = goal.infcx().interner;
        let obligation = Obligation::new(
            interner,
            self.obligation.cause.clone(),
            goal.goal().param_env,
            alias.trait_ref(interner),
        );
        self.with_derived_obligation(obligation, |this| {
            goal.infcx().visit_proof_tree_at_depth(
                goal.goal().with(interner, alias.trait_ref(interner)),
                goal.depth() + 1,
                this,
            )
        })
    }

    /// If we have no candidates, then it's likely that there is a
    /// non-well-formed alias in the goal.
    fn detect_error_from_empty_candidates(
        &mut self,
        goal: &inspect::InspectGoal<'_, 'db>,
    ) -> ControlFlow<PredicateObligation<'db>> {
        let interner = goal.infcx().interner;
        let pred_kind = goal.goal().predicate.kind();

        match pred_kind.no_bound_vars() {
            Some(PredicateKind::Clause(ClauseKind::Trait(pred))) => {
                self.detect_error_in_self_ty_normalization(goal, pred.self_ty())?;
            }
            Some(PredicateKind::NormalizesTo(pred)) => {
                if let AliasTermKind::ProjectionTy | AliasTermKind::ProjectionConst =
                    pred.alias.kind(interner)
                {
                    self.detect_error_in_self_ty_normalization(goal, pred.alias.self_ty())?;
                    self.detect_non_well_formed_assoc_item(goal, pred.alias)?;
                }
            }
            Some(_) | None => {}
        }

        ControlFlow::Break(self.obligation.clone())
    }
}

impl<'db> ProofTreeVisitor<'db> for BestObligation<'db> {
    type Result = ControlFlow<PredicateObligation<'db>>;

    #[instrument(level = "trace", skip(self, goal), fields(goal = ?goal.goal()))]
    fn visit_goal(&mut self, goal: &inspect::InspectGoal<'_, 'db>) -> Self::Result {
        let interner = goal.infcx().interner;
        // Skip goals that aren't the *reason* for our goal's failure.
        match (self.consider_ambiguities, goal.result()) {
            (true, Ok(Certainty::Maybe { cause: MaybeCause::Ambiguity, .. })) | (false, Err(_)) => {
            }
            _ => return ControlFlow::Continue(()),
        }

        let pred = goal.goal().predicate;

        let candidates = self.non_trivial_candidates(goal);
        let candidate = match candidates.as_slice() {
            [candidate] => candidate,
            [] => return self.detect_error_from_empty_candidates(goal),
            _ => return ControlFlow::Break(self.obligation.clone()),
        };

        // // Don't walk into impls that have `do_not_recommend`.
        // if let inspect::ProbeKind::TraitCandidate {
        //     source: CandidateSource::Impl(impl_def_id),
        //     result: _,
        // } = candidate.kind()
        //     && interner.do_not_recommend_impl(impl_def_id)
        // {
        //     trace!("#[do_not_recommend] -> exit");
        //     return ControlFlow::Break(self.obligation.clone());
        // }

        // FIXME: Also, what about considering >1 layer up the stack? May be necessary
        // for normalizes-to.
        let child_mode = match pred.kind().skip_binder() {
            PredicateKind::Clause(ClauseKind::Trait(trait_pred)) => {
                ChildMode::Trait(pred.kind().rebind(trait_pred))
            }
            PredicateKind::Clause(ClauseKind::HostEffect(host_pred)) => {
                ChildMode::Host(pred.kind().rebind(host_pred))
            }
            PredicateKind::NormalizesTo(normalizes_to)
                if matches!(
                    normalizes_to.alias.kind(interner),
                    AliasTermKind::ProjectionTy | AliasTermKind::ProjectionConst
                ) =>
            {
                ChildMode::Trait(pred.kind().rebind(TraitPredicate {
                    trait_ref: normalizes_to.alias.trait_ref(interner),
                    polarity: PredicatePolarity::Positive,
                }))
            }
            PredicateKind::Clause(ClauseKind::WellFormed(term)) => {
                return self.visit_well_formed_goal(candidate, term);
            }
            _ => ChildMode::PassThrough,
        };

        let nested_goals = candidate.instantiate_nested_goals();

        // If the candidate requires some `T: FnPtr` bound which does not hold should not be treated as
        // an actual candidate, instead we should treat them as if the impl was never considered to
        // have potentially applied. As if `impl<A, R> Trait for for<..> fn(..A) -> R` was written
        // instead of `impl<T: FnPtr> Trait for T`.
        //
        // We do this as a separate loop so that we do not choose to tell the user about some nested
        // goal before we encounter a `T: FnPtr` nested goal.
        for nested_goal in &nested_goals {
            if let Some(poly_trait_pred) = nested_goal.goal().predicate.as_trait_clause()
                && interner
                    .is_trait_lang_item(poly_trait_pred.def_id(), SolverTraitLangItem::FnPtrTrait)
                && let Err(NoSolution) = nested_goal.result()
            {
                return ControlFlow::Break(self.obligation.clone());
            }
        }

        for nested_goal in nested_goals {
            trace!(nested_goal = ?(nested_goal.goal(), nested_goal.source(), nested_goal.result()));

            let nested_pred = nested_goal.goal().predicate;

            let make_obligation = || Obligation {
                cause: ObligationCause::dummy(),
                param_env: nested_goal.goal().param_env,
                predicate: nested_pred,
                recursion_depth: self.obligation.recursion_depth + 1,
            };

            let obligation = match (child_mode, nested_goal.source()) {
                (
                    ChildMode::Trait(_) | ChildMode::Host(_),
                    GoalSource::Misc | GoalSource::TypeRelating | GoalSource::NormalizeGoal(_),
                ) => {
                    continue;
                }
                (ChildMode::Trait(_parent_trait_pred), GoalSource::ImplWhereBound) => {
                    make_obligation()
                }
                (
                    ChildMode::Host(_parent_host_pred),
                    GoalSource::ImplWhereBound | GoalSource::AliasBoundConstCondition,
                ) => make_obligation(),
                // Skip over a higher-ranked predicate.
                (_, GoalSource::InstantiateHigherRanked) => self.obligation.clone(),
                (ChildMode::PassThrough, _)
                | (_, GoalSource::AliasWellFormed | GoalSource::AliasBoundConstCondition) => {
                    make_obligation()
                }
            };

            self.with_derived_obligation(obligation, |this| nested_goal.visit_with(this))?;
        }

        // alias-relate may fail because the lhs or rhs can't be normalized,
        // and therefore is treated as rigid.
        if let Some(PredicateKind::AliasRelate(lhs, rhs, _)) = pred.kind().no_bound_vars() {
            goal.infcx().visit_proof_tree_at_depth(
                goal.goal().with(interner, ClauseKind::WellFormed(lhs)),
                goal.depth() + 1,
                self,
            )?;
            goal.infcx().visit_proof_tree_at_depth(
                goal.goal().with(interner, ClauseKind::WellFormed(rhs)),
                goal.depth() + 1,
                self,
            )?;
        }

        self.detect_trait_error_in_higher_ranked_projection(goal)?;

        ControlFlow::Break(self.obligation.clone())
    }
}

#[derive(Debug, Copy, Clone)]
enum ChildMode<'db> {
    // Try to derive an `ObligationCause::{ImplDerived,BuiltinDerived}`,
    // and skip all `GoalSource::Misc`, which represent useless obligations
    // such as alias-eq which may not hold.
    Trait(PolyTraitPredicate<'db>),
    // Try to derive an `ObligationCause::{ImplDerived,BuiltinDerived}`,
    // and skip all `GoalSource::Misc`, which represent useless obligations
    // such as alias-eq which may not hold.
    Host(Binder<'db, HostEffectPredicate<DbInterner<'db>>>),
    // Skip trying to derive an `ObligationCause` from this obligation, and
    // report *all* sub-obligations as if they came directly from the parent
    // obligation.
    PassThrough,
}

impl<'db> NextSolverError<'db> {
    pub fn to_debuggable_error(&self, infcx: &InferCtxt<'db>) -> FulfillmentError<'db> {
        match self {
            NextSolverError::TrueError(obligation) => {
                fulfillment_error_for_no_solution(infcx, obligation.clone())
            }
            NextSolverError::Ambiguity(obligation) => {
                fulfillment_error_for_stalled(infcx, obligation.clone())
            }
            NextSolverError::Overflow(obligation) => {
                fulfillment_error_for_overflow(infcx, obligation.clone())
            }
        }
    }
}

mod wf {
    use hir_def::ItemContainerId;
    use rustc_type_ir::inherent::{
        AdtDef, BoundExistentialPredicates, GenericArgs as _, IntoKind, SliceLike, Term as _,
        Ty as _,
    };
    use rustc_type_ir::lang_items::SolverTraitLangItem;
    use rustc_type_ir::{
        Interner, TypeSuperVisitable, TypeVisitable, TypeVisitableExt, TypeVisitor,
    };
    use tracing::{debug, instrument};

    use crate::next_solver::infer::InferCtxt;
    use crate::next_solver::infer::traits::{Obligation, ObligationCause, PredicateObligations};
    use crate::next_solver::{
        Binder, ClauseKind, Const, ConstKind, Ctor, DbInterner, ExistentialPredicate, GenericArgs,
        ParamEnv, Predicate, PredicateKind, Region, SolverDefId, Term, TraitRef, Ty, TyKind,
    };

    /// Compute the predicates that are required for a type to be well-formed.
    ///
    /// This is only intended to be used in the new solver, since it does not
    /// take into account recursion depth or proper error-reporting spans.
    pub(crate) fn unnormalized_obligations<'db>(
        infcx: &InferCtxt<'db>,
        param_env: ParamEnv<'db>,
        term: Term<'db>,
    ) -> Option<PredicateObligations<'db>> {
        debug_assert_eq!(term, infcx.resolve_vars_if_possible(term));

        // However, if `arg` IS an unresolved inference variable, returns `None`,
        // because we are not able to make any progress at all. This is to prevent
        // cycles where we say "?0 is WF if ?0 is WF".
        if term.is_infer() {
            return None;
        }

        let mut wf =
            WfPredicates { infcx, param_env, out: PredicateObligations::new(), recursion_depth: 0 };
        wf.add_wf_preds_for_term(term);
        Some(wf.out)
    }

    struct WfPredicates<'a, 'db> {
        infcx: &'a InferCtxt<'db>,
        param_env: ParamEnv<'db>,
        out: PredicateObligations<'db>,
        recursion_depth: usize,
    }

    impl<'a, 'db> WfPredicates<'a, 'db> {
        fn interner(&self) -> DbInterner<'db> {
            self.infcx.interner
        }

        fn require_sized(&mut self, subty: Ty<'db>) {
            if !subty.has_escaping_bound_vars() {
                let cause = ObligationCause::new();
                let trait_ref = TraitRef::new(
                    self.interner(),
                    self.interner().require_trait_lang_item(SolverTraitLangItem::Sized),
                    [subty],
                );
                self.out.push(Obligation::with_depth(
                    self.interner(),
                    cause,
                    self.recursion_depth,
                    self.param_env,
                    Binder::dummy(trait_ref),
                ));
            }
        }

        /// Pushes all the predicates needed to validate that `term` is WF into `out`.
        #[instrument(level = "debug", skip(self))]
        fn add_wf_preds_for_term(&mut self, term: Term<'db>) {
            term.visit_with(self);
            debug!(?self.out);
        }

        #[instrument(level = "debug", skip(self))]
        fn nominal_obligations(
            &mut self,
            def_id: SolverDefId,
            args: GenericArgs<'db>,
        ) -> PredicateObligations<'db> {
            // PERF: `Sized`'s predicates include `MetaSized`, but both are compiler implemented marker
            // traits, so `MetaSized` will always be WF if `Sized` is WF and vice-versa. Determining
            // the nominal obligations of `Sized` would in-effect just elaborate `MetaSized` and make
            // the compiler do a bunch of work needlessly.
            if let SolverDefId::TraitId(def_id) = def_id
                && self.interner().is_trait_lang_item(def_id.into(), SolverTraitLangItem::Sized)
            {
                return Default::default();
            }

            self.interner()
                .predicates_of(def_id)
                .iter_instantiated(self.interner(), args)
                .map(|pred| {
                    let cause = ObligationCause::new();
                    Obligation::with_depth(
                        self.interner(),
                        cause,
                        self.recursion_depth,
                        self.param_env,
                        pred,
                    )
                })
                .filter(|pred| !pred.has_escaping_bound_vars())
                .collect()
        }

        fn add_wf_preds_for_dyn_ty(
            &mut self,
            _ty: Ty<'db>,
            data: &[Binder<'db, ExistentialPredicate<'db>>],
            region: Region<'db>,
        ) {
            // Imagine a type like this:
            //
            //     trait Foo { }
            //     trait Bar<'c> : 'c { }
            //
            //     &'b (Foo+'c+Bar<'d>)
            //         ^
            //
            // In this case, the following relationships must hold:
            //
            //     'b <= 'c
            //     'd <= 'c
            //
            // The first conditions is due to the normal region pointer
            // rules, which say that a reference cannot outlive its
            // referent.
            //
            // The final condition may be a bit surprising. In particular,
            // you may expect that it would have been `'c <= 'd`, since
            // usually lifetimes of outer things are conservative
            // approximations for inner things. However, it works somewhat
            // differently with trait objects: here the idea is that if the
            // user specifies a region bound (`'c`, in this case) it is the
            // "master bound" that *implies* that bounds from other traits are
            // all met. (Remember that *all bounds* in a type like
            // `Foo+Bar+Zed` must be met, not just one, hence if we write
            // `Foo<'x>+Bar<'y>`, we know that the type outlives *both* 'x and
            // 'y.)
            //
            // Note: in fact we only permit builtin traits, not `Bar<'d>`, I
            // am looking forward to the future here.
            if !data.has_escaping_bound_vars() && !region.has_escaping_bound_vars() {
                let implicit_bounds = object_region_bounds(self.interner(), data);

                let explicit_bound = region;

                self.out.reserve(implicit_bounds.len());
                for implicit_bound in implicit_bounds {
                    let cause = ObligationCause::new();
                    let outlives = Binder::dummy(rustc_type_ir::OutlivesPredicate(
                        explicit_bound,
                        implicit_bound,
                    ));
                    self.out.push(Obligation::with_depth(
                        self.interner(),
                        cause,
                        self.recursion_depth,
                        self.param_env,
                        outlives,
                    ));
                }

                // We don't add any wf predicates corresponding to the trait ref's generic arguments
                // which allows code like this to compile:
                // ```rust
                // trait Trait<T: Sized> {}
                // fn foo(_: &dyn Trait<[u32]>) {}
                // ```
            }
        }
    }

    impl<'a, 'db> TypeVisitor<DbInterner<'db>> for WfPredicates<'a, 'db> {
        type Result = ();

        fn visit_ty(&mut self, t: Ty<'db>) -> Self::Result {
            debug!("wf bounds for t={:?} t.kind={:#?}", t, t.kind());

            let tcx = self.interner();

            match t.kind() {
                TyKind::Bool
                | TyKind::Char
                | TyKind::Int(..)
                | TyKind::Uint(..)
                | TyKind::Float(..)
                | TyKind::Error(_)
                | TyKind::Str
                | TyKind::CoroutineWitness(..)
                | TyKind::Never
                | TyKind::Param(_)
                | TyKind::Bound(..)
                | TyKind::Placeholder(..)
                | TyKind::Foreign(..) => {
                    // WfScalar, WfParameter, etc
                }

                // Can only infer to `TyKind::Int(_) | TyKind::Uint(_)`.
                TyKind::Infer(rustc_type_ir::IntVar(_)) => {}

                // Can only infer to `TyKind::Float(_)`.
                TyKind::Infer(rustc_type_ir::FloatVar(_)) => {}

                TyKind::Slice(subty) => {
                    self.require_sized(subty);
                }

                TyKind::Array(subty, len) => {
                    self.require_sized(subty);
                    // Note that the len being WF is implicitly checked while visiting.
                    // Here we just check that it's of type usize.
                    let cause = ObligationCause::new();
                    self.out.push(Obligation::with_depth(
                        tcx,
                        cause,
                        self.recursion_depth,
                        self.param_env,
                        Binder::dummy(PredicateKind::Clause(ClauseKind::ConstArgHasType(
                            len,
                            Ty::new_unit(self.interner()),
                        ))),
                    ));
                }

                TyKind::Pat(base_ty, _pat) => {
                    self.require_sized(base_ty);
                }

                TyKind::Tuple(tys) => {
                    if let Some((_last, rest)) = tys.split_last() {
                        for &elem in rest {
                            self.require_sized(elem);
                        }
                    }
                }

                TyKind::RawPtr(_, _) => {
                    // Simple cases that are WF if their type args are WF.
                }

                TyKind::Alias(
                    rustc_type_ir::Projection | rustc_type_ir::Opaque | rustc_type_ir::Free,
                    data,
                ) => {
                    let obligations = self.nominal_obligations(data.def_id, data.args);
                    self.out.extend(obligations);
                }
                TyKind::Alias(rustc_type_ir::Inherent, _data) => {
                    return;
                }

                TyKind::Adt(def, args) => {
                    // WfNominalType
                    let obligations = self.nominal_obligations(def.def_id().0.into(), args);
                    self.out.extend(obligations);
                }

                TyKind::FnDef(did, args) => {
                    // HACK: Check the return type of function definitions for
                    // well-formedness to mostly fix #84533. This is still not
                    // perfect and there may be ways to abuse the fact that we
                    // ignore requirements with escaping bound vars. That's a
                    // more general issue however.
                    let fn_sig = tcx.fn_sig(did).instantiate(tcx, args);
                    fn_sig.output().skip_binder().visit_with(self);

                    let did = match did.0 {
                        hir_def::CallableDefId::FunctionId(id) => id.into(),
                        hir_def::CallableDefId::StructId(id) => SolverDefId::Ctor(Ctor::Struct(id)),
                        hir_def::CallableDefId::EnumVariantId(id) => {
                            SolverDefId::Ctor(Ctor::Enum(id))
                        }
                    };
                    let obligations = self.nominal_obligations(did, args);
                    self.out.extend(obligations);
                }

                TyKind::Ref(r, rty, _) => {
                    // WfReference
                    if !r.has_escaping_bound_vars() && !rty.has_escaping_bound_vars() {
                        let cause = ObligationCause::new();
                        self.out.push(Obligation::with_depth(
                            tcx,
                            cause,
                            self.recursion_depth,
                            self.param_env,
                            Binder::dummy(PredicateKind::Clause(ClauseKind::TypeOutlives(
                                rustc_type_ir::OutlivesPredicate(rty, r),
                            ))),
                        ));
                    }
                }

                TyKind::Coroutine(did, args, ..) => {
                    // Walk ALL the types in the coroutine: this will
                    // include the upvar types as well as the yield
                    // type. Note that this is mildly distinct from
                    // the closure case, where we have to be careful
                    // about the signature of the closure. We don't
                    // have the problem of implied bounds here since
                    // coroutines don't take arguments.
                    let obligations = self.nominal_obligations(did.0.into(), args);
                    self.out.extend(obligations);
                }

                TyKind::Closure(did, args) => {
                    // Note that we cannot skip the generic types
                    // types. Normally, within the fn
                    // body where they are created, the generics will
                    // always be WF, and outside of that fn body we
                    // are not directly inspecting closure types
                    // anyway, except via auto trait matching (which
                    // only inspects the upvar types).
                    // But when a closure is part of a type-alias-impl-trait
                    // then the function that created the defining site may
                    // have had more bounds available than the type alias
                    // specifies. This may cause us to have a closure in the
                    // hidden type that is not actually well formed and
                    // can cause compiler crashes when the user abuses unsafe
                    // code to procure such a closure.
                    // See tests/ui/type-alias-impl-trait/wf_check_closures.rs
                    let obligations = self.nominal_obligations(did.0.into(), args);
                    self.out.extend(obligations);
                    // Only check the upvar types for WF, not the rest
                    // of the types within. This is needed because we
                    // capture the signature and it may not be WF
                    // without the implied bounds. Consider a closure
                    // like `|x: &'a T|` -- it may be that `T: 'a` is
                    // not known to hold in the creator's context (and
                    // indeed the closure may not be invoked by its
                    // creator, but rather turned to someone who *can*
                    // verify that).
                    //
                    // The special treatment of closures here really
                    // ought not to be necessary either; the problem
                    // is related to #25860 -- there is no way for us
                    // to express a fn type complete with the implied
                    // bounds that it is assuming. I think in reality
                    // the WF rules around fn are a bit messed up, and
                    // that is the rot problem: `fn(&'a T)` should
                    // probably always be WF, because it should be
                    // shorthand for something like `where(T: 'a) {
                    // fn(&'a T) }`, as discussed in #25860.
                    let upvars = args.as_closure().tupled_upvars_ty();
                    return upvars.visit_with(self);
                }

                TyKind::CoroutineClosure(did, args) => {
                    // See the above comments. The same apply to coroutine-closures.
                    let obligations = self.nominal_obligations(did.0.into(), args);
                    self.out.extend(obligations);
                    let upvars = args.as_coroutine_closure().tupled_upvars_ty();
                    return upvars.visit_with(self);
                }

                TyKind::FnPtr(..) => {
                    // Let the visitor iterate into the argument/return
                    // types appearing in the fn signature.
                }
                TyKind::UnsafeBinder(_ty) => {}

                TyKind::Dynamic(data, r) => {
                    // WfObject
                    //
                    // Here, we defer WF checking due to higher-ranked
                    // regions. This is perhaps not ideal.
                    self.add_wf_preds_for_dyn_ty(t, data.as_slice(), r);

                    // FIXME(#27579) RFC also considers adding trait
                    // obligations that don't refer to Self and
                    // checking those
                    if let Some(principal) = data.principal_def_id() {
                        self.out.push(Obligation::with_depth(
                            tcx,
                            ObligationCause::new(),
                            self.recursion_depth,
                            self.param_env,
                            Binder::dummy(PredicateKind::DynCompatible(principal)),
                        ));
                    }
                }

                // Inference variables are the complicated case, since we don't
                // know what type they are. We do two things:
                //
                // 1. Check if they have been resolved, and if so proceed with
                //    THAT type.
                // 2. If not, we've at least simplified things (e.g., we went
                //    from `Vec?0>: WF` to `?0: WF`), so we can
                //    register a pending obligation and keep
                //    moving. (Goal is that an "inductive hypothesis"
                //    is satisfied to ensure termination.)
                // See also the comment on `fn obligations`, describing cycle
                // prevention, which happens before this can be reached.
                TyKind::Infer(_) => {
                    let cause = ObligationCause::new();
                    self.out.push(Obligation::with_depth(
                        tcx,
                        cause,
                        self.recursion_depth,
                        self.param_env,
                        Binder::dummy(PredicateKind::Clause(ClauseKind::WellFormed(t.into()))),
                    ));
                }
            }

            t.super_visit_with(self)
        }

        fn visit_const(&mut self, c: Const<'db>) -> Self::Result {
            let tcx = self.interner();

            match c.kind() {
                ConstKind::Unevaluated(uv) => {
                    if !c.has_escaping_bound_vars() {
                        let predicate =
                            Binder::dummy(PredicateKind::Clause(ClauseKind::ConstEvaluatable(c)));
                        let cause = ObligationCause::new();
                        self.out.push(Obligation::with_depth(
                            tcx,
                            cause,
                            self.recursion_depth,
                            self.param_env,
                            predicate,
                        ));

                        if let SolverDefId::ConstId(uv_def) = uv.def
                            && let ItemContainerId::ImplId(impl_) =
                                uv_def.loc(self.interner().db).container
                            && self.interner().db.impl_signature(impl_).target_trait.is_none()
                        {
                            return; // Subtree is handled by above function
                        } else {
                            let obligations = self.nominal_obligations(uv.def, uv.args);
                            self.out.extend(obligations);
                        }
                    }
                }
                ConstKind::Infer(_) => {
                    let cause = ObligationCause::new();

                    self.out.push(Obligation::with_depth(
                        tcx,
                        cause,
                        self.recursion_depth,
                        self.param_env,
                        Binder::dummy(PredicateKind::Clause(ClauseKind::WellFormed(c.into()))),
                    ));
                }
                ConstKind::Expr(_) => {
                    // FIXME(generic_const_exprs): this doesn't verify that given `Expr(N + 1)` the
                    // trait bound `typeof(N): Add<typeof(1)>` holds. This is currently unnecessary
                    // as `ConstKind::Expr` is only produced via normalization of `ConstKind::Unevaluated`
                    // which means that the `DefId` would have been typeck'd elsewhere. However in
                    // the future we may allow directly lowering to `ConstKind::Expr` in which case
                    // we would not be proving bounds we should.

                    let predicate =
                        Binder::dummy(PredicateKind::Clause(ClauseKind::ConstEvaluatable(c)));
                    let cause = ObligationCause::new();
                    self.out.push(Obligation::with_depth(
                        tcx,
                        cause,
                        self.recursion_depth,
                        self.param_env,
                        predicate,
                    ));
                }

                ConstKind::Error(_)
                | ConstKind::Param(_)
                | ConstKind::Bound(..)
                | ConstKind::Placeholder(..) => {
                    // These variants are trivially WF, so nothing to do here.
                }
                ConstKind::Value(..) => {
                    // FIXME: Enforce that values are structurally-matchable.
                }
            }

            c.super_visit_with(self)
        }

        fn visit_predicate(&mut self, _p: Predicate<'db>) -> Self::Result {
            panic!("predicate should not be checked for well-formedness");
        }
    }

    /// Given an object type like `SomeTrait + Send`, computes the lifetime
    /// bounds that must hold on the elided self type. These are derived
    /// from the declarations of `SomeTrait`, `Send`, and friends -- if
    /// they declare `trait SomeTrait : 'static`, for example, then
    /// `'static` would appear in the list.
    ///
    /// N.B., in some cases, particularly around higher-ranked bounds,
    /// this function returns a kind of conservative approximation.
    /// That is, all regions returned by this function are definitely
    /// required, but there may be other region bounds that are not
    /// returned, as well as requirements like `for<'a> T: 'a`.
    ///
    /// Requires that trait definitions have been processed so that we can
    /// elaborate predicates and walk supertraits.
    pub(crate) fn object_region_bounds<'db>(
        interner: DbInterner<'db>,
        existential_predicates: &[Binder<'db, ExistentialPredicate<'db>>],
    ) -> Vec<Region<'db>> {
        let erased_self_ty = Ty::new_unit(interner);

        let predicates = existential_predicates
            .iter()
            .map(|predicate| predicate.with_self_ty(interner, erased_self_ty));

        rustc_type_ir::elaborate::elaborate(interner, predicates)
            .filter_map(|pred| {
                debug!(?pred);
                match pred.kind().skip_binder() {
                    ClauseKind::TypeOutlives(rustc_type_ir::OutlivesPredicate(ref t, ref r)) => {
                        // Search for a bound of the form `erased_self_ty
                        // : 'a`, but be wary of something like `for<'a>
                        // erased_self_ty : 'a` (we interpret a
                        // higher-ranked bound like that as 'static,
                        // though at present the code in `fulfill.rs`
                        // considers such bounds to be unsatisfiable, so
                        // it's kind of a moot point since you could never
                        // construct such an object, but this seems
                        // correct even if that code changes).
                        if t == &erased_self_ty && !r.has_escaping_bound_vars() {
                            Some(*r)
                        } else {
                            None
                        }
                    }
                    ClauseKind::Trait(_)
                    | ClauseKind::HostEffect(..)
                    | ClauseKind::RegionOutlives(_)
                    | ClauseKind::Projection(_)
                    | ClauseKind::ConstArgHasType(_, _)
                    | ClauseKind::WellFormed(_)
                    | ClauseKind::UnstableFeature(_)
                    | ClauseKind::ConstEvaluatable(_) => None,
                }
            })
            .collect()
    }
}
