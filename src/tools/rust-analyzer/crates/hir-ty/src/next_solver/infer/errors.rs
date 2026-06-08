use std::{fmt, ops::ControlFlow};

use hir_def::attrs::AttrFlags;
use rustc_next_trait_solver::solve::{GoalEvaluation, SolverDelegateEvalExt};
use rustc_type_ir::{
    AliasRelationDirection, AliasTermKind, PredicatePolarity,
    error::ExpectedFound,
    inherent::IntoKind as _,
    solve::{CandidateSource, Certainty, GoalSource, MaybeCause, MaybeInfo, NoSolution},
};
use tracing::{instrument, trace};

use crate::{
    Span,
    db::GeneralConstId,
    next_solver::{
        AliasTerm, AnyImplId, Binder, ClauseKind, Const, ConstKind, DbInterner,
        HostEffectPredicate, PolyTraitPredicate, PredicateKind, SolverContext, Term,
        TraitPredicate, Ty, TyKind, TypeError,
        fulfill::NextSolverError,
        infer::{
            InferCtxt,
            select::SelectionError,
            traits::{Obligation, ObligationCause, PredicateObligation, PredicateObligations},
        },
        inspect::{self, ProofTreeVisitor},
        normalize::deeply_normalize_for_diagnostics,
    },
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

#[derive(Clone)]
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

impl<'db> fmt::Debug for FulfillmentErrorCode<'db> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            FulfillmentErrorCode::Select(ref e) => write!(f, "{e:?}"),
            FulfillmentErrorCode::Project(ref e) => write!(f, "{e:?}"),
            FulfillmentErrorCode::Subtype(ref a, ref b) => {
                write!(f, "CodeSubtypeError({a:?}, {b:?})")
            }
            FulfillmentErrorCode::ConstEquate(ref a, ref b) => {
                write!(f, "CodeConstEquateError({a:?}, {b:?})")
            }
            FulfillmentErrorCode::Ambiguity { overflow: None } => write!(f, "Ambiguity"),
            FulfillmentErrorCode::Ambiguity { overflow: Some(suggest_increasing_limit) } => {
                write!(f, "Overflow({suggest_increasing_limit})")
            }
            FulfillmentErrorCode::Cycle(ref cycle) => write!(f, "Cycle({cycle:?})"),
        }
    }
}

#[derive(Clone)]
pub struct MismatchedProjectionTypes<'db> {
    pub err: TypeError<'db>,
}

impl<'db> fmt::Debug for MismatchedProjectionTypes<'db> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MismatchedProjectionTypes({:?})", self.err)
    }
}

impl<'db> NextSolverError<'db> {
    pub fn into_fulfillment_error(self, infcx: &InferCtxt<'db>) -> FulfillmentError<'db> {
        match self {
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

fn fulfillment_error_for_no_solution<'db>(
    infcx: &InferCtxt<'db>,
    root_obligation: PredicateObligation<'db>,
) -> FulfillmentError<'db> {
    let interner = infcx.interner;
    let db = interner.db;
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
                    let ct_ty = match uv.def.0 {
                        GeneralConstId::ConstId(konst) => db.value_ty(konst.into()).unwrap(),
                        GeneralConstId::StaticId(statik) => db.value_ty(statik.into()).unwrap(),
                        GeneralConstId::AnonConstId(konst) => konst.loc(db).ty.get(),
                    };
                    ct_ty.instantiate(interner, uv.args).skip_norm_wip()
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

fn fulfillment_error_for_stalled<'db>(
    infcx: &InferCtxt<'db>,
    root_obligation: PredicateObligation<'db>,
) -> FulfillmentError<'db> {
    let (code, refine_obligation) = infcx.probe(|_| {
        match <&SolverContext<'db>>::from(infcx).evaluate_root_goal(
            root_obligation.as_goal(),
            root_obligation.cause.span(),
            None,
        ) {
            Ok(GoalEvaluation {
                certainty: Certainty::Maybe(MaybeInfo { cause: MaybeCause::Ambiguity, .. }),
                ..
            }) => (FulfillmentErrorCode::Ambiguity { overflow: None }, true),
            Ok(GoalEvaluation {
                certainty:
                    Certainty::Maybe(MaybeInfo {
                        cause:
                            MaybeCause::Overflow { suggest_increasing_limit, keep_constraints: _ },
                        ..
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

fn fulfillment_error_for_overflow<'db>(
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
                // walk around the fact that the cause in `Obligation` is ignored by folders so that
                // we can properly fudge the infer vars in cause code.
                .map(|o| (o.cause, o))
        })
        .map(|(cause, o)| PredicateObligation { cause, ..o })
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
                            candidate.instantiate_nested_goals(self.span()).iter().any(
                                |nested_goal| {
                                    matches!(
                                        nested_goal.source(),
                                        GoalSource::ImplWhereBound
                                            | GoalSource::AliasBoundConstCondition
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
        candidate: &inspect::InspectCandidate<'_, 'db>,
        term: Term<'db>,
    ) -> ControlFlow<PredicateObligation<'db>> {
        let _ = (candidate, term);
        // FIXME: rustc does this, but we don't process WF obligations yet:
        // let infcx = candidate.goal().infcx();
        // let param_env = candidate.goal().goal().param_env;
        // let body_id = self.obligation.cause.body_id;

        // for obligation in wf::unnormalized_obligations(infcx, param_env, term, self.span(), body_id)
        //     .into_iter()
        //     .flatten()
        // {
        //     let nested_goal = candidate.instantiate_proof_tree_for_nested_goal(
        //         GoalSource::Misc,
        //         obligation.as_goal(),
        //         self.span(),
        //     );
        //     // Skip nested goals that aren't the *reason* for our goal's failure.
        //     match (self.consider_ambiguities, nested_goal.result()) {
        //         (true, Ok(Certainty::Maybe { cause: MaybeCause::Ambiguity, .. }))
        //         | (false, Err(_)) => {}
        //         _ => continue,
        //     }

        //     self.with_derived_obligation(obligation, |this| nested_goal.visit_with(this))?;
        // }

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
            let infer_term = goal.infcx().next_ty_var(self.obligation.cause.span());
            let pred = PredicateKind::AliasRelate(
                self_ty.into(),
                infer_term.into(),
                AliasRelationDirection::Equate,
            );
            let obligation =
                Obligation::new(interner, self.obligation.cause, goal.goal().param_env, pred);
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
                self.obligation.cause,
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
            self.obligation.cause,
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
            Some(PredicateKind::NormalizesTo(pred))
                if let AliasTermKind::ProjectionTy { .. }
                | AliasTermKind::ProjectionConst { .. } = pred.alias.kind(interner) =>
            {
                self.detect_error_in_self_ty_normalization(goal, pred.alias.self_ty())?;
                self.detect_non_well_formed_assoc_item(goal, pred.alias)?;
            }
            Some(_) | None => {}
        }

        ControlFlow::Break(self.obligation.clone())
    }
}

impl<'db> ProofTreeVisitor<'db> for BestObligation<'db> {
    type Result = ControlFlow<PredicateObligation<'db>>;

    fn span(&self) -> Span {
        self.obligation.cause.span()
    }

    #[instrument(level = "trace", skip(self, goal), fields(goal = ?goal.goal()))]
    fn visit_goal(&mut self, goal: &inspect::InspectGoal<'_, 'db>) -> Self::Result {
        let interner = goal.infcx().interner;
        // Skip goals that aren't the *reason* for our goal's failure.
        match (self.consider_ambiguities, goal.result()) {
            (true, Ok(Certainty::Maybe(MaybeInfo { cause: MaybeCause::Ambiguity, .. })))
            | (false, Err(_)) => {}
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
            && let AnyImplId::ImplId(impl_def_id) = impl_def_id
            && AttrFlags::query(interner.db, impl_def_id.into())
                .contains(AttrFlags::DIAGNOSTIC_DO_NOT_RECOMMEND)
        {
            trace!("#[diagnostic::do_not_recommend] -> exit");
            return ControlFlow::Break(self.obligation.clone());
        }

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
                    AliasTermKind::ProjectionTy { .. } | AliasTermKind::ProjectionConst { .. }
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
                && Some(poly_trait_pred.def_id().0) == interner.lang_items().FnPtrTrait
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
                        interner,
                        candidate.kind(),
                        self.obligation.cause,
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
                        interner,
                        candidate.kind(),
                        self.obligation.cause,
                        impl_where_bound_count,
                        parent_host_pred,
                    ));
                    impl_where_bound_count += 1;
                }
                (ChildMode::PassThrough, _)
                | (_, GoalSource::AliasWellFormed | GoalSource::AliasBoundConstCondition) => {
                    obligation = make_obligation(self.obligation.cause);
                }
            }

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
    Host(Binder<'db, HostEffectPredicate<'db>>),
    // Skip trying to derive an `ObligationCause` from this obligation, and
    // report *all* sub-obligations as if they came directly from the parent
    // obligation.
    PassThrough,
}

fn derive_cause<'db>(
    _interner: DbInterner<'db>,
    _candidate_kind: inspect::ProbeKind<DbInterner<'db>>,
    cause: ObligationCause,
    _idx: usize,
    _parent_trait_pred: PolyTraitPredicate<'db>,
) -> ObligationCause {
    cause
}

fn derive_host_cause<'db>(
    _interner: DbInterner<'db>,
    _candidate_kind: inspect::ProbeKind<DbInterner<'db>>,
    cause: ObligationCause,
    _idx: usize,
    _parent_host_pred: Binder<'db, HostEffectPredicate<'db>>,
) -> ObligationCause {
    cause
}
