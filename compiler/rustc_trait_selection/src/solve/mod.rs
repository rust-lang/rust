//! The new trait solver, currently still WIP.
//!
//! As a user of the trait system, you can use `TyCtxt::evaluate_goal` to
//! interact with this solver.
//!
//! For a high-level overview of how this solver works, check out the relevant
//! section of the rustc-dev-guide.
//!
//! FIXME(@lcnr): Write that section. If you read this before then ask me
//! about it on zulip.

// FIXME: Instead of using `infcx.canonicalize_query` we have to add a new routine which
// preserves universes and creates a unique var (in the highest universe) for each
// appearance of a region.

// FIXME: uses of `infcx.at` need to enable deferred projection equality once that's implemented.

use std::mem;

use rustc_hir::def_id::DefId;
use rustc_infer::infer::canonical::{Canonical, CanonicalVarValues};
use rustc_infer::infer::{DefineOpaqueTypes, InferCtxt, InferOk, TyCtxtInferExt};
use rustc_infer::traits::query::NoSolution;
use rustc_middle::traits::solve::{
    CanonicalGoal, CanonicalResponse, Certainty, ExternalConstraints, ExternalConstraintsData,
    Goal, MaybeCause, QueryResult, Response,
};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_middle::ty::{
    CoercePredicate, RegionOutlivesPredicate, SubtypePredicate, TypeOutlivesPredicate,
};
use rustc_span::DUMMY_SP;

use crate::solve::search_graph::OverflowHandler;
use crate::traits::ObligationCause;

mod assembly;
mod canonical;
mod eval_ctxt;
mod fulfill;
mod project_goals;
mod search_graph;
mod trait_goals;

pub use eval_ctxt::EvalCtxt;
pub use fulfill::FulfillmentCtxt;

use self::eval_ctxt::{IsNormalizesToHack, NestedGoals};

trait CanonicalResponseExt {
    fn has_no_inference_or_external_constraints(&self) -> bool;
}

impl<'tcx> CanonicalResponseExt for Canonical<'tcx, Response<'tcx>> {
    fn has_no_inference_or_external_constraints(&self) -> bool {
        self.value.external_constraints.region_constraints.is_empty()
            && self.value.var_values.is_identity()
            && self.value.external_constraints.opaque_types.is_empty()
    }
}

pub trait InferCtxtEvalExt<'tcx> {
    /// Evaluates a goal from **outside** of the trait solver.
    ///
    /// Using this while inside of the solver is wrong as it uses a new
    /// search graph which would break cycle detection.
    fn evaluate_root_goal(
        &self,
        goal: Goal<'tcx, ty::Predicate<'tcx>>,
    ) -> Result<(bool, Certainty), NoSolution>;
}

impl<'tcx> InferCtxtEvalExt<'tcx> for InferCtxt<'tcx> {
    #[instrument(level = "debug", skip(self))]
    fn evaluate_root_goal(
        &self,
        goal: Goal<'tcx, ty::Predicate<'tcx>>,
    ) -> Result<(bool, Certainty), NoSolution> {
        let mut search_graph = search_graph::SearchGraph::new(self.tcx);

        let mut ecx = EvalCtxt {
            search_graph: &mut search_graph,
            infcx: self,
            // Only relevant when canonicalizing the response.
            max_input_universe: ty::UniverseIndex::ROOT,
            var_values: CanonicalVarValues::dummy(),
            nested_goals: NestedGoals::new(),
        };
        let result = ecx.evaluate_goal(IsNormalizesToHack::No, goal);

        assert!(
            ecx.nested_goals.is_empty(),
            "root `EvalCtxt` should not have any goals added to it"
        );

        assert!(search_graph.is_empty());
        result
    }
}

impl<'a, 'tcx> EvalCtxt<'a, 'tcx> {
    /// The entry point of the solver.
    ///
    /// This function deals with (coinductive) cycles, overflow, and caching
    /// and then calls [`EvalCtxt::compute_goal`] which contains the actual
    /// logic of the solver.
    ///
    /// Instead of calling this function directly, use either [EvalCtxt::evaluate_goal]
    /// if you're inside of the solver or [InferCtxtEvalExt::evaluate_root_goal] if you're
    /// outside of it.
    #[instrument(level = "debug", skip(tcx, search_graph), ret)]
    fn evaluate_canonical_goal(
        tcx: TyCtxt<'tcx>,
        search_graph: &'a mut search_graph::SearchGraph<'tcx>,
        canonical_goal: CanonicalGoal<'tcx>,
    ) -> QueryResult<'tcx> {
        // Deal with overflow, caching, and coinduction.
        //
        // The actual solver logic happens in `ecx.compute_goal`.
        search_graph.with_new_goal(tcx, canonical_goal, |search_graph| {
            let (ref infcx, goal, var_values) =
                tcx.infer_ctxt().build_with_canonical(DUMMY_SP, &canonical_goal);
            let mut ecx = EvalCtxt {
                infcx,
                var_values,
                max_input_universe: canonical_goal.max_universe,
                search_graph,
                nested_goals: NestedGoals::new(),
            };
            ecx.compute_goal(goal)
        })
    }

    /// Recursively evaluates `goal`, returning whether any inference vars have
    /// been constrained and the certainty of the result.
    fn evaluate_goal(
        &mut self,
        is_normalizes_to_hack: IsNormalizesToHack,
        goal: Goal<'tcx, ty::Predicate<'tcx>>,
    ) -> Result<(bool, Certainty), NoSolution> {
        let (orig_values, canonical_goal) = self.canonicalize_goal(goal);
        let canonical_response =
            EvalCtxt::evaluate_canonical_goal(self.tcx(), self.search_graph, canonical_goal)?;

        let has_changed = !canonical_response.value.var_values.is_identity();
        let certainty = self.instantiate_and_apply_query_response(
            goal.param_env,
            orig_values,
            canonical_response,
        )?;

        // Check that rerunning this query with its inference constraints applied
        // doesn't result in new inference constraints and has the same result.
        //
        // If we have projection goals like `<T as Trait>::Assoc == u32` we recursively
        // call `exists<U> <T as Trait>::Assoc == U` to enable better caching. This goal
        // could constrain `U` to `u32` which would cause this check to result in a
        // solver cycle.
        if cfg!(debug_assertions)
            && has_changed
            && is_normalizes_to_hack == IsNormalizesToHack::No
            && !self.search_graph.in_cycle()
        {
            debug!("rerunning goal to check result is stable");
            let (_orig_values, canonical_goal) = self.canonicalize_goal(goal);
            let canonical_response =
                EvalCtxt::evaluate_canonical_goal(self.tcx(), self.search_graph, canonical_goal)?;
            if !canonical_response.value.var_values.is_identity() {
                bug!("unstable result: {goal:?} {canonical_goal:?} {canonical_response:?}");
            }
            assert_eq!(certainty, canonical_response.value.certainty);
        }

        Ok((has_changed, certainty))
    }

    fn compute_goal(&mut self, goal: Goal<'tcx, ty::Predicate<'tcx>>) -> QueryResult<'tcx> {
        let Goal { param_env, predicate } = goal;
        let kind = predicate.kind();
        if let Some(kind) = kind.no_bound_vars() {
            match kind {
                ty::PredicateKind::Clause(ty::Clause::Trait(predicate)) => {
                    self.compute_trait_goal(Goal { param_env, predicate })
                }
                ty::PredicateKind::Clause(ty::Clause::Projection(predicate)) => {
                    self.compute_projection_goal(Goal { param_env, predicate })
                }
                ty::PredicateKind::Clause(ty::Clause::TypeOutlives(predicate)) => {
                    self.compute_type_outlives_goal(Goal { param_env, predicate })
                }
                ty::PredicateKind::Clause(ty::Clause::RegionOutlives(predicate)) => {
                    self.compute_region_outlives_goal(Goal { param_env, predicate })
                }
                ty::PredicateKind::Clause(ty::Clause::ConstArgHasType(ct, ty)) => {
                    self.compute_const_arg_has_type_goal(Goal { param_env, predicate: (ct, ty) })
                }
                ty::PredicateKind::Subtype(predicate) => {
                    self.compute_subtype_goal(Goal { param_env, predicate })
                }
                ty::PredicateKind::Coerce(predicate) => {
                    self.compute_coerce_goal(Goal { param_env, predicate })
                }
                ty::PredicateKind::ClosureKind(def_id, substs, kind) => self
                    .compute_closure_kind_goal(Goal {
                        param_env,
                        predicate: (def_id, substs, kind),
                    }),
                ty::PredicateKind::ObjectSafe(trait_def_id) => {
                    self.compute_object_safe_goal(trait_def_id)
                }
                ty::PredicateKind::WellFormed(arg) => {
                    self.compute_well_formed_goal(Goal { param_env, predicate: arg })
                }
                ty::PredicateKind::Ambiguous => {
                    self.evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS)
                }
                // FIXME: implement these predicates :)
                ty::PredicateKind::ConstEvaluatable(_) | ty::PredicateKind::ConstEquate(_, _) => {
                    self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
                }
                ty::PredicateKind::TypeWellFormedFromEnv(..) => {
                    bug!("TypeWellFormedFromEnv is only used for Chalk")
                }
                ty::PredicateKind::AliasEq(lhs, rhs) => {
                    self.compute_alias_eq_goal(Goal { param_env, predicate: (lhs, rhs) })
                }
            }
        } else {
            let kind = self.infcx.instantiate_binder_with_placeholders(kind);
            let goal = goal.with(self.tcx(), ty::Binder::dummy(kind));
            self.add_goal(goal);
            self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn compute_type_outlives_goal(
        &mut self,
        goal: Goal<'tcx, TypeOutlivesPredicate<'tcx>>,
    ) -> QueryResult<'tcx> {
        let ty::OutlivesPredicate(ty, lt) = goal.predicate;
        self.infcx.register_region_obligation_with_cause(ty, lt, &ObligationCause::dummy());
        self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
    }

    #[instrument(level = "debug", skip(self))]
    fn compute_region_outlives_goal(
        &mut self,
        goal: Goal<'tcx, RegionOutlivesPredicate<'tcx>>,
    ) -> QueryResult<'tcx> {
        self.infcx.region_outlives_predicate(
            &ObligationCause::dummy(),
            ty::Binder::dummy(goal.predicate),
        );
        self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
    }

    #[instrument(level = "debug", skip(self))]
    fn compute_coerce_goal(
        &mut self,
        goal: Goal<'tcx, CoercePredicate<'tcx>>,
    ) -> QueryResult<'tcx> {
        self.compute_subtype_goal(Goal {
            param_env: goal.param_env,
            predicate: SubtypePredicate {
                a_is_expected: false,
                a: goal.predicate.a,
                b: goal.predicate.b,
            },
        })
    }

    #[instrument(level = "debug", skip(self))]
    fn compute_subtype_goal(
        &mut self,
        goal: Goal<'tcx, SubtypePredicate<'tcx>>,
    ) -> QueryResult<'tcx> {
        if goal.predicate.a.is_ty_var() && goal.predicate.b.is_ty_var() {
            // FIXME: Do we want to register a subtype relation between these vars?
            // That won't actually reflect in the query response, so it seems moot.
            self.evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS)
        } else {
            let InferOk { value: (), obligations } = self
                .infcx
                .at(&ObligationCause::dummy(), goal.param_env)
                .sub(DefineOpaqueTypes::No, goal.predicate.a, goal.predicate.b)?;
            self.add_goals(obligations.into_iter().map(|pred| pred.into()));
            self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn compute_closure_kind_goal(
        &mut self,
        goal: Goal<'tcx, (DefId, ty::SubstsRef<'tcx>, ty::ClosureKind)>,
    ) -> QueryResult<'tcx> {
        let (_, substs, expected_kind) = goal.predicate;
        let found_kind = substs.as_closure().kind_ty().to_opt_closure_kind();

        let Some(found_kind) = found_kind else {
            return self.evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS);
        };
        if found_kind.extends(expected_kind) {
            self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        } else {
            Err(NoSolution)
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn compute_object_safe_goal(&mut self, trait_def_id: DefId) -> QueryResult<'tcx> {
        if self.tcx().check_is_object_safe(trait_def_id) {
            self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        } else {
            Err(NoSolution)
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn compute_well_formed_goal(
        &mut self,
        goal: Goal<'tcx, ty::GenericArg<'tcx>>,
    ) -> QueryResult<'tcx> {
        match crate::traits::wf::unnormalized_obligations(
            self.infcx,
            goal.param_env,
            goal.predicate,
        ) {
            Some(obligations) => {
                self.add_goals(obligations.into_iter().map(|o| o.into()));
                self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
            }
            None => self.evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS),
        }
    }

    #[instrument(level = "debug", skip(self), ret)]
    fn compute_alias_eq_goal(
        &mut self,
        goal: Goal<'tcx, (ty::Term<'tcx>, ty::Term<'tcx>)>,
    ) -> QueryResult<'tcx> {
        let tcx = self.tcx();

        let evaluate_normalizes_to = |ecx: &mut EvalCtxt<'_, 'tcx>, alias, other| {
            debug!("evaluate_normalizes_to(alias={:?}, other={:?})", alias, other);
            let r = ecx.probe(|ecx| {
                ecx.add_goal(goal.with(
                    tcx,
                    ty::Binder::dummy(ty::ProjectionPredicate {
                        projection_ty: alias,
                        term: other,
                    }),
                ));
                ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
            });
            debug!("evaluate_normalizes_to(..) -> {:?}", r);
            r
        };

        if goal.predicate.0.is_infer() || goal.predicate.1.is_infer() {
            bug!(
                "`AliasEq` goal with an infer var on lhs or rhs which should have been instantiated"
            );
        }

        match (
            goal.predicate.0.to_alias_term_no_opaque(tcx),
            goal.predicate.1.to_alias_term_no_opaque(tcx),
        ) {
            (None, None) => bug!("`AliasEq` goal without an alias on either lhs or rhs"),
            (Some(alias), None) => evaluate_normalizes_to(self, alias, goal.predicate.1),
            (None, Some(alias)) => evaluate_normalizes_to(self, alias, goal.predicate.0),
            (Some(alias_lhs), Some(alias_rhs)) => {
                debug!("compute_alias_eq_goal: both sides are aliases");

                let mut candidates = Vec::with_capacity(3);

                // Evaluate all 3 potential candidates for the alias' being equal
                candidates.push(evaluate_normalizes_to(self, alias_lhs, goal.predicate.1));
                candidates.push(evaluate_normalizes_to(self, alias_rhs, goal.predicate.0));
                candidates.push(self.probe(|ecx| {
                    debug!("compute_alias_eq_goal: alias defids are equal, equating substs");
                    ecx.eq(goal.param_env, alias_lhs, alias_rhs)?;
                    ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
                }));

                debug!(?candidates);

                self.try_merge_responses(candidates.into_iter())
            }
        }
    }

    #[instrument(level = "debug", skip(self), ret)]
    fn compute_const_arg_has_type_goal(
        &mut self,
        goal: Goal<'tcx, (ty::Const<'tcx>, Ty<'tcx>)>,
    ) -> QueryResult<'tcx> {
        let (ct, ty) = goal.predicate;
        self.eq(goal.param_env, ct.ty(), ty)?;
        self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
    }
}

impl<'tcx> EvalCtxt<'_, 'tcx> {
    #[instrument(level = "debug", skip(self))]
    fn set_normalizes_to_hack_goal(&mut self, goal: Goal<'tcx, ty::ProjectionPredicate<'tcx>>) {
        assert!(
            self.nested_goals.normalizes_to_hack_goal.is_none(),
            "attempted to set the projection eq hack goal when one already exists"
        );
        self.nested_goals.normalizes_to_hack_goal = Some(goal);
    }

    #[instrument(level = "debug", skip(self))]
    fn add_goal(&mut self, goal: Goal<'tcx, ty::Predicate<'tcx>>) {
        self.nested_goals.goals.push(goal);
    }

    #[instrument(level = "debug", skip(self, goals))]
    fn add_goals(&mut self, goals: impl IntoIterator<Item = Goal<'tcx, ty::Predicate<'tcx>>>) {
        let current_len = self.nested_goals.goals.len();
        self.nested_goals.goals.extend(goals);
        debug!("added_goals={:?}", &self.nested_goals.goals[current_len..]);
    }

    // Recursively evaluates all the goals added to this `EvalCtxt` to completion, returning
    // the certainty of all the goals.
    #[instrument(level = "debug", skip(self))]
    fn try_evaluate_added_goals(&mut self) -> Result<Certainty, NoSolution> {
        let mut goals = core::mem::replace(&mut self.nested_goals, NestedGoals::new());
        let mut new_goals = NestedGoals::new();

        let response = self.repeat_while_none(
            |_| Ok(Certainty::Maybe(MaybeCause::Overflow)),
            |this| {
                let mut has_changed = Err(Certainty::Yes);

                if let Some(goal) = goals.normalizes_to_hack_goal.take() {
                    let (_, certainty) = match this.evaluate_goal(
                        IsNormalizesToHack::Yes,
                        goal.with(this.tcx(), ty::Binder::dummy(goal.predicate)),
                    ) {
                        Ok(r) => r,
                        Err(NoSolution) => return Some(Err(NoSolution)),
                    };

                    if goal.predicate.projection_ty
                        != this.resolve_vars_if_possible(goal.predicate.projection_ty)
                    {
                        has_changed = Ok(())
                    }

                    match certainty {
                        Certainty::Yes => {}
                        Certainty::Maybe(_) => {
                            let goal = this.resolve_vars_if_possible(goal);

                            // The rhs of this `normalizes-to` must always be an unconstrained infer var as it is
                            // the hack used by `normalizes-to` to ensure that every `normalizes-to` behaves the same
                            // regardless of the rhs.
                            //
                            // However it is important not to unconditionally replace the rhs with a new infer var
                            // as otherwise we may replace the original unconstrained infer var with a new infer var
                            // and never propagate any constraints on the new var back to the original var.
                            let term = this
                                .term_is_fully_unconstrained(goal)
                                .then_some(goal.predicate.term)
                                .unwrap_or_else(|| {
                                    this.next_term_infer_of_kind(goal.predicate.term)
                                });
                            let projection_pred = ty::ProjectionPredicate {
                                term,
                                projection_ty: goal.predicate.projection_ty,
                            };
                            new_goals.normalizes_to_hack_goal =
                                Some(goal.with(this.tcx(), projection_pred));

                            has_changed = has_changed.map_err(|c| c.unify_and(certainty));
                        }
                    }
                }

                for nested_goal in goals.goals.drain(..) {
                    let (changed, certainty) =
                        match this.evaluate_goal(IsNormalizesToHack::No, nested_goal) {
                            Ok(result) => result,
                            Err(NoSolution) => return Some(Err(NoSolution)),
                        };

                    if changed {
                        has_changed = Ok(());
                    }

                    match certainty {
                        Certainty::Yes => {}
                        Certainty::Maybe(_) => {
                            new_goals.goals.push(nested_goal);
                            has_changed = has_changed.map_err(|c| c.unify_and(certainty));
                        }
                    }
                }

                mem::swap(&mut new_goals, &mut goals);
                match has_changed {
                    Ok(()) => None,
                    Err(certainty) => Some(Ok(certainty)),
                }
            },
        );

        self.nested_goals = goals;
        response
    }

    fn try_merge_responses(
        &mut self,
        responses: impl Iterator<Item = QueryResult<'tcx>>,
    ) -> QueryResult<'tcx> {
        let candidates = responses.into_iter().flatten().collect::<Box<[_]>>();

        if candidates.is_empty() {
            return Err(NoSolution);
        }

        // FIXME(-Ztreat-solver=next): We should instead try to find a `Certainty::Yes` response with
        // a subset of the constraints that all the other responses have.
        let one = candidates[0];
        if candidates[1..].iter().all(|resp| resp == &one) {
            return Ok(one);
        }

        if let Some(response) = candidates.iter().find(|response| {
            response.value.certainty == Certainty::Yes
                && response.has_no_inference_or_external_constraints()
        }) {
            return Ok(*response);
        }

        let certainty = candidates.iter().fold(Certainty::AMBIGUOUS, |certainty, response| {
            certainty.unify_and(response.value.certainty)
        });
        // FIXME(-Ztrait-solver=next): We should take the intersection of the constraints on all the
        // responses and use that for the constraints of this ambiguous response.
        let response = self.evaluate_added_goals_and_make_canonical_response(certainty);
        if let Ok(response) = &response {
            assert!(response.has_no_inference_or_external_constraints());
        }

        response
    }
}

pub(super) fn response_no_constraints<'tcx>(
    tcx: TyCtxt<'tcx>,
    goal: Canonical<'tcx, impl Sized>,
    certainty: Certainty,
) -> QueryResult<'tcx> {
    Ok(Canonical {
        max_universe: goal.max_universe,
        variables: goal.variables,
        value: Response {
            var_values: CanonicalVarValues::make_identity(tcx, goal.variables),
            // FIXME: maybe we should store the "no response" version in tcx, like
            // we do for tcx.types and stuff.
            external_constraints: tcx.mk_external_constraints(ExternalConstraintsData::default()),
            certainty,
        },
    })
}
