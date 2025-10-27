pub(crate) use rustc_next_trait_solver::solve::inspect::*;

use rustc_ast_ir::try_visit;
use rustc_next_trait_solver::{
    canonical::instantiate_canonical_state,
    resolve::eager_resolve_vars,
    solve::{SolverDelegateEvalExt, inspect},
};
use rustc_type_ir::{
    VisitorResult,
    inherent::{IntoKind, Span as _},
    solve::{Certainty, GoalSource, MaybeCause, NoSolution},
};

use crate::next_solver::{
    DbInterner, GenericArg, GenericArgs, Goal, NormalizesTo, ParamEnv, Predicate, PredicateKind,
    QueryResult, SolverContext, Span, Term,
    fulfill::NextSolverError,
    infer::{
        InferCtxt,
        traits::{Obligation, ObligationCause},
    },
    obligation_ctxt::ObligationCtxt,
};

pub(crate) struct InspectConfig {
    pub(crate) max_depth: usize,
}

pub(crate) struct InspectGoal<'a, 'db> {
    infcx: &'a SolverContext<'db>,
    depth: usize,
    orig_values: Vec<GenericArg<'db>>,
    goal: Goal<'db, Predicate<'db>>,
    result: Result<Certainty, NoSolution>,
    final_revision: inspect::Probe<DbInterner<'db>>,
    normalizes_to_term_hack: Option<NormalizesToTermHack<'db>>,
    source: GoalSource,
}

impl<'a, 'db> std::fmt::Debug for InspectGoal<'a, 'db> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InspectGoal")
            .field("depth", &self.depth)
            .field("orig_values", &self.orig_values)
            .field("goal", &self.goal)
            .field("result", &self.result)
            .field("final_revision", &self.final_revision)
            .field("normalizes_to_term_hack", &self.normalizes_to_term_hack)
            .field("source", &self.source)
            .finish()
    }
}

impl<'a, 'db> std::fmt::Debug for InspectCandidate<'a, 'db> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InspectCandidate")
            .field("kind", &self.kind)
            .field("steps", &self.steps)
            .field("final_state", &self.final_state)
            .field("result", &self.result)
            .field("shallow_certainty", &self.shallow_certainty)
            .finish()
    }
}

/// The expected term of a `NormalizesTo` goal gets replaced
/// with an unconstrained inference variable when computing
/// `NormalizesTo` goals and we return the nested goals to the
/// caller, who also equates the actual term with the expected.
///
/// This is an implementation detail of the trait solver and
/// not something we want to leak to users. We therefore
/// treat `NormalizesTo` goals as if they apply the expected
/// type at the end of each candidate.
#[derive(Debug, Copy, Clone)]
struct NormalizesToTermHack<'db> {
    term: Term<'db>,
    unconstrained_term: Term<'db>,
}

impl<'db> NormalizesToTermHack<'db> {
    /// Relate the `term` with the new `unconstrained_term` created
    /// when computing the proof tree for this `NormalizesTo` goals.
    /// This handles nested obligations.
    fn constrain_and(
        &self,
        infcx: &InferCtxt<'db>,
        param_env: ParamEnv<'db>,
        f: impl FnOnce(&mut ObligationCtxt<'_, 'db>),
    ) -> Result<Certainty, NoSolution> {
        let mut ocx = ObligationCtxt::new(infcx);
        ocx.eq(&ObligationCause::dummy(), param_env, self.term, self.unconstrained_term)?;
        f(&mut ocx);
        let errors = ocx.evaluate_obligations_error_on_ambiguity();
        if errors.is_empty() {
            Ok(Certainty::Yes)
        } else if errors.iter().all(|e| !matches!(e, NextSolverError::TrueError(_))) {
            Ok(Certainty::AMBIGUOUS)
        } else {
            Err(NoSolution)
        }
    }
}

pub(crate) struct InspectCandidate<'a, 'db> {
    goal: &'a InspectGoal<'a, 'db>,
    kind: inspect::ProbeKind<DbInterner<'db>>,
    steps: Vec<&'a inspect::ProbeStep<DbInterner<'db>>>,
    final_state: inspect::CanonicalState<DbInterner<'db>, ()>,
    result: QueryResult<'db>,
    shallow_certainty: Certainty,
}

impl<'a, 'db> InspectCandidate<'a, 'db> {
    pub(crate) fn kind(&self) -> inspect::ProbeKind<DbInterner<'db>> {
        self.kind
    }

    pub(crate) fn result(&self) -> Result<Certainty, NoSolution> {
        self.result.map(|c| c.value.certainty)
    }

    pub(crate) fn goal(&self) -> &'a InspectGoal<'a, 'db> {
        self.goal
    }

    /// Certainty passed into `evaluate_added_goals_and_make_canonical_response`.
    ///
    /// If this certainty is `Yes`, then we must be confident that the candidate
    /// must hold iff it's nested goals hold. This is not true if the certainty is
    /// `Maybe(..)`, which suggests we forced ambiguity instead.
    ///
    /// This is *not* the certainty of the candidate's full nested evaluation, which
    /// can be accessed with [`Self::result`] instead.
    pub(crate) fn shallow_certainty(&self) -> Certainty {
        self.shallow_certainty
    }

    /// Visit all nested goals of this candidate without rolling
    /// back their inference constraints. This function modifies
    /// the state of the `infcx`.
    pub(crate) fn visit_nested_no_probe<V: ProofTreeVisitor<'db>>(
        &self,
        visitor: &mut V,
    ) -> V::Result {
        for goal in self.instantiate_nested_goals() {
            try_visit!(goal.visit_with(visitor));
        }

        V::Result::output()
    }

    /// Instantiate the nested goals for the candidate without rolling back their
    /// inference constraints. This function modifies the state of the `infcx`.
    ///
    /// See [`Self::instantiate_impl_args`] if you need the impl args too.
    pub(crate) fn instantiate_nested_goals(&self) -> Vec<InspectGoal<'a, 'db>> {
        let infcx = self.goal.infcx;
        let param_env = self.goal.goal.param_env;
        let mut orig_values = self.goal.orig_values.to_vec();

        let mut instantiated_goals = vec![];
        for step in &self.steps {
            match **step {
                inspect::ProbeStep::AddGoal(source, goal) => instantiated_goals.push((
                    source,
                    instantiate_canonical_state(
                        infcx,
                        Span::dummy(),
                        param_env,
                        &mut orig_values,
                        goal,
                    ),
                )),
                inspect::ProbeStep::RecordImplArgs { .. } => {}
                inspect::ProbeStep::MakeCanonicalResponse { .. }
                | inspect::ProbeStep::NestedProbe(_) => unreachable!(),
            }
        }

        let () = instantiate_canonical_state(
            infcx,
            Span::dummy(),
            param_env,
            &mut orig_values,
            self.final_state,
        );

        if let Some(term_hack) = &self.goal.normalizes_to_term_hack {
            // FIXME: We ignore the expected term of `NormalizesTo` goals
            // when computing the result of its candidates. This is
            // scuffed.
            let _ = term_hack.constrain_and(infcx, param_env, |_| {});
        }

        instantiated_goals
            .into_iter()
            .map(|(source, goal)| self.instantiate_proof_tree_for_nested_goal(source, goal))
            .collect()
    }

    /// Instantiate the args of an impl if this candidate came from a
    /// `CandidateSource::Impl`. This function modifies the state of the
    /// `infcx`.
    pub(crate) fn instantiate_impl_args(&self) -> GenericArgs<'db> {
        let infcx = self.goal.infcx;
        let param_env = self.goal.goal.param_env;
        let mut orig_values = self.goal.orig_values.to_vec();

        for step in &self.steps {
            match **step {
                inspect::ProbeStep::RecordImplArgs { impl_args } => {
                    let impl_args = instantiate_canonical_state(
                        infcx,
                        Span::dummy(),
                        param_env,
                        &mut orig_values,
                        impl_args,
                    );

                    let () = instantiate_canonical_state(
                        infcx,
                        Span::dummy(),
                        param_env,
                        &mut orig_values,
                        self.final_state,
                    );

                    // No reason we couldn't support this, but we don't need to for select.
                    assert!(
                        self.goal.normalizes_to_term_hack.is_none(),
                        "cannot use `instantiate_impl_args` with a `NormalizesTo` goal"
                    );

                    return eager_resolve_vars(infcx, impl_args);
                }
                inspect::ProbeStep::AddGoal(..) => {}
                inspect::ProbeStep::MakeCanonicalResponse { .. }
                | inspect::ProbeStep::NestedProbe(_) => unreachable!(),
            }
        }

        panic!("expected impl args probe step for `instantiate_impl_args`");
    }

    pub(crate) fn instantiate_proof_tree_for_nested_goal(
        &self,
        source: GoalSource,
        goal: Goal<'db, Predicate<'db>>,
    ) -> InspectGoal<'a, 'db> {
        let infcx = self.goal.infcx;
        match goal.predicate.kind().no_bound_vars() {
            Some(PredicateKind::NormalizesTo(NormalizesTo { alias, term })) => {
                let unconstrained_term = infcx.next_term_var_of_kind(term);
                let goal =
                    goal.with(infcx.interner, NormalizesTo { alias, term: unconstrained_term });
                // We have to use a `probe` here as evaluating a `NormalizesTo` can constrain the
                // expected term. This means that candidates which only fail due to nested goals
                // and which normalize to a different term then the final result could ICE: when
                // building their proof tree, the expected term was unconstrained, but when
                // instantiating the candidate it is already constrained to the result of another
                // candidate.
                let normalizes_to_term_hack = NormalizesToTermHack { term, unconstrained_term };
                let (proof_tree, nested_goals_result) = infcx.probe(|_| {
                    // Here, if we have any nested goals, then we make sure to apply them
                    // considering the constrained RHS, and pass the resulting certainty to
                    // `InspectGoal::new` so that the goal has the right result (and maintains
                    // the impression that we don't do this normalizes-to infer hack at all).
                    let (nested, proof_tree) =
                        infcx.evaluate_root_goal_for_proof_tree(goal, Span::dummy());
                    let nested_goals_result = nested.and_then(|nested| {
                        normalizes_to_term_hack.constrain_and(
                            infcx,
                            proof_tree.uncanonicalized_goal.param_env,
                            |ocx| {
                                ocx.register_obligations(nested.0.into_iter().map(|(_, goal)| {
                                    Obligation::new(
                                        infcx.interner,
                                        ObligationCause::dummy(),
                                        goal.param_env,
                                        goal.predicate,
                                    )
                                }));
                            },
                        )
                    });
                    (proof_tree, nested_goals_result)
                });
                InspectGoal::new(
                    infcx,
                    self.goal.depth + 1,
                    proof_tree,
                    Some((normalizes_to_term_hack, nested_goals_result)),
                    source,
                )
            }
            _ => {
                // We're using a probe here as evaluating a goal could constrain
                // inference variables by choosing one candidate. If we then recurse
                // into another candidate who ends up with different inference
                // constraints, we get an ICE if we already applied the constraints
                // from the chosen candidate.
                let proof_tree =
                    infcx.probe(|_| infcx.evaluate_root_goal_for_proof_tree(goal, Span::dummy()).1);
                InspectGoal::new(infcx, self.goal.depth + 1, proof_tree, None, source)
            }
        }
    }

    /// Visit all nested goals of this candidate, rolling back
    /// all inference constraints.
    #[expect(dead_code, reason = "used in rustc")]
    pub(crate) fn visit_nested_in_probe<V: ProofTreeVisitor<'db>>(
        &self,
        visitor: &mut V,
    ) -> V::Result {
        self.goal.infcx.probe(|_| self.visit_nested_no_probe(visitor))
    }
}

impl<'a, 'db> InspectGoal<'a, 'db> {
    pub(crate) fn infcx(&self) -> &'a InferCtxt<'db> {
        self.infcx
    }

    pub(crate) fn goal(&self) -> Goal<'db, Predicate<'db>> {
        self.goal
    }

    pub(crate) fn result(&self) -> Result<Certainty, NoSolution> {
        self.result
    }

    pub(crate) fn source(&self) -> GoalSource {
        self.source
    }

    pub(crate) fn depth(&self) -> usize {
        self.depth
    }

    fn candidates_recur(
        &'a self,
        candidates: &mut Vec<InspectCandidate<'a, 'db>>,
        steps: &mut Vec<&'a inspect::ProbeStep<DbInterner<'db>>>,
        probe: &'a inspect::Probe<DbInterner<'db>>,
    ) {
        let mut shallow_certainty = None;
        for step in &probe.steps {
            match *step {
                inspect::ProbeStep::AddGoal(..) | inspect::ProbeStep::RecordImplArgs { .. } => {
                    steps.push(step)
                }
                inspect::ProbeStep::MakeCanonicalResponse { shallow_certainty: c } => {
                    assert!(matches!(
                        shallow_certainty.replace(c),
                        None | Some(Certainty::Maybe { cause: MaybeCause::Ambiguity, .. })
                    ));
                }
                inspect::ProbeStep::NestedProbe(ref probe) => {
                    match probe.kind {
                        // These never assemble candidates for the goal we're trying to solve.
                        inspect::ProbeKind::ProjectionCompatibility
                        | inspect::ProbeKind::ShadowedEnvProbing => continue,

                        inspect::ProbeKind::NormalizedSelfTyAssembly
                        | inspect::ProbeKind::UnsizeAssembly
                        | inspect::ProbeKind::Root { .. }
                        | inspect::ProbeKind::TraitCandidate { .. }
                        | inspect::ProbeKind::OpaqueTypeStorageLookup { .. }
                        | inspect::ProbeKind::RigidAlias { .. } => {
                            // Nested probes have to prove goals added in their parent
                            // but do not leak them, so we truncate the added goals
                            // afterwards.
                            let num_steps = steps.len();
                            self.candidates_recur(candidates, steps, probe);
                            steps.truncate(num_steps);
                        }
                    }
                }
            }
        }

        match probe.kind {
            inspect::ProbeKind::ProjectionCompatibility
            | inspect::ProbeKind::ShadowedEnvProbing => {
                panic!()
            }

            inspect::ProbeKind::NormalizedSelfTyAssembly | inspect::ProbeKind::UnsizeAssembly => {}

            // We add a candidate even for the root evaluation if there
            // is only one way to prove a given goal, e.g. for `WellFormed`.
            inspect::ProbeKind::Root { result }
            | inspect::ProbeKind::TraitCandidate { source: _, result }
            | inspect::ProbeKind::OpaqueTypeStorageLookup { result }
            | inspect::ProbeKind::RigidAlias { result } => {
                // We only add a candidate if `shallow_certainty` was set, which means
                // that we ended up calling `evaluate_added_goals_and_make_canonical_response`.
                if let Some(shallow_certainty) = shallow_certainty {
                    candidates.push(InspectCandidate {
                        goal: self,
                        kind: probe.kind,
                        steps: steps.clone(),
                        final_state: probe.final_state,
                        shallow_certainty,
                        result,
                    });
                }
            }
        }
    }

    pub(crate) fn candidates(&'a self) -> Vec<InspectCandidate<'a, 'db>> {
        let mut candidates = vec![];
        let mut nested_goals = vec![];
        self.candidates_recur(&mut candidates, &mut nested_goals, &self.final_revision);
        candidates
    }

    /// Returns the single candidate applicable for the current goal, if it exists.
    ///
    /// Returns `None` if there are either no or multiple applicable candidates.
    pub(crate) fn unique_applicable_candidate(&'a self) -> Option<InspectCandidate<'a, 'db>> {
        // FIXME(-Znext-solver): This does not handle impl candidates
        // hidden by env candidates.
        let mut candidates = self.candidates();
        candidates.retain(|c| c.result().is_ok());
        candidates.pop().filter(|_| candidates.is_empty())
    }

    fn new(
        infcx: &'a InferCtxt<'db>,
        depth: usize,
        root: inspect::GoalEvaluation<DbInterner<'db>>,
        term_hack_and_nested_certainty: Option<(
            NormalizesToTermHack<'db>,
            Result<Certainty, NoSolution>,
        )>,
        source: GoalSource,
    ) -> Self {
        let infcx = <&SolverContext<'db>>::from(infcx);

        let inspect::GoalEvaluation { uncanonicalized_goal, orig_values, final_revision, result } =
            root;
        // If there's a normalizes-to goal, AND the evaluation result with the result of
        // constraining the normalizes-to RHS and computing the nested goals.
        let result = result.and_then(|ok| {
            let nested_goals_certainty =
                term_hack_and_nested_certainty.map_or(Ok(Certainty::Yes), |(_, c)| c)?;
            Ok(ok.value.certainty.and(nested_goals_certainty))
        });

        InspectGoal {
            infcx,
            depth,
            orig_values,
            goal: eager_resolve_vars(infcx, uncanonicalized_goal),
            result,
            final_revision,
            normalizes_to_term_hack: term_hack_and_nested_certainty.map(|(n, _)| n),
            source,
        }
    }

    pub(crate) fn visit_with<V: ProofTreeVisitor<'db>>(&self, visitor: &mut V) -> V::Result {
        if self.depth < visitor.config().max_depth {
            try_visit!(visitor.visit_goal(self));
        }

        V::Result::output()
    }
}

/// The public API to interact with proof trees.
pub(crate) trait ProofTreeVisitor<'db> {
    type Result: VisitorResult;

    fn config(&self) -> InspectConfig {
        InspectConfig { max_depth: 10 }
    }

    fn visit_goal(&mut self, goal: &InspectGoal<'_, 'db>) -> Self::Result;
}

impl<'db> InferCtxt<'db> {
    pub(crate) fn visit_proof_tree<V: ProofTreeVisitor<'db>>(
        &self,
        goal: Goal<'db, Predicate<'db>>,
        visitor: &mut V,
    ) -> V::Result {
        self.visit_proof_tree_at_depth(goal, 0, visitor)
    }

    pub(crate) fn visit_proof_tree_at_depth<V: ProofTreeVisitor<'db>>(
        &self,
        goal: Goal<'db, Predicate<'db>>,
        depth: usize,
        visitor: &mut V,
    ) -> V::Result {
        let (_, proof_tree) = <&SolverContext<'db>>::from(self)
            .evaluate_root_goal_for_proof_tree(goal, Span::dummy());
        visitor.visit_goal(&InspectGoal::new(self, depth, proof_tree, None, GoalSource::Misc))
    }
}
