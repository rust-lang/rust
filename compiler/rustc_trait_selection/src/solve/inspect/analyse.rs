//! An infrastructure to mechanically analyse proof trees.
//!
//! It is unavoidable that this representation is somewhat
//! lossy as it should hide quite a few semantically relevant things,
//! e.g. canonicalization and the order of nested goals.
//!
//! @lcnr: However, a lot of the weirdness here is not strictly necessary
//! and could be improved in the future. This is mostly good enough for
//! coherence right now and was annoying to implement, so I am leaving it
//! as is until we start using it for something else.

use std::assert_matches::assert_matches;

use rustc_infer::infer::{DefineOpaqueTypes, InferCtxt, InferOk};
use rustc_macros::extension;
use rustc_middle::traits::ObligationCause;
use rustc_middle::traits::solve::{Certainty, Goal, GoalSource, NoSolution, QueryResult};
use rustc_middle::ty::{TyCtxt, VisitorResult, try_visit};
use rustc_middle::{bug, ty};
use rustc_next_trait_solver::resolve::eager_resolve_vars;
use rustc_next_trait_solver::solve::inspect::{self, instantiate_canonical_state};
use rustc_next_trait_solver::solve::{GenerateProofTree, MaybeCause, SolverDelegateEvalExt as _};
use rustc_span::{DUMMY_SP, Span};
use tracing::instrument;

use crate::solve::delegate::SolverDelegate;
use crate::traits::ObligationCtxt;

pub struct InspectConfig {
    pub max_depth: usize,
}

pub struct InspectGoal<'a, 'tcx> {
    infcx: &'a SolverDelegate<'tcx>,
    depth: usize,
    orig_values: Vec<ty::GenericArg<'tcx>>,
    goal: Goal<'tcx, ty::Predicate<'tcx>>,
    result: Result<Certainty, NoSolution>,
    evaluation_kind: inspect::CanonicalGoalEvaluationKind<TyCtxt<'tcx>>,
    normalizes_to_term_hack: Option<NormalizesToTermHack<'tcx>>,
    source: GoalSource,
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
#[derive(Copy, Clone)]
struct NormalizesToTermHack<'tcx> {
    term: ty::Term<'tcx>,
    unconstrained_term: ty::Term<'tcx>,
}

impl<'tcx> NormalizesToTermHack<'tcx> {
    /// Relate the `term` with the new `unconstrained_term` created
    /// when computing the proof tree for this `NormalizesTo` goals.
    /// This handles nested obligations.
    fn constrain(
        self,
        infcx: &InferCtxt<'tcx>,
        span: Span,
        param_env: ty::ParamEnv<'tcx>,
    ) -> Result<Certainty, NoSolution> {
        infcx
            .at(&ObligationCause::dummy_with_span(span), param_env)
            .eq(DefineOpaqueTypes::Yes, self.term, self.unconstrained_term)
            .map_err(|_| NoSolution)
            .and_then(|InferOk { value: (), obligations }| {
                let ocx = ObligationCtxt::new(infcx);
                ocx.register_obligations(obligations);
                let errors = ocx.select_all_or_error();
                if errors.is_empty() {
                    Ok(Certainty::Yes)
                } else if errors.iter().all(|e| !e.is_true_error()) {
                    Ok(Certainty::AMBIGUOUS)
                } else {
                    Err(NoSolution)
                }
            })
    }
}

pub struct InspectCandidate<'a, 'tcx> {
    goal: &'a InspectGoal<'a, 'tcx>,
    kind: inspect::ProbeKind<TyCtxt<'tcx>>,
    steps: Vec<&'a inspect::ProbeStep<TyCtxt<'tcx>>>,
    final_state: inspect::CanonicalState<TyCtxt<'tcx>, ()>,
    result: QueryResult<'tcx>,
    shallow_certainty: Certainty,
}

impl<'a, 'tcx> InspectCandidate<'a, 'tcx> {
    pub fn kind(&self) -> inspect::ProbeKind<TyCtxt<'tcx>> {
        self.kind
    }

    pub fn result(&self) -> Result<Certainty, NoSolution> {
        self.result.map(|c| c.value.certainty)
    }

    pub fn goal(&self) -> &'a InspectGoal<'a, 'tcx> {
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
    pub fn shallow_certainty(&self) -> Certainty {
        self.shallow_certainty
    }

    /// Visit all nested goals of this candidate without rolling
    /// back their inference constraints. This function modifies
    /// the state of the `infcx`.
    pub fn visit_nested_no_probe<V: ProofTreeVisitor<'tcx>>(&self, visitor: &mut V) -> V::Result {
        for goal in self.instantiate_nested_goals(visitor.span()) {
            try_visit!(goal.visit_with(visitor));
        }

        V::Result::output()
    }

    /// Instantiate the nested goals for the candidate without rolling back their
    /// inference constraints. This function modifies the state of the `infcx`.
    ///
    /// See [`Self::instantiate_impl_args`] if you need the impl args too.
    #[instrument(
        level = "debug",
        skip_all,
        fields(goal = ?self.goal.goal, steps = ?self.steps)
    )]
    pub fn instantiate_nested_goals(&self, span: Span) -> Vec<InspectGoal<'a, 'tcx>> {
        let infcx = self.goal.infcx;
        let param_env = self.goal.goal.param_env;
        let mut orig_values = self.goal.orig_values.to_vec();

        let mut instantiated_goals = vec![];
        for step in &self.steps {
            match **step {
                inspect::ProbeStep::AddGoal(source, goal) => instantiated_goals.push((
                    source,
                    instantiate_canonical_state(infcx, span, param_env, &mut orig_values, goal),
                )),
                inspect::ProbeStep::RecordImplArgs { .. } => {}
                inspect::ProbeStep::MakeCanonicalResponse { .. }
                | inspect::ProbeStep::NestedProbe(_) => unreachable!(),
            }
        }

        let () =
            instantiate_canonical_state(infcx, span, param_env, &mut orig_values, self.final_state);

        if let Some(term_hack) = self.goal.normalizes_to_term_hack {
            // FIXME: We ignore the expected term of `NormalizesTo` goals
            // when computing the result of its candidates. This is
            // scuffed.
            let _ = term_hack.constrain(infcx, span, param_env);
        }

        instantiated_goals
            .into_iter()
            .map(|(source, goal)| self.instantiate_proof_tree_for_nested_goal(source, goal, span))
            .collect()
    }

    /// Instantiate the args of an impl if this candidate came from a
    /// `CandidateSource::Impl`. This function modifies the state of the
    /// `infcx`.
    #[instrument(
        level = "debug",
        skip_all,
        fields(goal = ?self.goal.goal, steps = ?self.steps)
    )]
    pub fn instantiate_impl_args(&self, span: Span) -> ty::GenericArgsRef<'tcx> {
        let infcx = self.goal.infcx;
        let param_env = self.goal.goal.param_env;
        let mut orig_values = self.goal.orig_values.to_vec();

        for step in &self.steps {
            match **step {
                inspect::ProbeStep::RecordImplArgs { impl_args } => {
                    let impl_args = instantiate_canonical_state(
                        infcx,
                        span,
                        param_env,
                        &mut orig_values,
                        impl_args,
                    );

                    let () = instantiate_canonical_state(
                        infcx,
                        span,
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

        bug!("expected impl args probe step for `instantiate_impl_args`");
    }

    pub fn instantiate_proof_tree_for_nested_goal(
        &self,
        source: GoalSource,
        goal: Goal<'tcx, ty::Predicate<'tcx>>,
        span: Span,
    ) -> InspectGoal<'a, 'tcx> {
        let infcx = self.goal.infcx;
        match goal.predicate.kind().no_bound_vars() {
            Some(ty::PredicateKind::NormalizesTo(ty::NormalizesTo { alias, term })) => {
                let unconstrained_term = infcx.next_term_var_of_kind(term, span);
                let goal =
                    goal.with(infcx.tcx, ty::NormalizesTo { alias, term: unconstrained_term });
                // We have to use a `probe` here as evaluating a `NormalizesTo` can constrain the
                // expected term. This means that candidates which only fail due to nested goals
                // and which normalize to a different term then the final result could ICE: when
                // building their proof tree, the expected term was unconstrained, but when
                // instantiating the candidate it is already constrained to the result of another
                // candidate.
                let proof_tree = infcx
                    .probe(|_| infcx.evaluate_root_goal_raw(goal, GenerateProofTree::Yes, None).1);
                InspectGoal::new(
                    infcx,
                    self.goal.depth + 1,
                    proof_tree.unwrap(),
                    Some(NormalizesToTermHack { term, unconstrained_term }),
                    source,
                )
            }
            _ => {
                // We're using a probe here as evaluating a goal could constrain
                // inference variables by choosing one candidate. If we then recurse
                // into another candidate who ends up with different inference
                // constraints, we get an ICE if we already applied the constraints
                // from the chosen candidate.
                let proof_tree = infcx
                    .probe(|_| infcx.evaluate_root_goal(goal, GenerateProofTree::Yes, span, None).1)
                    .unwrap();
                InspectGoal::new(infcx, self.goal.depth + 1, proof_tree, None, source)
            }
        }
    }

    /// Visit all nested goals of this candidate, rolling back
    /// all inference constraints.
    pub fn visit_nested_in_probe<V: ProofTreeVisitor<'tcx>>(&self, visitor: &mut V) -> V::Result {
        self.goal.infcx.probe(|_| self.visit_nested_no_probe(visitor))
    }
}

impl<'a, 'tcx> InspectGoal<'a, 'tcx> {
    pub fn infcx(&self) -> &'a InferCtxt<'tcx> {
        self.infcx
    }

    pub fn goal(&self) -> Goal<'tcx, ty::Predicate<'tcx>> {
        self.goal
    }

    pub fn result(&self) -> Result<Certainty, NoSolution> {
        self.result
    }

    pub fn source(&self) -> GoalSource {
        self.source
    }

    pub fn depth(&self) -> usize {
        self.depth
    }

    fn candidates_recur(
        &'a self,
        candidates: &mut Vec<InspectCandidate<'a, 'tcx>>,
        steps: &mut Vec<&'a inspect::ProbeStep<TyCtxt<'tcx>>>,
        probe: &'a inspect::Probe<TyCtxt<'tcx>>,
    ) {
        let mut shallow_certainty = None;
        for step in &probe.steps {
            match *step {
                inspect::ProbeStep::AddGoal(..) | inspect::ProbeStep::RecordImplArgs { .. } => {
                    steps.push(step)
                }
                inspect::ProbeStep::MakeCanonicalResponse { shallow_certainty: c } => {
                    assert_matches!(
                        shallow_certainty.replace(c),
                        None | Some(Certainty::Maybe(MaybeCause::Ambiguity))
                    );
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
                bug!()
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

    pub fn candidates(&'a self) -> Vec<InspectCandidate<'a, 'tcx>> {
        let mut candidates = vec![];
        let last_eval_step = match &self.evaluation_kind {
            // An annoying edge case in case the recursion limit is 0.
            inspect::CanonicalGoalEvaluationKind::Overflow => return vec![],
            inspect::CanonicalGoalEvaluationKind::Evaluation { final_revision } => final_revision,
        };

        let mut nested_goals = vec![];
        self.candidates_recur(&mut candidates, &mut nested_goals, &last_eval_step);

        candidates
    }

    /// Returns the single candidate applicable for the current goal, if it exists.
    ///
    /// Returns `None` if there are either no or multiple applicable candidates.
    pub fn unique_applicable_candidate(&'a self) -> Option<InspectCandidate<'a, 'tcx>> {
        // FIXME(-Znext-solver): This does not handle impl candidates
        // hidden by env candidates.
        let mut candidates = self.candidates();
        candidates.retain(|c| c.result().is_ok());
        candidates.pop().filter(|_| candidates.is_empty())
    }

    fn new(
        infcx: &'a InferCtxt<'tcx>,
        depth: usize,
        root: inspect::GoalEvaluation<TyCtxt<'tcx>>,
        normalizes_to_term_hack: Option<NormalizesToTermHack<'tcx>>,
        source: GoalSource,
    ) -> Self {
        let infcx = <&SolverDelegate<'tcx>>::from(infcx);

        let inspect::GoalEvaluation { uncanonicalized_goal, orig_values, evaluation } = root;
        let result = evaluation.result.and_then(|ok| {
            if let Some(term_hack) = normalizes_to_term_hack {
                infcx
                    .probe(|_| term_hack.constrain(infcx, DUMMY_SP, uncanonicalized_goal.param_env))
                    .map(|certainty| ok.value.certainty.and(certainty))
            } else {
                Ok(ok.value.certainty)
            }
        });

        InspectGoal {
            infcx,
            depth,
            orig_values,
            goal: eager_resolve_vars(infcx, uncanonicalized_goal),
            result,
            evaluation_kind: evaluation.kind,
            normalizes_to_term_hack,
            source,
        }
    }

    pub(crate) fn visit_with<V: ProofTreeVisitor<'tcx>>(&self, visitor: &mut V) -> V::Result {
        if self.depth < visitor.config().max_depth {
            try_visit!(visitor.visit_goal(self));
        }

        V::Result::output()
    }
}

/// The public API to interact with proof trees.
pub trait ProofTreeVisitor<'tcx> {
    type Result: VisitorResult = ();

    fn span(&self) -> Span;

    fn config(&self) -> InspectConfig {
        InspectConfig { max_depth: 10 }
    }

    fn visit_goal(&mut self, goal: &InspectGoal<'_, 'tcx>) -> Self::Result;
}

#[extension(pub trait ProofTreeInferCtxtExt<'tcx>)]
impl<'tcx> InferCtxt<'tcx> {
    fn visit_proof_tree<V: ProofTreeVisitor<'tcx>>(
        &self,
        goal: Goal<'tcx, ty::Predicate<'tcx>>,
        visitor: &mut V,
    ) -> V::Result {
        self.visit_proof_tree_at_depth(goal, 0, visitor)
    }

    fn visit_proof_tree_at_depth<V: ProofTreeVisitor<'tcx>>(
        &self,
        goal: Goal<'tcx, ty::Predicate<'tcx>>,
        depth: usize,
        visitor: &mut V,
    ) -> V::Result {
        let (_, proof_tree) = <&SolverDelegate<'tcx>>::from(self).evaluate_root_goal(
            goal,
            GenerateProofTree::Yes,
            visitor.span(),
            None,
        );
        let proof_tree = proof_tree.unwrap();
        visitor.visit_goal(&InspectGoal::new(self, depth, proof_tree, None, GoalSource::Misc))
    }
}
