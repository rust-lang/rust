/// An infrastructure to mechanically analyse proof trees.
///
/// It is unavoidable that this representation is somewhat
/// lossy as it should hide quite a few semantically relevant things,
/// e.g. canonicalization and the order of nested goals.
///
/// @lcnr: However, a lot of the weirdness here is not strictly necessary
/// and could be improved in the future. This is mostly good enough for
/// coherence right now and was annoying to implement, so I am leaving it
/// as is until we start using it for something else.
use std::ops::ControlFlow;

use rustc_infer::infer::InferCtxt;
use rustc_middle::traits::query::NoSolution;
use rustc_middle::traits::solve::{inspect, QueryResult};
use rustc_middle::traits::solve::{Certainty, Goal};
use rustc_middle::ty;

use crate::solve::inspect::ProofTreeBuilder;
use crate::solve::{GenerateProofTree, InferCtxtEvalExt};

pub struct InspectGoal<'a, 'tcx> {
    infcx: &'a InferCtxt<'tcx>,
    depth: usize,
    orig_values: &'a [ty::GenericArg<'tcx>],
    goal: Goal<'tcx, ty::Predicate<'tcx>>,
    evaluation: &'a inspect::GoalEvaluation<'tcx>,
}

pub struct InspectCandidate<'a, 'tcx> {
    goal: &'a InspectGoal<'a, 'tcx>,
    kind: inspect::ProbeKind<'tcx>,
    nested_goals: Vec<inspect::CanonicalState<'tcx, Goal<'tcx, ty::Predicate<'tcx>>>>,
    result: QueryResult<'tcx>,
}

impl<'a, 'tcx> InspectCandidate<'a, 'tcx> {
    pub fn infcx(&self) -> &'a InferCtxt<'tcx> {
        self.goal.infcx
    }

    pub fn kind(&self) -> inspect::ProbeKind<'tcx> {
        self.kind
    }

    pub fn result(&self) -> Result<Certainty, NoSolution> {
        self.result.map(|c| c.value.certainty)
    }

    /// Visit the nested goals of this candidate.
    ///
    /// FIXME(@lcnr): we have to slightly adapt this API
    /// to also use it to compute the most relevant goal
    /// for fulfillment errors. Will do that once we actually
    /// need it.
    pub fn visit_nested<V: ProofTreeVisitor<'tcx>>(
        &self,
        visitor: &mut V,
    ) -> ControlFlow<V::BreakTy> {
        // HACK: An arbitrary cutoff to avoid dealing with overflow and cycles.
        if self.goal.depth <= 10 {
            let infcx = self.goal.infcx;
            infcx.probe(|_| {
                let mut instantiated_goals = vec![];
                for goal in &self.nested_goals {
                    let goal = ProofTreeBuilder::instantiate_canonical_state(
                        infcx,
                        self.goal.goal.param_env,
                        self.goal.orig_values,
                        *goal,
                    );
                    instantiated_goals.push(goal);
                }

                for &goal in &instantiated_goals {
                    let (_, proof_tree) = infcx.evaluate_root_goal(goal, GenerateProofTree::Yes);
                    let proof_tree = proof_tree.unwrap();
                    visitor.visit_goal(&InspectGoal::new(
                        infcx,
                        self.goal.depth + 1,
                        &proof_tree,
                    ))?;
                }

                ControlFlow::Continue(())
            })?;
        }
        ControlFlow::Continue(())
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
        self.evaluation.evaluation.result.map(|c| c.value.certainty)
    }

    fn candidates_recur(
        &'a self,
        candidates: &mut Vec<InspectCandidate<'a, 'tcx>>,
        nested_goals: &mut Vec<inspect::CanonicalState<'tcx, Goal<'tcx, ty::Predicate<'tcx>>>>,
        probe: &inspect::Probe<'tcx>,
    ) {
        for step in &probe.steps {
            match step {
                &inspect::ProbeStep::AddGoal(_source, goal) => nested_goals.push(goal),
                inspect::ProbeStep::NestedProbe(ref probe) => {
                    // Nested probes have to prove goals added in their parent
                    // but do not leak them, so we truncate the added goals
                    // afterwards.
                    let num_goals = nested_goals.len();
                    self.candidates_recur(candidates, nested_goals, probe);
                    nested_goals.truncate(num_goals);
                }
                inspect::ProbeStep::EvaluateGoals(_)
                | inspect::ProbeStep::CommitIfOkStart
                | inspect::ProbeStep::CommitIfOkSuccess => (),
            }
        }

        match probe.kind {
            inspect::ProbeKind::NormalizedSelfTyAssembly
            | inspect::ProbeKind::UnsizeAssembly
            | inspect::ProbeKind::UpcastProjectionCompatibility
            | inspect::ProbeKind::CommitIfOk => (),
            // We add a candidate for the root evaluation if there
            // is only one way to prove a given goal, e.g. for `WellFormed`.
            //
            // FIXME: This is currently wrong if we don't even try any
            // candidates, e.g. for a trait goal, as in this case `candidates` is
            // actually supposed to be empty.
            inspect::ProbeKind::Root { result } => {
                if candidates.is_empty() {
                    candidates.push(InspectCandidate {
                        goal: self,
                        kind: probe.kind,
                        nested_goals: nested_goals.clone(),
                        result,
                    });
                }
            }
            inspect::ProbeKind::MiscCandidate { name: _, result }
            | inspect::ProbeKind::TraitCandidate { source: _, result } => {
                candidates.push(InspectCandidate {
                    goal: self,
                    kind: probe.kind,
                    nested_goals: nested_goals.clone(),
                    result,
                });
            }
        }
    }

    pub fn candidates(&'a self) -> Vec<InspectCandidate<'a, 'tcx>> {
        let mut candidates = vec![];
        let last_eval_step = match self.evaluation.evaluation.kind {
            inspect::CanonicalGoalEvaluationKind::Overflow
            | inspect::CanonicalGoalEvaluationKind::CycleInStack
            | inspect::CanonicalGoalEvaluationKind::ProvisionalCacheHit => {
                warn!("unexpected root evaluation: {:?}", self.evaluation);
                return vec![];
            }
            inspect::CanonicalGoalEvaluationKind::Evaluation { revisions } => {
                if let Some(last) = revisions.last() {
                    last
                } else {
                    return vec![];
                }
            }
        };

        let mut nested_goals = vec![];
        self.candidates_recur(&mut candidates, &mut nested_goals, &last_eval_step.evaluation);

        candidates
    }

    fn new(
        infcx: &'a InferCtxt<'tcx>,
        depth: usize,
        root: &'a inspect::GoalEvaluation<'tcx>,
    ) -> Self {
        match root.kind {
            inspect::GoalEvaluationKind::Root { ref orig_values } => InspectGoal {
                infcx,
                depth,
                orig_values,
                goal: infcx.resolve_vars_if_possible(root.uncanonicalized_goal),
                evaluation: root,
            },
            inspect::GoalEvaluationKind::Nested { .. } => unreachable!(),
        }
    }
}

/// The public API to interact with proof trees.
pub trait ProofTreeVisitor<'tcx> {
    type BreakTy;

    fn visit_goal(&mut self, goal: &InspectGoal<'_, 'tcx>) -> ControlFlow<Self::BreakTy>;
}

#[extension(pub trait ProofTreeInferCtxtExt<'tcx>)]
impl<'tcx> InferCtxt<'tcx> {
    fn visit_proof_tree<V: ProofTreeVisitor<'tcx>>(
        &self,
        goal: Goal<'tcx, ty::Predicate<'tcx>>,
        visitor: &mut V,
    ) -> ControlFlow<V::BreakTy> {
        self.probe(|_| {
            let (_, proof_tree) = self.evaluate_root_goal(goal, GenerateProofTree::Yes);
            let proof_tree = proof_tree.unwrap();
            visitor.visit_goal(&InspectGoal::new(self, 0, &proof_tree))
        })
    }
}
