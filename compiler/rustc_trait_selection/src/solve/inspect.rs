use rustc_middle::traits::query::NoSolution;
use rustc_middle::traits::solve::inspect::{self, CacheHit, CandidateKind};
use rustc_middle::traits::solve::{
    CanonicalInput, Certainty, Goal, IsNormalizesToHack, QueryInput, QueryResult,
};
use rustc_middle::ty;

pub mod dump;

#[derive(Eq, PartialEq, Debug, Hash, HashStable)]
pub struct WipGoalEvaluation<'tcx> {
    pub uncanonicalized_goal: Goal<'tcx, ty::Predicate<'tcx>>,
    pub canonicalized_goal: Option<CanonicalInput<'tcx>>,

    pub evaluation_steps: Vec<WipGoalEvaluationStep<'tcx>>,

    pub cache_hit: Option<CacheHit>,
    pub is_normalizes_to_hack: IsNormalizesToHack,
    pub returned_goals: Vec<Goal<'tcx, ty::Predicate<'tcx>>>,

    pub result: Option<QueryResult<'tcx>>,
}

impl<'tcx> WipGoalEvaluation<'tcx> {
    pub fn finalize(self) -> inspect::GoalEvaluation<'tcx> {
        inspect::GoalEvaluation {
            uncanonicalized_goal: self.uncanonicalized_goal,
            canonicalized_goal: self.canonicalized_goal.unwrap(),
            kind: match self.cache_hit {
                Some(hit) => inspect::GoalEvaluationKind::CacheHit(hit),
                None => inspect::GoalEvaluationKind::Uncached {
                    revisions: self
                        .evaluation_steps
                        .into_iter()
                        .map(WipGoalEvaluationStep::finalize)
                        .collect(),
                },
            },
            is_normalizes_to_hack: self.is_normalizes_to_hack,
            returned_goals: self.returned_goals,
            result: self.result.unwrap(),
        }
    }
}

#[derive(Eq, PartialEq, Debug, Hash, HashStable)]
pub struct WipAddedGoalsEvaluation<'tcx> {
    pub evaluations: Vec<Vec<WipGoalEvaluation<'tcx>>>,
    pub result: Option<Result<Certainty, NoSolution>>,
}

impl<'tcx> WipAddedGoalsEvaluation<'tcx> {
    pub fn finalize(self) -> inspect::AddedGoalsEvaluation<'tcx> {
        inspect::AddedGoalsEvaluation {
            evaluations: self
                .evaluations
                .into_iter()
                .map(|evaluations| {
                    evaluations.into_iter().map(WipGoalEvaluation::finalize).collect()
                })
                .collect(),
            result: self.result.unwrap(),
        }
    }
}

#[derive(Eq, PartialEq, Debug, Hash, HashStable)]
pub struct WipGoalEvaluationStep<'tcx> {
    pub instantiated_goal: QueryInput<'tcx, ty::Predicate<'tcx>>,

    pub nested_goal_evaluations: Vec<WipAddedGoalsEvaluation<'tcx>>,
    pub candidates: Vec<WipGoalCandidate<'tcx>>,

    pub result: Option<QueryResult<'tcx>>,
}

impl<'tcx> WipGoalEvaluationStep<'tcx> {
    pub fn finalize(self) -> inspect::GoalEvaluationStep<'tcx> {
        inspect::GoalEvaluationStep {
            instantiated_goal: self.instantiated_goal,
            nested_goal_evaluations: self
                .nested_goal_evaluations
                .into_iter()
                .map(WipAddedGoalsEvaluation::finalize)
                .collect(),
            candidates: self.candidates.into_iter().map(WipGoalCandidate::finalize).collect(),
            result: self.result.unwrap(),
        }
    }
}

#[derive(Eq, PartialEq, Debug, Hash, HashStable)]
pub struct WipGoalCandidate<'tcx> {
    pub nested_goal_evaluations: Vec<WipAddedGoalsEvaluation<'tcx>>,
    pub candidates: Vec<WipGoalCandidate<'tcx>>,
    pub kind: Option<CandidateKind<'tcx>>,
}

impl<'tcx> WipGoalCandidate<'tcx> {
    pub fn finalize(self) -> inspect::GoalCandidate<'tcx> {
        inspect::GoalCandidate {
            nested_goal_evaluations: self
                .nested_goal_evaluations
                .into_iter()
                .map(WipAddedGoalsEvaluation::finalize)
                .collect(),
            candidates: self.candidates.into_iter().map(WipGoalCandidate::finalize).collect(),
            kind: self.kind.unwrap(),
        }
    }
}

#[derive(Debug)]
pub enum DebugSolver<'tcx> {
    Root,
    GoalEvaluation(WipGoalEvaluation<'tcx>),
    AddedGoalsEvaluation(WipAddedGoalsEvaluation<'tcx>),
    GoalEvaluationStep(WipGoalEvaluationStep<'tcx>),
    GoalCandidate(WipGoalCandidate<'tcx>),
}

impl<'tcx> From<WipGoalEvaluation<'tcx>> for DebugSolver<'tcx> {
    fn from(g: WipGoalEvaluation<'tcx>) -> DebugSolver<'tcx> {
        DebugSolver::GoalEvaluation(g)
    }
}

impl<'tcx> From<WipAddedGoalsEvaluation<'tcx>> for DebugSolver<'tcx> {
    fn from(g: WipAddedGoalsEvaluation<'tcx>) -> DebugSolver<'tcx> {
        DebugSolver::AddedGoalsEvaluation(g)
    }
}

impl<'tcx> From<WipGoalEvaluationStep<'tcx>> for DebugSolver<'tcx> {
    fn from(g: WipGoalEvaluationStep<'tcx>) -> DebugSolver<'tcx> {
        DebugSolver::GoalEvaluationStep(g)
    }
}

impl<'tcx> From<WipGoalCandidate<'tcx>> for DebugSolver<'tcx> {
    fn from(g: WipGoalCandidate<'tcx>) -> DebugSolver<'tcx> {
        DebugSolver::GoalCandidate(g)
    }
}

pub struct ProofTreeBuilder<'tcx> {
    state: Option<Box<DebugSolver<'tcx>>>,
}

impl<'tcx> ProofTreeBuilder<'tcx> {
    fn new(state: impl Into<DebugSolver<'tcx>>) -> ProofTreeBuilder<'tcx> {
        ProofTreeBuilder { state: Some(Box::new(state.into())) }
    }

    fn as_mut(&mut self) -> Option<&mut DebugSolver<'tcx>> {
        self.state.as_mut().map(|boxed| &mut **boxed)
    }

    pub fn finalize(self) -> Option<inspect::GoalEvaluation<'tcx>> {
        match *(self.state?) {
            DebugSolver::GoalEvaluation(wip_goal_evaluation) => {
                Some(wip_goal_evaluation.finalize())
            }
            root => unreachable!("unexpected proof tree builder root node: {:?}", root),
        }
    }

    pub fn new_root() -> ProofTreeBuilder<'tcx> {
        ProofTreeBuilder::new(DebugSolver::Root)
    }

    pub fn new_noop() -> ProofTreeBuilder<'tcx> {
        ProofTreeBuilder { state: None }
    }

    pub fn is_noop(&self) -> bool {
        self.state.is_none()
    }

    pub fn new_goal_evaluation(
        &mut self,
        goal: Goal<'tcx, ty::Predicate<'tcx>>,
        is_normalizes_to_hack: IsNormalizesToHack,
    ) -> ProofTreeBuilder<'tcx> {
        if self.state.is_none() {
            return ProofTreeBuilder { state: None };
        }

        ProofTreeBuilder::new(WipGoalEvaluation {
            uncanonicalized_goal: goal,
            canonicalized_goal: None,
            evaluation_steps: vec![],
            is_normalizes_to_hack,
            cache_hit: None,
            returned_goals: vec![],
            result: None,
        })
    }

    pub fn canonicalized_goal(&mut self, canonical_goal: CanonicalInput<'tcx>) {
        if let Some(this) = self.as_mut() {
            match this {
                DebugSolver::GoalEvaluation(goal_evaluation) => {
                    assert_eq!(goal_evaluation.canonicalized_goal.replace(canonical_goal), None);
                }
                _ => unreachable!(),
            }
        }
    }

    pub fn cache_hit(&mut self, cache_hit: CacheHit) {
        if let Some(this) = self.as_mut() {
            match this {
                DebugSolver::GoalEvaluation(goal_evaluation) => {
                    assert_eq!(goal_evaluation.cache_hit.replace(cache_hit), None);
                }
                _ => unreachable!(),
            };
        }
    }

    pub fn returned_goals(&mut self, goals: &[Goal<'tcx, ty::Predicate<'tcx>>]) {
        if let Some(this) = self.as_mut() {
            match this {
                DebugSolver::GoalEvaluation(evaluation) => {
                    assert!(evaluation.returned_goals.is_empty());
                    evaluation.returned_goals.extend(goals);
                }
                _ => unreachable!(),
            }
        }
    }
    pub fn goal_evaluation(&mut self, goal_evaluation: ProofTreeBuilder<'tcx>) {
        if let Some(this) = self.as_mut() {
            match (this, *goal_evaluation.state.unwrap()) {
                (
                    DebugSolver::AddedGoalsEvaluation(WipAddedGoalsEvaluation {
                        evaluations, ..
                    }),
                    DebugSolver::GoalEvaluation(goal_evaluation),
                ) => evaluations.last_mut().unwrap().push(goal_evaluation),
                (this @ DebugSolver::Root, goal_evaluation) => *this = goal_evaluation,
                _ => unreachable!(),
            }
        }
    }

    pub fn new_goal_evaluation_step(
        &mut self,
        instantiated_goal: QueryInput<'tcx, ty::Predicate<'tcx>>,
    ) -> ProofTreeBuilder<'tcx> {
        if self.state.is_none() {
            return ProofTreeBuilder { state: None };
        }

        ProofTreeBuilder::new(WipGoalEvaluationStep {
            instantiated_goal,
            nested_goal_evaluations: vec![],
            candidates: vec![],
            result: None,
        })
    }
    pub fn goal_evaluation_step(&mut self, goal_eval_step: ProofTreeBuilder<'tcx>) {
        if let Some(this) = self.as_mut() {
            match (this, *goal_eval_step.state.unwrap()) {
                (DebugSolver::GoalEvaluation(goal_eval), DebugSolver::GoalEvaluationStep(step)) => {
                    goal_eval.evaluation_steps.push(step);
                }
                _ => unreachable!(),
            }
        }
    }

    pub fn new_goal_candidate(&mut self) -> ProofTreeBuilder<'tcx> {
        if self.state.is_none() {
            return ProofTreeBuilder { state: None };
        }

        ProofTreeBuilder::new(WipGoalCandidate {
            nested_goal_evaluations: vec![],
            candidates: vec![],
            kind: None,
        })
    }

    pub fn candidate_kind(&mut self, candidate_kind: CandidateKind<'tcx>) {
        if let Some(this) = self.as_mut() {
            match this {
                DebugSolver::GoalCandidate(this) => {
                    assert_eq!(this.kind.replace(candidate_kind), None)
                }
                _ => unreachable!(),
            }
        }
    }

    pub fn goal_candidate(&mut self, candidate: ProofTreeBuilder<'tcx>) {
        if let Some(this) = self.as_mut() {
            match (this, *candidate.state.unwrap()) {
                (
                    DebugSolver::GoalCandidate(WipGoalCandidate { candidates, .. })
                    | DebugSolver::GoalEvaluationStep(WipGoalEvaluationStep { candidates, .. }),
                    DebugSolver::GoalCandidate(candidate),
                ) => candidates.push(candidate),
                _ => unreachable!(),
            }
        }
    }

    pub fn new_evaluate_added_goals(&mut self) -> ProofTreeBuilder<'tcx> {
        if self.state.is_none() {
            return ProofTreeBuilder { state: None };
        }

        ProofTreeBuilder::new(WipAddedGoalsEvaluation { evaluations: vec![], result: None })
    }

    pub fn evaluate_added_goals_loop_start(&mut self) {
        if let Some(this) = self.as_mut() {
            match this {
                DebugSolver::AddedGoalsEvaluation(this) => {
                    this.evaluations.push(vec![]);
                }
                _ => unreachable!(),
            }
        }
    }

    pub fn eval_added_goals_result(&mut self, result: Result<Certainty, NoSolution>) {
        if let Some(this) = self.as_mut() {
            match this {
                DebugSolver::AddedGoalsEvaluation(this) => {
                    assert_eq!(this.result.replace(result), None);
                }
                _ => unreachable!(),
            }
        }
    }

    pub fn added_goals_evaluation(&mut self, goals_evaluation: ProofTreeBuilder<'tcx>) {
        if let Some(this) = self.as_mut() {
            match (this, *goals_evaluation.state.unwrap()) {
                (
                    DebugSolver::GoalEvaluationStep(WipGoalEvaluationStep {
                        nested_goal_evaluations,
                        ..
                    })
                    | DebugSolver::GoalCandidate(WipGoalCandidate {
                        nested_goal_evaluations, ..
                    }),
                    DebugSolver::AddedGoalsEvaluation(added_goals_evaluation),
                ) => nested_goal_evaluations.push(added_goals_evaluation),
                _ => unreachable!(),
            }
        }
    }

    pub fn query_result(&mut self, result: QueryResult<'tcx>) {
        if let Some(this) = self.as_mut() {
            match this {
                DebugSolver::GoalEvaluation(goal_evaluation) => {
                    assert_eq!(goal_evaluation.result.replace(result), None);
                }
                DebugSolver::GoalEvaluationStep(evaluation_step) => {
                    assert_eq!(evaluation_step.result.replace(result), None);
                }
                DebugSolver::Root
                | DebugSolver::AddedGoalsEvaluation(_)
                | DebugSolver::GoalCandidate(_) => unreachable!(),
            }
        }
    }
}
