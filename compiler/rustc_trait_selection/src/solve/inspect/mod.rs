use rustc_middle::{
    traits::{
        query::NoSolution,
        solve::{inspect::*, CanonicalInput, Certainty, Goal, QueryInput, QueryResult},
    },
    ty,
};

#[derive(Debug)]
pub enum DebugSolver<'tcx> {
    Root,
    GoalEvaluation(GoalEvaluation<'tcx>),
    AddedGoalsEvaluation(AddedGoalsEvaluation<'tcx>),
    GoalEvaluationStep(GoalEvaluationStep<'tcx>),
    GoalCandidate(GoalCandidate<'tcx>),
}

pub struct ProofTreeBuilder<'tcx>(Option<Box<DebugSolver<'tcx>>>);

impl<'tcx> ProofTreeBuilder<'tcx> {
    pub fn into_proof_tree(self) -> Option<DebugSolver<'tcx>> {
        self.0.map(|tree| *tree)
    }

    pub fn new_root() -> ProofTreeBuilder<'tcx> {
        Self(Some(Box::new(DebugSolver::Root)))
    }

    pub fn new_noop() -> ProofTreeBuilder<'tcx> {
        Self(None)
    }

    pub fn new_goal_evaluation(
        &mut self,
        goal: Goal<'tcx, ty::Predicate<'tcx>>,
    ) -> ProofTreeBuilder<'tcx> {
        if self.0.is_none() {
            return ProofTreeBuilder(None);
        }

        Self(Some(Box::new(DebugSolver::GoalEvaluation(GoalEvaluation {
            uncanonicalized_goal: goal,
            canonicalized_goal: None,
            evaluation_steps: vec![],
            cache_hit: None,
            result: None,
        }))))
    }
    pub fn canonicalized_goal(&mut self, canonical_goal: CanonicalInput<'tcx>) {
        let this = match self.0.as_mut() {
            None => return,
            Some(this) => &mut **this,
        };

        match this {
            DebugSolver::GoalEvaluation(goal_evaluation) => {
                assert!(goal_evaluation.canonicalized_goal.is_none());
                goal_evaluation.canonicalized_goal = Some(canonical_goal)
            }
            _ => unreachable!(),
        }
    }
    pub fn cache_hit(&mut self, cache_hit: CacheHit) {
        let this = match self.0.as_mut() {
            None => return,
            Some(this) => &mut **this,
        };

        match this {
            DebugSolver::GoalEvaluation(goal_evaluation) => {
                goal_evaluation.cache_hit = Some(cache_hit)
            }
            _ => unreachable!(),
        };
    }
    pub fn goal_evaluation(&mut self, goal_evaluation: ProofTreeBuilder<'tcx>) {
        let this = match self.0.as_mut() {
            None => return,
            Some(this) => &mut **this,
        };

        match (this, *goal_evaluation.0.unwrap()) {
            (
                DebugSolver::AddedGoalsEvaluation(AddedGoalsEvaluation { evaluations, .. }),
                DebugSolver::GoalEvaluation(goal_evaluation),
            ) => evaluations.last_mut().unwrap().push(goal_evaluation),
            (this @ DebugSolver::Root, goal_evaluation) => *this = goal_evaluation,
            _ => unreachable!(),
        }
    }

    pub fn new_goal_evaluation_step(
        &mut self,
        instantiated_goal: QueryInput<'tcx, ty::Predicate<'tcx>>,
    ) -> ProofTreeBuilder<'tcx> {
        Self(Some(Box::new(DebugSolver::GoalEvaluationStep(GoalEvaluationStep {
            instantiated_goal,
            nested_goal_evaluations: vec![],
            candidates: vec![],
            result: None,
        }))))
    }
    pub fn goal_evaluation_step(&mut self, goal_eval_step: ProofTreeBuilder<'tcx>) {
        let this = match self.0.as_mut() {
            None => return,
            Some(this) => &mut **this,
        };

        match (this, *goal_eval_step.0.unwrap()) {
            (DebugSolver::GoalEvaluation(goal_eval), DebugSolver::GoalEvaluationStep(step)) => {
                goal_eval.evaluation_steps.push(step);
            }
            _ => unreachable!(),
        }
    }

    pub fn new_goal_candidate(&mut self) -> ProofTreeBuilder<'tcx> {
        Self(Some(Box::new(DebugSolver::GoalCandidate(GoalCandidate {
            nested_goal_evaluations: vec![],
            candidates: vec![],
            name: None,
            result: None,
        }))))
    }
    pub fn candidate_name(&mut self, f: &mut dyn FnMut() -> String) {
        let this = match self.0.as_mut() {
            None => return,
            Some(this) => &mut **this,
        };

        match this {
            DebugSolver::GoalCandidate(goal_candidate) => {
                let name = f();
                assert!(goal_candidate.name.is_none());
                goal_candidate.name = Some(name);
            }
            _ => unreachable!(),
        }
    }
    pub fn goal_candidate(&mut self, candidate: ProofTreeBuilder<'tcx>) {
        let this = match self.0.as_mut() {
            None => return,
            Some(this) => &mut **this,
        };

        match (this, *candidate.0.unwrap()) {
            (
                DebugSolver::GoalCandidate(GoalCandidate { candidates, .. })
                | DebugSolver::GoalEvaluationStep(GoalEvaluationStep { candidates, .. }),
                DebugSolver::GoalCandidate(candidate),
            ) => candidates.push(candidate),
            _ => unreachable!(),
        }
    }

    pub fn new_evaluate_added_goals(&mut self) -> ProofTreeBuilder<'tcx> {
        Self(Some(Box::new(DebugSolver::AddedGoalsEvaluation(AddedGoalsEvaluation {
            evaluations: vec![],
            result: None,
        }))))
    }
    pub fn evaluate_added_goals_loop_start(&mut self) {
        let this = match self.0.as_mut() {
            None => return,
            Some(this) => &mut **this,
        };

        match this {
            DebugSolver::AddedGoalsEvaluation(this) => {
                this.evaluations.push(vec![]);
            }
            _ => unreachable!(),
        }
    }
    pub fn eval_added_goals_result(&mut self, result: Result<Certainty, NoSolution>) {
        let this = match self.0.as_mut() {
            None => return,
            Some(this) => &mut **this,
        };

        match this {
            DebugSolver::AddedGoalsEvaluation(this) => {
                assert!(this.result.is_none());
                this.result = Some(result);
            }
            _ => unreachable!(),
        }
    }
    pub fn added_goals_evaluation(&mut self, goals_evaluation: ProofTreeBuilder<'tcx>) {
        let this = match self.0.as_mut() {
            None => return,
            Some(this) => &mut **this,
        };

        match (this, *goals_evaluation.0.unwrap()) {
            (
                DebugSolver::GoalEvaluationStep(GoalEvaluationStep {
                    nested_goal_evaluations, ..
                })
                | DebugSolver::GoalCandidate(GoalCandidate { nested_goal_evaluations, .. }),
                DebugSolver::AddedGoalsEvaluation(added_goals_evaluation),
            ) => nested_goal_evaluations.push(added_goals_evaluation),
            _ => unreachable!(),
        }
    }

    pub fn query_result(&mut self, result: QueryResult<'tcx>) {
        let this = match self.0.as_mut() {
            None => return,
            Some(this) => &mut **this,
        };

        match this {
            DebugSolver::GoalEvaluation(goal_evaluation) => {
                assert!(goal_evaluation.result.is_none());
                goal_evaluation.result = Some(result);
            }
            DebugSolver::Root | DebugSolver::AddedGoalsEvaluation(_) => unreachable!(),
            DebugSolver::GoalEvaluationStep(evaluation_step) => {
                assert!(evaluation_step.result.is_none());
                evaluation_step.result = Some(result);
            }
            DebugSolver::GoalCandidate(candidate) => {
                assert!(candidate.result.is_none());
                candidate.result = Some(result);
            }
        }
    }
}
