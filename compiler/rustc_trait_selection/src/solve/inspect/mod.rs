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

pub trait InspectSolve<'tcx> {
    fn into_debug_solver(self: Box<Self>) -> Option<Box<DebugSolver<'tcx>>>;

    fn new_goal_evaluation(
        &mut self,
        goal: Goal<'tcx, ty::Predicate<'tcx>>,
    ) -> Box<dyn InspectSolve<'tcx> + 'tcx>;
    fn canonicalized_goal(&mut self, canonical_goal: CanonicalInput<'tcx>);
    fn cache_hit(&mut self);
    fn goal_evaluation(&mut self, goal_evaluation: Box<dyn InspectSolve<'tcx> + 'tcx>);

    fn new_goal_evaluation_step(
        &mut self,
        instantiated_goal: QueryInput<'tcx, ty::Predicate<'tcx>>,
    ) -> Box<dyn InspectSolve<'tcx> + 'tcx>;
    fn goal_evaluation_step(&mut self, goal_eval_step: Box<dyn InspectSolve<'tcx> + 'tcx>);

    fn new_goal_candidate(&mut self) -> Box<dyn InspectSolve<'tcx> + 'tcx>;
    fn candidate_name(&mut self, f: &mut dyn FnMut() -> String);
    fn goal_candidate(&mut self, candidate: Box<dyn InspectSolve<'tcx> + 'tcx>);

    fn new_evaluate_added_goals(&mut self) -> Box<dyn InspectSolve<'tcx> + 'tcx>;
    fn evaluate_added_goals_loop_start(&mut self);
    fn eval_added_goals_result(&mut self, result: Result<Certainty, NoSolution>);
    fn added_goals_evaluation(&mut self, goals_evaluation: Box<dyn InspectSolve<'tcx> + 'tcx>);

    fn query_result(&mut self, result: QueryResult<'tcx>);
}

/// No-op `InspectSolve` impl to use for normal trait solving when we do not want
/// to take a performance hit from recording information about how things are being
/// proven.
impl<'tcx> InspectSolve<'tcx> for () {
    fn into_debug_solver(self: Box<Self>) -> Option<Box<DebugSolver<'tcx>>> {
        None
    }

    fn new_goal_evaluation(
        &mut self,
        _goal: Goal<'tcx, ty::Predicate<'tcx>>,
    ) -> Box<dyn InspectSolve<'tcx> + 'tcx> {
        Box::new(())
    }
    fn canonicalized_goal(&mut self, _canonical_goal: CanonicalInput<'tcx>) {}
    fn cache_hit(&mut self) {}
    fn goal_evaluation(&mut self, _goal_evaluation: Box<dyn InspectSolve<'tcx> + 'tcx>) {}

    fn new_goal_evaluation_step(
        &mut self,
        _instantiated_goal: QueryInput<'tcx, ty::Predicate<'tcx>>,
    ) -> Box<dyn InspectSolve<'tcx> + 'tcx> {
        Box::new(())
    }
    fn goal_evaluation_step(&mut self, _goal_eval_step: Box<dyn InspectSolve<'tcx> + 'tcx>) {}

    fn new_goal_candidate(&mut self) -> Box<dyn InspectSolve<'tcx> + 'tcx> {
        Box::new(())
    }
    fn candidate_name(&mut self, _f: &mut dyn FnMut() -> String) {}
    fn goal_candidate(&mut self, _candidate: Box<dyn InspectSolve<'tcx> + 'tcx>) {}

    fn new_evaluate_added_goals(&mut self) -> Box<dyn InspectSolve<'tcx> + 'tcx> {
        Box::new(())
    }
    fn evaluate_added_goals_loop_start(&mut self) {}
    fn eval_added_goals_result(&mut self, _result: Result<Certainty, NoSolution>) {}
    fn added_goals_evaluation(&mut self, _goals_evaluation: Box<dyn InspectSolve<'tcx> + 'tcx>) {}

    fn query_result(&mut self, _result: QueryResult<'tcx>) {}
}

impl<'tcx> DebugSolver<'tcx> {
    pub fn new() -> Self {
        Self::Root
    }
}
impl<'tcx> InspectSolve<'tcx> for DebugSolver<'tcx> {
    fn into_debug_solver(self: Box<Self>) -> Option<Box<DebugSolver<'tcx>>> {
        Some(self)
    }

    fn new_goal_evaluation(
        &mut self,
        goal: Goal<'tcx, ty::Predicate<'tcx>>,
    ) -> Box<dyn InspectSolve<'tcx> + 'tcx> {
        Box::new(DebugSolver::GoalEvaluation(GoalEvaluation {
            uncanonicalized_goal: goal,
            canonicalized_goal: None,
            evaluation_steps: vec![],
            cache_hit: false,
            result: None,
        }))
    }
    fn canonicalized_goal(&mut self, canonical_goal: CanonicalInput<'tcx>) {
        match self {
            DebugSolver::GoalEvaluation(goal_evaluation) => {
                assert!(goal_evaluation.canonicalized_goal.is_none());
                goal_evaluation.canonicalized_goal = Some(canonical_goal)
            }
            _ => unreachable!(),
        }
    }
    fn cache_hit(&mut self) {
        match self {
            DebugSolver::GoalEvaluation(goal_evaluation) => goal_evaluation.cache_hit = true,
            _ => unreachable!(),
        };
    }
    fn goal_evaluation(&mut self, goal_evaluation: Box<dyn InspectSolve<'tcx> + 'tcx>) {
        let goal_evaluation = goal_evaluation.into_debug_solver().unwrap();
        match (self, *goal_evaluation) {
            (
                DebugSolver::AddedGoalsEvaluation(AddedGoalsEvaluation { evaluations, .. }),
                DebugSolver::GoalEvaluation(goal_evaluation),
            ) => evaluations.last_mut().unwrap().push(goal_evaluation),
            (this @ DebugSolver::Root, goal_evaluation) => *this = goal_evaluation,
            _ => unreachable!(),
        }
    }

    fn new_goal_evaluation_step(
        &mut self,
        instantiated_goal: QueryInput<'tcx, ty::Predicate<'tcx>>,
    ) -> Box<dyn InspectSolve<'tcx> + 'tcx> {
        Box::new(DebugSolver::GoalEvaluationStep(GoalEvaluationStep {
            instantiated_goal,
            nested_goal_evaluations: vec![],
            candidates: vec![],
            result: None,
        }))
    }
    fn goal_evaluation_step(&mut self, goal_eval_step: Box<dyn InspectSolve<'tcx> + 'tcx>) {
        let goal_eval_step = goal_eval_step.into_debug_solver().unwrap();
        match (self, *goal_eval_step) {
            (DebugSolver::GoalEvaluation(goal_eval), DebugSolver::GoalEvaluationStep(step)) => {
                goal_eval.evaluation_steps.push(step);
            }
            _ => unreachable!(),
        }
    }

    fn new_goal_candidate(&mut self) -> Box<dyn InspectSolve<'tcx> + 'tcx> {
        Box::new(DebugSolver::GoalCandidate(GoalCandidate {
            nested_goal_evaluations: vec![],
            candidates: vec![],
            name: None,
            result: None,
        }))
    }
    fn candidate_name(&mut self, f: &mut dyn FnMut() -> String) {
        let name = f();

        match self {
            DebugSolver::GoalCandidate(goal_candidate) => {
                assert!(goal_candidate.name.is_none());
                goal_candidate.name = Some(name);
            }
            _ => unreachable!(),
        }
    }
    fn goal_candidate(&mut self, candidate: Box<dyn InspectSolve<'tcx> + 'tcx>) {
        let candidate = candidate.into_debug_solver().unwrap();
        match (self, *candidate) {
            (
                DebugSolver::GoalCandidate(GoalCandidate { candidates, .. })
                | DebugSolver::GoalEvaluationStep(GoalEvaluationStep { candidates, .. }),
                DebugSolver::GoalCandidate(candidate),
            ) => candidates.push(candidate),
            _ => unreachable!(),
        }
    }

    fn new_evaluate_added_goals(&mut self) -> Box<dyn InspectSolve<'tcx> + 'tcx> {
        Box::new(DebugSolver::AddedGoalsEvaluation(AddedGoalsEvaluation {
            evaluations: vec![],
            result: None,
        }))
    }
    fn evaluate_added_goals_loop_start(&mut self) {
        match self {
            DebugSolver::AddedGoalsEvaluation(this) => {
                this.evaluations.push(vec![]);
            }
            _ => unreachable!(),
        }
    }
    fn eval_added_goals_result(&mut self, result: Result<Certainty, NoSolution>) {
        match self {
            DebugSolver::AddedGoalsEvaluation(this) => {
                assert!(this.result.is_none());
                this.result = Some(result);
            }
            _ => unreachable!(),
        }
    }
    fn added_goals_evaluation(&mut self, goals_evaluation: Box<dyn InspectSolve<'tcx> + 'tcx>) {
        let goals_evaluation = goals_evaluation.into_debug_solver().unwrap();
        match (self, *goals_evaluation) {
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

    fn query_result(&mut self, result: QueryResult<'tcx>) {
        match self {
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
