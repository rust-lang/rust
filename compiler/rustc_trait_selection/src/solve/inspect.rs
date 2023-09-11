use rustc_middle::traits::query::NoSolution;
use rustc_middle::traits::solve::inspect::{self, CacheHit, ProbeKind};
use rustc_middle::traits::solve::{
    CanonicalInput, Certainty, Goal, IsNormalizesToHack, QueryInput, QueryResult,
};
use rustc_middle::ty::{self, TyCtxt};
use rustc_session::config::DumpSolverProofTree;

use super::eval_ctxt::UseGlobalCache;
use super::GenerateProofTree;

#[derive(Eq, PartialEq, Debug)]
pub struct WipGoalEvaluation<'tcx> {
    pub uncanonicalized_goal: Goal<'tcx, ty::Predicate<'tcx>>,
    pub evaluation: Option<WipCanonicalGoalEvaluation<'tcx>>,
    pub is_normalizes_to_hack: IsNormalizesToHack,
    pub returned_goals: Vec<Goal<'tcx, ty::Predicate<'tcx>>>,
}

impl<'tcx> WipGoalEvaluation<'tcx> {
    pub fn finalize(self) -> inspect::GoalEvaluation<'tcx> {
        inspect::GoalEvaluation {
            uncanonicalized_goal: self.uncanonicalized_goal,
            evaluation: self.evaluation.unwrap().finalize(),
            is_normalizes_to_hack: self.is_normalizes_to_hack,
            returned_goals: self.returned_goals,
        }
    }
}

#[derive(Eq, PartialEq, Debug)]
pub enum WipGoalEvaluationKind {
    Overflow,
    CacheHit(CacheHit),
}

#[derive(Eq, PartialEq, Debug)]
pub struct WipCanonicalGoalEvaluation<'tcx> {
    pub goal: CanonicalInput<'tcx>,
    pub kind: Option<WipGoalEvaluationKind>,
    pub revisions: Vec<WipGoalEvaluationStep<'tcx>>,
    pub result: Option<QueryResult<'tcx>>,
}

impl<'tcx> WipCanonicalGoalEvaluation<'tcx> {
    pub fn finalize(self) -> inspect::CanonicalGoalEvaluation<'tcx> {
        let kind = match self.kind {
            Some(WipGoalEvaluationKind::Overflow) => inspect::GoalEvaluationKind::Overflow,
            Some(WipGoalEvaluationKind::CacheHit(hit)) => {
                inspect::GoalEvaluationKind::CacheHit(hit)
            }
            None => inspect::GoalEvaluationKind::Uncached {
                revisions: self
                    .revisions
                    .into_iter()
                    .map(WipGoalEvaluationStep::finalize)
                    .collect(),
            },
        };

        inspect::CanonicalGoalEvaluation { goal: self.goal, kind, result: self.result.unwrap() }
    }
}

#[derive(Eq, PartialEq, Debug)]
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

#[derive(Eq, PartialEq, Debug)]
pub struct WipGoalEvaluationStep<'tcx> {
    pub instantiated_goal: QueryInput<'tcx, ty::Predicate<'tcx>>,

    pub evaluation: WipGoalCandidate<'tcx>,
}

impl<'tcx> WipGoalEvaluationStep<'tcx> {
    pub fn finalize(self) -> inspect::GoalEvaluationStep<'tcx> {
        let evaluation = self.evaluation.finalize();
        match evaluation.kind {
            ProbeKind::Root { .. } => (),
            _ => unreachable!("unexpected root evaluation: {evaluation:?}"),
        }
        inspect::GoalEvaluationStep { instantiated_goal: self.instantiated_goal, evaluation }
    }
}

#[derive(Eq, PartialEq, Debug)]
pub struct WipGoalCandidate<'tcx> {
    pub added_goals_evaluations: Vec<WipAddedGoalsEvaluation<'tcx>>,
    pub candidates: Vec<WipGoalCandidate<'tcx>>,
    pub kind: Option<ProbeKind<'tcx>>,
}

impl<'tcx> WipGoalCandidate<'tcx> {
    pub fn finalize(self) -> inspect::GoalCandidate<'tcx> {
        inspect::GoalCandidate {
            added_goals_evaluations: self
                .added_goals_evaluations
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
    CanonicalGoalEvaluation(WipCanonicalGoalEvaluation<'tcx>),
    AddedGoalsEvaluation(WipAddedGoalsEvaluation<'tcx>),
    GoalEvaluationStep(WipGoalEvaluationStep<'tcx>),
    GoalCandidate(WipGoalCandidate<'tcx>),
}

impl<'tcx> From<WipGoalEvaluation<'tcx>> for DebugSolver<'tcx> {
    fn from(g: WipGoalEvaluation<'tcx>) -> DebugSolver<'tcx> {
        DebugSolver::GoalEvaluation(g)
    }
}

impl<'tcx> From<WipCanonicalGoalEvaluation<'tcx>> for DebugSolver<'tcx> {
    fn from(g: WipCanonicalGoalEvaluation<'tcx>) -> DebugSolver<'tcx> {
        DebugSolver::CanonicalGoalEvaluation(g)
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
    state: Option<Box<BuilderData<'tcx>>>,
}

struct BuilderData<'tcx> {
    tree: DebugSolver<'tcx>,
    use_global_cache: UseGlobalCache,
}

impl<'tcx> ProofTreeBuilder<'tcx> {
    fn new(
        state: impl Into<DebugSolver<'tcx>>,
        use_global_cache: UseGlobalCache,
    ) -> ProofTreeBuilder<'tcx> {
        ProofTreeBuilder {
            state: Some(Box::new(BuilderData { tree: state.into(), use_global_cache })),
        }
    }

    fn nested<T: Into<DebugSolver<'tcx>>>(&self, state: impl FnOnce() -> T) -> Self {
        match &self.state {
            Some(prev_state) => Self {
                state: Some(Box::new(BuilderData {
                    tree: state().into(),
                    use_global_cache: prev_state.use_global_cache,
                })),
            },
            None => Self { state: None },
        }
    }

    fn as_mut(&mut self) -> Option<&mut DebugSolver<'tcx>> {
        self.state.as_mut().map(|boxed| &mut boxed.tree)
    }

    pub fn finalize(self) -> Option<inspect::GoalEvaluation<'tcx>> {
        match self.state?.tree {
            DebugSolver::GoalEvaluation(wip_goal_evaluation) => {
                Some(wip_goal_evaluation.finalize())
            }
            root => unreachable!("unexpected proof tree builder root node: {:?}", root),
        }
    }

    pub fn use_global_cache(&self) -> bool {
        self.state
            .as_ref()
            .map(|state| matches!(state.use_global_cache, UseGlobalCache::Yes))
            .unwrap_or(true)
    }

    pub fn new_maybe_root(
        tcx: TyCtxt<'tcx>,
        generate_proof_tree: GenerateProofTree,
    ) -> ProofTreeBuilder<'tcx> {
        match generate_proof_tree {
            GenerateProofTree::Never => ProofTreeBuilder::new_noop(),
            GenerateProofTree::IfEnabled => {
                let opts = &tcx.sess.opts.unstable_opts;
                match opts.dump_solver_proof_tree {
                    DumpSolverProofTree::Always => {
                        let use_cache = opts.dump_solver_proof_tree_use_cache.unwrap_or(true);
                        ProofTreeBuilder::new_root(UseGlobalCache::from_bool(use_cache))
                    }
                    // `OnError` is handled by reevaluating goals in error
                    // reporting with `GenerateProofTree::Yes`.
                    DumpSolverProofTree::OnError | DumpSolverProofTree::Never => {
                        ProofTreeBuilder::new_noop()
                    }
                }
            }
            GenerateProofTree::Yes(use_cache) => ProofTreeBuilder::new_root(use_cache),
        }
    }

    pub fn new_root(use_global_cache: UseGlobalCache) -> ProofTreeBuilder<'tcx> {
        ProofTreeBuilder::new(DebugSolver::Root, use_global_cache)
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
        self.nested(|| WipGoalEvaluation {
            uncanonicalized_goal: goal,
            evaluation: None,
            is_normalizes_to_hack,
            returned_goals: vec![],
        })
    }

    pub fn new_canonical_goal_evaluation(
        &mut self,
        goal: CanonicalInput<'tcx>,
    ) -> ProofTreeBuilder<'tcx> {
        self.nested(|| WipCanonicalGoalEvaluation {
            goal,
            kind: None,
            revisions: vec![],
            result: None,
        })
    }

    pub fn canonical_goal_evaluation(&mut self, canonical_goal_evaluation: ProofTreeBuilder<'tcx>) {
        if let Some(this) = self.as_mut() {
            match (this, canonical_goal_evaluation.state.unwrap().tree) {
                (
                    DebugSolver::GoalEvaluation(goal_evaluation),
                    DebugSolver::CanonicalGoalEvaluation(canonical_goal_evaluation),
                ) => goal_evaluation.evaluation = Some(canonical_goal_evaluation),
                _ => unreachable!(),
            }
        }
    }

    pub fn goal_evaluation_kind(&mut self, kind: WipGoalEvaluationKind) {
        if let Some(this) = self.as_mut() {
            match this {
                DebugSolver::CanonicalGoalEvaluation(canonical_goal_evaluation) => {
                    assert_eq!(canonical_goal_evaluation.kind.replace(kind), None);
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
            match (this, goal_evaluation.state.unwrap().tree) {
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
        self.nested(|| WipGoalEvaluationStep {
            instantiated_goal,
            evaluation: WipGoalCandidate {
                added_goals_evaluations: vec![],
                candidates: vec![],
                kind: None,
            },
        })
    }
    pub fn goal_evaluation_step(&mut self, goal_evaluation_step: ProofTreeBuilder<'tcx>) {
        if let Some(this) = self.as_mut() {
            match (this, goal_evaluation_step.state.unwrap().tree) {
                (
                    DebugSolver::CanonicalGoalEvaluation(canonical_goal_evaluations),
                    DebugSolver::GoalEvaluationStep(goal_evaluation_step),
                ) => {
                    canonical_goal_evaluations.revisions.push(goal_evaluation_step);
                }
                _ => unreachable!(),
            }
        }
    }

    pub fn new_goal_candidate(&mut self) -> ProofTreeBuilder<'tcx> {
        self.nested(|| WipGoalCandidate {
            added_goals_evaluations: vec![],
            candidates: vec![],
            kind: None,
        })
    }

    pub fn probe_kind(&mut self, probe_kind: ProbeKind<'tcx>) {
        if let Some(this) = self.as_mut() {
            match this {
                DebugSolver::GoalCandidate(this) => {
                    assert_eq!(this.kind.replace(probe_kind), None)
                }
                _ => unreachable!(),
            }
        }
    }

    pub fn goal_candidate(&mut self, candidate: ProofTreeBuilder<'tcx>) {
        if let Some(this) = self.as_mut() {
            match (this, candidate.state.unwrap().tree) {
                (
                    DebugSolver::GoalCandidate(WipGoalCandidate { candidates, .. })
                    | DebugSolver::GoalEvaluationStep(WipGoalEvaluationStep {
                        evaluation: WipGoalCandidate { candidates, .. },
                        ..
                    }),
                    DebugSolver::GoalCandidate(candidate),
                ) => candidates.push(candidate),
                _ => unreachable!(),
            }
        }
    }

    pub fn new_evaluate_added_goals(&mut self) -> ProofTreeBuilder<'tcx> {
        self.nested(|| WipAddedGoalsEvaluation { evaluations: vec![], result: None })
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

    pub fn added_goals_evaluation(&mut self, added_goals_evaluation: ProofTreeBuilder<'tcx>) {
        if let Some(this) = self.as_mut() {
            match (this, added_goals_evaluation.state.unwrap().tree) {
                (
                    DebugSolver::GoalEvaluationStep(WipGoalEvaluationStep {
                        evaluation: WipGoalCandidate { added_goals_evaluations, .. },
                        ..
                    })
                    | DebugSolver::GoalCandidate(WipGoalCandidate {
                        added_goals_evaluations, ..
                    }),
                    DebugSolver::AddedGoalsEvaluation(added_goals_evaluation),
                ) => added_goals_evaluations.push(added_goals_evaluation),
                _ => unreachable!(),
            }
        }
    }

    pub fn query_result(&mut self, result: QueryResult<'tcx>) {
        if let Some(this) = self.as_mut() {
            match this {
                DebugSolver::CanonicalGoalEvaluation(canonical_goal_evaluation) => {
                    assert_eq!(canonical_goal_evaluation.result.replace(result), None);
                }
                DebugSolver::GoalEvaluationStep(evaluation_step) => {
                    assert_eq!(
                        evaluation_step.evaluation.kind.replace(ProbeKind::Root { result }),
                        None
                    );
                }
                _ => unreachable!(),
            }
        }
    }
}
