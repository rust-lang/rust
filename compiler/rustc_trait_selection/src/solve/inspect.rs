use rustc_middle::traits::query::NoSolution;
use rustc_middle::traits::solve::{CanonicalInput, Certainty, Goal, QueryResult};
use rustc_middle::ty::{self, TyCtxt};
use rustc_session::config::DumpSolverProofTree;

use super::eval_ctxt::UseGlobalCache;
use super::{EvalCtxt, GenerateProofTree, GoalEvaluationKind};

pub use rustc_middle::traits::solve::inspect::*;

pub(super) mod analyse;

#[derive(Debug)]
pub struct WipRootGoalEvaluation<'tcx> {
    goal: Goal<'tcx, ty::Predicate<'tcx>>,
    orig_values: Vec<ty::GenericArg<'tcx>>,
    evaluation: Option<WipCanonicalGoalEvaluation<'tcx>>,
    returned_goals: Vec<Goal<'tcx, ty::Predicate<'tcx>>>,
}

impl<'tcx> WipRootGoalEvaluation<'tcx> {
    pub(super) fn finalize(self) -> RootGoalEvaluation<'tcx> {
        RootGoalEvaluation {
            goal: self.goal,
            orig_values: self.orig_values,
            evaluation: self.evaluation.unwrap().finalize(),
            returned_goals: self.returned_goals,
        }
    }
}

#[derive(Debug)]
pub struct WipNestedGoalEvaluation<'tcx> {
    pub goal: CanonicalState<'tcx, Goal<'tcx, ty::Predicate<'tcx>>>,
    pub orig_values: CanonicalState<'tcx, Vec<ty::GenericArg<'tcx>>>,
    pub evaluation: Option<WipCanonicalGoalEvaluation<'tcx>>,
    pub is_normalizes_to_hack: IsNormalizesToHack,
    pub returned_goals: Vec<CanonicalState<'tcx, Goal<'tcx, ty::Predicate<'tcx>>>>,
}

impl<'tcx> WipNestedGoalEvaluation<'tcx> {
    pub(super) fn finalize(self) -> NestedGoalEvaluation<'tcx> {
        NestedGoalEvaluation {
            goal: self.goal,
            orig_values: self.orig_values,
            evaluation: self.evaluation.unwrap().finalize(),
            is_normalizes_to_hack: self.is_normalizes_to_hack,
            returned_goals: self.returned_goals,
        }
    }
}

#[derive(Debug)]
pub struct WipCanonicalGoalEvaluation<'tcx> {
    pub goal: CanonicalInput<'tcx>,
    pub cache_hit: Option<CacheHit>,
    pub evaluation_steps: Vec<WipGoalEvaluationStep<'tcx>>,
    pub result: Option<QueryResult<'tcx>>,
}

impl<'tcx> WipCanonicalGoalEvaluation<'tcx> {
    pub(super) fn finalize(self) -> CanonicalGoalEvaluation<'tcx> {
        let data = match self.cache_hit {
            Some(hit) => GoalEvaluationData::CacheHit(hit),
            None => {
                assert!(!self.evaluation_steps.is_empty());
                GoalEvaluationData::Uncached {
                    revisions: self
                        .evaluation_steps
                        .into_iter()
                        .map(WipGoalEvaluationStep::finalize)
                        .collect(),
                }
            }
        };

        CanonicalGoalEvaluation { goal: self.goal, data, result: self.result.unwrap() }
    }
}

#[derive(Debug)]
pub struct WipAddedGoalsEvaluation<'tcx> {
    pub evaluations: Vec<Vec<WipNestedGoalEvaluation<'tcx>>>,
    pub result: Option<Result<Certainty, NoSolution>>,
}

impl<'tcx> WipAddedGoalsEvaluation<'tcx> {
    pub(super) fn finalize(self) -> AddedGoalsEvaluation<'tcx> {
        AddedGoalsEvaluation {
            evaluations: self
                .evaluations
                .into_iter()
                .map(|evaluations| evaluations.into_iter().map(|e| e.finalize()).collect())
                .collect(),
            result: self.result.unwrap(),
        }
    }
}

#[derive(Debug)]
pub struct WipGoalEvaluationStep<'tcx> {
    pub added_goals_evaluations: Vec<WipAddedGoalsEvaluation<'tcx>>,
    pub candidates: Vec<WipGoalCandidate<'tcx>>,
    pub result: Option<QueryResult<'tcx>>,
}

impl<'tcx> WipGoalEvaluationStep<'tcx> {
    pub(super) fn finalize(self) -> GoalEvaluationStep<'tcx> {
        GoalEvaluationStep {
            added_goals_evaluations: self
                .added_goals_evaluations
                .into_iter()
                .map(WipAddedGoalsEvaluation::finalize)
                .collect(),
            candidates: self.candidates.into_iter().map(WipGoalCandidate::finalize).collect(),
            result: self.result.unwrap(),
        }
    }
}

#[derive(Debug)]
pub struct WipGoalCandidate<'tcx> {
    pub added_goals_evaluations: Vec<WipAddedGoalsEvaluation<'tcx>>,
    pub candidates: Vec<WipGoalCandidate<'tcx>>,
    pub kind: Option<ProbeKind<'tcx>>,
}

impl<'tcx> WipGoalCandidate<'tcx> {
    pub(super) fn finalize(self) -> GoalCandidate<'tcx> {
        GoalCandidate {
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
    RootGoalEvaluation(WipRootGoalEvaluation<'tcx>),
    NestedGoalEvaluation(WipNestedGoalEvaluation<'tcx>),
    CanonicalGoalEvaluation(WipCanonicalGoalEvaluation<'tcx>),
    AddedGoalsEvaluation(WipAddedGoalsEvaluation<'tcx>),
    GoalEvaluationStep(WipGoalEvaluationStep<'tcx>),
    GoalCandidate(WipGoalCandidate<'tcx>),
}

impl<'tcx> From<WipRootGoalEvaluation<'tcx>> for DebugSolver<'tcx> {
    fn from(g: WipRootGoalEvaluation<'tcx>) -> DebugSolver<'tcx> {
        DebugSolver::RootGoalEvaluation(g)
    }
}

impl<'tcx> From<WipNestedGoalEvaluation<'tcx>> for DebugSolver<'tcx> {
    fn from(g: WipNestedGoalEvaluation<'tcx>) -> DebugSolver<'tcx> {
        DebugSolver::NestedGoalEvaluation(g)
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

    pub(super) fn finalize(self) -> Option<RootGoalEvaluation<'tcx>> {
        match self.state?.tree {
            DebugSolver::RootGoalEvaluation(root_goal_evaluation) => {
                Some(root_goal_evaluation.finalize())
            }
            root => unreachable!("unexpected proof tree builder root node: {:?}", root),
        }
    }

    pub(super) fn use_global_cache(&self) -> bool {
        self.state
            .as_ref()
            .map(|state| matches!(state.use_global_cache, UseGlobalCache::Yes))
            .unwrap_or(true)
    }

    pub(super) fn new_maybe_root(
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

    pub(super) fn new_root(use_global_cache: UseGlobalCache) -> ProofTreeBuilder<'tcx> {
        ProofTreeBuilder::new(DebugSolver::Root, use_global_cache)
    }

    pub(super) fn new_noop() -> ProofTreeBuilder<'tcx> {
        ProofTreeBuilder { state: None }
    }

    pub(super) fn is_noop(&self) -> bool {
        self.state.is_none()
    }

    pub(super) fn new_goal_evaluation(
        ecx: &EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, ty::Predicate<'tcx>>,
        orig_values: &[ty::GenericArg<'tcx>],
        goal_evaluation_kind: GoalEvaluationKind,
    ) -> ProofTreeBuilder<'tcx> {
        let is_normalizes_to_hack = match goal_evaluation_kind {
            GoalEvaluationKind::Root => {
                return ecx.inspect.nested(|| WipRootGoalEvaluation {
                    goal,
                    orig_values: orig_values.to_vec(),
                    evaluation: None,
                    returned_goals: vec![],
                });
            }
            GoalEvaluationKind::NormalizesToHack => IsNormalizesToHack::Yes,
            GoalEvaluationKind::Nested => IsNormalizesToHack::No,
        };

        ecx.inspect.nested(|| WipNestedGoalEvaluation {
            goal: Self::make_canonical_state(ecx, goal),
            orig_values: Self::make_canonical_state(ecx, orig_values.to_vec()),
            evaluation: None,
            is_normalizes_to_hack,
            returned_goals: vec![],
        })
    }

    pub(super) fn new_canonical_goal_evaluation(
        &mut self,
        goal: CanonicalInput<'tcx>,
    ) -> ProofTreeBuilder<'tcx> {
        self.nested(|| WipCanonicalGoalEvaluation {
            goal,
            cache_hit: None,
            evaluation_steps: vec![],
            result: None,
        })
    }

    pub(super) fn canonical_goal_evaluation(
        &mut self,
        canonical_goal_evaluation: ProofTreeBuilder<'tcx>,
    ) {
        if let Some(this) = self.as_mut() {
            match (this, canonical_goal_evaluation.state.unwrap().tree) {
                (
                    DebugSolver::RootGoalEvaluation(WipRootGoalEvaluation { evaluation, .. })
                    | DebugSolver::NestedGoalEvaluation(WipNestedGoalEvaluation {
                        evaluation, ..
                    }),
                    DebugSolver::CanonicalGoalEvaluation(canonical_goal_evaluation),
                ) => *evaluation = Some(canonical_goal_evaluation),
                _ => unreachable!(),
            }
        }
    }

    pub(super) fn cache_hit(&mut self, cache_hit: CacheHit) {
        if let Some(this) = self.as_mut() {
            match this {
                DebugSolver::CanonicalGoalEvaluation(canonical_goal_evaluation) => {
                    assert_eq!(canonical_goal_evaluation.cache_hit.replace(cache_hit), None);
                }
                _ => unreachable!(),
            };
        }
    }

    pub(super) fn goal_evaluation(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal_evaluation: ProofTreeBuilder<'tcx>,
        goals: &[Goal<'tcx, ty::Predicate<'tcx>>],
    ) {
        // Can't use `if let Some(this) = ecx.inspect.as_mut()` here because
        // we have to immutably use the `EvalCtxt` for `make_canonical_state`.
        if ecx.inspect.is_noop() {
            return;
        }

        match goal_evaluation.state.unwrap().tree {
            DebugSolver::NestedGoalEvaluation(mut nested_goal_evaluation) => {
                assert!(nested_goal_evaluation.returned_goals.is_empty());
                for &goal in goals {
                    let goal = Self::make_canonical_state(ecx, goal);
                    nested_goal_evaluation.returned_goals.push(goal);
                }

                if let DebugSolver::AddedGoalsEvaluation(added_goals_evaluation) =
                    ecx.inspect.as_mut().unwrap()
                {
                    added_goals_evaluation
                        .evaluations
                        .last_mut()
                        .unwrap()
                        .push(nested_goal_evaluation);
                } else {
                    unreachable!()
                }
            }
            DebugSolver::RootGoalEvaluation(mut root_goal_evaluation) => {
                assert!(root_goal_evaluation.returned_goals.is_empty());
                root_goal_evaluation.returned_goals.extend(goals);

                let this = ecx.inspect.as_mut().unwrap();
                assert!(matches!(this, DebugSolver::Root));
                *this = DebugSolver::RootGoalEvaluation(root_goal_evaluation);
            }

            _ => unreachable!(),
        }
    }

    pub(super) fn new_goal_evaluation_step(&mut self) -> ProofTreeBuilder<'tcx> {
        self.nested(|| WipGoalEvaluationStep {
            added_goals_evaluations: vec![],
            candidates: vec![],
            result: None,
        })
    }
    pub(super) fn goal_evaluation_step(&mut self, goal_evaluation_step: ProofTreeBuilder<'tcx>) {
        if let Some(this) = self.as_mut() {
            match (this, goal_evaluation_step.state.unwrap().tree) {
                (
                    DebugSolver::CanonicalGoalEvaluation(canonical_goal_evaluations),
                    DebugSolver::GoalEvaluationStep(goal_evaluation_step),
                ) => {
                    canonical_goal_evaluations.evaluation_steps.push(goal_evaluation_step);
                }
                _ => unreachable!(),
            }
        }
    }

    pub(super) fn new_goal_candidate(&mut self) -> ProofTreeBuilder<'tcx> {
        self.nested(|| WipGoalCandidate {
            added_goals_evaluations: vec![],
            candidates: vec![],
            kind: None,
        })
    }

    pub(super) fn probe_kind(&mut self, probe_kind: ProbeKind<'tcx>) {
        if let Some(this) = self.as_mut() {
            match this {
                DebugSolver::GoalCandidate(this) => {
                    assert_eq!(this.kind.replace(probe_kind), None)
                }
                _ => unreachable!(),
            }
        }
    }

    pub(super) fn goal_candidate(&mut self, candidate: ProofTreeBuilder<'tcx>) {
        if let Some(this) = self.as_mut() {
            match (this, candidate.state.unwrap().tree) {
                (
                    DebugSolver::GoalCandidate(WipGoalCandidate { candidates, .. })
                    | DebugSolver::GoalEvaluationStep(WipGoalEvaluationStep { candidates, .. }),
                    DebugSolver::GoalCandidate(candidate),
                ) => candidates.push(candidate),
                _ => unreachable!(),
            }
        }
    }

    pub(super) fn new_evaluate_added_goals(&mut self) -> ProofTreeBuilder<'tcx> {
        self.nested(|| WipAddedGoalsEvaluation { evaluations: vec![], result: None })
    }

    pub(super) fn evaluate_added_goals_loop_start(&mut self) {
        if let Some(this) = self.as_mut() {
            match this {
                DebugSolver::AddedGoalsEvaluation(this) => {
                    this.evaluations.push(vec![]);
                }
                _ => unreachable!(),
            }
        }
    }

    pub(super) fn eval_added_goals_result(&mut self, result: Result<Certainty, NoSolution>) {
        if let Some(this) = self.as_mut() {
            match this {
                DebugSolver::AddedGoalsEvaluation(this) => {
                    assert_eq!(this.result.replace(result), None);
                }
                _ => unreachable!(),
            }
        }
    }

    pub(super) fn added_goals_evaluation(
        &mut self,
        added_goals_evaluation: ProofTreeBuilder<'tcx>,
    ) {
        if let Some(this) = self.as_mut() {
            match (this, added_goals_evaluation.state.unwrap().tree) {
                (
                    DebugSolver::GoalEvaluationStep(WipGoalEvaluationStep {
                        added_goals_evaluations,
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

    pub(super) fn query_result(&mut self, result: QueryResult<'tcx>) {
        if let Some(this) = self.as_mut() {
            match this {
                DebugSolver::CanonicalGoalEvaluation(canonical_goal_evaluation) => {
                    assert_eq!(canonical_goal_evaluation.result.replace(result), None);
                }
                DebugSolver::GoalEvaluationStep(evaluation_step) => {
                    assert_eq!(evaluation_step.result.replace(result), None);
                }
                _ => unreachable!(),
            }
        }
    }
}
