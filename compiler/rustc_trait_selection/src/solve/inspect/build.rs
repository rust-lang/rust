//! Building proof trees incrementally during trait solving.
//!
//! This code is *a bit* of a mess and can hopefully be
//! mostly ignored. For a general overview of how it works,
//! see the comment on [ProofTreeBuilder].
use std::mem;

use rustc_middle::traits::query::NoSolution;
use rustc_middle::traits::solve::{
    CanonicalInput, Certainty, Goal, IsNormalizesToHack, QueryInput, QueryResult,
};
use rustc_middle::ty::{self, TyCtxt};
use rustc_session::config::DumpSolverProofTree;

use crate::solve::{self, inspect, EvalCtxt, GenerateProofTree};

/// The core data structure when building proof trees.
///
/// In case the current evaluation does not generate a proof
/// tree, `state` is simply `None` and we avoid any work.
///
/// The possible states of the solver are represented via
/// variants of [DebugSolver]. For any nested computation we call
/// `ProofTreeBuilder::new_nested_computation_kind` which
/// creates a new `ProofTreeBuilder` to temporarily replace the
/// current one. Once that nested computation is done,
/// `ProofTreeBuilder::nested_computation_kind` is called
/// to add the finished nested evaluation to the parent.
///
/// We provide additional information to the current state
/// by calling methods such as `ProofTreeBuilder::probe_kind`.
///
/// The actual structure closely mirrors the finished proof
/// trees. At the end of trait solving `ProofTreeBuilder::finalize`
/// is called to recursively convert the whole structure to a
/// finished proof tree.
pub(in crate::solve) struct ProofTreeBuilder<'tcx> {
    state: Option<Box<DebugSolver<'tcx>>>,
}

/// The current state of the proof tree builder, at most places
/// in the code, only one or two variants are actually possible.
///
/// We simply ICE in case that assumption is broken.
#[derive(Debug)]
enum DebugSolver<'tcx> {
    Root,
    GoalEvaluation(WipGoalEvaluation<'tcx>),
    CanonicalGoalEvaluation(WipCanonicalGoalEvaluation<'tcx>),
    AddedGoalsEvaluation(WipAddedGoalsEvaluation<'tcx>),
    GoalEvaluationStep(WipGoalEvaluationStep<'tcx>),
    Probe(WipProbe<'tcx>),
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

impl<'tcx> From<WipProbe<'tcx>> for DebugSolver<'tcx> {
    fn from(p: WipProbe<'tcx>) -> DebugSolver<'tcx> {
        DebugSolver::Probe(p)
    }
}

#[derive(Eq, PartialEq, Debug)]
struct WipGoalEvaluation<'tcx> {
    pub uncanonicalized_goal: Goal<'tcx, ty::Predicate<'tcx>>,
    pub kind: WipGoalEvaluationKind<'tcx>,
    pub evaluation: Option<WipCanonicalGoalEvaluation<'tcx>>,
    pub returned_goals: Vec<Goal<'tcx, ty::Predicate<'tcx>>>,
}

impl<'tcx> WipGoalEvaluation<'tcx> {
    fn finalize(self) -> inspect::GoalEvaluation<'tcx> {
        inspect::GoalEvaluation {
            uncanonicalized_goal: self.uncanonicalized_goal,
            kind: match self.kind {
                WipGoalEvaluationKind::Root { orig_values } => {
                    inspect::GoalEvaluationKind::Root { orig_values }
                }
                WipGoalEvaluationKind::Nested { is_normalizes_to_hack } => {
                    inspect::GoalEvaluationKind::Nested { is_normalizes_to_hack }
                }
            },
            evaluation: self.evaluation.unwrap().finalize(),
            returned_goals: self.returned_goals,
        }
    }
}

#[derive(Eq, PartialEq, Debug)]
pub(in crate::solve) enum WipGoalEvaluationKind<'tcx> {
    Root { orig_values: Vec<ty::GenericArg<'tcx>> },
    Nested { is_normalizes_to_hack: IsNormalizesToHack },
}

#[derive(Eq, PartialEq)]
pub(in crate::solve) enum WipCanonicalGoalEvaluationKind<'tcx> {
    Overflow,
    CycleInStack,
    Interned { revisions: &'tcx [inspect::GoalEvaluationStep<'tcx>] },
}

impl std::fmt::Debug for WipCanonicalGoalEvaluationKind<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Overflow => write!(f, "Overflow"),
            Self::CycleInStack => write!(f, "CycleInStack"),
            Self::Interned { revisions: _ } => f.debug_struct("Interned").finish_non_exhaustive(),
        }
    }
}

#[derive(Eq, PartialEq, Debug)]
struct WipCanonicalGoalEvaluation<'tcx> {
    goal: CanonicalInput<'tcx>,
    kind: Option<WipCanonicalGoalEvaluationKind<'tcx>>,
    /// Only used for uncached goals. After we finished evaluating
    /// the goal, this is interned and moved into `kind`.
    revisions: Vec<WipGoalEvaluationStep<'tcx>>,
    result: Option<QueryResult<'tcx>>,
}

impl<'tcx> WipCanonicalGoalEvaluation<'tcx> {
    fn finalize(self) -> inspect::CanonicalGoalEvaluation<'tcx> {
        assert!(self.revisions.is_empty());
        let kind = match self.kind.unwrap() {
            WipCanonicalGoalEvaluationKind::Overflow => {
                inspect::CanonicalGoalEvaluationKind::Overflow
            }
            WipCanonicalGoalEvaluationKind::CycleInStack => {
                inspect::CanonicalGoalEvaluationKind::CycleInStack
            }
            WipCanonicalGoalEvaluationKind::Interned { revisions } => {
                inspect::CanonicalGoalEvaluationKind::Evaluation { revisions }
            }
        };

        inspect::CanonicalGoalEvaluation { goal: self.goal, kind, result: self.result.unwrap() }
    }
}

#[derive(Eq, PartialEq, Debug)]
struct WipAddedGoalsEvaluation<'tcx> {
    evaluations: Vec<Vec<WipGoalEvaluation<'tcx>>>,
    result: Option<Result<Certainty, NoSolution>>,
}

impl<'tcx> WipAddedGoalsEvaluation<'tcx> {
    fn finalize(self) -> inspect::AddedGoalsEvaluation<'tcx> {
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
struct WipGoalEvaluationStep<'tcx> {
    instantiated_goal: QueryInput<'tcx, ty::Predicate<'tcx>>,

    evaluation: WipProbe<'tcx>,
}

impl<'tcx> WipGoalEvaluationStep<'tcx> {
    fn finalize(self) -> inspect::GoalEvaluationStep<'tcx> {
        let evaluation = self.evaluation.finalize();
        match evaluation.kind {
            inspect::ProbeKind::Root { .. } => (),
            _ => unreachable!("unexpected root evaluation: {evaluation:?}"),
        }
        inspect::GoalEvaluationStep { instantiated_goal: self.instantiated_goal, evaluation }
    }
}

#[derive(Eq, PartialEq, Debug)]
struct WipProbe<'tcx> {
    pub steps: Vec<WipProbeStep<'tcx>>,
    pub kind: Option<inspect::ProbeKind<'tcx>>,
}

impl<'tcx> WipProbe<'tcx> {
    fn finalize(self) -> inspect::Probe<'tcx> {
        inspect::Probe {
            steps: self.steps.into_iter().map(WipProbeStep::finalize).collect(),
            kind: self.kind.unwrap(),
        }
    }
}

#[derive(Eq, PartialEq, Debug)]
enum WipProbeStep<'tcx> {
    AddGoal(inspect::CanonicalState<'tcx, Goal<'tcx, ty::Predicate<'tcx>>>),
    EvaluateGoals(WipAddedGoalsEvaluation<'tcx>),
    NestedProbe(WipProbe<'tcx>),
    CommitIfOkStart,
    CommitIfOkSuccess,
}

impl<'tcx> WipProbeStep<'tcx> {
    fn finalize(self) -> inspect::ProbeStep<'tcx> {
        match self {
            WipProbeStep::AddGoal(goal) => inspect::ProbeStep::AddGoal(goal),
            WipProbeStep::EvaluateGoals(eval) => inspect::ProbeStep::EvaluateGoals(eval.finalize()),
            WipProbeStep::NestedProbe(probe) => inspect::ProbeStep::NestedProbe(probe.finalize()),
            WipProbeStep::CommitIfOkStart => inspect::ProbeStep::CommitIfOkStart,
            WipProbeStep::CommitIfOkSuccess => inspect::ProbeStep::CommitIfOkSuccess,
        }
    }
}

impl<'tcx> ProofTreeBuilder<'tcx> {
    fn new(state: impl Into<DebugSolver<'tcx>>) -> ProofTreeBuilder<'tcx> {
        ProofTreeBuilder { state: Some(Box::new(state.into())) }
    }

    fn nested<T: Into<DebugSolver<'tcx>>>(&self, state: impl FnOnce() -> T) -> Self {
        ProofTreeBuilder { state: self.state.as_ref().map(|_| Box::new(state().into())) }
    }

    fn as_mut(&mut self) -> Option<&mut DebugSolver<'tcx>> {
        self.state.as_deref_mut()
    }

    pub fn finalize(self) -> Option<inspect::GoalEvaluation<'tcx>> {
        match *self.state? {
            DebugSolver::GoalEvaluation(wip_goal_evaluation) => {
                Some(wip_goal_evaluation.finalize())
            }
            root => unreachable!("unexpected proof tree builder root node: {:?}", root),
        }
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
                    DumpSolverProofTree::Always => ProofTreeBuilder::new_root(),
                    // `OnError` is handled by reevaluating goals in error
                    // reporting with `GenerateProofTree::Yes`.
                    DumpSolverProofTree::OnError | DumpSolverProofTree::Never => {
                        ProofTreeBuilder::new_noop()
                    }
                }
            }
            GenerateProofTree::Yes => ProofTreeBuilder::new_root(),
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

    pub(in crate::solve) fn new_goal_evaluation(
        &mut self,
        goal: Goal<'tcx, ty::Predicate<'tcx>>,
        orig_values: &[ty::GenericArg<'tcx>],
        kind: solve::GoalEvaluationKind,
    ) -> ProofTreeBuilder<'tcx> {
        self.nested(|| WipGoalEvaluation {
            uncanonicalized_goal: goal,
            kind: match kind {
                solve::GoalEvaluationKind::Root => {
                    WipGoalEvaluationKind::Root { orig_values: orig_values.to_vec() }
                }
                solve::GoalEvaluationKind::Nested { is_normalizes_to_hack } => {
                    WipGoalEvaluationKind::Nested { is_normalizes_to_hack }
                }
            },
            evaluation: None,
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

    pub fn finalize_evaluation(
        &mut self,
        tcx: TyCtxt<'tcx>,
    ) -> Option<&'tcx [inspect::GoalEvaluationStep<'tcx>]> {
        self.as_mut().map(|this| match this {
            DebugSolver::CanonicalGoalEvaluation(evaluation) => {
                let revisions = mem::take(&mut evaluation.revisions)
                    .into_iter()
                    .map(WipGoalEvaluationStep::finalize);
                let revisions = &*tcx.arena.alloc_from_iter(revisions);
                let kind = WipCanonicalGoalEvaluationKind::Interned { revisions };
                assert_eq!(evaluation.kind.replace(kind), None);
                revisions
            }
            _ => unreachable!(),
        })
    }

    pub fn canonical_goal_evaluation(&mut self, canonical_goal_evaluation: ProofTreeBuilder<'tcx>) {
        if let Some(this) = self.as_mut() {
            match (this, *canonical_goal_evaluation.state.unwrap()) {
                (
                    DebugSolver::GoalEvaluation(goal_evaluation),
                    DebugSolver::CanonicalGoalEvaluation(canonical_goal_evaluation),
                ) => goal_evaluation.evaluation = Some(canonical_goal_evaluation),
                _ => unreachable!(),
            }
        }
    }

    pub fn goal_evaluation_kind(&mut self, kind: WipCanonicalGoalEvaluationKind<'tcx>) {
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
        self.nested(|| WipGoalEvaluationStep {
            instantiated_goal,
            evaluation: WipProbe { steps: vec![], kind: None },
        })
    }
    pub fn goal_evaluation_step(&mut self, goal_evaluation_step: ProofTreeBuilder<'tcx>) {
        if let Some(this) = self.as_mut() {
            match (this, *goal_evaluation_step.state.unwrap()) {
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

    pub fn new_probe(&mut self) -> ProofTreeBuilder<'tcx> {
        self.nested(|| WipProbe { steps: vec![], kind: None })
    }

    pub fn probe_kind(&mut self, probe_kind: inspect::ProbeKind<'tcx>) {
        if let Some(this) = self.as_mut() {
            match this {
                DebugSolver::Probe(this) => {
                    assert_eq!(this.kind.replace(probe_kind), None)
                }
                _ => unreachable!(),
            }
        }
    }

    pub fn add_goal(ecx: &mut EvalCtxt<'_, 'tcx>, goal: Goal<'tcx, ty::Predicate<'tcx>>) {
        // Can't use `if let Some(this) = ecx.inspect.as_mut()` here because
        // we have to immutably use the `EvalCtxt` for `make_canonical_state`.
        if ecx.inspect.is_noop() {
            return;
        }

        let goal = Self::make_canonical_state(ecx, goal);

        match ecx.inspect.as_mut().unwrap() {
            DebugSolver::GoalEvaluationStep(WipGoalEvaluationStep {
                evaluation: WipProbe { steps, .. },
                ..
            })
            | DebugSolver::Probe(WipProbe { steps, .. }) => steps.push(WipProbeStep::AddGoal(goal)),
            s => unreachable!("tried to add {goal:?} to {s:?}"),
        }
    }

    pub fn finish_probe(&mut self, probe: ProofTreeBuilder<'tcx>) {
        if let Some(this) = self.as_mut() {
            match (this, *probe.state.unwrap()) {
                (
                    DebugSolver::Probe(WipProbe { steps, .. })
                    | DebugSolver::GoalEvaluationStep(WipGoalEvaluationStep {
                        evaluation: WipProbe { steps, .. },
                        ..
                    }),
                    DebugSolver::Probe(probe),
                ) => steps.push(WipProbeStep::NestedProbe(probe)),
                _ => unreachable!(),
            }
        }
    }

    /// Used by `EvalCtxt::commit_if_ok` to flatten the work done inside
    /// of the probe into the parent.
    pub fn integrate_snapshot(&mut self, probe: ProofTreeBuilder<'tcx>) {
        if let Some(this) = self.as_mut() {
            match (this, *probe.state.unwrap()) {
                (
                    DebugSolver::Probe(WipProbe { steps, .. })
                    | DebugSolver::GoalEvaluationStep(WipGoalEvaluationStep {
                        evaluation: WipProbe { steps, .. },
                        ..
                    }),
                    DebugSolver::Probe(probe),
                ) => {
                    steps.push(WipProbeStep::CommitIfOkStart);
                    assert_eq!(probe.kind, None);
                    steps.extend(probe.steps);
                    steps.push(WipProbeStep::CommitIfOkSuccess);
                }
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
            match (this, *added_goals_evaluation.state.unwrap()) {
                (
                    DebugSolver::GoalEvaluationStep(WipGoalEvaluationStep {
                        evaluation: WipProbe { steps, .. },
                        ..
                    })
                    | DebugSolver::Probe(WipProbe { steps, .. }),
                    DebugSolver::AddedGoalsEvaluation(added_goals_evaluation),
                ) => steps.push(WipProbeStep::EvaluateGoals(added_goals_evaluation)),
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
                        evaluation_step
                            .evaluation
                            .kind
                            .replace(inspect::ProbeKind::Root { result }),
                        None
                    );
                }
                _ => unreachable!(),
            }
        }
    }
}
