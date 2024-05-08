//! Building proof trees incrementally during trait solving.
//!
//! This code is *a bit* of a mess and can hopefully be
//! mostly ignored. For a general overview of how it works,
//! see the comment on [ProofTreeBuilder].
use std::mem;

use rustc_infer::infer::InferCtxt;
use rustc_middle::bug;
use rustc_middle::infer::canonical::CanonicalVarValues;
use rustc_middle::traits::query::NoSolution;
use rustc_middle::traits::solve::{
    CanonicalInput, Certainty, Goal, GoalSource, QueryInput, QueryResult,
};
use rustc_middle::ty::{self, TyCtxt};
use rustc_session::config::DumpSolverProofTree;

use crate::solve::eval_ctxt::canonical;
use crate::solve::{self, inspect, GenerateProofTree};

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
    GoalEvaluationStep(WipGoalEvaluationStep<'tcx>),
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

impl<'tcx> From<WipGoalEvaluationStep<'tcx>> for DebugSolver<'tcx> {
    fn from(g: WipGoalEvaluationStep<'tcx>) -> DebugSolver<'tcx> {
        DebugSolver::GoalEvaluationStep(g)
    }
}

#[derive(Eq, PartialEq, Debug)]
struct WipGoalEvaluation<'tcx> {
    pub uncanonicalized_goal: Goal<'tcx, ty::Predicate<'tcx>>,
    pub kind: WipGoalEvaluationKind<'tcx>,
    pub evaluation: Option<WipCanonicalGoalEvaluation<'tcx>>,
}

impl<'tcx> WipGoalEvaluation<'tcx> {
    fn finalize(self) -> inspect::GoalEvaluation<'tcx> {
        inspect::GoalEvaluation {
            uncanonicalized_goal: self.uncanonicalized_goal,
            kind: match self.kind {
                WipGoalEvaluationKind::Root { orig_values } => {
                    inspect::GoalEvaluationKind::Root { orig_values }
                }
                WipGoalEvaluationKind::Nested => inspect::GoalEvaluationKind::Nested,
            },
            evaluation: self.evaluation.unwrap().finalize(),
        }
    }
}

#[derive(Eq, PartialEq, Debug)]
pub(in crate::solve) enum WipGoalEvaluationKind<'tcx> {
    Root { orig_values: Vec<ty::GenericArg<'tcx>> },
    Nested,
}

#[derive(Eq, PartialEq)]
pub(in crate::solve) enum WipCanonicalGoalEvaluationKind<'tcx> {
    Overflow,
    CycleInStack,
    ProvisionalCacheHit,
    Interned { revisions: &'tcx [inspect::GoalEvaluationStep<'tcx>] },
}

impl std::fmt::Debug for WipCanonicalGoalEvaluationKind<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Overflow => write!(f, "Overflow"),
            Self::CycleInStack => write!(f, "CycleInStack"),
            Self::ProvisionalCacheHit => write!(f, "ProvisionalCacheHit"),
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
            WipCanonicalGoalEvaluationKind::ProvisionalCacheHit => {
                inspect::CanonicalGoalEvaluationKind::ProvisionalCacheHit
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
    /// Unlike `EvalCtxt::var_values`, we append a new
    /// generic arg here whenever we create a new inference
    /// variable.
    ///
    /// This is necessary as we otherwise don't unify these
    /// vars when instantiating multiple `CanonicalState`.
    var_values: Vec<ty::GenericArg<'tcx>>,
    instantiated_goal: QueryInput<'tcx, ty::Predicate<'tcx>>,
    probe_depth: usize,
    evaluation: WipProbe<'tcx>,
}

impl<'tcx> WipGoalEvaluationStep<'tcx> {
    fn current_evaluation_scope(&mut self) -> &mut WipProbe<'tcx> {
        let mut current = &mut self.evaluation;
        for _ in 0..self.probe_depth {
            match current.steps.last_mut() {
                Some(WipProbeStep::NestedProbe(p)) => current = p,
                _ => bug!(),
            }
        }
        current
    }

    fn added_goals_evaluation(&mut self) -> &mut WipAddedGoalsEvaluation<'tcx> {
        let mut current = &mut self.evaluation;
        loop {
            match current.steps.last_mut() {
                Some(WipProbeStep::NestedProbe(p)) => current = p,
                Some(WipProbeStep::EvaluateGoals(evaluation)) => return evaluation,
                _ => bug!(),
            }
        }
    }

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
    initial_num_var_values: usize,
    steps: Vec<WipProbeStep<'tcx>>,
    kind: Option<inspect::ProbeKind<'tcx>>,
    final_state: Option<inspect::CanonicalState<'tcx, ()>>,
}

impl<'tcx> WipProbe<'tcx> {
    fn finalize(self) -> inspect::Probe<'tcx> {
        inspect::Probe {
            steps: self.steps.into_iter().map(WipProbeStep::finalize).collect(),
            kind: self.kind.unwrap(),
            final_state: self.final_state.unwrap(),
        }
    }
}

#[derive(Eq, PartialEq, Debug)]
enum WipProbeStep<'tcx> {
    AddGoal(GoalSource, inspect::CanonicalState<'tcx, Goal<'tcx, ty::Predicate<'tcx>>>),
    EvaluateGoals(WipAddedGoalsEvaluation<'tcx>),
    NestedProbe(WipProbe<'tcx>),
    MakeCanonicalResponse { shallow_certainty: Certainty },
    RecordImplArgs { impl_args: inspect::CanonicalState<'tcx, ty::GenericArgsRef<'tcx>> },
}

impl<'tcx> WipProbeStep<'tcx> {
    fn finalize(self) -> inspect::ProbeStep<'tcx> {
        match self {
            WipProbeStep::AddGoal(source, goal) => inspect::ProbeStep::AddGoal(source, goal),
            WipProbeStep::EvaluateGoals(eval) => inspect::ProbeStep::EvaluateGoals(eval.finalize()),
            WipProbeStep::NestedProbe(probe) => inspect::ProbeStep::NestedProbe(probe.finalize()),
            WipProbeStep::RecordImplArgs { impl_args } => {
                inspect::ProbeStep::RecordImplArgs { impl_args }
            }
            WipProbeStep::MakeCanonicalResponse { shallow_certainty } => {
                inspect::ProbeStep::MakeCanonicalResponse { shallow_certainty }
            }
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

    pub fn take_and_enter_probe(&mut self) -> ProofTreeBuilder<'tcx> {
        let mut nested = ProofTreeBuilder { state: self.state.take() };
        nested.enter_probe();
        nested
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
                match opts.next_solver.map(|c| c.dump_tree).unwrap_or_default() {
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
                solve::GoalEvaluationKind::Nested => WipGoalEvaluationKind::Nested,
            },
            evaluation: None,
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
                ) => {
                    let prev = goal_evaluation.evaluation.replace(canonical_goal_evaluation);
                    assert_eq!(prev, None);
                }
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

    pub fn goal_evaluation(&mut self, goal_evaluation: ProofTreeBuilder<'tcx>) {
        if let Some(this) = self.as_mut() {
            match (this, *goal_evaluation.state.unwrap()) {
                (
                    DebugSolver::GoalEvaluationStep(state),
                    DebugSolver::GoalEvaluation(goal_evaluation),
                ) => state
                    .added_goals_evaluation()
                    .evaluations
                    .last_mut()
                    .unwrap()
                    .push(goal_evaluation),
                (this @ DebugSolver::Root, goal_evaluation) => *this = goal_evaluation,
                _ => unreachable!(),
            }
        }
    }

    pub fn new_goal_evaluation_step(
        &mut self,
        var_values: CanonicalVarValues<'tcx>,
        instantiated_goal: QueryInput<'tcx, ty::Predicate<'tcx>>,
    ) -> ProofTreeBuilder<'tcx> {
        self.nested(|| WipGoalEvaluationStep {
            var_values: var_values.var_values.to_vec(),
            instantiated_goal,
            evaluation: WipProbe {
                initial_num_var_values: var_values.len(),
                steps: vec![],
                kind: None,
                final_state: None,
            },
            probe_depth: 0,
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

    pub fn add_var_value<T: Into<ty::GenericArg<'tcx>>>(&mut self, arg: T) {
        match self.as_mut() {
            None => {}
            Some(DebugSolver::GoalEvaluationStep(state)) => {
                state.var_values.push(arg.into());
            }
            Some(s) => bug!("tried to add var values to {s:?}"),
        }
    }

    pub fn enter_probe(&mut self) {
        match self.as_mut() {
            None => {}
            Some(DebugSolver::GoalEvaluationStep(state)) => {
                let initial_num_var_values = state.var_values.len();
                state.current_evaluation_scope().steps.push(WipProbeStep::NestedProbe(WipProbe {
                    initial_num_var_values,
                    steps: vec![],
                    kind: None,
                    final_state: None,
                }));
                state.probe_depth += 1;
            }
            Some(s) => bug!("tried to start probe to {s:?}"),
        }
    }

    pub fn probe_kind(&mut self, probe_kind: inspect::ProbeKind<'tcx>) {
        match self.as_mut() {
            None => {}
            Some(DebugSolver::GoalEvaluationStep(state)) => {
                let prev = state.current_evaluation_scope().kind.replace(probe_kind);
                assert_eq!(prev, None);
            }
            _ => bug!(),
        }
    }

    pub fn probe_final_state(
        &mut self,
        infcx: &InferCtxt<'tcx>,
        max_input_universe: ty::UniverseIndex,
    ) {
        match self.as_mut() {
            None => {}
            Some(DebugSolver::GoalEvaluationStep(state)) => {
                let final_state = canonical::make_canonical_state(
                    infcx,
                    &state.var_values,
                    max_input_universe,
                    (),
                );
                let prev = state.current_evaluation_scope().final_state.replace(final_state);
                assert_eq!(prev, None);
            }
            _ => bug!(),
        }
    }

    pub fn add_normalizes_to_goal(
        &mut self,
        infcx: &InferCtxt<'tcx>,
        max_input_universe: ty::UniverseIndex,
        goal: Goal<'tcx, ty::NormalizesTo<'tcx>>,
    ) {
        self.add_goal(
            infcx,
            max_input_universe,
            GoalSource::Misc,
            goal.with(infcx.tcx, goal.predicate),
        );
    }

    pub fn add_goal(
        &mut self,
        infcx: &InferCtxt<'tcx>,
        max_input_universe: ty::UniverseIndex,
        source: GoalSource,
        goal: Goal<'tcx, ty::Predicate<'tcx>>,
    ) {
        match self.as_mut() {
            None => {}
            Some(DebugSolver::GoalEvaluationStep(state)) => {
                let goal = canonical::make_canonical_state(
                    infcx,
                    &state.var_values,
                    max_input_universe,
                    goal,
                );
                state.current_evaluation_scope().steps.push(WipProbeStep::AddGoal(source, goal))
            }
            _ => bug!(),
        }
    }

    pub(crate) fn record_impl_args(
        &mut self,
        infcx: &InferCtxt<'tcx>,
        max_input_universe: ty::UniverseIndex,
        impl_args: ty::GenericArgsRef<'tcx>,
    ) {
        match self.as_mut() {
            Some(DebugSolver::GoalEvaluationStep(state)) => {
                let impl_args = canonical::make_canonical_state(
                    infcx,
                    &state.var_values,
                    max_input_universe,
                    impl_args,
                );
                state
                    .current_evaluation_scope()
                    .steps
                    .push(WipProbeStep::RecordImplArgs { impl_args });
            }
            None => {}
            _ => bug!(),
        }
    }

    pub fn make_canonical_response(&mut self, shallow_certainty: Certainty) {
        match self.as_mut() {
            Some(DebugSolver::GoalEvaluationStep(state)) => {
                state
                    .current_evaluation_scope()
                    .steps
                    .push(WipProbeStep::MakeCanonicalResponse { shallow_certainty });
            }
            None => {}
            _ => bug!(),
        }
    }

    pub fn finish_probe(mut self) -> ProofTreeBuilder<'tcx> {
        match self.as_mut() {
            None => {}
            Some(DebugSolver::GoalEvaluationStep(state)) => {
                assert_ne!(state.probe_depth, 0);
                let num_var_values = state.current_evaluation_scope().initial_num_var_values;
                state.var_values.truncate(num_var_values);
                state.probe_depth -= 1;
            }
            _ => bug!(),
        }

        self
    }

    pub fn start_evaluate_added_goals(&mut self) {
        match self.as_mut() {
            None => {}
            Some(DebugSolver::GoalEvaluationStep(state)) => {
                state.current_evaluation_scope().steps.push(WipProbeStep::EvaluateGoals(
                    WipAddedGoalsEvaluation { evaluations: vec![], result: None },
                ));
            }
            _ => bug!(),
        }
    }

    pub fn start_evaluate_added_goals_step(&mut self) {
        match self.as_mut() {
            None => {}
            Some(DebugSolver::GoalEvaluationStep(state)) => {
                state.added_goals_evaluation().evaluations.push(vec![]);
            }
            _ => bug!(),
        }
    }

    pub fn evaluate_added_goals_result(&mut self, result: Result<Certainty, NoSolution>) {
        match self.as_mut() {
            None => {}
            Some(DebugSolver::GoalEvaluationStep(state)) => {
                let prev = state.added_goals_evaluation().result.replace(result);
                assert_eq!(prev, None);
            }
            _ => bug!(),
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
