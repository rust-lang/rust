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
use rustc_middle::ty::{self, TyCtxt};
use rustc_next_trait_solver::solve::{
    CanonicalInput, Certainty, Goal, GoalSource, QueryInput, QueryResult,
};
use rustc_session::config::DumpSolverProofTree;
use rustc_type_ir::Interner;

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
pub(in crate::solve) struct ProofTreeBuilder<I: Interner> {
    state: Option<Box<DebugSolver<I>>>,
}

/// The current state of the proof tree builder, at most places
/// in the code, only one or two variants are actually possible.
///
/// We simply ICE in case that assumption is broken.
#[derive(derivative::Derivative)]
#[derivative(Debug(bound = ""))]
enum DebugSolver<I: Interner> {
    Root,
    GoalEvaluation(WipGoalEvaluation<I>),
    CanonicalGoalEvaluation(WipCanonicalGoalEvaluation<I>),
    GoalEvaluationStep(WipGoalEvaluationStep<I>),
}

impl<I: Interner> From<WipGoalEvaluation<I>> for DebugSolver<I> {
    fn from(g: WipGoalEvaluation<I>) -> DebugSolver<I> {
        DebugSolver::GoalEvaluation(g)
    }
}

impl<I: Interner> From<WipCanonicalGoalEvaluation<I>> for DebugSolver<I> {
    fn from(g: WipCanonicalGoalEvaluation<I>) -> DebugSolver<I> {
        DebugSolver::CanonicalGoalEvaluation(g)
    }
}

impl<I: Interner> From<WipGoalEvaluationStep<I>> for DebugSolver<I> {
    fn from(g: WipGoalEvaluationStep<I>) -> DebugSolver<I> {
        DebugSolver::GoalEvaluationStep(g)
    }
}

#[derive(derivative::Derivative)]
#[derivative(PartialEq(bound = ""), Eq(bound = ""), Debug(bound = ""))]
struct WipGoalEvaluation<I: Interner> {
    pub uncanonicalized_goal: Goal<I, I::Predicate>,
    pub kind: WipGoalEvaluationKind<I>,
    pub evaluation: Option<WipCanonicalGoalEvaluation<I>>,
}

impl<I: Interner> WipGoalEvaluation<I> {
    fn finalize(self) -> inspect::GoalEvaluation<I> {
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

#[derive(derivative::Derivative)]
#[derivative(PartialEq(bound = ""), Eq(bound = ""), Debug(bound = ""))]
pub(in crate::solve) enum WipGoalEvaluationKind<I: Interner> {
    Root { orig_values: Vec<I::GenericArg> },
    Nested,
}

#[derive(derivative::Derivative)]
#[derivative(PartialEq(bound = ""), Eq(bound = ""))]
pub(in crate::solve) enum WipCanonicalGoalEvaluationKind<I: Interner> {
    Overflow,
    CycleInStack,
    ProvisionalCacheHit,
    Interned { revisions: I::GoalEvaluationSteps },
}

impl<I: Interner> std::fmt::Debug for WipCanonicalGoalEvaluationKind<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Overflow => write!(f, "Overflow"),
            Self::CycleInStack => write!(f, "CycleInStack"),
            Self::ProvisionalCacheHit => write!(f, "ProvisionalCacheHit"),
            Self::Interned { revisions: _ } => f.debug_struct("Interned").finish_non_exhaustive(),
        }
    }
}

#[derive(derivative::Derivative)]
#[derivative(PartialEq(bound = ""), Eq(bound = ""), Debug(bound = ""))]
struct WipCanonicalGoalEvaluation<I: Interner> {
    goal: CanonicalInput<I>,
    kind: Option<WipCanonicalGoalEvaluationKind<I>>,
    /// Only used for uncached goals. After we finished evaluating
    /// the goal, this is interned and moved into `kind`.
    revisions: Vec<WipGoalEvaluationStep<I>>,
    result: Option<QueryResult<I>>,
}

impl<I: Interner> WipCanonicalGoalEvaluation<I> {
    fn finalize(self) -> inspect::CanonicalGoalEvaluation<I> {
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

#[derive(derivative::Derivative)]
#[derivative(PartialEq(bound = ""), Eq(bound = ""), Debug(bound = ""))]
struct WipAddedGoalsEvaluation<I: Interner> {
    evaluations: Vec<Vec<WipGoalEvaluation<I>>>,
    result: Option<Result<Certainty, NoSolution>>,
}

impl<I: Interner> WipAddedGoalsEvaluation<I> {
    fn finalize(self) -> inspect::AddedGoalsEvaluation<I> {
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

#[derive(derivative::Derivative)]
#[derivative(PartialEq(bound = ""), Eq(bound = ""), Debug(bound = ""))]
struct WipGoalEvaluationStep<I: Interner> {
    /// Unlike `EvalCtxt::var_values`, we append a new
    /// generic arg here whenever we create a new inference
    /// variable.
    ///
    /// This is necessary as we otherwise don't unify these
    /// vars when instantiating multiple `CanonicalState`.
    var_values: Vec<I::GenericArg>,
    instantiated_goal: QueryInput<I, I::Predicate>,
    probe_depth: usize,
    evaluation: WipProbe<I>,
}

impl<I: Interner> WipGoalEvaluationStep<I> {
    fn current_evaluation_scope(&mut self) -> &mut WipProbe<I> {
        let mut current = &mut self.evaluation;
        for _ in 0..self.probe_depth {
            match current.steps.last_mut() {
                Some(WipProbeStep::NestedProbe(p)) => current = p,
                _ => bug!(),
            }
        }
        current
    }

    fn added_goals_evaluation(&mut self) -> &mut WipAddedGoalsEvaluation<I> {
        let mut current = &mut self.evaluation;
        loop {
            match current.steps.last_mut() {
                Some(WipProbeStep::NestedProbe(p)) => current = p,
                Some(WipProbeStep::EvaluateGoals(evaluation)) => return evaluation,
                _ => bug!(),
            }
        }
    }

    fn finalize(self) -> inspect::GoalEvaluationStep<I> {
        let evaluation = self.evaluation.finalize();
        match evaluation.kind {
            inspect::ProbeKind::Root { .. } => (),
            _ => unreachable!("unexpected root evaluation: {evaluation:?}"),
        }
        inspect::GoalEvaluationStep { instantiated_goal: self.instantiated_goal, evaluation }
    }
}

#[derive(derivative::Derivative)]
#[derivative(PartialEq(bound = ""), Eq(bound = ""), Debug(bound = ""))]
struct WipProbe<I: Interner> {
    initial_num_var_values: usize,
    steps: Vec<WipProbeStep<I>>,
    kind: Option<inspect::ProbeKind<I>>,
    final_state: Option<inspect::CanonicalState<I, ()>>,
}

impl<I: Interner> WipProbe<I> {
    fn finalize(self) -> inspect::Probe<I> {
        inspect::Probe {
            steps: self.steps.into_iter().map(WipProbeStep::finalize).collect(),
            kind: self.kind.unwrap(),
            final_state: self.final_state.unwrap(),
        }
    }
}

#[derive(derivative::Derivative)]
#[derivative(PartialEq(bound = ""), Eq(bound = ""), Debug(bound = ""))]
enum WipProbeStep<I: Interner> {
    AddGoal(GoalSource, inspect::CanonicalState<I, Goal<I, I::Predicate>>),
    EvaluateGoals(WipAddedGoalsEvaluation<I>),
    NestedProbe(WipProbe<I>),
    MakeCanonicalResponse { shallow_certainty: Certainty },
    RecordImplArgs { impl_args: inspect::CanonicalState<I, I::GenericArgs> },
}

impl<I: Interner> WipProbeStep<I> {
    fn finalize(self) -> inspect::ProbeStep<I> {
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

// FIXME: Genericize this impl.
impl<'tcx> ProofTreeBuilder<TyCtxt<'tcx>> {
    fn new(state: impl Into<DebugSolver<TyCtxt<'tcx>>>) -> ProofTreeBuilder<TyCtxt<'tcx>> {
        ProofTreeBuilder { state: Some(Box::new(state.into())) }
    }

    fn nested<T: Into<DebugSolver<TyCtxt<'tcx>>>>(&self, state: impl FnOnce() -> T) -> Self {
        ProofTreeBuilder { state: self.state.as_ref().map(|_| Box::new(state().into())) }
    }

    fn as_mut(&mut self) -> Option<&mut DebugSolver<TyCtxt<'tcx>>> {
        self.state.as_deref_mut()
    }

    pub fn take_and_enter_probe(&mut self) -> ProofTreeBuilder<TyCtxt<'tcx>> {
        let mut nested = ProofTreeBuilder { state: self.state.take() };
        nested.enter_probe();
        nested
    }

    pub fn finalize(self) -> Option<inspect::GoalEvaluation<TyCtxt<'tcx>>> {
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
    ) -> ProofTreeBuilder<TyCtxt<'tcx>> {
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

    pub fn new_root() -> ProofTreeBuilder<TyCtxt<'tcx>> {
        ProofTreeBuilder::new(DebugSolver::Root)
    }

    pub fn new_noop() -> ProofTreeBuilder<TyCtxt<'tcx>> {
        ProofTreeBuilder { state: None }
    }

    pub fn is_noop(&self) -> bool {
        self.state.is_none()
    }

    pub(in crate::solve) fn new_goal_evaluation(
        &mut self,
        goal: Goal<TyCtxt<'tcx>, ty::Predicate<'tcx>>,
        orig_values: &[ty::GenericArg<'tcx>],
        kind: solve::GoalEvaluationKind,
    ) -> ProofTreeBuilder<TyCtxt<'tcx>> {
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
        goal: CanonicalInput<TyCtxt<'tcx>>,
    ) -> ProofTreeBuilder<TyCtxt<'tcx>> {
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
    ) -> Option<&'tcx [inspect::GoalEvaluationStep<TyCtxt<'tcx>>]> {
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

    pub fn canonical_goal_evaluation(
        &mut self,
        canonical_goal_evaluation: ProofTreeBuilder<TyCtxt<'tcx>>,
    ) {
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

    pub fn goal_evaluation_kind(&mut self, kind: WipCanonicalGoalEvaluationKind<TyCtxt<'tcx>>) {
        if let Some(this) = self.as_mut() {
            match this {
                DebugSolver::CanonicalGoalEvaluation(canonical_goal_evaluation) => {
                    assert_eq!(canonical_goal_evaluation.kind.replace(kind), None);
                }
                _ => unreachable!(),
            };
        }
    }

    pub fn goal_evaluation(&mut self, goal_evaluation: ProofTreeBuilder<TyCtxt<'tcx>>) {
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
        instantiated_goal: QueryInput<TyCtxt<'tcx>, ty::Predicate<'tcx>>,
    ) -> ProofTreeBuilder<TyCtxt<'tcx>> {
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

    pub fn goal_evaluation_step(&mut self, goal_evaluation_step: ProofTreeBuilder<TyCtxt<'tcx>>) {
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

    pub fn probe_kind(&mut self, probe_kind: inspect::ProbeKind<TyCtxt<'tcx>>) {
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
        goal: Goal<TyCtxt<'tcx>, ty::NormalizesTo<'tcx>>,
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
        goal: Goal<TyCtxt<'tcx>, ty::Predicate<'tcx>>,
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

    pub fn finish_probe(mut self) -> ProofTreeBuilder<TyCtxt<'tcx>> {
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

    pub fn query_result(&mut self, result: QueryResult<TyCtxt<'tcx>>) {
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
