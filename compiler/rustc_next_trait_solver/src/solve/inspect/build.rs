//! Building proof trees incrementally during trait solving.
//!
//! This code is *a bit* of a mess and can hopefully be
//! mostly ignored. For a general overview of how it works,
//! see the comment on [ProofTreeBuilder].

use std::marker::PhantomData;
use std::mem;

use derive_where::derive_where;
use rustc_type_ir::inherent::*;
use rustc_type_ir::{self as ty, search_graph, Interner};

use crate::delegate::SolverDelegate;
use crate::solve::eval_ctxt::canonical;
use crate::solve::inspect;
use crate::solve::{
    CanonicalInput, Certainty, GenerateProofTree, Goal, GoalEvaluationKind, GoalSource, QueryInput,
    QueryResult,
};

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
pub(crate) struct ProofTreeBuilder<D, I = <D as SolverDelegate>::Interner>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    _infcx: PhantomData<D>,
    state: Option<Box<DebugSolver<I>>>,
}

/// The current state of the proof tree builder, at most places
/// in the code, only one or two variants are actually possible.
///
/// We simply ICE in case that assumption is broken.
#[derive_where(Debug; I: Interner)]
enum DebugSolver<I: Interner> {
    Root,
    GoalEvaluation(WipGoalEvaluation<I>),
    CanonicalGoalEvaluation(WipCanonicalGoalEvaluation<I>),
    CanonicalGoalEvaluationStep(WipCanonicalGoalEvaluationStep<I>),
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

impl<I: Interner> From<WipCanonicalGoalEvaluationStep<I>> for DebugSolver<I> {
    fn from(g: WipCanonicalGoalEvaluationStep<I>) -> DebugSolver<I> {
        DebugSolver::CanonicalGoalEvaluationStep(g)
    }
}

#[derive_where(PartialEq, Eq, Debug; I: Interner)]
struct WipGoalEvaluation<I: Interner> {
    pub uncanonicalized_goal: Goal<I, I::Predicate>,
    pub orig_values: Vec<I::GenericArg>,
    pub evaluation: Option<WipCanonicalGoalEvaluation<I>>,
}

impl<I: Interner> WipGoalEvaluation<I> {
    fn finalize(self) -> inspect::GoalEvaluation<I> {
        inspect::GoalEvaluation {
            uncanonicalized_goal: self.uncanonicalized_goal,
            orig_values: self.orig_values,
            evaluation: self.evaluation.unwrap().finalize(),
        }
    }
}

#[derive_where(PartialEq, Eq; I: Interner)]
pub(in crate::solve) enum WipCanonicalGoalEvaluationKind<I: Interner> {
    Overflow,
    CycleInStack,
    ProvisionalCacheHit,
    Interned { final_revision: I::CanonicalGoalEvaluationStepRef },
}

impl<I: Interner> std::fmt::Debug for WipCanonicalGoalEvaluationKind<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Overflow => write!(f, "Overflow"),
            Self::CycleInStack => write!(f, "CycleInStack"),
            Self::ProvisionalCacheHit => write!(f, "ProvisionalCacheHit"),
            Self::Interned { final_revision: _ } => {
                f.debug_struct("Interned").finish_non_exhaustive()
            }
        }
    }
}

#[derive_where(PartialEq, Eq, Debug; I: Interner)]
struct WipCanonicalGoalEvaluation<I: Interner> {
    goal: CanonicalInput<I>,
    kind: Option<WipCanonicalGoalEvaluationKind<I>>,
    /// Only used for uncached goals. After we finished evaluating
    /// the goal, this is interned and moved into `kind`.
    final_revision: Option<WipCanonicalGoalEvaluationStep<I>>,
    result: Option<QueryResult<I>>,
}

impl<I: Interner> WipCanonicalGoalEvaluation<I> {
    fn finalize(self) -> inspect::CanonicalGoalEvaluation<I> {
        // We've already interned the final revision in
        // `fn finalize_canonical_goal_evaluation`.
        assert!(self.final_revision.is_none());
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
            WipCanonicalGoalEvaluationKind::Interned { final_revision } => {
                inspect::CanonicalGoalEvaluationKind::Evaluation { final_revision }
            }
        };

        inspect::CanonicalGoalEvaluation { goal: self.goal, kind, result: self.result.unwrap() }
    }
}

#[derive_where(PartialEq, Eq, Debug; I: Interner)]
struct WipCanonicalGoalEvaluationStep<I: Interner> {
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

impl<I: Interner> WipCanonicalGoalEvaluationStep<I> {
    fn current_evaluation_scope(&mut self) -> &mut WipProbe<I> {
        let mut current = &mut self.evaluation;
        for _ in 0..self.probe_depth {
            match current.steps.last_mut() {
                Some(WipProbeStep::NestedProbe(p)) => current = p,
                _ => panic!(),
            }
        }
        current
    }

    fn finalize(self) -> inspect::CanonicalGoalEvaluationStep<I> {
        let evaluation = self.evaluation.finalize();
        match evaluation.kind {
            inspect::ProbeKind::Root { .. } => (),
            _ => unreachable!("unexpected root evaluation: {evaluation:?}"),
        }
        inspect::CanonicalGoalEvaluationStep {
            instantiated_goal: self.instantiated_goal,
            evaluation,
        }
    }
}

#[derive_where(PartialEq, Eq, Debug; I: Interner)]
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

#[derive_where(PartialEq, Eq, Debug; I: Interner)]
enum WipProbeStep<I: Interner> {
    AddGoal(GoalSource, inspect::CanonicalState<I, Goal<I, I::Predicate>>),
    NestedProbe(WipProbe<I>),
    MakeCanonicalResponse { shallow_certainty: Certainty },
    RecordImplArgs { impl_args: inspect::CanonicalState<I, I::GenericArgs> },
}

impl<I: Interner> WipProbeStep<I> {
    fn finalize(self) -> inspect::ProbeStep<I> {
        match self {
            WipProbeStep::AddGoal(source, goal) => inspect::ProbeStep::AddGoal(source, goal),
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

impl<D: SolverDelegate<Interner = I>, I: Interner> ProofTreeBuilder<D> {
    fn new(state: impl Into<DebugSolver<I>>) -> ProofTreeBuilder<D> {
        ProofTreeBuilder { state: Some(Box::new(state.into())), _infcx: PhantomData }
    }

    fn opt_nested<T: Into<DebugSolver<I>>>(&self, state: impl FnOnce() -> Option<T>) -> Self {
        ProofTreeBuilder {
            state: self.state.as_ref().and_then(|_| Some(state()?.into())).map(Box::new),
            _infcx: PhantomData,
        }
    }

    fn nested<T: Into<DebugSolver<I>>>(&self, state: impl FnOnce() -> T) -> Self {
        ProofTreeBuilder {
            state: self.state.as_ref().map(|_| Box::new(state().into())),
            _infcx: PhantomData,
        }
    }

    fn as_mut(&mut self) -> Option<&mut DebugSolver<I>> {
        self.state.as_deref_mut()
    }

    pub fn take_and_enter_probe(&mut self) -> ProofTreeBuilder<D> {
        let mut nested = ProofTreeBuilder { state: self.state.take(), _infcx: PhantomData };
        nested.enter_probe();
        nested
    }

    pub fn finalize(self) -> Option<inspect::GoalEvaluation<I>> {
        match *self.state? {
            DebugSolver::GoalEvaluation(wip_goal_evaluation) => {
                Some(wip_goal_evaluation.finalize())
            }
            root => unreachable!("unexpected proof tree builder root node: {:?}", root),
        }
    }

    pub fn new_maybe_root(generate_proof_tree: GenerateProofTree) -> ProofTreeBuilder<D> {
        match generate_proof_tree {
            GenerateProofTree::No => ProofTreeBuilder::new_noop(),
            GenerateProofTree::Yes => ProofTreeBuilder::new_root(),
        }
    }

    pub fn new_root() -> ProofTreeBuilder<D> {
        ProofTreeBuilder::new(DebugSolver::Root)
    }

    pub fn new_noop() -> ProofTreeBuilder<D> {
        ProofTreeBuilder { state: None, _infcx: PhantomData }
    }

    pub fn is_noop(&self) -> bool {
        self.state.is_none()
    }

    pub(in crate::solve) fn new_goal_evaluation(
        &mut self,
        goal: Goal<I, I::Predicate>,
        orig_values: &[I::GenericArg],
        kind: GoalEvaluationKind,
    ) -> ProofTreeBuilder<D> {
        self.opt_nested(|| match kind {
            GoalEvaluationKind::Root => Some(WipGoalEvaluation {
                uncanonicalized_goal: goal,
                orig_values: orig_values.to_vec(),
                evaluation: None,
            }),
            GoalEvaluationKind::Nested => None,
        })
    }

    pub fn new_canonical_goal_evaluation(
        &mut self,
        goal: CanonicalInput<I>,
    ) -> ProofTreeBuilder<D> {
        self.nested(|| WipCanonicalGoalEvaluation {
            goal,
            kind: None,
            final_revision: None,
            result: None,
        })
    }

    pub fn canonical_goal_evaluation(&mut self, canonical_goal_evaluation: ProofTreeBuilder<D>) {
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

    pub fn canonical_goal_evaluation_kind(&mut self, kind: WipCanonicalGoalEvaluationKind<I>) {
        if let Some(this) = self.as_mut() {
            match this {
                DebugSolver::CanonicalGoalEvaluation(canonical_goal_evaluation) => {
                    assert_eq!(canonical_goal_evaluation.kind.replace(kind), None);
                }
                _ => unreachable!(),
            };
        }
    }

    pub fn goal_evaluation(&mut self, goal_evaluation: ProofTreeBuilder<D>) {
        if let Some(this) = self.as_mut() {
            match this {
                DebugSolver::Root => *this = *goal_evaluation.state.unwrap(),
                DebugSolver::CanonicalGoalEvaluationStep(_) => {
                    assert!(goal_evaluation.state.is_none())
                }
                _ => unreachable!(),
            }
        }
    }

    pub fn new_goal_evaluation_step(
        &mut self,
        var_values: ty::CanonicalVarValues<I>,
        instantiated_goal: QueryInput<I, I::Predicate>,
    ) -> ProofTreeBuilder<D> {
        self.nested(|| WipCanonicalGoalEvaluationStep {
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

    pub fn goal_evaluation_step(&mut self, goal_evaluation_step: ProofTreeBuilder<D>) {
        if let Some(this) = self.as_mut() {
            match (this, *goal_evaluation_step.state.unwrap()) {
                (
                    DebugSolver::CanonicalGoalEvaluation(canonical_goal_evaluations),
                    DebugSolver::CanonicalGoalEvaluationStep(goal_evaluation_step),
                ) => {
                    canonical_goal_evaluations.final_revision = Some(goal_evaluation_step);
                }
                _ => unreachable!(),
            }
        }
    }

    pub fn add_var_value<T: Into<I::GenericArg>>(&mut self, arg: T) {
        match self.as_mut() {
            None => {}
            Some(DebugSolver::CanonicalGoalEvaluationStep(state)) => {
                state.var_values.push(arg.into());
            }
            Some(s) => panic!("tried to add var values to {s:?}"),
        }
    }

    pub fn enter_probe(&mut self) {
        match self.as_mut() {
            None => {}
            Some(DebugSolver::CanonicalGoalEvaluationStep(state)) => {
                let initial_num_var_values = state.var_values.len();
                state.current_evaluation_scope().steps.push(WipProbeStep::NestedProbe(WipProbe {
                    initial_num_var_values,
                    steps: vec![],
                    kind: None,
                    final_state: None,
                }));
                state.probe_depth += 1;
            }
            Some(s) => panic!("tried to start probe to {s:?}"),
        }
    }

    pub fn probe_kind(&mut self, probe_kind: inspect::ProbeKind<I>) {
        match self.as_mut() {
            None => {}
            Some(DebugSolver::CanonicalGoalEvaluationStep(state)) => {
                let prev = state.current_evaluation_scope().kind.replace(probe_kind);
                assert_eq!(prev, None);
            }
            _ => panic!(),
        }
    }

    pub fn probe_final_state(&mut self, delegate: &D, max_input_universe: ty::UniverseIndex) {
        match self.as_mut() {
            None => {}
            Some(DebugSolver::CanonicalGoalEvaluationStep(state)) => {
                let final_state = canonical::make_canonical_state(
                    delegate,
                    &state.var_values,
                    max_input_universe,
                    (),
                );
                let prev = state.current_evaluation_scope().final_state.replace(final_state);
                assert_eq!(prev, None);
            }
            _ => panic!(),
        }
    }

    pub fn add_normalizes_to_goal(
        &mut self,
        delegate: &D,
        max_input_universe: ty::UniverseIndex,
        goal: Goal<I, ty::NormalizesTo<I>>,
    ) {
        self.add_goal(
            delegate,
            max_input_universe,
            GoalSource::Misc,
            goal.with(delegate.cx(), goal.predicate),
        );
    }

    pub fn add_goal(
        &mut self,
        delegate: &D,
        max_input_universe: ty::UniverseIndex,
        source: GoalSource,
        goal: Goal<I, I::Predicate>,
    ) {
        match self.as_mut() {
            None => {}
            Some(DebugSolver::CanonicalGoalEvaluationStep(state)) => {
                let goal = canonical::make_canonical_state(
                    delegate,
                    &state.var_values,
                    max_input_universe,
                    goal,
                );
                state.current_evaluation_scope().steps.push(WipProbeStep::AddGoal(source, goal))
            }
            _ => panic!(),
        }
    }

    pub(crate) fn record_impl_args(
        &mut self,
        delegate: &D,
        max_input_universe: ty::UniverseIndex,
        impl_args: I::GenericArgs,
    ) {
        match self.as_mut() {
            Some(DebugSolver::CanonicalGoalEvaluationStep(state)) => {
                let impl_args = canonical::make_canonical_state(
                    delegate,
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
            _ => panic!(),
        }
    }

    pub fn make_canonical_response(&mut self, shallow_certainty: Certainty) {
        match self.as_mut() {
            Some(DebugSolver::CanonicalGoalEvaluationStep(state)) => {
                state
                    .current_evaluation_scope()
                    .steps
                    .push(WipProbeStep::MakeCanonicalResponse { shallow_certainty });
            }
            None => {}
            _ => panic!(),
        }
    }

    pub fn finish_probe(mut self) -> ProofTreeBuilder<D> {
        match self.as_mut() {
            None => {}
            Some(DebugSolver::CanonicalGoalEvaluationStep(state)) => {
                assert_ne!(state.probe_depth, 0);
                let num_var_values = state.current_evaluation_scope().initial_num_var_values;
                state.var_values.truncate(num_var_values);
                state.probe_depth -= 1;
            }
            _ => panic!(),
        }

        self
    }

    pub fn query_result(&mut self, result: QueryResult<I>) {
        if let Some(this) = self.as_mut() {
            match this {
                DebugSolver::CanonicalGoalEvaluation(canonical_goal_evaluation) => {
                    assert_eq!(canonical_goal_evaluation.result.replace(result), None);
                }
                DebugSolver::CanonicalGoalEvaluationStep(evaluation_step) => {
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

impl<D, I> search_graph::ProofTreeBuilder<I> for ProofTreeBuilder<D>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    fn try_apply_proof_tree(
        &mut self,
        proof_tree: Option<I::CanonicalGoalEvaluationStepRef>,
    ) -> bool {
        if !self.is_noop() {
            if let Some(final_revision) = proof_tree {
                let kind = WipCanonicalGoalEvaluationKind::Interned { final_revision };
                self.canonical_goal_evaluation_kind(kind);
                true
            } else {
                false
            }
        } else {
            true
        }
    }

    fn on_provisional_cache_hit(&mut self) {
        self.canonical_goal_evaluation_kind(WipCanonicalGoalEvaluationKind::ProvisionalCacheHit);
    }

    fn on_cycle_in_stack(&mut self) {
        self.canonical_goal_evaluation_kind(WipCanonicalGoalEvaluationKind::CycleInStack);
    }

    fn finalize_canonical_goal_evaluation(
        &mut self,
        tcx: I,
    ) -> Option<I::CanonicalGoalEvaluationStepRef> {
        self.as_mut().map(|this| match this {
            DebugSolver::CanonicalGoalEvaluation(evaluation) => {
                let final_revision = mem::take(&mut evaluation.final_revision).unwrap();
                let final_revision =
                    tcx.intern_canonical_goal_evaluation_step(final_revision.finalize());
                let kind = WipCanonicalGoalEvaluationKind::Interned { final_revision };
                assert_eq!(evaluation.kind.replace(kind), None);
                final_revision
            }
            _ => unreachable!(),
        })
    }
}
