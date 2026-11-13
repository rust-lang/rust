//! Building proof trees incrementally during trait solving.
//!
//! This code is *a bit* of a mess and can hopefully be
//! mostly ignored. For a general overview of how it works,
//! see the comment on [ProofTreeBuilder].

use std::marker::PhantomData;

use derive_where::derive_where;
use rustc_type_ir::inherent::*;
use rustc_type_ir::{self as ty, Interner};

use crate::canonical;
use crate::delegate::SolverDelegate;
use crate::solve::{Certainty, Goal, GoalSource, QueryResult, inspect};

/// We need to know whether to build a prove tree while evaluating. We
/// pass a `ProofTreeBuilder` with `state: Some(None)` into the search
/// graph which then causes the initial `EvalCtxt::compute_goal` to actually
/// build a proof tree which then gets written into the `state`.
///
/// Building the proof tree for a single evaluation step happens via the
/// [EvaluationStepBuilder] which is updated by the `EvalCtxt` when
/// appropriate.
pub(crate) struct ProofTreeBuilder<D, I = <D as SolverDelegate>::Interner>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    state: Option<Box<Option<inspect::Probe<I>>>>,
    _infcx: PhantomData<D>,
}

impl<D: SolverDelegate<Interner = I>, I: Interner> ProofTreeBuilder<D> {
    pub(crate) fn new() -> ProofTreeBuilder<D> {
        ProofTreeBuilder { state: Some(Box::new(None)), _infcx: PhantomData }
    }

    pub(crate) fn new_noop() -> ProofTreeBuilder<D> {
        ProofTreeBuilder { state: None, _infcx: PhantomData }
    }

    pub(crate) fn is_noop(&self) -> bool {
        self.state.is_none()
    }

    pub(crate) fn new_evaluation_step(
        &mut self,
        var_values: ty::CanonicalVarValues<I>,
    ) -> EvaluationStepBuilder<D> {
        if self.is_noop() {
            EvaluationStepBuilder { state: None, _infcx: PhantomData }
        } else {
            EvaluationStepBuilder {
                state: Some(Box::new(WipEvaluationStep {
                    var_values: var_values.var_values.to_vec(),
                    evaluation: WipProbe {
                        initial_num_var_values: var_values.len(),
                        steps: vec![],
                        kind: None,
                        final_state: None,
                    },
                    probe_depth: 0,
                })),
                _infcx: PhantomData,
            }
        }
    }

    pub(crate) fn finish_evaluation_step(
        &mut self,
        goal_evaluation_step: EvaluationStepBuilder<D>,
    ) {
        if let Some(this) = self.state.as_deref_mut() {
            *this = Some(goal_evaluation_step.state.unwrap().finalize());
        }
    }

    pub(crate) fn unwrap(self) -> inspect::Probe<I> {
        self.state.unwrap().unwrap()
    }
}

pub(crate) struct EvaluationStepBuilder<D, I = <D as SolverDelegate>::Interner>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    state: Option<Box<WipEvaluationStep<I>>>,
    _infcx: PhantomData<D>,
}

#[derive_where(PartialEq, Eq, Debug; I: Interner)]
struct WipEvaluationStep<I: Interner> {
    /// Unlike `EvalCtxt::var_values`, we append a new
    /// generic arg here whenever we create a new inference
    /// variable.
    ///
    /// This is necessary as we otherwise don't unify these
    /// vars when instantiating multiple `CanonicalState`.
    var_values: Vec<I::GenericArg>,
    probe_depth: usize,
    evaluation: WipProbe<I>,
}

impl<I: Interner> WipEvaluationStep<I> {
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

    fn finalize(self) -> inspect::Probe<I> {
        let evaluation = self.evaluation.finalize();
        match evaluation.kind {
            inspect::ProbeKind::Root { .. } => evaluation,
            _ => unreachable!("unexpected root evaluation: {evaluation:?}"),
        }
    }
}

#[derive_where(PartialEq, Debug; I: Interner)]
struct WipProbe<I: Interner> {
    initial_num_var_values: usize,
    steps: Vec<WipProbeStep<I>>,
    kind: Option<inspect::ProbeKind<I>>,
    final_state: Option<inspect::CanonicalState<I, ()>>,
}

impl<I: Interner> Eq for WipProbe<I> {}

impl<I: Interner> WipProbe<I> {
    fn finalize(self) -> inspect::Probe<I> {
        inspect::Probe {
            steps: self.steps.into_iter().map(WipProbeStep::finalize).collect(),
            kind: self.kind.unwrap(),
            final_state: self.final_state.unwrap(),
        }
    }
}

#[derive_where(PartialEq, Debug; I: Interner)]
enum WipProbeStep<I: Interner> {
    AddGoal(GoalSource, inspect::CanonicalState<I, Goal<I, I::Predicate>>),
    NestedProbe(WipProbe<I>),
    MakeCanonicalResponse { shallow_certainty: Certainty },
    RecordImplArgs { impl_args: inspect::CanonicalState<I, I::GenericArgs> },
}

impl<I: Interner> Eq for WipProbeStep<I> {}

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

impl<D: SolverDelegate<Interner = I>, I: Interner> EvaluationStepBuilder<D> {
    pub(crate) fn new_noop() -> EvaluationStepBuilder<D> {
        EvaluationStepBuilder { state: None, _infcx: PhantomData }
    }

    pub(crate) fn is_noop(&self) -> bool {
        self.state.is_none()
    }

    fn as_mut(&mut self) -> Option<&mut WipEvaluationStep<I>> {
        self.state.as_deref_mut()
    }

    pub(crate) fn take_and_enter_probe(&mut self) -> EvaluationStepBuilder<D> {
        let mut nested = EvaluationStepBuilder { state: self.state.take(), _infcx: PhantomData };
        nested.enter_probe();
        nested
    }

    pub(crate) fn add_var_value<T: Into<I::GenericArg>>(&mut self, arg: T) {
        if let Some(this) = self.as_mut() {
            this.var_values.push(arg.into());
        }
    }

    fn enter_probe(&mut self) {
        if let Some(this) = self.as_mut() {
            let initial_num_var_values = this.var_values.len();
            this.current_evaluation_scope().steps.push(WipProbeStep::NestedProbe(WipProbe {
                initial_num_var_values,
                steps: vec![],
                kind: None,
                final_state: None,
            }));
            this.probe_depth += 1;
        }
    }

    pub(crate) fn probe_kind(&mut self, probe_kind: inspect::ProbeKind<I>) {
        if let Some(this) = self.as_mut() {
            let prev = this.current_evaluation_scope().kind.replace(probe_kind);
            assert_eq!(prev, None);
        }
    }

    pub(crate) fn probe_final_state(
        &mut self,
        delegate: &D,
        max_input_universe: ty::UniverseIndex,
    ) {
        if let Some(this) = self.as_mut() {
            let final_state =
                canonical::make_canonical_state(delegate, &this.var_values, max_input_universe, ());
            let prev = this.current_evaluation_scope().final_state.replace(final_state);
            assert_eq!(prev, None);
        }
    }

    pub(crate) fn add_goal(
        &mut self,
        delegate: &D,
        max_input_universe: ty::UniverseIndex,
        source: GoalSource,
        goal: Goal<I, I::Predicate>,
    ) {
        if let Some(this) = self.as_mut() {
            let goal = canonical::make_canonical_state(
                delegate,
                &this.var_values,
                max_input_universe,
                goal,
            );
            this.current_evaluation_scope().steps.push(WipProbeStep::AddGoal(source, goal))
        }
    }

    pub(crate) fn record_impl_args(
        &mut self,
        delegate: &D,
        max_input_universe: ty::UniverseIndex,
        impl_args: I::GenericArgs,
    ) {
        if let Some(this) = self.as_mut() {
            let impl_args = canonical::make_canonical_state(
                delegate,
                &this.var_values,
                max_input_universe,
                impl_args,
            );
            this.current_evaluation_scope().steps.push(WipProbeStep::RecordImplArgs { impl_args });
        }
    }

    pub(crate) fn make_canonical_response(&mut self, shallow_certainty: Certainty) {
        if let Some(this) = self.as_mut() {
            this.current_evaluation_scope()
                .steps
                .push(WipProbeStep::MakeCanonicalResponse { shallow_certainty });
        }
    }

    pub(crate) fn finish_probe(mut self) -> EvaluationStepBuilder<D> {
        if let Some(this) = self.as_mut() {
            assert_ne!(this.probe_depth, 0);
            let num_var_values = this.current_evaluation_scope().initial_num_var_values;
            this.var_values.truncate(num_var_values);
            this.probe_depth -= 1;
        }

        self
    }

    pub(crate) fn query_result(&mut self, result: QueryResult<I>) {
        if let Some(this) = self.as_mut() {
            assert_eq!(this.evaluation.kind.replace(inspect::ProbeKind::Root { result }), None);
        }
    }
}
