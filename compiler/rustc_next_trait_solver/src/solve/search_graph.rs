use std::marker::PhantomData;

use rustc_type_ir::inherent::*;
use rustc_type_ir::search_graph::{self, CycleKind, UsageKind};
use rustc_type_ir::solve::{CanonicalInput, Certainty, QueryResult};
use rustc_type_ir::Interner;

use super::inspect::{self, ProofTreeBuilder};
use super::FIXPOINT_STEP_LIMIT;
use crate::delegate::SolverDelegate;

/// This type is never constructed. We only use it to implement `search_graph::Delegate`
/// for all types which impl `SolverDelegate` and doing it directly fails in coherence.
pub(super) struct SearchGraphDelegate<D: SolverDelegate> {
    _marker: PhantomData<D>,
}
pub(super) type SearchGraph<D> = search_graph::SearchGraph<SearchGraphDelegate<D>>;
impl<D, I> search_graph::Delegate for SearchGraphDelegate<D>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    type Cx = D::Interner;

    const FIXPOINT_STEP_LIMIT: usize = FIXPOINT_STEP_LIMIT;

    type ProofTreeBuilder = ProofTreeBuilder<D>;

    fn recursion_limit(cx: I) -> usize {
        cx.recursion_limit()
    }

    fn initial_provisional_result(
        cx: I,
        kind: CycleKind,
        input: CanonicalInput<I>,
    ) -> QueryResult<I> {
        match kind {
            CycleKind::Coinductive => response_no_constraints(cx, input, Certainty::Yes),
            CycleKind::Inductive => response_no_constraints(cx, input, Certainty::overflow(false)),
        }
    }

    fn reached_fixpoint(
        cx: I,
        kind: UsageKind,
        input: CanonicalInput<I>,
        provisional_result: Option<QueryResult<I>>,
        result: QueryResult<I>,
    ) -> bool {
        if let Some(r) = provisional_result {
            r == result
        } else {
            match kind {
                UsageKind::Single(CycleKind::Coinductive) => {
                    response_no_constraints(cx, input, Certainty::Yes) == result
                }
                UsageKind::Single(CycleKind::Inductive) => {
                    response_no_constraints(cx, input, Certainty::overflow(false)) == result
                }
                UsageKind::Mixed => false,
            }
        }
    }

    fn on_stack_overflow(
        cx: I,
        inspect: &mut ProofTreeBuilder<D>,
        input: CanonicalInput<I>,
    ) -> QueryResult<I> {
        inspect.canonical_goal_evaluation_kind(inspect::WipCanonicalGoalEvaluationKind::Overflow);
        response_no_constraints(cx, input, Certainty::overflow(true))
    }

    fn on_fixpoint_overflow(cx: I, input: CanonicalInput<I>) -> QueryResult<I> {
        response_no_constraints(cx, input, Certainty::overflow(false))
    }

    fn step_is_coinductive(cx: I, input: CanonicalInput<I>) -> bool {
        input.value.goal.predicate.is_coinductive(cx)
    }
}

fn response_no_constraints<I: Interner>(
    cx: I,
    goal: CanonicalInput<I>,
    certainty: Certainty,
) -> QueryResult<I> {
    Ok(super::response_no_constraints_raw(cx, goal.max_universe, goal.variables, certainty))
}
