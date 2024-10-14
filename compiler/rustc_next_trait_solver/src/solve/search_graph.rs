use std::convert::Infallible;
use std::marker::PhantomData;

use rustc_type_ir::Interner;
use rustc_type_ir::inherent::*;
use rustc_type_ir::search_graph::{self, PathKind};
use rustc_type_ir::solve::{CanonicalInput, Certainty, QueryResult};

use super::inspect::ProofTreeBuilder;
use super::{FIXPOINT_STEP_LIMIT, has_no_inference_or_external_constraints};
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

    const ENABLE_PROVISIONAL_CACHE: bool = true;
    type ValidationScope = Infallible;
    fn enter_validation_scope(
        _cx: Self::Cx,
        _input: CanonicalInput<I>,
    ) -> Option<Self::ValidationScope> {
        None
    }

    const FIXPOINT_STEP_LIMIT: usize = FIXPOINT_STEP_LIMIT;

    type ProofTreeBuilder = ProofTreeBuilder<D>;
    fn inspect_is_noop(inspect: &mut Self::ProofTreeBuilder) -> bool {
        inspect.is_noop()
    }

    const DIVIDE_AVAILABLE_DEPTH_ON_OVERFLOW: usize = 4;

    fn initial_provisional_result(
        cx: I,
        kind: PathKind,
        input: CanonicalInput<I>,
    ) -> QueryResult<I> {
        match kind {
            PathKind::Coinductive => response_no_constraints(cx, input, Certainty::Yes),
            PathKind::Inductive => response_no_constraints(cx, input, Certainty::overflow(false)),
        }
    }

    fn is_initial_provisional_result(
        cx: Self::Cx,
        kind: PathKind,
        input: CanonicalInput<I>,
        result: QueryResult<I>,
    ) -> bool {
        match kind {
            PathKind::Coinductive => response_no_constraints(cx, input, Certainty::Yes) == result,
            PathKind::Inductive => {
                response_no_constraints(cx, input, Certainty::overflow(false)) == result
            }
        }
    }

    fn on_stack_overflow(
        cx: I,
        inspect: &mut ProofTreeBuilder<D>,
        input: CanonicalInput<I>,
    ) -> QueryResult<I> {
        inspect.canonical_goal_evaluation_overflow();
        response_no_constraints(cx, input, Certainty::overflow(true))
    }

    fn on_fixpoint_overflow(cx: I, input: CanonicalInput<I>) -> QueryResult<I> {
        response_no_constraints(cx, input, Certainty::overflow(false))
    }

    fn is_ambiguous_result(result: QueryResult<I>) -> bool {
        result.is_ok_and(|response| {
            has_no_inference_or_external_constraints(response)
                && matches!(response.value.certainty, Certainty::Maybe(_))
        })
    }

    fn propagate_ambiguity(
        cx: I,
        for_input: CanonicalInput<I>,
        from_result: QueryResult<I>,
    ) -> QueryResult<I> {
        let certainty = from_result.unwrap().value.certainty;
        response_no_constraints(cx, for_input, certainty)
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
