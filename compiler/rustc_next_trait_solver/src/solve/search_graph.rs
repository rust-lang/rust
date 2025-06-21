use std::convert::Infallible;
use std::marker::PhantomData;

use rustc_type_ir::data_structures::ensure_sufficient_stack;
use rustc_type_ir::search_graph::{self, PathKind};
use rustc_type_ir::solve::{CanonicalInput, Certainty, NoSolution, QueryResult};
use rustc_type_ir::{Interner, TypingMode};

use crate::delegate::SolverDelegate;
use crate::solve::inspect::ProofTreeBuilder;
use crate::solve::{EvalCtxt, FIXPOINT_STEP_LIMIT, has_no_inference_or_external_constraints};

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
            PathKind::Unknown => response_no_constraints(cx, input, Certainty::overflow(false)),
            // Even though we know these cycles to be unproductive, we still return
            // overflow during coherence. This is both as we are not 100% confident in
            // the implementation yet and any incorrect errors would be unsound there.
            // The affected cases are also fairly artificial and not necessarily desirable
            // so keeping this as ambiguity is fine for now.
            //
            // See `tests/ui/traits/next-solver/cycles/unproductive-in-coherence.rs` for an
            // example where this would matter. We likely should change these cycles to `NoSolution`
            // even in coherence once this is a bit more settled.
            PathKind::Inductive => match input.typing_mode {
                TypingMode::Coherence => {
                    response_no_constraints(cx, input, Certainty::overflow(false))
                }
                TypingMode::Analysis { .. }
                | TypingMode::Borrowck { .. }
                | TypingMode::PostBorrowckAnalysis { .. }
                | TypingMode::PostAnalysis => Err(NoSolution),
            },
        }
    }

    fn is_initial_provisional_result(
        cx: Self::Cx,
        kind: PathKind,
        input: CanonicalInput<I>,
        result: QueryResult<I>,
    ) -> bool {
        Self::initial_provisional_result(cx, kind, input) == result
    }

    fn on_stack_overflow(
        cx: I,
        input: CanonicalInput<I>,
        inspect: &mut ProofTreeBuilder<D>,
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

    fn compute_goal(
        search_graph: &mut SearchGraph<D>,
        cx: I,
        input: CanonicalInput<I>,
        inspect: &mut Self::ProofTreeBuilder,
    ) -> QueryResult<I> {
        ensure_sufficient_stack(|| {
            EvalCtxt::enter_canonical(cx, search_graph, input, inspect, |ecx, goal| {
                let result = ecx.compute_goal(goal);
                ecx.inspect.query_result(result);
                result
            })
        })
    }
}

fn response_no_constraints<I: Interner>(
    cx: I,
    input: CanonicalInput<I>,
    certainty: Certainty,
) -> QueryResult<I> {
    Ok(super::response_no_constraints_raw(
        cx,
        input.canonical.max_universe,
        input.canonical.variables,
        certainty,
    ))
}
