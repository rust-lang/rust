use rustc::mir::{self, BasicBlock, Location};
use rustc_index::bit_set::BitSet;

use super::{Analysis, Results};
use crate::dataflow::impls::{borrows::Borrows, EverInitializedPlaces, MaybeUninitializedPlaces};

/// Calls the corresponding method in `ResultsVisitor` for every location in a `mir::Body` with the
/// dataflow state at that location.
pub fn visit_results<F>(
    body: &'mir mir::Body<'tcx>,
    blocks: impl IntoIterator<Item = BasicBlock>,
    results: &impl ResultsVisitable<'tcx, FlowState = F>,
    vis: &mut impl ResultsVisitor<'mir, 'tcx, FlowState = F>,
) {
    let mut state = results.new_flow_state(body);

    for block in blocks {
        let block_data = &body[block];
        results.reset_to_block_start(&mut state, block);

        for (statement_index, stmt) in block_data.statements.iter().enumerate() {
            let loc = Location { block, statement_index };

            results.reconstruct_before_statement_effect(&mut state, stmt, loc);
            vis.visit_statement(&state, stmt, loc);

            results.reconstruct_statement_effect(&mut state, stmt, loc);
            vis.visit_statement_exit(&state, stmt, loc);
        }

        let loc = body.terminator_loc(block);
        let term = block_data.terminator();

        results.reconstruct_before_terminator_effect(&mut state, term, loc);
        vis.visit_terminator(&state, term, loc);

        results.reconstruct_terminator_effect(&mut state, term, loc);
        vis.visit_terminator_exit(&state, term, loc);
    }
}

pub trait ResultsVisitor<'mir, 'tcx> {
    type FlowState;

    /// Called with the `before_statement_effect` of the given statement applied to `state` but not
    /// its `statement_effect`.
    fn visit_statement(
        &mut self,
        _state: &Self::FlowState,
        _statement: &'mir mir::Statement<'tcx>,
        _location: Location,
    ) {
    }

    /// Called with both the `before_statement_effect` and the `statement_effect` of the given
    /// statement applied to `state`.
    fn visit_statement_exit(
        &mut self,
        _state: &Self::FlowState,
        _statement: &'mir mir::Statement<'tcx>,
        _location: Location,
    ) {
    }

    /// Called with the `before_terminator_effect` of the given terminator applied to `state` but not
    /// its `terminator_effect`.
    fn visit_terminator(
        &mut self,
        _state: &Self::FlowState,
        _terminator: &'mir mir::Terminator<'tcx>,
        _location: Location,
    ) {
    }

    /// Called with both the `before_terminator_effect` and the `terminator_effect` of the given
    /// terminator applied to `state`.
    ///
    /// The `call_return_effect` (if one exists) will *not* be applied to `state`.
    fn visit_terminator_exit(
        &mut self,
        _state: &Self::FlowState,
        _terminator: &'mir mir::Terminator<'tcx>,
        _location: Location,
    ) {
    }
}

/// Things that can be visited by a `ResultsVisitor`.
///
/// This trait exists so that we can visit the results of multiple dataflow analyses simultaneously.
/// DO NOT IMPLEMENT MANUALLY. Instead, use the `impl_visitable` macro below.
pub trait ResultsVisitable<'tcx> {
    type FlowState;

    /// Creates an empty `FlowState` to hold the transient state for these dataflow results.
    ///
    /// The value of the newly created `FlowState` will be overwritten by `reset_to_block_start`
    /// before it can be observed by a `ResultsVisitor`.
    fn new_flow_state(&self, body: &mir::Body<'tcx>) -> Self::FlowState;

    fn reset_to_block_start(&self, state: &mut Self::FlowState, block: BasicBlock);

    fn reconstruct_before_statement_effect(
        &self,
        state: &mut Self::FlowState,
        statement: &mir::Statement<'tcx>,
        location: Location,
    );

    fn reconstruct_statement_effect(
        &self,
        state: &mut Self::FlowState,
        statement: &mir::Statement<'tcx>,
        location: Location,
    );

    fn reconstruct_before_terminator_effect(
        &self,
        state: &mut Self::FlowState,
        terminator: &mir::Terminator<'tcx>,
        location: Location,
    );

    fn reconstruct_terminator_effect(
        &self,
        state: &mut Self::FlowState,
        terminator: &mir::Terminator<'tcx>,
        location: Location,
    );
}

impl<'tcx, A> ResultsVisitable<'tcx> for Results<'tcx, A>
where
    A: Analysis<'tcx>,
{
    type FlowState = BitSet<A::Idx>;

    fn new_flow_state(&self, body: &mir::Body<'tcx>) -> Self::FlowState {
        BitSet::new_empty(self.analysis.bits_per_block(body))
    }

    fn reset_to_block_start(&self, state: &mut Self::FlowState, block: BasicBlock) {
        state.overwrite(&self.entry_set_for_block(block));
    }

    fn reconstruct_before_statement_effect(
        &self,
        state: &mut Self::FlowState,
        stmt: &mir::Statement<'tcx>,
        loc: Location,
    ) {
        self.analysis.apply_before_statement_effect(state, stmt, loc);
    }

    fn reconstruct_statement_effect(
        &self,
        state: &mut Self::FlowState,
        stmt: &mir::Statement<'tcx>,
        loc: Location,
    ) {
        self.analysis.apply_statement_effect(state, stmt, loc);
    }

    fn reconstruct_before_terminator_effect(
        &self,
        state: &mut Self::FlowState,
        term: &mir::Terminator<'tcx>,
        loc: Location,
    ) {
        self.analysis.apply_before_terminator_effect(state, term, loc);
    }

    fn reconstruct_terminator_effect(
        &self,
        state: &mut Self::FlowState,
        term: &mir::Terminator<'tcx>,
        loc: Location,
    ) {
        self.analysis.apply_terminator_effect(state, term, loc);
    }
}

/// A tuple with named fields that can hold either the results or the transient state of the
/// dataflow analyses used by the borrow checker.
#[derive(Debug)]
pub struct BorrowckAnalyses<B, U, E> {
    pub borrows: B,
    pub uninits: U,
    pub ever_inits: E,
}

/// The results of the dataflow analyses used by the borrow checker.
pub type BorrowckResults<'mir, 'tcx> = BorrowckAnalyses<
    Results<'tcx, Borrows<'mir, 'tcx>>,
    Results<'tcx, MaybeUninitializedPlaces<'mir, 'tcx>>,
    Results<'tcx, EverInitializedPlaces<'mir, 'tcx>>,
>;

/// The transient state of the dataflow analyses used by the borrow checker.
pub type BorrowckFlowState<'mir, 'tcx> =
    <BorrowckResults<'mir, 'tcx> as ResultsVisitable<'tcx>>::FlowState;

macro_rules! impl_visitable {
    ( $(
        $T:ident { $( $field:ident : $A:ident ),* $(,)? }
    )* ) => { $(
        impl<'tcx, $($A),*> ResultsVisitable<'tcx> for $T<$( Results<'tcx, $A> ),*>
        where
            $( $A: Analysis<'tcx>, )*
        {
            type FlowState = $T<$( BitSet<$A::Idx> ),*>;

            fn new_flow_state(&self, body: &mir::Body<'tcx>) -> Self::FlowState {
                $T {
                    $( $field: BitSet::new_empty(self.$field.analysis.bits_per_block(body)) ),*
                }
            }

            fn reset_to_block_start(
                &self,
                state: &mut Self::FlowState,
                block: BasicBlock,
            ) {
                $( state.$field.overwrite(&self.$field.entry_sets[block]); )*
            }

            fn reconstruct_before_statement_effect(
                &self,
                state: &mut Self::FlowState,
                stmt: &mir::Statement<'tcx>,
                loc: Location,
            ) {
                $( self.$field.analysis
                    .apply_before_statement_effect(&mut state.$field, stmt, loc); )*
            }

            fn reconstruct_statement_effect(
                &self,
                state: &mut Self::FlowState,
                stmt: &mir::Statement<'tcx>,
                loc: Location,
            ) {
                $( self.$field.analysis
                    .apply_statement_effect(&mut state.$field, stmt, loc); )*
            }

            fn reconstruct_before_terminator_effect(
                &self,
                state: &mut Self::FlowState,
                term: &mir::Terminator<'tcx>,
                loc: Location,
            ) {
                $( self.$field.analysis
                    .apply_before_terminator_effect(&mut state.$field, term, loc); )*
            }

            fn reconstruct_terminator_effect(
                &self,
                state: &mut Self::FlowState,
                term: &mir::Terminator<'tcx>,
                loc: Location,
            ) {
                $( self.$field.analysis
                    .apply_terminator_effect(&mut state.$field, term, loc); )*
            }
        }
    )* }
}

impl_visitable! {
    BorrowckAnalyses { borrows: B, uninits: U, ever_inits: E }
}
