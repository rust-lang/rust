use rustc_middle::mir::{self, BasicBlock, Location};

use super::{Analysis, Direction, Results};

/// Calls the corresponding method in `ResultsVisitor` for every location in a `mir::Body` with the
/// dataflow state at that location.
pub fn visit_results<'mir, 'tcx, D, R>(
    body: &'mir mir::Body<'tcx>,
    blocks: impl IntoIterator<Item = BasicBlock>,
    results: &mut R,
    vis: &mut impl ResultsVisitor<'mir, 'tcx, R, Domain = D>,
) where
    R: ResultsVisitable<'tcx, Domain = D>,
{
    let mut state = results.bottom_value(body);

    #[cfg(debug_assertions)]
    let reachable_blocks = mir::traversal::reachable_as_bitset(body);

    for block in blocks {
        #[cfg(debug_assertions)]
        assert!(reachable_blocks.contains(block));

        let block_data = &body[block];
        R::Direction::visit_results_in_block(&mut state, block, block_data, results, vis);
    }
}

/// A visitor over the results of an `Analysis`. The type parameter `R` is the results type being
/// visited.
pub trait ResultsVisitor<'mir, 'tcx, R> {
    type Domain;

    fn visit_block_start(&mut self, _state: &Self::Domain) {}

    /// Called with the `before_statement_effect` of the given statement applied to `state` but not
    /// its `statement_effect`.
    fn visit_statement_before_primary_effect(
        &mut self,
        _results: &mut R,
        _state: &Self::Domain,
        _statement: &'mir mir::Statement<'tcx>,
        _location: Location,
    ) {
    }

    /// Called with both the `before_statement_effect` and the `statement_effect` of the given
    /// statement applied to `state`.
    fn visit_statement_after_primary_effect(
        &mut self,
        _results: &mut R,
        _state: &Self::Domain,
        _statement: &'mir mir::Statement<'tcx>,
        _location: Location,
    ) {
    }

    /// Called with the `before_terminator_effect` of the given terminator applied to `state` but not
    /// its `terminator_effect`.
    fn visit_terminator_before_primary_effect(
        &mut self,
        _results: &mut R,
        _state: &Self::Domain,
        _terminator: &'mir mir::Terminator<'tcx>,
        _location: Location,
    ) {
    }

    /// Called with both the `before_terminator_effect` and the `terminator_effect` of the given
    /// terminator applied to `state`.
    ///
    /// The `call_return_effect` (if one exists) will *not* be applied to `state`.
    fn visit_terminator_after_primary_effect(
        &mut self,
        _results: &mut R,
        _state: &Self::Domain,
        _terminator: &'mir mir::Terminator<'tcx>,
        _location: Location,
    ) {
    }

    fn visit_block_end(&mut self, _state: &Self::Domain) {}
}

/// Things that can be visited by a `ResultsVisitor`.
///
/// This trait exists so that we can visit the results of one or more dataflow analyses
/// simultaneously.
pub trait ResultsVisitable<'tcx> {
    type Direction: Direction;
    type Domain;

    /// Creates an empty `Domain` to hold the transient state for these dataflow results.
    ///
    /// The value of the newly created `Domain` will be overwritten by `reset_to_block_entry`
    /// before it can be observed by a `ResultsVisitor`.
    fn bottom_value(&self, body: &mir::Body<'tcx>) -> Self::Domain;

    fn reset_to_block_entry(&self, state: &mut Self::Domain, block: BasicBlock);

    fn reconstruct_before_statement_effect(
        &mut self,
        state: &mut Self::Domain,
        statement: &mir::Statement<'tcx>,
        location: Location,
    );

    fn reconstruct_statement_effect(
        &mut self,
        state: &mut Self::Domain,
        statement: &mir::Statement<'tcx>,
        location: Location,
    );

    fn reconstruct_before_terminator_effect(
        &mut self,
        state: &mut Self::Domain,
        terminator: &mir::Terminator<'tcx>,
        location: Location,
    );

    fn reconstruct_terminator_effect(
        &mut self,
        state: &mut Self::Domain,
        terminator: &mir::Terminator<'tcx>,
        location: Location,
    );
}

impl<'tcx, A> ResultsVisitable<'tcx> for Results<'tcx, A>
where
    A: Analysis<'tcx>,
{
    type Domain = A::Domain;
    type Direction = A::Direction;

    fn bottom_value(&self, body: &mir::Body<'tcx>) -> Self::Domain {
        self.analysis.bottom_value(body)
    }

    fn reset_to_block_entry(&self, state: &mut Self::Domain, block: BasicBlock) {
        state.clone_from(self.entry_set_for_block(block));
    }

    fn reconstruct_before_statement_effect(
        &mut self,
        state: &mut Self::Domain,
        stmt: &mir::Statement<'tcx>,
        loc: Location,
    ) {
        self.analysis.apply_before_statement_effect(state, stmt, loc);
    }

    fn reconstruct_statement_effect(
        &mut self,
        state: &mut Self::Domain,
        stmt: &mir::Statement<'tcx>,
        loc: Location,
    ) {
        self.analysis.apply_statement_effect(state, stmt, loc);
    }

    fn reconstruct_before_terminator_effect(
        &mut self,
        state: &mut Self::Domain,
        term: &mir::Terminator<'tcx>,
        loc: Location,
    ) {
        self.analysis.apply_before_terminator_effect(state, term, loc);
    }

    fn reconstruct_terminator_effect(
        &mut self,
        state: &mut Self::Domain,
        term: &mir::Terminator<'tcx>,
        loc: Location,
    ) {
        self.analysis.apply_terminator_effect(state, term, loc);
    }
}
