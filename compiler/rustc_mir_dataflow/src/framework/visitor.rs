use rustc_middle::mir::{self, BasicBlock, Location};

use super::{Analysis, Direction, Results};

/// Calls the corresponding method in `ResultsVisitor` for every location in a `mir::Body` with the
/// dataflow state at that location.
pub fn visit_results<'mir, 'tcx, A>(
    body: &'mir mir::Body<'tcx>,
    blocks: impl IntoIterator<Item = BasicBlock>,
    results: &mut Results<'tcx, A>,
    vis: &mut impl ResultsVisitor<'mir, 'tcx, A>,
) where
    A: Analysis<'tcx>,
{
    let mut state = results.analysis.bottom_value(body);

    #[cfg(debug_assertions)]
    let reachable_blocks = mir::traversal::reachable_as_bitset(body);

    for block in blocks {
        #[cfg(debug_assertions)]
        assert!(reachable_blocks.contains(block));

        let block_data = &body[block];
        A::Direction::visit_results_in_block(&mut state, block, block_data, results, vis);
    }
}

/// A visitor over the results of an `Analysis`. Use this when you want to inspect domain values in
/// many or all locations; use `ResultsCursor` if you want to inspect domain values only in certain
/// locations.
pub trait ResultsVisitor<'mir, 'tcx, A>
where
    A: Analysis<'tcx>,
{
    fn visit_block_start(&mut self, _state: &A::Domain) {}

    /// Called with the `before_statement_effect` of the given statement applied to `state` but not
    /// its `statement_effect`.
    fn visit_statement_before_primary_effect(
        &mut self,
        _results: &mut Results<'tcx, A>,
        _state: &A::Domain,
        _statement: &'mir mir::Statement<'tcx>,
        _location: Location,
    ) {
    }

    /// Called with both the `before_statement_effect` and the `statement_effect` of the given
    /// statement applied to `state`.
    fn visit_statement_after_primary_effect(
        &mut self,
        _results: &mut Results<'tcx, A>,
        _state: &A::Domain,
        _statement: &'mir mir::Statement<'tcx>,
        _location: Location,
    ) {
    }

    /// Called with the `before_terminator_effect` of the given terminator applied to `state` but
    /// not its `terminator_effect`.
    fn visit_terminator_before_primary_effect(
        &mut self,
        _results: &mut Results<'tcx, A>,
        _state: &A::Domain,
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
        _results: &mut Results<'tcx, A>,
        _state: &A::Domain,
        _terminator: &'mir mir::Terminator<'tcx>,
        _location: Location,
    ) {
    }

    fn visit_block_end(&mut self, _state: &A::Domain) {}
}
