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

    /// Called after the "early" effect of the given statement is applied to `state`.
    fn visit_after_early_statement_effect(
        &mut self,
        _results: &mut Results<'tcx, A>,
        _state: &A::Domain,
        _statement: &'mir mir::Statement<'tcx>,
        _location: Location,
    ) {
    }

    /// Called after the "primary" effect of the given statement is applied to `state`.
    fn visit_after_primary_statement_effect(
        &mut self,
        _results: &mut Results<'tcx, A>,
        _state: &A::Domain,
        _statement: &'mir mir::Statement<'tcx>,
        _location: Location,
    ) {
    }

    /// Called after the "early" effect of the given terminator is applied to `state`.
    fn visit_after_early_terminator_effect(
        &mut self,
        _results: &mut Results<'tcx, A>,
        _state: &A::Domain,
        _terminator: &'mir mir::Terminator<'tcx>,
        _location: Location,
    ) {
    }

    /// Called after the "primary" effect of the given terminator is applied to `state`.
    ///
    /// The `call_return_effect` (if one exists) will *not* be applied to `state`.
    fn visit_after_primary_terminator_effect(
        &mut self,
        _results: &mut Results<'tcx, A>,
        _state: &A::Domain,
        _terminator: &'mir mir::Terminator<'tcx>,
        _location: Location,
    ) {
    }

    fn visit_block_end(&mut self, _state: &A::Domain) {}
}
