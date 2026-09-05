use rustc_middle::mir::{self, BasicBlock, Location};

use super::{Analysis, Direction, Results};

/// Calls the visitor methods in `vis` for every location in every block in `blocks`. Note that
/// every block in `blocks` must be reachable, and a `debug_assert` checks this.
pub fn visit_results<'mir, 'tcx, A>(
    body: &'mir mir::Body<'tcx>,
    blocks: impl IntoIterator<Item = BasicBlock>,
    results: &Results<'tcx, A>,
    vis: &mut impl ResultsVisitor<'tcx, A>,
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
        state.clone_from(&results.entry_states[block]);
        A::Direction::visit_results_in_block(&results.analysis, &mut state, block, block_data, vis);
    }
}

/// A visitor over the results of an `Analysis`. Use this when you want to inspect domain values in
/// many or all locations; use `ResultsCursor` if you want to inspect domain values only in certain
/// locations.
pub trait ResultsVisitor<'tcx, A>
where
    A: Analysis<'tcx>,
{
    /// Called after all effects in a block have been applied in the direction
    /// of the analysis.
    ///
    /// In a forwards analysis, `state` is from the block's end. In a backwards
    /// analysis, `state` is from the block's start.
    fn visit_block_exit(&mut self, _state: &A::Domain, _block: BasicBlock) {}

    /// Called after the "early" effect of the given statement is applied to `state`.
    fn visit_after_early_statement_effect(
        &mut self,
        _state: &A::Domain,
        _statement: &mir::Statement<'tcx>,
        _location: Location,
    ) {
    }

    /// Called after the "primary" effect of the given statement is applied to `state`.
    fn visit_after_primary_statement_effect(
        &mut self,
        _state: &A::Domain,
        _statement: &mir::Statement<'tcx>,
        _location: Location,
    ) {
    }

    /// Called after the "early" effect of the given terminator is applied to `state`.
    fn visit_after_early_terminator_effect(
        &mut self,
        _state: &A::Domain,
        _terminator: &mir::Terminator<'tcx>,
        _location: Location,
    ) {
    }

    /// Called after the "primary" effect of the given terminator is applied to `state`.
    ///
    /// The `call_return_effect` (if one exists) will *not* be applied to `state`.
    fn visit_after_primary_terminator_effect(
        &mut self,
        _state: &A::Domain,
        _terminator: &mir::Terminator<'tcx>,
        _location: Location,
    ) {
    }
}
