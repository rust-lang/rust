//! Dataflow analysis results.

use rustc_index::IndexVec;
use rustc_middle::mir::{BasicBlock, Body, traversal};

use super::{Analysis, ResultsCursor, ResultsVisitor, visit_results};
use crate::framework::cursor::ResultsHandle;

pub type EntryStates<'tcx, A> = IndexVec<BasicBlock, <A as Analysis<'tcx>>::Domain>;

/// A dataflow analysis that has converged to fixpoint. It only holds the domain values at the
/// entry of each basic block. Domain values in other parts of the block are recomputed on the fly
/// by visitors (i.e. `ResultsCursor`, or `ResultsVisitor` impls).
#[derive(Clone)]
pub struct Results<'tcx, A>
where
    A: Analysis<'tcx>,
{
    pub analysis: A,
    pub entry_states: EntryStates<'tcx, A>,
}

impl<'tcx, A> Results<'tcx, A>
where
    A: Analysis<'tcx>,
{
    /// Creates a `ResultsCursor` that mutably borrows the `Results`, which is appropriate when the
    /// `Results` is also used outside the cursor.
    pub fn as_results_cursor<'mir>(
        &'mir mut self,
        body: &'mir Body<'tcx>,
    ) -> ResultsCursor<'mir, 'tcx, A> {
        ResultsCursor::new(body, ResultsHandle::BorrowedMut(self))
    }

    /// Creates a `ResultsCursor` that takes ownership of the `Results`.
    pub fn into_results_cursor<'mir>(self, body: &'mir Body<'tcx>) -> ResultsCursor<'mir, 'tcx, A> {
        ResultsCursor::new(body, ResultsHandle::Owned(self))
    }

    /// Gets the dataflow state for the given block.
    pub fn entry_set_for_block(&self, block: BasicBlock) -> &A::Domain {
        &self.entry_states[block]
    }

    pub fn visit_with<'mir>(
        &mut self,
        body: &'mir Body<'tcx>,
        blocks: impl IntoIterator<Item = BasicBlock>,
        vis: &mut impl ResultsVisitor<'tcx, A>,
    ) {
        visit_results(body, blocks, self, vis)
    }

    pub fn visit_reachable_with<'mir>(
        &mut self,
        body: &'mir Body<'tcx>,
        vis: &mut impl ResultsVisitor<'tcx, A>,
    ) {
        let blocks = traversal::reachable(body);
        visit_results(body, blocks.map(|(bb, _)| bb), self, vis)
    }
}
