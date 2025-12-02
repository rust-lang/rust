//! Dataflow analysis results.

use rustc_index::IndexVec;
use rustc_middle::mir::{BasicBlock, Body};

use super::{Analysis, ResultsCursor};

pub type EntryStates<D> = IndexVec<BasicBlock, D>;

/// The results of a dataflow analysis that has converged to fixpoint. It holds the domain values
/// (states) at the entry of each basic block. Domain values in other parts of the block are
/// recomputed on the fly by visitors (i.e. `ResultsCursor`, or `ResultsVisitor` impls). The
/// analysis is also present because it's often needed alongside the entry states.
pub struct Results<'tcx, A>
where
    A: Analysis<'tcx>,
{
    pub analysis: A,
    pub entry_states: EntryStates<A::Domain>,
}

impl<'tcx, A> Results<'tcx, A>
where
    A: Analysis<'tcx>,
{
    /// Creates a `ResultsCursor` that takes ownership of `self`.
    pub fn into_results_cursor<'mir>(self, body: &'mir Body<'tcx>) -> ResultsCursor<'mir, 'tcx, A> {
        ResultsCursor::new_owning(body, self)
    }
}
