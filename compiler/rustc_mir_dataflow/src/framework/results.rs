//! Dataflow analysis results.

use rustc_index::IndexVec;
use rustc_middle::mir::{BasicBlock, Body};

use super::{Analysis, ResultsCursor};

/// The results of a dataflow analysis that has converged to fixpoint. It only holds the domain
/// values at the entry of each basic block. Domain values in other parts of the block are
/// recomputed on the fly by visitors (i.e. `ResultsCursor`, or `ResultsVisitor` impls).
pub type Results<D> = IndexVec<BasicBlock, D>;

/// Utility type used in a few places where it's convenient to bundle an analysis with its results.
pub struct AnalysisAndResults<'tcx, A>
where
    A: Analysis<'tcx>,
{
    pub analysis: A,
    pub results: Results<A::Domain>,
}

impl<'tcx, A> AnalysisAndResults<'tcx, A>
where
    A: Analysis<'tcx>,
{
    /// Creates a `ResultsCursor` that takes ownership of `self`.
    pub fn into_results_cursor<'mir>(self, body: &'mir Body<'tcx>) -> ResultsCursor<'mir, 'tcx, A> {
        ResultsCursor::new_owning(body, self.analysis, self.results)
    }
}
