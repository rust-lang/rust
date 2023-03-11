use rustc_infer::infer::canonical::Canonical;
use rustc_infer::traits::query::NoSolution;
use rustc_middle::traits::solve::{Certainty, MaybeCause, QueryResult};
use rustc_middle::ty::TyCtxt;
use rustc_session::Limit;

use super::SearchGraph;
use crate::solve::{response_no_constraints, EvalCtxt};

/// When detecting a solver overflow, we return ambiguity. Overflow can be
/// *hidden* by either a fatal error in an **AND** or a trivial success in an **OR**.
///
/// This is in issue in case of exponential blowup, e.g. if each goal on the stack
/// has multiple nested (overflowing) candidates. To deal with this, we reduce the limit
/// used by the solver when hitting the default limit for the first time.
///
/// FIXME: Get tests where always using the `default_limit` results in a hang and refer
/// to them here. We can also improve the overflow strategy if necessary.
pub(super) struct OverflowData {
    default_limit: Limit,
    current_limit: Limit,
    /// When proving an **AND** we have to repeatedly iterate over the yet unproven goals.
    ///
    /// Because of this each iteration also increases the depth in addition to the stack
    /// depth.
    additional_depth: usize,
}

impl OverflowData {
    pub(super) fn new(tcx: TyCtxt<'_>) -> OverflowData {
        let default_limit = tcx.recursion_limit();
        OverflowData { default_limit, current_limit: default_limit, additional_depth: 0 }
    }

    #[inline]
    pub(super) fn did_overflow(&self) -> bool {
        self.default_limit.0 != self.current_limit.0
    }

    #[inline]
    pub(super) fn has_overflow(&self, depth: usize) -> bool {
        !self.current_limit.value_within_limit(depth + self.additional_depth)
    }

    /// Updating the current limit when hitting overflow.
    fn deal_with_overflow(&mut self) {
        // When first hitting overflow we reduce the overflow limit
        // for all future goals to prevent hangs if there's an exponental
        // blowup.
        self.current_limit.0 = self.default_limit.0 / 8;
    }
}

pub(in crate::solve) trait OverflowHandler<'tcx> {
    fn search_graph(&mut self) -> &mut SearchGraph<'tcx>;

    fn repeat_while_none<T>(
        &mut self,
        on_overflow: impl FnOnce(&mut Self) -> Result<T, NoSolution>,
        mut loop_body: impl FnMut(&mut Self) -> Option<Result<T, NoSolution>>,
    ) -> Result<T, NoSolution> {
        let start_depth = self.search_graph().overflow_data.additional_depth;
        let depth = self.search_graph().stack.len();
        while !self.search_graph().overflow_data.has_overflow(depth) {
            if let Some(result) = loop_body(self) {
                self.search_graph().overflow_data.additional_depth = start_depth;
                return result;
            }

            self.search_graph().overflow_data.additional_depth += 1;
        }
        self.search_graph().overflow_data.additional_depth = start_depth;
        self.search_graph().overflow_data.deal_with_overflow();
        on_overflow(self)
    }
}

impl<'tcx> OverflowHandler<'tcx> for EvalCtxt<'_, 'tcx> {
    fn search_graph(&mut self) -> &mut SearchGraph<'tcx> {
        &mut self.search_graph
    }
}

impl<'tcx> OverflowHandler<'tcx> for SearchGraph<'tcx> {
    fn search_graph(&mut self) -> &mut SearchGraph<'tcx> {
        self
    }
}

impl<'tcx> SearchGraph<'tcx> {
    pub fn deal_with_overflow(
        &mut self,
        tcx: TyCtxt<'tcx>,
        goal: Canonical<'tcx, impl Sized>,
    ) -> QueryResult<'tcx> {
        self.overflow_data.deal_with_overflow();
        response_no_constraints(tcx, goal, Certainty::Maybe(MaybeCause::Overflow))
    }
}
