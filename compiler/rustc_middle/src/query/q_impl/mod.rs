use rustc_span::Span;

use crate::query::q_impl::execution::{execute_query_incr_inner, execute_query_non_incr_inner};
use crate::query::{QueryCache, QueryMode, QueryVTable};
use crate::ty::TyCtxt;

pub mod diagnostics;
pub mod execution;
pub mod handle_cycle_error;
pub mod incremental;
pub mod job;

pub struct QueryExecutionMethods(pub bool);

#[inline(never)]
fn __rust_end_short_backtrace<F, T>(f: F) -> T
where
    F: FnOnce() -> T,
{
    let result = f();

    // prevent this frame from being tail-call optimised away
    std::hint::black_box(());

    result
}

impl QueryExecutionMethods {
    #[inline(never)]
    pub fn execute_query<'tcx, C: QueryCache>(
        &self,
        query: &'tcx QueryVTable<'tcx, C>,
        tcx: TyCtxt<'tcx>,
        span: Span,
        key: C::Key,
        mode: QueryMode,
    ) -> Option<C::Value> {
        __rust_end_short_backtrace(|| {
            if self.0 {
                execute_query_incr_inner(query, tcx, span, key, mode)
            } else {
                Some(execute_query_non_incr_inner(query, tcx, span, key))
            }
        })
    }
}
