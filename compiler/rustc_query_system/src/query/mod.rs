mod plumbing;
pub use self::plumbing::*;

mod job;
#[cfg(parallel_compiler)]
pub use self::job::deadlock;
pub use self::job::{print_query_stack, QueryInfo, QueryJob, QueryJobId, QueryJobInfo, QueryMap};

mod caches;
pub use self::caches::{
    ArenaCacheSelector, CacheSelector, DefaultCacheSelector, QueryCache, QueryStorage,
};

mod config;
pub use self::config::{QueryConfig, QueryDescription};

use crate::dep_graph::HasDepContext;

use rustc_hir::def::DefKind;
use rustc_span::Span;

/// Description of a frame in the query stack.
///
/// This is mostly used in case of cycles for error reporting.
#[derive(Clone, Debug)]
pub struct QueryStackFrame {
    pub name: &'static str,
    pub description: String,
    span: Option<Span>,
    def_kind: Option<DefKind>,
    /// This hash is used to deterministically pick
    /// a query to remove cycles in the parallel compiler.
    #[cfg(parallel_compiler)]
    hash: u64,
}

impl QueryStackFrame {
    #[inline]
    pub fn new(
        name: &'static str,
        description: String,
        span: Option<Span>,
        def_kind: Option<DefKind>,
        _hash: impl FnOnce() -> u64,
    ) -> Self {
        Self {
            name,
            description,
            span,
            def_kind,
            #[cfg(parallel_compiler)]
            hash: _hash(),
        }
    }

    // FIXME(eddyb) Get more valid `Span`s on queries.
    #[inline]
    pub fn default_span(&self, span: Span) -> Span {
        if !span.is_dummy() {
            return span;
        }
        self.span.unwrap_or(span)
    }
}

pub trait QueryContext: HasDepContext {
    fn next_job_id(&self) -> QueryJobId;

    /// Get the query information from the TLS context.
    fn current_query_job(&self) -> Option<QueryJobId>;

    fn try_collect_active_jobs(&self) -> Option<QueryMap>;
    /// Executes a job by changing the `ImplicitCtxt` to point to the
    /// new query job while it executes. It returns the diagnostics
    /// captured during execution and the actual result.
    fn start_query<R>(&self, token: QueryJobId, compute: impl FnOnce() -> R) -> R;
}
