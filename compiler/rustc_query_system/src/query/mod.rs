mod plumbing;
pub use self::plumbing::*;

mod job;
pub use self::job::{
    QueryInfo, QueryJob, QueryJobId, QueryJobInfo, QueryMap, break_query_cycles, print_query_stack,
    report_cycle,
};

mod caches;
pub use self::caches::{DefIdCache, DefaultCache, QueryCache, SingleCache, VecCache};

mod config;
use rustc_errors::DiagInner;
use rustc_hashes::Hash64;
use rustc_hir::def::DefKind;
use rustc_macros::{Decodable, Encodable};
use rustc_span::Span;
use rustc_span::def_id::DefId;

pub use self::config::{HashResult, QueryConfig};
use crate::dep_graph::{DepKind, DepNodeIndex, HasDepContext, SerializedDepNodeIndex};

/// Description of a frame in the query stack.
///
/// This is mostly used in case of cycles for error reporting.
#[derive(Clone, Debug)]
pub struct QueryStackFrame {
    pub description: String,
    span: Option<Span>,
    pub def_id: Option<DefId>,
    pub def_kind: Option<DefKind>,
    /// A def-id that is extracted from a `Ty` in a query key
    pub def_id_for_ty_in_cycle: Option<DefId>,
    pub dep_kind: DepKind,
    /// This hash is used to deterministically pick
    /// a query to remove cycles in the parallel compiler.
    hash: Hash64,
}

impl QueryStackFrame {
    #[inline]
    pub fn new(
        description: String,
        span: Option<Span>,
        def_id: Option<DefId>,
        def_kind: Option<DefKind>,
        dep_kind: DepKind,
        def_id_for_ty_in_cycle: Option<DefId>,
        hash: impl FnOnce() -> Hash64,
    ) -> Self {
        Self { description, span, def_id, def_kind, def_id_for_ty_in_cycle, dep_kind, hash: hash() }
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

/// Track a 'side effects' for a particular query.
/// This struct is saved to disk along with the query result,
/// and loaded from disk if we mark the query as green.
/// This allows us to 'replay' changes to global state
/// that would otherwise only occur if we actually
/// executed the query method.
///
/// Each side effect gets an unique dep node index which is added
/// as a dependency of the query which had the effect.
#[derive(Debug, Encodable, Decodable)]
pub enum QuerySideEffect {
    /// Stores a diagnostic emitted during query execution.
    /// This diagnostic will be re-emitted if we mark
    /// the query as green, as that query will have the side
    /// effect dep node as a dependency.
    Diagnostic(DiagInner),
}

pub trait QueryContext: HasDepContext {
    fn next_job_id(self) -> QueryJobId;

    /// Get the query information from the TLS context.
    fn current_query_job(self) -> Option<QueryJobId>;

    fn collect_active_jobs(self) -> QueryMap;

    /// Load a side effect associated to the node in the previous session.
    fn load_side_effect(
        self,
        prev_dep_node_index: SerializedDepNodeIndex,
    ) -> Option<QuerySideEffect>;

    /// Register a side effect for the given node, for use in next session.
    fn store_side_effect(self, dep_node_index: DepNodeIndex, side_effect: QuerySideEffect);

    /// Executes a job by changing the `ImplicitCtxt` to point to the
    /// new query job while it executes.
    fn start_query<R>(self, token: QueryJobId, depth_limit: bool, compute: impl FnOnce() -> R)
    -> R;

    fn depth_limit_error(self, job: QueryJobId);
}
