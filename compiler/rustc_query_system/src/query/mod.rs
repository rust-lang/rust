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
pub use self::config::{QueryAccessors, QueryConfig, QueryDescription};

use crate::dep_graph::{DepNode, DepNodeIndex, HasDepContext, SerializedDepNodeIndex};

use rustc_data_structures::sync::Lock;
use rustc_data_structures::thin_vec::ThinVec;
use rustc_errors::Diagnostic;
use rustc_span::Span;

/// Description of a frame in the query stack.
///
/// This is mostly used in case of cycles for error reporting.
#[derive(Clone, Debug)]
pub struct QueryStackFrame {
    pub name: &'static str,
    pub description: String,
    span: Option<Span>,
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
        _hash: impl FnOnce() -> u64,
    ) -> Self {
        Self {
            name,
            description,
            span,
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

/// Tracks 'side effects' for a particular query.
/// This struct is saved to disk along with the query result,
/// and loaded from disk if we mark the query as green.
/// This allows us to 'replay' changes to global state
/// that would otherwise only occur if we actually
/// executed the query method.
#[derive(Debug, Clone, Default, Encodable, Decodable)]
pub struct QuerySideEffects {
    /// Stores any diagnostics emitted during query execution.
    /// These diagnostics will be re-emitted if we mark
    /// the query as green.
    pub(super) diagnostics: ThinVec<Diagnostic>,
}

impl QuerySideEffects {
    pub fn is_empty(&self) -> bool {
        let QuerySideEffects { diagnostics } = self;
        diagnostics.is_empty()
    }
    pub fn append(&mut self, other: QuerySideEffects) {
        let QuerySideEffects { diagnostics } = self;
        diagnostics.extend(other.diagnostics);
    }
}

pub trait QueryContext: HasDepContext {
    /// Get the query information from the TLS context.
    fn current_query_job(&self) -> Option<QueryJobId<Self::DepKind>>;

    fn try_collect_active_jobs(&self) -> Option<QueryMap<Self::DepKind>>;

    /// Load data from the on-disk cache.
    fn try_load_from_on_disk_cache(&self, dep_node: &DepNode<Self::DepKind>);

    /// Try to force a dep node to execute and see if it's green.
    fn try_force_from_dep_node(&self, dep_node: &DepNode<Self::DepKind>) -> bool;

    /// Load side effects associated to the node in the previous session.
    fn load_side_effects(&self, prev_dep_node_index: SerializedDepNodeIndex) -> QuerySideEffects;

    /// Register diagnostics for the given node, for use in next session.
    fn store_side_effects(&self, dep_node_index: DepNodeIndex, side_effects: QuerySideEffects);

    /// Register diagnostics for the given node, for use in next session.
    fn store_side_effects_for_anon_node(
        &self,
        dep_node_index: DepNodeIndex,
        side_effects: QuerySideEffects,
    );

    /// Executes a job by changing the `ImplicitCtxt` to point to the
    /// new query job while it executes. It returns the diagnostics
    /// captured during execution and the actual result.
    fn start_query<R>(
        &self,
        token: QueryJobId<Self::DepKind>,
        diagnostics: Option<&Lock<ThinVec<Diagnostic>>>,
        compute: impl FnOnce() -> R,
    ) -> R;
}
