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
pub use self::config::{QueryConfig, QueryDescription, QueryVtable};

use crate::dep_graph::{DepNodeIndex, HasDepContext, SerializedDepNodeIndex};

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
    /// The `DefKind` this query frame is associated with, if applicable.
    ///
    /// We can't use `rustc_hir::def::DefKind` because `rustc_hir` is not
    /// available in `rustc_query_system`. Instead, we have a simplified
    /// custom version of it, called [`SimpleDefKind`].
    def_kind: Option<SimpleDefKind>,
    /// This hash is used to deterministically pick
    /// a query to remove cycles in the parallel compiler.
    #[cfg(parallel_compiler)]
    hash: u64,
}

/// A simplified version of `rustc_hir::def::DefKind`.
///
/// It was added to help improve cycle errors caused by recursive type aliases.
/// As of August 2021, `rustc_query_system` cannot depend on `rustc_hir`
/// because it would create a dependency cycle. So, instead, a simplified
/// version of `DefKind` was added to `rustc_query_system`.
///
/// `DefKind`s are converted to `SimpleDefKind`s in `rustc_query_impl`.
#[derive(Debug, Copy, Clone)]
pub enum SimpleDefKind {
    Struct,
    Enum,
    Union,
    Trait,
    TyAlias,
    TraitAlias,

    // FIXME: add more from `rustc_hir::def::DefKind` and then remove `Other`
    Other,
}

impl QueryStackFrame {
    #[inline]
    pub fn new(
        name: &'static str,
        description: String,
        span: Option<Span>,
        def_kind: Option<SimpleDefKind>,
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
    fn next_job_id(&self) -> QueryJobId;

    /// Get the query information from the TLS context.
    fn current_query_job(&self) -> Option<QueryJobId>;

    fn try_collect_active_jobs(&self) -> Option<QueryMap>;

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
        token: QueryJobId,
        diagnostics: Option<&Lock<ThinVec<Diagnostic>>>,
        compute: impl FnOnce() -> R,
    ) -> R;
}
