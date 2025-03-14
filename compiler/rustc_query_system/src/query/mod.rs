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
use rustc_data_structures::sync::Lock;
use rustc_errors::DiagInner;
use rustc_hashes::Hash64;
use rustc_hir::def::DefKind;
use rustc_macros::{Decodable, Encodable};
use rustc_span::Span;
use rustc_span::def_id::DefId;
use thin_vec::ThinVec;

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
    pub(super) diagnostics: ThinVec<DiagInner>,
}

impl QuerySideEffects {
    /// Returns true if there might be side effects.
    #[inline]
    pub fn maybe_any(&self) -> bool {
        let QuerySideEffects { diagnostics } = self;
        // Use `has_capacity` so that the destructor for `self.diagnostics` can be skipped
        // if `maybe_any` is known to be false.
        diagnostics.has_capacity()
    }
    pub fn append(&mut self, other: QuerySideEffects) {
        let QuerySideEffects { diagnostics } = self;
        diagnostics.extend(other.diagnostics);
    }
}

pub trait QueryContext: HasDepContext {
    fn next_job_id(self) -> QueryJobId;

    /// Get the query information from the TLS context.
    fn current_query_job(self) -> Option<QueryJobId>;

    fn collect_active_jobs(self) -> QueryMap;

    /// Load side effects associated to the node in the previous session.
    fn load_side_effects(self, prev_dep_node_index: SerializedDepNodeIndex) -> QuerySideEffects;

    /// Register diagnostics for the given node, for use in next session.
    fn store_side_effects(self, dep_node_index: DepNodeIndex, side_effects: QuerySideEffects);

    /// Register diagnostics for the given node, for use in next session.
    fn store_side_effects_for_anon_node(
        self,
        dep_node_index: DepNodeIndex,
        side_effects: QuerySideEffects,
    );

    /// Executes a job by changing the `ImplicitCtxt` to point to the
    /// new query job while it executes. It returns the diagnostics
    /// captured during execution and the actual result.
    fn start_query<R>(
        self,
        token: QueryJobId,
        depth_limit: bool,
        diagnostics: Option<&Lock<ThinVec<DiagInner>>>,
        compute: impl FnOnce() -> R,
    ) -> R;

    fn depth_limit_error(self, job: QueryJobId);
}
