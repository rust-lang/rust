mod plumbing;
pub use self::plumbing::*;

mod job;
#[cfg(parallel_compiler)]
pub use self::job::deadlock;
pub use self::job::{QueryInfo, QueryJob, QueryJobId, QueryJobInfo, QueryMap};

mod caches;
pub use self::caches::{
    ArenaCacheSelector, CacheSelector, DefaultCacheSelector, QueryCache, QueryStorage,
};

mod config;
pub use self::config::{QueryAccessors, QueryConfig, QueryDescription};

use crate::dep_graph::{DepNode, DepNodeIndex, HasDepContext, SerializedDepNodeIndex};

use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::sync::Lock;
use rustc_data_structures::thin_vec::ThinVec;
use rustc_errors::Diagnostic;
use rustc_span::def_id::DefId;
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
    hash: Fingerprint,
}

impl QueryStackFrame {
    #[inline]
    pub fn new(
        name: &'static str,
        description: String,
        span: Option<Span>,
        hash: Fingerprint,
    ) -> Self {
        Self { name, hash, description, span }
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

impl<CTX> HashStable<CTX> for QueryStackFrame {
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
        self.hash.hash_stable(hcx, hasher)
    }
}

pub trait QueryContext: HasDepContext {
    /// Get string representation from DefPath.
    fn def_path_str(&self, def_id: DefId) -> String;

    /// Get the query information from the TLS context.
    fn current_query_job(&self) -> Option<QueryJobId<Self::DepKind>>;

    fn try_collect_active_jobs(&self) -> Option<QueryMap<Self::DepKind>>;

    /// Load data from the on-disk cache.
    fn try_load_from_on_disk_cache(&self, dep_node: &DepNode<Self::DepKind>);

    /// Try to force a dep node to execute and see if it's green.
    fn try_force_from_dep_node(&self, dep_node: &DepNode<Self::DepKind>) -> bool;

    /// Load diagnostics associated to the node in the previous session.
    fn load_diagnostics(&self, prev_dep_node_index: SerializedDepNodeIndex) -> Vec<Diagnostic>;

    /// Register diagnostics for the given node, for use in next session.
    fn store_diagnostics(&self, dep_node_index: DepNodeIndex, diagnostics: ThinVec<Diagnostic>);

    /// Register diagnostics for the given node, for use in next session.
    fn store_diagnostics_for_anon_node(
        &self,
        dep_node_index: DepNodeIndex,
        diagnostics: ThinVec<Diagnostic>,
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
