mod plumbing;
pub use self::plumbing::*;

mod job;
#[cfg(parallel_compiler)]
pub use self::job::deadlock;
pub use self::job::{QueryInfo, QueryJob, QueryJobId, QueryJobInfo};

mod caches;
pub use self::caches::{
    ArenaCacheSelector, CacheSelector, DefaultCacheSelector, QueryCache, QueryStorage,
};

mod config;
pub use self::config::{QueryAccessors, QueryConfig, QueryDescription};

use crate::dep_graph::{DepNode, DepNodeIndex, HasDepContext, SerializedDepNodeIndex};
use crate::query::job::QueryMap;

use rustc_data_structures::stable_hasher::HashStable;
use rustc_data_structures::sync::Lock;
use rustc_data_structures::thin_vec::ThinVec;
use rustc_errors::Diagnostic;
use rustc_span::def_id::DefId;

pub trait QueryContext: HasDepContext {
    type Query: Clone + HashStable<Self::StableHashingContext>;

    fn incremental_verify_ich(&self) -> bool;
    fn verbose(&self) -> bool;

    /// Get string representation from DefPath.
    fn def_path_str(&self, def_id: DefId) -> String;

    /// Get the query information from the TLS context.
    fn current_query_job(&self) -> Option<QueryJobId<Self::DepKind>>;

    fn try_collect_active_jobs(&self) -> Option<QueryMap<Self::DepKind, Self::Query>>;

    /// Load data from the on-disk cache.
    fn try_load_from_on_disk_cache(&self, dep_node: &DepNode<Self::DepKind>);

    /// Try to force a dep node to execute and see if it's green.
    fn try_force_from_dep_node(&self, dep_node: &DepNode<Self::DepKind>) -> bool;

    /// Return whether the current session is tainted by errors.
    fn has_errors_or_delayed_span_bugs(&self) -> bool;

    /// Return the diagnostic handler.
    fn diagnostic(&self) -> &rustc_errors::Handler;

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
