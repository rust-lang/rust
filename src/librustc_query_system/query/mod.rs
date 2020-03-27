mod plumbing;
pub use self::plumbing::*;

mod job;
#[cfg(parallel_compiler)]
pub use self::job::deadlock;
pub use self::job::{QueryInfo, QueryJob, QueryJobId, QueryJobInfo};

mod caches;
pub use self::caches::{CacheSelector, DefaultCacheSelector, QueryCache};

mod config;
pub use self::config::{QueryAccessors, QueryConfig, QueryDescription};

use crate::dep_graph::{DepContext, DepGraph};

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stable_hasher::HashStable;
use rustc_data_structures::sync::Lock;
use rustc_data_structures::thin_vec::ThinVec;
use rustc_errors::Diagnostic;
use rustc_span::def_id::DefId;

pub trait QueryContext: DepContext {
    type Query: Clone + HashStable<Self::StableHashingContext>;

    fn incremental_verify_ich(&self) -> bool;
    fn verbose(&self) -> bool;

    /// Get string representation from DefPath.
    fn def_path_str(&self, def_id: DefId) -> String;

    /// Access the DepGraph.
    fn dep_graph(&self) -> &DepGraph<Self::DepKind>;

    /// Get the query information from the TLS context.
    fn current_query_job(&self) -> Option<QueryJobId<Self::DepKind>>;

    fn try_collect_active_jobs(
        &self,
    ) -> Option<FxHashMap<QueryJobId<Self::DepKind>, QueryJobInfo<Self>>>;

    /// Executes a job by changing the `ImplicitCtxt` to point to the
    /// new query job while it executes. It returns the diagnostics
    /// captured during execution and the actual result.
    fn start_query<R>(
        &self,
        token: QueryJobId<Self::DepKind>,
        diagnostics: Option<&Lock<ThinVec<Diagnostic>>>,
        compute: impl FnOnce(Self) -> R,
    ) -> R;
}
