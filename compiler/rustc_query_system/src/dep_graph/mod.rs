pub mod debug;
mod dep_node;
mod graph;
mod prev;
mod query;
mod serialized;

pub use dep_node::{DepNode, DepNodeParams, WorkProductId};
pub use graph::{hash_result, DepGraph, DepNodeColor, DepNodeIndex, TaskDeps, WorkProduct};
pub use prev::PreviousDepGraph;
pub use query::DepGraphQuery;
pub use serialized::{SerializedDepGraph, SerializedDepNodeIndex};

use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_data_structures::sync::Lock;
use rustc_data_structures::thin_vec::ThinVec;
use rustc_errors::Diagnostic;
use rustc_serialize::opaque;
use rustc_session::Session;
use rustc_span::def_id::DefPathHash;

use std::fmt;
use std::hash::Hash;

pub trait DepContext: Copy {
    type DepKind: self::DepKind;
    type OnDiskCache: self::OnDiskCache<Self>;
    type StableHashingContext;

    /// Create a hashing context for hashing new results.
    fn create_stable_hashing_context(&self) -> Self::StableHashingContext;

    /// Try to force a dep node to execute and see if it's green.
    fn try_force_from_dep_node(&self, dep_node: &DepNode<Self::DepKind>) -> bool;

    fn register_reused_dep_path_hash(&self, hash: DefPathHash);

    /// Load data from the on-disk cache.
    fn try_load_from_on_disk_cache(&self, dep_node: &DepNode<Self::DepKind>);

    /// Access the on_disk_cache.
    fn on_disk_cache(&self) -> Option<&Self::OnDiskCache>;

    /// Access the profiler.
    fn profiler(&self) -> &SelfProfilerRef;

    /// Access the compiler session.
    fn sess(&self) -> &Session;
}

pub trait OnDiskCache<CTX> {
    fn serialize(&self, tcx: CTX, encoder: &mut opaque::Encoder);

    /// Loads a diagnostic emitted during the previous compilation session.
    fn load_diagnostics(&self, tcx: CTX, dep_node_index: SerializedDepNodeIndex)
    -> Vec<Diagnostic>;

    /// Stores a diagnostic emitted during the current compilation session.
    /// Anything stored like this will be available via `load_diagnostics` in
    /// the next compilation session.
    fn store_diagnostics(&self, dep_node_index: DepNodeIndex, diagnostics: ThinVec<Diagnostic>);

    /// Stores a diagnostic emitted during computation of an anonymous query.
    /// Since many anonymous queries can share the same `DepNode`, we aggregate
    /// them -- as opposed to regular queries where we assume that there is a
    /// 1:1 relationship between query-key and `DepNode`.
    fn store_diagnostics_for_anon_node(
        &self,
        dep_node_index: DepNodeIndex,
        diagnostics: ThinVec<Diagnostic>,
    );
}

/// Describe the different families of dependency nodes.
pub trait DepKind: Copy + fmt::Debug + Eq + Ord + Hash {
    const NULL: Self;

    /// Return whether this kind always require evaluation.
    fn is_eval_always(&self) -> bool;

    /// Return whether this kind requires additional parameters to be executed.
    fn has_params(&self) -> bool;

    /// Implementation of `std::fmt::Debug` for `DepNode`.
    fn debug_node(node: &DepNode<Self>, f: &mut fmt::Formatter<'_>) -> fmt::Result;

    /// Execute the operation with provided dependencies.
    fn with_deps<OP, R>(deps: Option<&Lock<TaskDeps<Self>>>, op: OP) -> R
    where
        OP: FnOnce() -> R;

    /// Access dependencies from current implicit context.
    fn read_deps<OP>(op: OP)
    where
        OP: for<'a> FnOnce(Option<&'a Lock<TaskDeps<Self>>>);

    fn can_reconstruct_query_key(&self) -> bool;
}
