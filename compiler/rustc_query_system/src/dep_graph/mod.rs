pub mod debug;
mod dep_node;
pub mod dep_kind;
mod graph;
mod prev;
mod query;
mod serialized;

pub use dep_node::{DepNode, DepNodeParams, WorkProductId};
pub use dep_kind::{DepKindExt, DepKind};
pub use graph::{hash_result, DepGraph, DepNodeColor, DepNodeIndex, TaskDeps, WorkProduct};
pub use prev::PreviousDepGraph;
pub use query::DepGraphQuery;
pub use serialized::{SerializedDepGraph, SerializedDepNodeIndex};

use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_data_structures::thin_vec::ThinVec;
use rustc_errors::Diagnostic;

pub trait DepContext: Copy {
    type DepKind: self::DepKindExt;
    type StableHashingContext;

    /// Create a hashing context for hashing new results.
    fn create_stable_hashing_context(&self) -> Self::StableHashingContext;

    fn debug_dep_tasks(&self) -> bool;
    fn debug_dep_node(&self) -> bool;

    /// Try to force a dep node to execute and see if it's green.
    fn try_force_from_dep_node(&self, dep_node: &DepNode) -> bool;

    /// Return whether the current session is tainted by errors.
    fn has_errors_or_delayed_span_bugs(&self) -> bool;

    /// Return the diagnostic handler.
    fn diagnostic(&self) -> &rustc_errors::Handler;

    /// Load data from the on-disk cache.
    fn try_load_from_on_disk_cache(&self, dep_node: &DepNode);

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

    /// Access the profiler.
    fn profiler(&self) -> &SelfProfilerRef;
}
