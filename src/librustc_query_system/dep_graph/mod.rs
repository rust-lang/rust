pub mod debug;
mod dep_node;
mod graph;
mod prev;
mod query;
mod safe;
mod serialized;

pub use dep_node::{DepNode, DepNodeParams, WorkProductId};
pub use graph::WorkProductFileKind;
pub use graph::{hash_result, DepGraph, DepNodeColor, DepNodeIndex, TaskDeps, WorkProduct};
pub use prev::PreviousDepGraph;
pub use query::DepGraphQuery;
pub use safe::AssertDepGraphSafe;
pub use safe::DepGraphSafe;
pub use serialized::{SerializedDepGraph, SerializedDepNodeIndex};

use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_data_structures::sync::Lock;
use rustc_data_structures::thin_vec::ThinVec;
use rustc_errors::Diagnostic;
use rustc_hir::def_id::DefId;

use std::fmt;
use std::hash::Hash;

pub trait DepContext: Copy {
    type DepKind: self::DepKind;
    type StableHashingContext: crate::HashStableContext;

    /// Create a hashing context for hashing new results.
    fn create_stable_hashing_context(&self) -> Self::StableHashingContext;

    /// Try to force a dep node to execute and see if it's green.
    fn try_force_previous_green(&self, node: &DepNode<Self::DepKind>) -> bool;

    /// Extracts the DefId corresponding to this DepNode. This will work
    /// if two conditions are met:
    ///
    /// 1. The Fingerprint of the DepNode actually is a DefPathHash, and
    /// 2. the item that the DefPath refers to exists in the current tcx.
    ///
    /// Condition (1) is determined by the DepKind variant of the
    /// DepNode. Condition (2) might not be fulfilled if a DepNode
    /// refers to something from the previous compilation session that
    /// has been removed.
    fn extract_def_id(&self, node: &DepNode<Self::DepKind>) -> Option<DefId>;

    /// Return whether the current session is tainted by errors.
    fn has_errors_or_delayed_span_bugs(&self) -> bool;

    /// Return the diagnostic handler.
    fn diagnostic(&self) -> &rustc_errors::Handler;

    /// Load data from the on-disk cache.
    fn try_load_from_on_disk_cache(&self, dep_node: &DepNode<Self::DepKind>);

    /// Load diagnostics associated to the node in the previous session.
    fn load_diagnostics(&self, prev_dep_node_index: SerializedDepNodeIndex) -> Vec<Diagnostic>;

    /// Register diagnostics for the given node, for use in next session.
    fn store_diagnostics(&self, dep_node_index: DepNodeIndex, diagnostics: ThinVec<Diagnostic>);

    /// Access the profiler.
    fn profiler(&self) -> &SelfProfilerRef;
}

/// Describe the different families of dependency nodes.
pub trait DepKind: Copy + fmt::Debug + Eq + Ord + Hash {
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
    fn read_deps<OP>(op: OP) -> ()
    where
        OP: for<'a> FnOnce(Option<&'a Lock<TaskDeps<Self>>>) -> ();
}
