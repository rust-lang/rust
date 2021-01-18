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

use std::fmt;
use std::hash::Hash;

pub trait DepContext: Copy {
    type DepKind: self::DepKind;
    type StableHashingContext;

    /// Create a hashing context for hashing new results.
    fn create_stable_hashing_context(&self) -> Self::StableHashingContext;

    fn debug_dep_tasks(&self) -> bool;
    fn debug_dep_node(&self) -> bool;

    /// Access the DepGraph.
    fn dep_graph(&self) -> &DepGraph<Self::DepKind>;

    fn register_reused_dep_node(&self, dep_node: &DepNode<Self::DepKind>);

    /// Access the profiler.
    fn profiler(&self) -> &SelfProfilerRef;
}

pub trait HasDepContext: Copy {
    type DepKind: self::DepKind;
    type StableHashingContext;
    type DepContext: self::DepContext<
        DepKind = Self::DepKind,
        StableHashingContext = Self::StableHashingContext,
    >;

    fn dep_context(&self) -> &Self::DepContext;
}

impl<T: DepContext> HasDepContext for T {
    type DepKind = T::DepKind;
    type StableHashingContext = T::StableHashingContext;
    type DepContext = Self;

    fn dep_context(&self) -> &Self::DepContext {
        self
    }
}

/// Describe the different families of dependency nodes.
pub trait DepKind: Copy + fmt::Debug + Eq + Hash {
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
