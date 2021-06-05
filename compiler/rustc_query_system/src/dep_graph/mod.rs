pub mod debug;
mod dep_kind;
mod dep_node;
mod graph;
mod query;
mod serialized;

pub use dep_kind::{dep_kind_from_label_string, label_strs, DepKind};
pub use dep_node::{DepNode, DepNodeParams, WorkProductId, NODE_DEBUG};
pub use graph::{hash_result, DepGraph, DepNodeColor, DepNodeIndex, TaskDeps, WorkProduct};
pub use query::DepGraphQuery;
pub use serialized::{SerializedDepGraph, SerializedDepNodeIndex};

use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_session::Session;

pub trait DepContext: Copy {
    type StableHashingContext;

    /// Create a hashing context for hashing new results.
    fn create_stable_hashing_context(&self) -> Self::StableHashingContext;

    /// Access the DepGraph.
    fn dep_graph(&self) -> &DepGraph;

    fn register_reused_dep_node(&self, dep_node: &DepNode);

    /// Access the profiler.
    fn profiler(&self) -> &SelfProfilerRef;

    /// Access the compiler session.
    fn sess(&self) -> &Session;
}

pub trait HasDepContext: Copy {
    type StableHashingContext;
    type DepContext: self::DepContext<StableHashingContext = Self::StableHashingContext>;

    fn dep_context(&self) -> &Self::DepContext;
}

impl<T: DepContext> HasDepContext for T {
    type StableHashingContext = T::StableHashingContext;
    type DepContext = Self;

    fn dep_context(&self) -> &Self::DepContext {
        self
    }
}
