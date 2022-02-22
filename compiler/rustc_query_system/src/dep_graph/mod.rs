mod dep_node;
mod graph;

pub use dep_node::{DepNode, WorkProductId};
pub use graph::{DepGraph, DepNodeIndex, WorkProduct};

use crate::ich::StableHashingContext;
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_session::Session;

pub trait DepContext: Copy {
    /// Create a hashing context for hashing new results.
    fn create_stable_hashing_context(&self) -> StableHashingContext<'_>;

    /// Access the DepGraph.
    fn dep_graph(&self) -> &DepGraph;

    /// Access the profiler.
    fn profiler(&self) -> &SelfProfilerRef;

    /// Access the compiler session.
    fn sess(&self) -> &Session;
}

pub trait HasDepContext: Copy {
    type DepContext: self::DepContext;

    fn dep_context(&self) -> &Self::DepContext;
}

impl<T: DepContext> HasDepContext for T {
    type DepContext = Self;

    fn dep_context(&self) -> &Self::DepContext {
        self
    }
}
