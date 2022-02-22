use crate::ty::TyCtxt;
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_query_system::ich::StableHashingContext;
use rustc_session::Session;

#[macro_use]
mod dep_node;

pub use rustc_query_system::dep_graph::{DepContext, DepNodeIndex, WorkProduct, WorkProductId};

pub use dep_node::{label_strs, DepKind, DepNode};

pub type DepGraph = rustc_query_system::dep_graph::DepGraph;

impl<'tcx> DepContext for TyCtxt<'tcx> {
    #[inline]
    fn create_stable_hashing_context(&self) -> StableHashingContext<'_> {
        TyCtxt::create_stable_hashing_context(*self)
    }

    #[inline]
    fn dep_graph(&self) -> &DepGraph {
        &self.dep_graph
    }

    #[inline(always)]
    fn profiler(&self) -> &SelfProfilerRef {
        &self.prof
    }

    #[inline(always)]
    fn sess(&self) -> &Session {
        self.sess
    }
}
