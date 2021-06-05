use crate::ich::StableHashingContext;
use crate::ty::TyCtxt;
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_session::Session;

#[macro_use]
mod dep_node;

pub use rustc_query_system::dep_graph::{
    debug::DepNodeFilter, debug::EdgeFilter, hash_result, label_strs, DepContext, DepGraph,
    DepGraphQuery, DepKind, DepNode, DepNodeColor, DepNodeIndex, SerializedDepGraph,
    SerializedDepNodeIndex, TaskDeps, WorkProduct, WorkProductId,
};

pub use dep_node::DepNodeExt;
crate use dep_node::{make_compile_codegen_unit, make_compile_mono_item};

impl<'tcx> DepContext for TyCtxt<'tcx> {
    type StableHashingContext = StableHashingContext<'tcx>;

    fn register_reused_dep_node(&self, dep_node: &DepNode) {
        if let Some(cache) = self.on_disk_cache.as_ref() {
            cache.register_reused_dep_node(*self, dep_node)
        }
    }

    fn create_stable_hashing_context(&self) -> Self::StableHashingContext {
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
