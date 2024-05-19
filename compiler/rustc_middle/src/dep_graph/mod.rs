use crate::ty::{self, TyCtxt};
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_query_system::ich::StableHashingContext;
use rustc_session::Session;

#[macro_use]
mod dep_node;

pub use rustc_query_system::dep_graph::debug::EdgeFilter;
pub use rustc_query_system::dep_graph::{
    debug::DepNodeFilter, hash_result, DepContext, DepGraphQuery, DepNodeIndex, Deps,
    SerializedDepGraph, SerializedDepNodeIndex, TaskDepsRef, WorkProduct, WorkProductId,
    WorkProductMap,
};

pub use dep_node::{dep_kinds, label_strs, DepKind, DepNode, DepNodeExt};
pub(crate) use dep_node::{make_compile_codegen_unit, make_compile_mono_item};

pub type DepGraph = rustc_query_system::dep_graph::DepGraph<DepsType>;

pub type DepKindStruct<'tcx> = rustc_query_system::dep_graph::DepKindStruct<TyCtxt<'tcx>>;

#[derive(Clone)]
pub struct DepsType;

impl Deps for DepsType {
    fn with_deps<OP, R>(task_deps: TaskDepsRef<'_>, op: OP) -> R
    where
        OP: FnOnce() -> R,
    {
        ty::tls::with_context(|icx| {
            let icx = ty::tls::ImplicitCtxt { task_deps, ..icx.clone() };

            ty::tls::enter_context(&icx, op)
        })
    }

    fn read_deps<OP>(op: OP)
    where
        OP: for<'a> FnOnce(TaskDepsRef<'a>),
    {
        ty::tls::with_context_opt(|icx| {
            let Some(icx) = icx else { return };
            op(icx.task_deps)
        })
    }

    const DEP_KIND_NULL: DepKind = dep_kinds::Null;
    const DEP_KIND_RED: DepKind = dep_kinds::Red;
    const DEP_KIND_MAX: u16 = dep_node::DEP_KIND_VARIANTS - 1;
}

impl<'tcx> DepContext for TyCtxt<'tcx> {
    type Deps = DepsType;

    #[inline]
    fn with_stable_hashing_context<R>(self, f: impl FnOnce(StableHashingContext<'_>) -> R) -> R {
        TyCtxt::with_stable_hashing_context(self, f)
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

    #[inline]
    fn dep_kind_info(&self, dk: DepKind) -> &DepKindStruct<'tcx> {
        &self.query_kinds[dk.as_usize()]
    }
}
