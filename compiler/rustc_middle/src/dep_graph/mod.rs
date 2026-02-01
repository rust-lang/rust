use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_query_system::dep_graph::{DEP_KIND_UNUSED_BITS, unused_dep_kind_bits};
use rustc_query_system::ich::StableHashingContext;
use rustc_session::Session;

use crate::ty::print::with_reduced_queries;
use crate::ty::{self, TyCtxt};

#[macro_use]
mod dep_node;

pub use dep_node::{
    DEP_KIND_NAMES, DEP_KIND_VARIANTS, DepKind, DepNode, DepNodeExt, dep_kind_from_label,
    dep_kinds, label_strs,
};
pub(crate) use dep_node::{make_compile_codegen_unit, make_compile_mono_item, make_metadata};
pub use rustc_query_system::dep_graph::debug::{DepNodeFilter, EdgeFilter};
pub use rustc_query_system::dep_graph::{
    DepContext, DepGraphQuery, DepNodeIndex, Deps, SerializedDepGraph, SerializedDepNodeIndex,
    TaskDepsRef, WorkProduct, WorkProductId, WorkProductMap, hash_result,
};

pub type DepGraph = rustc_query_system::dep_graph::DepGraph;

pub type DepKindVTable<'tcx> = rustc_query_system::dep_graph::DepKindVTable<TyCtxt<'tcx>>;

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
}

/// Verify that the unused bits for the dep kind matches the hardcoded value in `rustc_query_system`.
const _: [(); unused_dep_kind_bits(dep_node::DEP_KIND_VARIANTS)] = [(); DEP_KIND_UNUSED_BITS];

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
    fn dep_kind_vtable(&self, dk: DepKind) -> &DepKindVTable<'tcx> {
        &self.dep_kind_vtables[dk.as_usize()]
    }

    fn with_reduced_queries<T>(self, f: impl FnOnce() -> T) -> T {
        with_reduced_queries!(f())
    }
}
