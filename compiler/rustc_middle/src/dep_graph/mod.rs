use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_query_system::ich::StableHashingContext;
use rustc_session::Session;

use crate::ty::{self, TyCtxt};

#[macro_use]
mod dep_node;

pub use dep_node::{DepKind, DepNode, DepNodeExt, dep_kinds, label_strs};
pub(crate) use dep_node::{make_compile_codegen_unit, make_compile_mono_item};
pub use rustc_query_system::dep_graph::debug::{DepNodeFilter, EdgeFilter};
pub use rustc_query_system::dep_graph::{
    DepContext, DepGraphQuery, DepNodeIndex, Deps, SerializedDepGraph, SerializedDepNodeIndex,
    TaskDepsRef, WorkProduct, WorkProductId, WorkProductMap, hash_result,
};

pub type DepGraph = rustc_query_system::dep_graph::DepGraph<DepsType>;

pub type DepKindStruct<'tcx> = rustc_query_system::dep_graph::DepKindStruct<TyCtxt<'tcx>>;

#[derive(Clone)]
pub struct DepsType {
    pub dep_names: Vec<&'static str>,
}

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

    fn name(&self, dep_kind: DepKind) -> &'static str {
        self.dep_names[dep_kind.as_usize()]
    }

    const DEP_KIND_NULL: DepKind = dep_kinds::Null;
    const DEP_KIND_RED: DepKind = dep_kinds::Red;
    const DEP_KIND_SIDE_EFFECT: DepKind = dep_kinds::SideEffect;
    const DEP_KIND_ANON_ZERO_DEPS: DepKind = dep_kinds::AnonZeroDeps;
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
