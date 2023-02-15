use crate::ty::{self, TyCtxt};
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_query_system::ich::StableHashingContext;
use rustc_session::Session;

#[macro_use]
mod dep_node;

pub use rustc_query_system::dep_graph::{
    debug::DepNodeFilter, hash_result, DepContext, DepNodeColor, DepNodeIndex,
    SerializedDepNodeIndex, WorkProduct, WorkProductId,
};

pub use dep_node::{label_strs, DepKind, DepNode, DepNodeExt};
pub(crate) use dep_node::{make_compile_codegen_unit, make_compile_mono_item};

pub type DepGraph = rustc_query_system::dep_graph::DepGraph<DepKind>;

pub type TaskDeps = rustc_query_system::dep_graph::TaskDeps<DepKind>;
pub type TaskDepsRef<'a> = rustc_query_system::dep_graph::TaskDepsRef<'a, DepKind>;
pub type DepGraphQuery = rustc_query_system::dep_graph::DepGraphQuery<DepKind>;
pub type SerializedDepGraph = rustc_query_system::dep_graph::SerializedDepGraph<DepKind>;
pub type EdgeFilter = rustc_query_system::dep_graph::debug::EdgeFilter<DepKind>;
pub type DepKindStruct<'tcx> = rustc_query_system::dep_graph::DepKindStruct<TyCtxt<'tcx>>;

impl rustc_query_system::dep_graph::DepKind for DepKind {
    const NULL: Self = DepKind::Null;
    const RED: Self = DepKind::Red;

    fn debug_node(node: &DepNode, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}(", node.kind)?;

        ty::tls::with_opt(|opt_tcx| {
            if let Some(tcx) = opt_tcx {
                if let Some(def_id) = node.extract_def_id(tcx) {
                    write!(f, "{}", tcx.def_path_debug_str(def_id))?;
                } else if let Some(ref s) = tcx.dep_graph.dep_node_debug_str(*node) {
                    write!(f, "{}", s)?;
                } else {
                    write!(f, "{}", node.hash)?;
                }
            } else {
                write!(f, "{}", node.hash)?;
            }
            Ok(())
        })?;

        write!(f, ")")
    }

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

impl<'tcx> DepContext for TyCtxt<'tcx> {
    type DepKind = DepKind;

    #[inline]
    fn with_stable_hashing_context<R>(&self, f: impl FnOnce(StableHashingContext<'_>) -> R) -> R {
        TyCtxt::with_stable_hashing_context(*self, f)
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
    fn dep_kind_info(&self, dep_kind: DepKind) -> &DepKindStruct<'tcx> {
        &self.query_kinds[dep_kind as usize]
    }
}
