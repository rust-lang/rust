use crate::ich::StableHashingContext;
use crate::ty::{self, TyCtxt};
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_data_structures::sync::Lock;
use rustc_session::Session;

#[macro_use]
mod dep_node;

pub use rustc_query_system::dep_graph::{
    debug::DepNodeFilter, hash_result, DepContext, DepNodeColor, DepNodeIndex,
    SerializedDepNodeIndex, WorkProduct, WorkProductId,
};

pub use dep_node::{label_strs, DepKind, DepNode, DepNodeExt};
crate use dep_node::{make_compile_codegen_unit, make_compile_mono_item};

pub type DepGraph = rustc_query_system::dep_graph::DepGraph<DepKind>;
pub type TaskDeps = rustc_query_system::dep_graph::TaskDeps<DepKind>;
pub type DepGraphQuery = rustc_query_system::dep_graph::DepGraphQuery<DepKind>;
pub type SerializedDepGraph = rustc_query_system::dep_graph::SerializedDepGraph<DepKind>;
pub type EdgeFilter = rustc_query_system::dep_graph::debug::EdgeFilter<DepKind>;

impl rustc_query_system::dep_graph::DepKind for DepKind {
    const NULL: Self = DepKind::Null;

    #[inline(always)]
    fn can_reconstruct_query_key(&self) -> bool {
        DepKind::can_reconstruct_query_key(self)
    }

    #[inline(always)]
    fn is_eval_always(&self) -> bool {
        self.is_eval_always
    }

    #[inline(always)]
    fn has_params(&self) -> bool {
        self.has_params
    }

    fn debug_node(node: &DepNode, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", node.kind)?;

        if !node.kind.has_params && !node.kind.is_anon {
            return Ok(());
        }

        write!(f, "(")?;

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

    fn with_deps<OP, R>(task_deps: Option<&Lock<TaskDeps>>, op: OP) -> R
    where
        OP: FnOnce() -> R,
    {
        ty::tls::with_context(|icx| {
            let icx = ty::tls::ImplicitCtxt { task_deps, ..icx.clone() };

            ty::tls::enter_context(&icx, |_| op())
        })
    }

    fn read_deps<OP>(op: OP)
    where
        OP: for<'a> FnOnce(Option<&'a Lock<TaskDeps>>),
    {
        ty::tls::with_context_opt(|icx| {
            let icx = if let Some(icx) = icx { icx } else { return };
            op(icx.task_deps)
        })
    }
}

impl<'tcx> DepContext for TyCtxt<'tcx> {
    type DepKind = DepKind;
    type StableHashingContext = StableHashingContext<'tcx>;

    #[inline]
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
