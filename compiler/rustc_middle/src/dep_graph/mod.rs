use crate::ty::{self, TyCtxt};
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_data_structures::sync::Lock;
use rustc_query_system::ich::StableHashingContext;
use rustc_session::Session;

#[macro_use]
mod dep_node;

pub use rustc_query_system::dep_graph::{
    debug::DepNodeFilter, hash_result, DepContext, DepNodeColor, DepNodeIndex,
    SerializedDepNodeIndex, WorkProduct, WorkProductId,
};

pub use dep_node::{label_strs, DepKind, DepKindStruct, DepNode, DepNodeExt};
crate use dep_node::{make_compile_codegen_unit, make_compile_mono_item};

pub type DepGraph = rustc_query_system::dep_graph::DepGraph<DepKind>;
pub type TaskDeps = rustc_query_system::dep_graph::TaskDeps<DepKind>;
pub type DepGraphQuery = rustc_query_system::dep_graph::DepGraphQuery<DepKind>;
pub type SerializedDepGraph = rustc_query_system::dep_graph::SerializedDepGraph<DepKind>;
pub type EdgeFilter = rustc_query_system::dep_graph::debug::EdgeFilter<DepKind>;

impl rustc_query_system::dep_graph::DepKind for DepKind {
    const NULL: Self = DepKind::Null;

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

    #[inline(always)]
    fn fingerprint_style(&self, kind: DepKind) -> rustc_query_system::dep_graph::FingerprintStyle {
        kind.fingerprint_style(*self)
    }

    #[inline(always)]
    fn is_eval_always(&self, kind: DepKind) -> bool {
        self.query_kind(kind).is_eval_always
    }

    fn try_force_from_dep_node(&self, dep_node: DepNode) -> bool {
        debug!("try_force_from_dep_node({:?}) --- trying to force", dep_node);

        // We must avoid ever having to call `force_from_dep_node()` for a
        // `DepNode::codegen_unit`:
        // Since we cannot reconstruct the query key of a `DepNode::codegen_unit`, we
        // would always end up having to evaluate the first caller of the
        // `codegen_unit` query that *is* reconstructible. This might very well be
        // the `compile_codegen_unit` query, thus re-codegenning the whole CGU just
        // to re-trigger calling the `codegen_unit` query with the right key. At
        // that point we would already have re-done all the work we are trying to
        // avoid doing in the first place.
        // The solution is simple: Just explicitly call the `codegen_unit` query for
        // each CGU, right after partitioning. This way `try_mark_green` will always
        // hit the cache instead of having to go through `force_from_dep_node`.
        // This assertion makes sure, we actually keep applying the solution above.
        debug_assert!(
            dep_node.kind != DepKind::codegen_unit,
            "calling force_from_dep_node() on DepKind::codegen_unit"
        );

        let cb = self.query_kind(dep_node.kind);
        if let Some(f) = cb.force_from_dep_node {
            f(*self, dep_node);
            true
        } else {
            false
        }
    }

    fn try_load_from_on_disk_cache(&self, dep_node: DepNode) {
        let cb = self.query_kind(dep_node.kind);
        if let Some(f) = cb.try_load_from_on_disk_cache {
            f(*self, dep_node)
        }
    }
}
