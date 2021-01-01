use crate::ich::StableHashingContext;
use crate::ty::query::try_load_from_on_disk_cache;
use crate::ty::{self, TyCtxt};
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_data_structures::sync::Lock;
use rustc_data_structures::thin_vec::ThinVec;
use rustc_errors::Diagnostic;
use rustc_hir::def_id::LocalDefId;

mod dep_node;

pub(crate) use rustc_query_system::dep_graph::DepNodeParams;
pub use rustc_query_system::dep_graph::{
    debug, hash_result, DepContext, DepNodeColor, DepNodeIndex, SerializedDepNodeIndex,
    WorkProduct, WorkProductId,
};

pub use dep_node::{label_strs, DepConstructor, DepKind, DepNode, DepNodeExt};

pub type DepGraph = rustc_query_system::dep_graph::DepGraph<DepKind>;
pub type TaskDeps = rustc_query_system::dep_graph::TaskDeps<DepKind>;
pub type DepGraphQuery = rustc_query_system::dep_graph::DepGraphQuery<DepKind>;
pub type PreviousDepGraph = rustc_query_system::dep_graph::PreviousDepGraph<DepKind>;
pub type SerializedDepGraph = rustc_query_system::dep_graph::SerializedDepGraph<DepKind>;

impl rustc_query_system::dep_graph::DepKind for DepKind {
    const NULL: Self = DepKind::Null;

    fn is_eval_always(&self) -> bool {
        DepKind::is_eval_always(self)
    }

    fn has_params(&self) -> bool {
        DepKind::has_params(self)
    }

    fn debug_node(node: &DepNode, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", node.kind)?;

        if !node.kind.has_params() && !node.kind.is_anon() {
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

    fn can_reconstruct_query_key(&self) -> bool {
        DepKind::can_reconstruct_query_key(self)
    }
}

impl<'tcx> DepContext for TyCtxt<'tcx> {
    type DepKind = DepKind;
    type StableHashingContext = StableHashingContext<'tcx>;

    fn register_reused_dep_node(&self, dep_node: &DepNode) {
        if let Some(cache) = self.queries.on_disk_cache.as_ref() {
            cache.register_reused_dep_node(*self, dep_node)
        }
    }

    fn create_stable_hashing_context(&self) -> Self::StableHashingContext {
        TyCtxt::create_stable_hashing_context(*self)
    }

    fn debug_dep_tasks(&self) -> bool {
        self.sess.opts.debugging_opts.dep_tasks
    }
    fn debug_dep_node(&self) -> bool {
        self.sess.opts.debugging_opts.incremental_info
            || self.sess.opts.debugging_opts.query_dep_graph
    }

    fn try_force_from_dep_node(&self, dep_node: &DepNode) -> bool {
        // FIXME: This match is just a workaround for incremental bugs and should
        // be removed. https://github.com/rust-lang/rust/issues/62649 is one such
        // bug that must be fixed before removing this.
        match dep_node.kind {
            DepKind::hir_owner | DepKind::hir_owner_nodes | DepKind::CrateMetadata => {
                if let Some(def_id) = dep_node.extract_def_id(*self) {
                    if def_id_corresponds_to_hir_dep_node(*self, def_id.expect_local()) {
                        if dep_node.kind == DepKind::CrateMetadata {
                            // The `DefPath` has corresponding node,
                            // and that node should have been marked
                            // either red or green in `data.colors`.
                            bug!(
                                "DepNode {:?} should have been \
                             pre-marked as red or green but wasn't.",
                                dep_node
                            );
                        }
                    } else {
                        // This `DefPath` does not have a
                        // corresponding `DepNode` (e.g. a
                        // struct field), and the ` DefPath`
                        // collided with the `DefPath` of a
                        // proper item that existed in the
                        // previous compilation session.
                        //
                        // Since the given `DefPath` does not
                        // denote the item that previously
                        // existed, we just fail to mark green.
                        return false;
                    }
                } else {
                    // If the node does not exist anymore, we
                    // just fail to mark green.
                    return false;
                }
            }
            _ => {
                // For other kinds of nodes it's OK to be
                // forced.
            }
        }

        debug!("try_force_from_dep_node({:?}) --- trying to force", dep_node);
        ty::query::force_from_dep_node(*self, dep_node)
    }

    fn has_errors_or_delayed_span_bugs(&self) -> bool {
        self.sess.has_errors_or_delayed_span_bugs()
    }

    fn diagnostic(&self) -> &rustc_errors::Handler {
        self.sess.diagnostic()
    }

    // Interactions with on_disk_cache
    fn try_load_from_on_disk_cache(&self, dep_node: &DepNode) {
        try_load_from_on_disk_cache(*self, dep_node)
    }

    fn load_diagnostics(&self, prev_dep_node_index: SerializedDepNodeIndex) -> Vec<Diagnostic> {
        self.queries
            .on_disk_cache
            .as_ref()
            .map(|c| c.load_diagnostics(*self, prev_dep_node_index))
            .unwrap_or_default()
    }

    fn store_diagnostics(&self, dep_node_index: DepNodeIndex, diagnostics: ThinVec<Diagnostic>) {
        if let Some(c) = self.queries.on_disk_cache.as_ref() {
            c.store_diagnostics(dep_node_index, diagnostics)
        }
    }

    fn store_diagnostics_for_anon_node(
        &self,
        dep_node_index: DepNodeIndex,
        diagnostics: ThinVec<Diagnostic>,
    ) {
        if let Some(c) = self.queries.on_disk_cache.as_ref() {
            c.store_diagnostics_for_anon_node(dep_node_index, diagnostics)
        }
    }

    fn profiler(&self) -> &SelfProfilerRef {
        &self.prof
    }
}

fn def_id_corresponds_to_hir_dep_node(tcx: TyCtxt<'_>, def_id: LocalDefId) -> bool {
    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);
    def_id == hir_id.owner
}
