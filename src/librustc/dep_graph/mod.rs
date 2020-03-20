use crate::hir::map::definitions::DefPathHash;
use crate::ich::StableHashingContext;
use crate::ty::{self, TyCtxt};
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_data_structures::sync::Lock;
use rustc_data_structures::thin_vec::ThinVec;
use rustc_errors::Diagnostic;
use rustc_hir::def_id::DefId;

mod dep_node;
mod safe;

pub(crate) use rustc_query_system::dep_graph::DepNodeParams;
pub use rustc_query_system::dep_graph::{
    debug, hash_result, DepContext, DepNodeColor, DepNodeIndex, SerializedDepNodeIndex,
    WorkProduct, WorkProductFileKind, WorkProductId,
};

pub use dep_node::{label_strs, DepConstructor, DepKind, DepNode, DepNodeExt};
pub use safe::AssertDepGraphSafe;
pub use safe::DepGraphSafe;

pub type DepGraph = rustc_query_system::dep_graph::DepGraph<DepKind>;
pub type TaskDeps = rustc_query_system::dep_graph::TaskDeps<DepKind>;
pub type DepGraphQuery = rustc_query_system::dep_graph::DepGraphQuery<DepKind>;
pub type PreviousDepGraph = rustc_query_system::dep_graph::PreviousDepGraph<DepKind>;
pub type SerializedDepGraph = rustc_query_system::dep_graph::SerializedDepGraph<DepKind>;

impl rustc_query_system::dep_graph::DepKind for DepKind {
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
                if let Some(def_id) = tcx.extract_def_id(node) {
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

    fn read_deps<OP>(op: OP) -> ()
    where
        OP: for<'a> FnOnce(Option<&'a Lock<TaskDeps>>) -> (),
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

    fn create_stable_hashing_context(&self) -> Self::StableHashingContext {
        TyCtxt::create_stable_hashing_context(*self)
    }

    /// Extracts the DefId corresponding to this DepNode. This will work
    /// if two conditions are met:
    ///
    /// 1. The Fingerprint of the DepNode actually is a DefPathHash, and
    /// 2. the item that the DefPath refers to exists in the current tcx.
    ///
    /// Condition (1) is determined by the DepKind variant of the
    /// DepNode. Condition (2) might not be fulfilled if a DepNode
    /// refers to something from the previous compilation session that
    /// has been removed.
    fn extract_def_id(&self, node: &DepNode) -> Option<DefId> {
        if node.kind.can_reconstruct_query_key() {
            let def_path_hash = DefPathHash(node.hash);
            self.def_path_hash_to_def_id.as_ref()?.get(&def_path_hash).cloned()
        } else {
            None
        }
    }

    fn try_force_previous_green(&self, dep_dep_node: &DepNode) -> bool {
        // FIXME: This match is just a workaround for incremental bugs and should
        // be removed. https://github.com/rust-lang/rust/issues/62649 is one such
        // bug that must be fixed before removing this.
        match dep_dep_node.kind {
            DepKind::hir_owner | DepKind::hir_owner_nodes | DepKind::CrateMetadata => {
                if let Some(def_id) = self.extract_def_id(dep_dep_node) {
                    if def_id_corresponds_to_hir_dep_node(*self, def_id) {
                        if dep_dep_node.kind == DepKind::CrateMetadata {
                            // The `DefPath` has corresponding node,
                            // and that node should have been marked
                            // either red or green in `data.colors`.
                            bug!(
                                "DepNode {:?} should have been \
                             pre-marked as red or green but wasn't.",
                                dep_dep_node
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

        debug!("try_force_previous_green({:?}) --- trying to force", dep_dep_node);
        ty::query::force_from_dep_node(*self, dep_dep_node)
    }

    fn has_errors_or_delayed_span_bugs(&self) -> bool {
        self.sess.has_errors_or_delayed_span_bugs()
    }

    fn diagnostic(&self) -> &rustc_errors::Handler {
        self.sess.diagnostic()
    }

    // Interactions with on_disk_cache
    fn try_load_from_on_disk_cache(&self, dep_node: &DepNode) {
        use crate::mir::interpret::GlobalId;
        use crate::ty::query::queries;
        use crate::ty::query::QueryDescription;
        rustc_dep_node_try_load_from_on_disk_cache!(dep_node, *self)
    }

    fn load_diagnostics(&self, prev_dep_node_index: SerializedDepNodeIndex) -> Vec<Diagnostic> {
        self.queries.on_disk_cache.load_diagnostics(*self, prev_dep_node_index)
    }

    fn store_diagnostics(&self, dep_node_index: DepNodeIndex, diagnostics: ThinVec<Diagnostic>) {
        self.queries.on_disk_cache.store_diagnostics(dep_node_index, diagnostics)
    }

    fn profiler(&self) -> &SelfProfilerRef {
        &self.prof
    }
}

fn def_id_corresponds_to_hir_dep_node(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    let hir_id = tcx.hir().as_local_hir_id(def_id).unwrap();
    def_id.index == hir_id.owner.local_def_index
}

impl rustc_query_system::HashStableContext for StableHashingContext<'_> {
    fn debug_dep_tasks(&self) -> bool {
        self.sess().opts.debugging_opts.dep_tasks
    }
}

impl rustc_query_system::HashStableContextProvider<StableHashingContext<'tcx>> for TyCtxt<'tcx> {
    fn get_stable_hashing_context(&self) -> StableHashingContext<'tcx> {
        self.create_stable_hashing_context()
    }
}

impl rustc_query_system::HashStableContextProvider<StableHashingContext<'a>>
    for StableHashingContext<'a>
{
    fn get_stable_hashing_context(&self) -> Self {
        self.clone()
    }
}
