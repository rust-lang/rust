use crate::ty::TyCtxt;
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_query_system::ich::StableHashingContext;
use rustc_session::Session;

#[macro_use]
mod dep_node;

pub use rustc_query_system::dep_graph::{
    debug::DepNodeFilter, debug::EdgeFilter, hash_result, label_strs, DepContext, DepGraph,
    DepGraphQuery, DepKind, DepNode, DepNodeColor, DepNodeIndex, SerializedDepGraph,
    SerializedDepNodeIndex, TaskDeps, WorkProduct, WorkProductId,
};

crate use dep_node::{fingerprint_style, make_compile_codegen_unit, make_compile_mono_item};
pub use dep_node::{DepKindStruct, DepNodeExt};

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

    #[inline(always)]
    fn fingerprint_style(&self, kind: DepKind) -> rustc_query_system::dep_graph::FingerprintStyle {
        fingerprint_style(*self, kind)
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
