use rustc_data_structures::fingerprint::Fingerprint;
use rustc_hir::def_id::{CrateNum, DefId, LOCAL_CRATE, LocalDefId, LocalModDefId, ModDefId};
use rustc_hir::definitions::DefPathHash;
use rustc_hir::{HirId, ItemLocalId, OwnerId};

use crate::dep_graph::{DepNode, DepNodeKey, FingerprintStyle};
use crate::ty::TyCtxt;

impl<'tcx> DepNodeKey<'tcx> for () {
    #[inline(always)]
    fn fingerprint_style() -> FingerprintStyle {
        FingerprintStyle::Unit
    }

    #[inline(always)]
    fn to_fingerprint(&self, _: TyCtxt<'tcx>) -> Fingerprint {
        Fingerprint::ZERO
    }

    #[inline(always)]
    fn recover(_: TyCtxt<'tcx>, _: &DepNode) -> Option<Self> {
        Some(())
    }
}

impl<'tcx> DepNodeKey<'tcx> for DefId {
    #[inline(always)]
    fn fingerprint_style() -> FingerprintStyle {
        FingerprintStyle::DefPathHash
    }

    #[inline(always)]
    fn to_fingerprint(&self, tcx: TyCtxt<'tcx>) -> Fingerprint {
        tcx.def_path_hash(*self).0
    }

    #[inline(always)]
    fn to_debug_str(&self, tcx: TyCtxt<'tcx>) -> String {
        tcx.def_path_str(*self)
    }

    #[inline(always)]
    fn recover(tcx: TyCtxt<'tcx>, dep_node: &DepNode) -> Option<Self> {
        dep_node.extract_def_id(tcx)
    }
}

impl<'tcx> DepNodeKey<'tcx> for LocalDefId {
    #[inline(always)]
    fn fingerprint_style() -> FingerprintStyle {
        FingerprintStyle::DefPathHash
    }

    #[inline(always)]
    fn to_fingerprint(&self, tcx: TyCtxt<'tcx>) -> Fingerprint {
        self.to_def_id().to_fingerprint(tcx)
    }

    #[inline(always)]
    fn to_debug_str(&self, tcx: TyCtxt<'tcx>) -> String {
        self.to_def_id().to_debug_str(tcx)
    }

    #[inline(always)]
    fn recover(tcx: TyCtxt<'tcx>, dep_node: &DepNode) -> Option<Self> {
        dep_node.extract_def_id(tcx).map(|id| id.expect_local())
    }
}

impl<'tcx> DepNodeKey<'tcx> for OwnerId {
    #[inline(always)]
    fn fingerprint_style() -> FingerprintStyle {
        FingerprintStyle::DefPathHash
    }

    #[inline(always)]
    fn to_fingerprint(&self, tcx: TyCtxt<'tcx>) -> Fingerprint {
        self.to_def_id().to_fingerprint(tcx)
    }

    #[inline(always)]
    fn to_debug_str(&self, tcx: TyCtxt<'tcx>) -> String {
        self.to_def_id().to_debug_str(tcx)
    }

    #[inline(always)]
    fn recover(tcx: TyCtxt<'tcx>, dep_node: &DepNode) -> Option<Self> {
        dep_node.extract_def_id(tcx).map(|id| OwnerId { def_id: id.expect_local() })
    }
}

impl<'tcx> DepNodeKey<'tcx> for CrateNum {
    #[inline(always)]
    fn fingerprint_style() -> FingerprintStyle {
        FingerprintStyle::DefPathHash
    }

    #[inline(always)]
    fn to_fingerprint(&self, tcx: TyCtxt<'tcx>) -> Fingerprint {
        let def_id = self.as_def_id();
        def_id.to_fingerprint(tcx)
    }

    #[inline(always)]
    fn to_debug_str(&self, tcx: TyCtxt<'tcx>) -> String {
        tcx.crate_name(*self).to_string()
    }

    #[inline(always)]
    fn recover(tcx: TyCtxt<'tcx>, dep_node: &DepNode) -> Option<Self> {
        dep_node.extract_def_id(tcx).map(|id| id.krate)
    }
}

impl<'tcx> DepNodeKey<'tcx> for (DefId, DefId) {
    #[inline(always)]
    fn fingerprint_style() -> FingerprintStyle {
        FingerprintStyle::Opaque
    }

    // We actually would not need to specialize the implementation of this
    // method but it's faster to combine the hashes than to instantiate a full
    // hashing context and stable-hashing state.
    #[inline(always)]
    fn to_fingerprint(&self, tcx: TyCtxt<'tcx>) -> Fingerprint {
        let (def_id_0, def_id_1) = *self;

        let def_path_hash_0 = tcx.def_path_hash(def_id_0);
        let def_path_hash_1 = tcx.def_path_hash(def_id_1);

        def_path_hash_0.0.combine(def_path_hash_1.0)
    }

    #[inline(always)]
    fn to_debug_str(&self, tcx: TyCtxt<'tcx>) -> String {
        let (def_id_0, def_id_1) = *self;

        format!("({}, {})", tcx.def_path_debug_str(def_id_0), tcx.def_path_debug_str(def_id_1))
    }
}

impl<'tcx> DepNodeKey<'tcx> for HirId {
    #[inline(always)]
    fn fingerprint_style() -> FingerprintStyle {
        FingerprintStyle::HirId
    }

    // We actually would not need to specialize the implementation of this
    // method but it's faster to combine the hashes than to instantiate a full
    // hashing context and stable-hashing state.
    #[inline(always)]
    fn to_fingerprint(&self, tcx: TyCtxt<'tcx>) -> Fingerprint {
        let HirId { owner, local_id } = *self;
        let def_path_hash = tcx.def_path_hash(owner.to_def_id());
        Fingerprint::new(
            // `owner` is local, so is completely defined by the local hash
            def_path_hash.local_hash(),
            local_id.as_u32() as u64,
        )
    }

    #[inline(always)]
    fn to_debug_str(&self, tcx: TyCtxt<'tcx>) -> String {
        let HirId { owner, local_id } = *self;
        format!("{}.{}", tcx.def_path_str(owner), local_id.as_u32())
    }

    #[inline(always)]
    fn recover(tcx: TyCtxt<'tcx>, dep_node: &DepNode) -> Option<Self> {
        if tcx.fingerprint_style(dep_node.kind) == FingerprintStyle::HirId {
            let (local_hash, local_id) = Fingerprint::from(dep_node.hash).split();
            let def_path_hash = DefPathHash::new(tcx.stable_crate_id(LOCAL_CRATE), local_hash);
            let def_id = tcx.def_path_hash_to_def_id(def_path_hash)?.expect_local();
            let local_id = local_id
                .as_u64()
                .try_into()
                .unwrap_or_else(|_| panic!("local id should be u32, found {local_id:?}"));
            Some(HirId { owner: OwnerId { def_id }, local_id: ItemLocalId::from_u32(local_id) })
        } else {
            None
        }
    }
}

impl<'tcx> DepNodeKey<'tcx> for ModDefId {
    #[inline(always)]
    fn fingerprint_style() -> FingerprintStyle {
        FingerprintStyle::DefPathHash
    }

    #[inline(always)]
    fn to_fingerprint(&self, tcx: TyCtxt<'tcx>) -> Fingerprint {
        self.to_def_id().to_fingerprint(tcx)
    }

    #[inline(always)]
    fn to_debug_str(&self, tcx: TyCtxt<'tcx>) -> String {
        self.to_def_id().to_debug_str(tcx)
    }

    #[inline(always)]
    fn recover(tcx: TyCtxt<'tcx>, dep_node: &DepNode) -> Option<Self> {
        DefId::recover(tcx, dep_node).map(ModDefId::new_unchecked)
    }
}

impl<'tcx> DepNodeKey<'tcx> for LocalModDefId {
    #[inline(always)]
    fn fingerprint_style() -> FingerprintStyle {
        FingerprintStyle::DefPathHash
    }

    #[inline(always)]
    fn to_fingerprint(&self, tcx: TyCtxt<'tcx>) -> Fingerprint {
        self.to_def_id().to_fingerprint(tcx)
    }

    #[inline(always)]
    fn to_debug_str(&self, tcx: TyCtxt<'tcx>) -> String {
        self.to_def_id().to_debug_str(tcx)
    }

    #[inline(always)]
    fn recover(tcx: TyCtxt<'tcx>, dep_node: &DepNode) -> Option<Self> {
        LocalDefId::recover(tcx, dep_node).map(LocalModDefId::new_unchecked)
    }
}
