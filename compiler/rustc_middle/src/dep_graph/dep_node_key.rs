use std::fmt::Debug;

use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_hir::def_id::{CrateNum, DefId, LOCAL_CRATE, LocalDefId, LocalModDefId, ModDefId};
use rustc_hir::definitions::DefPathHash;
use rustc_hir::{HirId, ItemLocalId, OwnerId};

use crate::dep_graph::{DepNode, KeyFingerprintStyle};
use crate::ich::StableHashingContext;
use crate::ty::TyCtxt;

/// Trait for query keys as seen by dependency-node tracking.
pub trait DepNodeKey<'tcx>: Debug + Sized {
    fn key_fingerprint_style() -> KeyFingerprintStyle;

    /// This method turns a query key into an opaque `Fingerprint` to be used
    /// in `DepNode`.
    fn to_fingerprint(&self, tcx: TyCtxt<'tcx>) -> Fingerprint;

    fn to_debug_str(&self, tcx: TyCtxt<'tcx>) -> String;

    /// This method tries to recover the query key from the given `DepNode`,
    /// something which is needed when forcing `DepNode`s during red-green
    /// evaluation. The query system will only call this method if
    /// `fingerprint_style()` is not `FingerprintStyle::Opaque`.
    /// It is always valid to return `None` here, in which case incremental
    /// compilation will treat the query as having changed instead of forcing it.
    fn try_recover_key(tcx: TyCtxt<'tcx>, dep_node: &DepNode) -> Option<Self>;
}

// Blanket impl of `DepNodeKey`, which is specialized by other impls elsewhere.
impl<'tcx, T> DepNodeKey<'tcx> for T
where
    T: for<'a> HashStable<StableHashingContext<'a>> + Debug,
{
    #[inline(always)]
    default fn key_fingerprint_style() -> KeyFingerprintStyle {
        KeyFingerprintStyle::Opaque
    }

    #[inline(always)]
    default fn to_fingerprint(&self, tcx: TyCtxt<'tcx>) -> Fingerprint {
        tcx.with_stable_hashing_context(|mut hcx| {
            let mut hasher = StableHasher::new();
            self.hash_stable(&mut hcx, &mut hasher);
            hasher.finish()
        })
    }

    #[inline(always)]
    default fn to_debug_str(&self, tcx: TyCtxt<'tcx>) -> String {
        // Make sure to print dep node params with reduced queries since printing
        // may themselves call queries, which may lead to (possibly untracked!)
        // query cycles.
        tcx.with_reduced_queries(|| format!("{self:?}"))
    }

    #[inline(always)]
    default fn try_recover_key(_: TyCtxt<'tcx>, _: &DepNode) -> Option<Self> {
        None
    }
}

impl<'tcx> DepNodeKey<'tcx> for () {
    #[inline(always)]
    fn key_fingerprint_style() -> KeyFingerprintStyle {
        KeyFingerprintStyle::Unit
    }

    #[inline(always)]
    fn to_fingerprint(&self, _: TyCtxt<'tcx>) -> Fingerprint {
        Fingerprint::ZERO
    }

    #[inline(always)]
    fn try_recover_key(_: TyCtxt<'tcx>, _: &DepNode) -> Option<Self> {
        Some(())
    }
}

impl<'tcx> DepNodeKey<'tcx> for DefId {
    #[inline(always)]
    fn key_fingerprint_style() -> KeyFingerprintStyle {
        KeyFingerprintStyle::DefPathHash
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
    fn try_recover_key(tcx: TyCtxt<'tcx>, dep_node: &DepNode) -> Option<Self> {
        dep_node.extract_def_id(tcx)
    }
}

impl<'tcx> DepNodeKey<'tcx> for LocalDefId {
    #[inline(always)]
    fn key_fingerprint_style() -> KeyFingerprintStyle {
        KeyFingerprintStyle::DefPathHash
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
    fn try_recover_key(tcx: TyCtxt<'tcx>, dep_node: &DepNode) -> Option<Self> {
        dep_node.extract_def_id(tcx).map(|id| id.expect_local())
    }
}

impl<'tcx> DepNodeKey<'tcx> for OwnerId {
    #[inline(always)]
    fn key_fingerprint_style() -> KeyFingerprintStyle {
        KeyFingerprintStyle::DefPathHash
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
    fn try_recover_key(tcx: TyCtxt<'tcx>, dep_node: &DepNode) -> Option<Self> {
        dep_node.extract_def_id(tcx).map(|id| OwnerId { def_id: id.expect_local() })
    }
}

impl<'tcx> DepNodeKey<'tcx> for CrateNum {
    #[inline(always)]
    fn key_fingerprint_style() -> KeyFingerprintStyle {
        KeyFingerprintStyle::DefPathHash
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
    fn try_recover_key(tcx: TyCtxt<'tcx>, dep_node: &DepNode) -> Option<Self> {
        dep_node.extract_def_id(tcx).map(|id| id.krate)
    }
}

impl<'tcx> DepNodeKey<'tcx> for (DefId, DefId) {
    #[inline(always)]
    fn key_fingerprint_style() -> KeyFingerprintStyle {
        KeyFingerprintStyle::Opaque
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
    fn key_fingerprint_style() -> KeyFingerprintStyle {
        KeyFingerprintStyle::HirId
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
    fn try_recover_key(tcx: TyCtxt<'tcx>, dep_node: &DepNode) -> Option<Self> {
        if tcx.key_fingerprint_style(dep_node.kind) == KeyFingerprintStyle::HirId {
            let (local_hash, local_id) = Fingerprint::from(dep_node.key_fingerprint).split();
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
    fn key_fingerprint_style() -> KeyFingerprintStyle {
        KeyFingerprintStyle::DefPathHash
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
    fn try_recover_key(tcx: TyCtxt<'tcx>, dep_node: &DepNode) -> Option<Self> {
        DefId::try_recover_key(tcx, dep_node).map(ModDefId::new_unchecked)
    }
}

impl<'tcx> DepNodeKey<'tcx> for LocalModDefId {
    #[inline(always)]
    fn key_fingerprint_style() -> KeyFingerprintStyle {
        KeyFingerprintStyle::DefPathHash
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
    fn try_recover_key(tcx: TyCtxt<'tcx>, dep_node: &DepNode) -> Option<Self> {
        LocalDefId::try_recover_key(tcx, dep_node).map(LocalModDefId::new_unchecked)
    }
}
