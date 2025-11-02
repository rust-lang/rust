use rustc_data_structures::stable_hasher::{HashStable, StableHasher, ToStableHashKey};
use rustc_span::def_id::DefPathHash;

use crate::HashIgnoredAttrId;
use crate::hir::{
    AttributeMap, BodyId, ForeignItemId, ImplItemId, ItemId, OwnerInfo, OwnerNodes, TraitItemId,
};
use crate::hir_id::ItemLocalId;
use crate::lints::DelayedLints;

/// Requirements for a `StableHashingContext` to be used in this crate.
/// This is a hack to allow using the `HashStable_Generic` derive macro
/// instead of implementing everything in `rustc_middle`.
pub trait HashStableContext: rustc_ast::HashStableContext + rustc_abi::HashStableContext {
    fn hash_attr_id(&mut self, id: &HashIgnoredAttrId, hasher: &mut StableHasher);
}

impl<HirCtx: crate::HashStableContext> ToStableHashKey<HirCtx> for BodyId {
    type KeyType = (DefPathHash, ItemLocalId);

    #[inline]
    fn to_stable_hash_key(&self, hcx: &HirCtx) -> (DefPathHash, ItemLocalId) {
        let BodyId { hir_id } = *self;
        hir_id.to_stable_hash_key(hcx)
    }
}

impl<HirCtx: crate::HashStableContext> ToStableHashKey<HirCtx> for ItemId {
    type KeyType = DefPathHash;

    #[inline]
    fn to_stable_hash_key(&self, hcx: &HirCtx) -> DefPathHash {
        self.owner_id.def_id.to_stable_hash_key(hcx)
    }
}

impl<HirCtx: crate::HashStableContext> ToStableHashKey<HirCtx> for TraitItemId {
    type KeyType = DefPathHash;

    #[inline]
    fn to_stable_hash_key(&self, hcx: &HirCtx) -> DefPathHash {
        self.owner_id.def_id.to_stable_hash_key(hcx)
    }
}

impl<HirCtx: crate::HashStableContext> ToStableHashKey<HirCtx> for ImplItemId {
    type KeyType = DefPathHash;

    #[inline]
    fn to_stable_hash_key(&self, hcx: &HirCtx) -> DefPathHash {
        self.owner_id.def_id.to_stable_hash_key(hcx)
    }
}

impl<HirCtx: crate::HashStableContext> ToStableHashKey<HirCtx> for ForeignItemId {
    type KeyType = DefPathHash;

    #[inline]
    fn to_stable_hash_key(&self, hcx: &HirCtx) -> DefPathHash {
        self.owner_id.def_id.to_stable_hash_key(hcx)
    }
}

// The following implementations of HashStable for `ItemId`, `TraitItemId`, and
// `ImplItemId` deserve special attention. Normally we do not hash `NodeId`s within
// the HIR, since they just signify a HIR nodes own path. But `ItemId` et al
// are used when another item in the HIR is *referenced* and we certainly
// want to pick up on a reference changing its target, so we hash the NodeIds
// in "DefPath Mode".

impl<'tcx, HirCtx: crate::HashStableContext> HashStable<HirCtx> for OwnerNodes<'tcx> {
    #[inline]
    fn hash_stable(&self, hcx: &mut HirCtx, hasher: &mut StableHasher) {
        // We ignore the other fields since these refer to information included in
        // `opt_hash` which is hashed in the collector and used for the crate hash.
        let OwnerNodes { opt_hash, .. } = *self;
        opt_hash.unwrap().hash_stable(hcx, hasher);
    }
}

impl<HirCtx: crate::HashStableContext> HashStable<HirCtx> for DelayedLints {
    #[inline]
    fn hash_stable(&self, hcx: &mut HirCtx, hasher: &mut StableHasher) {
        let DelayedLints { opt_hash, .. } = *self;
        opt_hash.unwrap().hash_stable(hcx, hasher);
    }
}

impl<'tcx, HirCtx: crate::HashStableContext> HashStable<HirCtx> for AttributeMap<'tcx> {
    #[inline]
    fn hash_stable(&self, hcx: &mut HirCtx, hasher: &mut StableHasher) {
        // We ignore the `map` since it refers to information included in `opt_hash` which is
        // hashed in the collector and used for the crate hash.
        let AttributeMap { opt_hash, define_opaque: _, map: _ } = *self;
        opt_hash.unwrap().hash_stable(hcx, hasher);
    }
}

impl<'tcx, HirCtx: crate::HashStableContext> HashStable<HirCtx> for OwnerInfo<'tcx> {
    #[inline]
    fn hash_stable(&self, hcx: &mut HirCtx, hasher: &mut StableHasher) {
        // We ignore the rest since it refers to information included in `opt_hash` which is
        // hashed in the collector and used for the crate hash.
        let OwnerInfo { opt_hash, .. } = *self;
        opt_hash.unwrap().hash_stable(hcx, hasher);
    }
}

impl<HirCtx: crate::HashStableContext> HashStable<HirCtx> for HashIgnoredAttrId {
    fn hash_stable(&self, hcx: &mut HirCtx, hasher: &mut StableHasher) {
        hcx.hash_attr_id(self, hasher)
    }
}
