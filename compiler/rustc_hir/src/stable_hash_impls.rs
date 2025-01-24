use rustc_data_structures::stable_hasher::{HashStable, StableHasher, ToStableHashKey};
use rustc_span::def_id::DefPathHash;

use crate::hir::{
    Attribute, AttributeMap, BodyId, Crate, ForeignItemId, ImplItemId, ItemId, OwnerNodes,
    TraitItemId,
};
use crate::hir_id::{HirId, ItemLocalId};

/// Requirements for a `StableHashingContext` to be used in this crate.
/// This is a hack to allow using the `HashStable_Generic` derive macro
/// instead of implementing everything in `rustc_middle`.
pub trait HashStableContext:
    rustc_ast::HashStableContext + rustc_target::HashStableContext
{
    fn hash_attr(&mut self, _: &Attribute, hasher: &mut StableHasher);
}

impl<HirCtx: crate::HashStableContext> ToStableHashKey<HirCtx> for HirId {
    type KeyType = (DefPathHash, ItemLocalId);

    #[inline]
    fn to_stable_hash_key(&self, hcx: &HirCtx) -> (DefPathHash, ItemLocalId) {
        let def_path_hash = self.owner.def_id.to_stable_hash_key(hcx);
        (def_path_hash, self.local_id)
    }
}

impl<HirCtx: crate::HashStableContext> ToStableHashKey<HirCtx> for ItemLocalId {
    type KeyType = ItemLocalId;

    #[inline]
    fn to_stable_hash_key(&self, _: &HirCtx) -> ItemLocalId {
        *self
    }
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
    fn hash_stable(&self, hcx: &mut HirCtx, hasher: &mut StableHasher) {
        // We ignore the `nodes` and `bodies` fields since these refer to information included in
        // `hash` which is hashed in the collector and used for the crate hash.
        // `local_id_to_def_id` is also ignored because is dependent on the body, then just hashing
        // the body satisfies the condition of two nodes being different have different
        // `hash_stable` results.
        let OwnerNodes { opt_hash_including_bodies, nodes: _, bodies: _ } = *self;
        opt_hash_including_bodies.unwrap().hash_stable(hcx, hasher);
    }
}

impl<'tcx, HirCtx: crate::HashStableContext> HashStable<HirCtx> for AttributeMap<'tcx> {
    fn hash_stable(&self, hcx: &mut HirCtx, hasher: &mut StableHasher) {
        // We ignore the `map` since it refers to information included in `opt_hash` which is
        // hashed in the collector and used for the crate hash.
        let AttributeMap { opt_hash, map: _ } = *self;
        opt_hash.unwrap().hash_stable(hcx, hasher);
    }
}

impl<HirCtx: crate::HashStableContext> HashStable<HirCtx> for Crate<'_> {
    fn hash_stable(&self, hcx: &mut HirCtx, hasher: &mut StableHasher) {
        let Crate { owners: _, opt_hir_hash } = self;
        opt_hir_hash.unwrap().hash_stable(hcx, hasher)
    }
}

impl<HirCtx: crate::HashStableContext> HashStable<HirCtx> for Attribute {
    fn hash_stable(&self, hcx: &mut HirCtx, hasher: &mut StableHasher) {
        hcx.hash_attr(self, hasher)
    }
}
