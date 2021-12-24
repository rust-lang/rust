use rustc_data_structures::stable_hasher::{HashStable, StableHasher, ToStableHashKey};

use crate::hir::{
    AttributeMap, BodyId, Crate, Expr, ForeignItem, ForeignItemId, ImplItem, ImplItemId, Item,
    ItemId, OwnerNodes, TraitCandidate, TraitItem, TraitItemId, Ty, VisibilityKind,
};
use crate::hir_id::{HirId, ItemLocalId};
use rustc_span::def_id::DefPathHash;

/// Requirements for a `StableHashingContext` to be used in this crate.
/// This is a hack to allow using the `HashStable_Generic` derive macro
/// instead of implementing everything in `rustc_middle`.
pub trait HashStableContext:
    rustc_ast::HashStableContext + rustc_target::HashStableContext
{
    fn hash_hir_id(&mut self, _: HirId, hasher: &mut StableHasher);
    fn hash_body_id(&mut self, _: BodyId, hasher: &mut StableHasher);
    fn hash_reference_to_item(&mut self, _: HirId, hasher: &mut StableHasher);
    fn hash_hir_expr(&mut self, _: &Expr<'_>, hasher: &mut StableHasher);
    fn hash_hir_ty(&mut self, _: &Ty<'_>, hasher: &mut StableHasher);
    fn hash_hir_visibility_kind(&mut self, _: &VisibilityKind<'_>, hasher: &mut StableHasher);
    fn hash_hir_item_like<F: FnOnce(&mut Self)>(&mut self, f: F);
    fn hash_hir_trait_candidate(&mut self, _: &TraitCandidate, hasher: &mut StableHasher);
}

impl<HirCtx: crate::HashStableContext> ToStableHashKey<HirCtx> for HirId {
    type KeyType = (DefPathHash, ItemLocalId);

    #[inline]
    fn to_stable_hash_key(&self, hcx: &HirCtx) -> (DefPathHash, ItemLocalId) {
        let def_path_hash = self.owner.to_stable_hash_key(hcx);
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
        self.def_id.to_stable_hash_key(hcx)
    }
}

impl<HirCtx: crate::HashStableContext> ToStableHashKey<HirCtx> for TraitItemId {
    type KeyType = DefPathHash;

    #[inline]
    fn to_stable_hash_key(&self, hcx: &HirCtx) -> DefPathHash {
        self.def_id.to_stable_hash_key(hcx)
    }
}

impl<HirCtx: crate::HashStableContext> ToStableHashKey<HirCtx> for ImplItemId {
    type KeyType = DefPathHash;

    #[inline]
    fn to_stable_hash_key(&self, hcx: &HirCtx) -> DefPathHash {
        self.def_id.to_stable_hash_key(hcx)
    }
}

impl<HirCtx: crate::HashStableContext> ToStableHashKey<HirCtx> for ForeignItemId {
    type KeyType = DefPathHash;

    #[inline]
    fn to_stable_hash_key(&self, hcx: &HirCtx) -> DefPathHash {
        self.def_id.to_stable_hash_key(hcx)
    }
}

impl<HirCtx: crate::HashStableContext> HashStable<HirCtx> for HirId {
    fn hash_stable(&self, hcx: &mut HirCtx, hasher: &mut StableHasher) {
        hcx.hash_hir_id(*self, hasher)
    }
}

impl<HirCtx: crate::HashStableContext> HashStable<HirCtx> for BodyId {
    fn hash_stable(&self, hcx: &mut HirCtx, hasher: &mut StableHasher) {
        hcx.hash_body_id(*self, hasher)
    }
}

// The following implementations of HashStable for `ItemId`, `TraitItemId`, and
// `ImplItemId` deserve special attention. Normally we do not hash `NodeId`s within
// the HIR, since they just signify a HIR nodes own path. But `ItemId` et al
// are used when another item in the HIR is *referenced* and we certainly
// want to pick up on a reference changing its target, so we hash the NodeIds
// in "DefPath Mode".

impl<HirCtx: crate::HashStableContext> HashStable<HirCtx> for ItemId {
    fn hash_stable(&self, hcx: &mut HirCtx, hasher: &mut StableHasher) {
        hcx.hash_reference_to_item(self.hir_id(), hasher)
    }
}

impl<HirCtx: crate::HashStableContext> HashStable<HirCtx> for ForeignItemId {
    fn hash_stable(&self, hcx: &mut HirCtx, hasher: &mut StableHasher) {
        hcx.hash_reference_to_item(self.hir_id(), hasher)
    }
}

impl<HirCtx: crate::HashStableContext> HashStable<HirCtx> for ImplItemId {
    fn hash_stable(&self, hcx: &mut HirCtx, hasher: &mut StableHasher) {
        hcx.hash_reference_to_item(self.hir_id(), hasher)
    }
}

impl<HirCtx: crate::HashStableContext> HashStable<HirCtx> for TraitItemId {
    fn hash_stable(&self, hcx: &mut HirCtx, hasher: &mut StableHasher) {
        hcx.hash_reference_to_item(self.hir_id(), hasher)
    }
}

impl<HirCtx: crate::HashStableContext> HashStable<HirCtx> for Expr<'_> {
    fn hash_stable(&self, hcx: &mut HirCtx, hasher: &mut StableHasher) {
        hcx.hash_hir_expr(self, hasher)
    }
}

impl<HirCtx: crate::HashStableContext> HashStable<HirCtx> for Ty<'_> {
    fn hash_stable(&self, hcx: &mut HirCtx, hasher: &mut StableHasher) {
        hcx.hash_hir_ty(self, hasher)
    }
}

impl<HirCtx: crate::HashStableContext> HashStable<HirCtx> for VisibilityKind<'_> {
    fn hash_stable(&self, hcx: &mut HirCtx, hasher: &mut StableHasher) {
        hcx.hash_hir_visibility_kind(self, hasher)
    }
}

impl<HirCtx: crate::HashStableContext> HashStable<HirCtx> for TraitItem<'_> {
    fn hash_stable(&self, hcx: &mut HirCtx, hasher: &mut StableHasher) {
        let TraitItem { def_id: _, ident, ref generics, ref kind, span } = *self;

        hcx.hash_hir_item_like(|hcx| {
            ident.name.hash_stable(hcx, hasher);
            generics.hash_stable(hcx, hasher);
            kind.hash_stable(hcx, hasher);
            span.hash_stable(hcx, hasher);
        });
    }
}

impl<HirCtx: crate::HashStableContext> HashStable<HirCtx> for ImplItem<'_> {
    fn hash_stable(&self, hcx: &mut HirCtx, hasher: &mut StableHasher) {
        let ImplItem { def_id: _, ident, ref vis, defaultness, ref generics, ref kind, span } =
            *self;

        hcx.hash_hir_item_like(|hcx| {
            ident.name.hash_stable(hcx, hasher);
            vis.hash_stable(hcx, hasher);
            defaultness.hash_stable(hcx, hasher);
            generics.hash_stable(hcx, hasher);
            kind.hash_stable(hcx, hasher);
            span.hash_stable(hcx, hasher);
        });
    }
}

impl<HirCtx: crate::HashStableContext> HashStable<HirCtx> for ForeignItem<'_> {
    fn hash_stable(&self, hcx: &mut HirCtx, hasher: &mut StableHasher) {
        let ForeignItem { def_id: _, ident, ref kind, span, ref vis } = *self;

        hcx.hash_hir_item_like(|hcx| {
            ident.name.hash_stable(hcx, hasher);
            kind.hash_stable(hcx, hasher);
            span.hash_stable(hcx, hasher);
            vis.hash_stable(hcx, hasher);
        });
    }
}

impl<HirCtx: crate::HashStableContext> HashStable<HirCtx> for Item<'_> {
    fn hash_stable(&self, hcx: &mut HirCtx, hasher: &mut StableHasher) {
        let Item { ident, def_id: _, ref kind, ref vis, span } = *self;

        hcx.hash_hir_item_like(|hcx| {
            ident.name.hash_stable(hcx, hasher);
            kind.hash_stable(hcx, hasher);
            vis.hash_stable(hcx, hasher);
            span.hash_stable(hcx, hasher);
        });
    }
}

impl<'tcx, HirCtx: crate::HashStableContext> HashStable<HirCtx> for OwnerNodes<'tcx> {
    fn hash_stable(&self, hcx: &mut HirCtx, hasher: &mut StableHasher) {
        // We ignore the `nodes` and `bodies` fields since these refer to information included in
        // `hash` which is hashed in the collector and used for the crate hash.
        let OwnerNodes { hash_including_bodies, hash_without_bodies: _, nodes: _, bodies: _ } =
            *self;
        hash_including_bodies.hash_stable(hcx, hasher);
    }
}

impl<'tcx, HirCtx: crate::HashStableContext> HashStable<HirCtx> for AttributeMap<'tcx> {
    fn hash_stable(&self, hcx: &mut HirCtx, hasher: &mut StableHasher) {
        // We ignore the `map` since it refers to information included in `hash` which is hashed in
        // the collector and used for the crate hash.
        let AttributeMap { hash, map: _ } = *self;
        hash.hash_stable(hcx, hasher);
    }
}

impl<HirCtx: crate::HashStableContext> HashStable<HirCtx> for Crate<'_> {
    fn hash_stable(&self, hcx: &mut HirCtx, hasher: &mut StableHasher) {
        let Crate { owners: _, hir_hash } = self;
        hir_hash.hash_stable(hcx, hasher)
    }
}

impl<HirCtx: crate::HashStableContext> HashStable<HirCtx> for TraitCandidate {
    fn hash_stable(&self, hcx: &mut HirCtx, hasher: &mut StableHasher) {
        hcx.hash_hir_trait_candidate(self, hasher)
    }
}
