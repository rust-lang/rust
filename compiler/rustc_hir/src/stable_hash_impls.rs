use rustc_data_structures::stable_hasher::{HashStable, StableHasher, ToStableHashKey};

use crate::hir::{
    BodyId, Expr, ForeignItemId, ImplItem, ImplItemId, Item, ItemId, Mod, TraitItem, TraitItemId,
    Ty, VisibilityKind,
};
use crate::hir_id::{HirId, ItemLocalId};
use rustc_span::def_id::{DefPathHash, LocalDefId};

/// Requirements for a `StableHashingContext` to be used in this crate.
/// This is a hack to allow using the `HashStable_Generic` derive macro
/// instead of implementing everything in librustc_middle.
pub trait HashStableContext:
    rustc_ast::HashStableContext + rustc_target::HashStableContext
{
    fn hash_hir_id(&mut self, _: HirId, hasher: &mut StableHasher);
    fn hash_body_id(&mut self, _: BodyId, hasher: &mut StableHasher);
    fn hash_reference_to_item(&mut self, _: HirId, hasher: &mut StableHasher);
    fn hash_hir_mod(&mut self, _: &Mod<'_>, hasher: &mut StableHasher);
    fn hash_hir_expr(&mut self, _: &Expr<'_>, hasher: &mut StableHasher);
    fn hash_hir_ty(&mut self, _: &Ty<'_>, hasher: &mut StableHasher);
    fn hash_hir_visibility_kind(&mut self, _: &VisibilityKind<'_>, hasher: &mut StableHasher);
    fn hash_hir_item_like<F: FnOnce(&mut Self)>(&mut self, f: F);
    fn local_def_path_hash(&self, def_id: LocalDefId) -> DefPathHash;
}

impl<HirCtx: crate::HashStableContext> ToStableHashKey<HirCtx> for HirId {
    type KeyType = (DefPathHash, ItemLocalId);

    #[inline]
    fn to_stable_hash_key(&self, hcx: &HirCtx) -> (DefPathHash, ItemLocalId) {
        let def_path_hash = hcx.local_def_path_hash(self.owner);
        (def_path_hash, self.local_id)
    }
}

impl<HirCtx: crate::HashStableContext> ToStableHashKey<HirCtx> for TraitItemId {
    type KeyType = (DefPathHash, ItemLocalId);

    #[inline]
    fn to_stable_hash_key(&self, hcx: &HirCtx) -> (DefPathHash, ItemLocalId) {
        self.hir_id.to_stable_hash_key(hcx)
    }
}

impl<HirCtx: crate::HashStableContext> ToStableHashKey<HirCtx> for ImplItemId {
    type KeyType = (DefPathHash, ItemLocalId);

    #[inline]
    fn to_stable_hash_key(&self, hcx: &HirCtx) -> (DefPathHash, ItemLocalId) {
        self.hir_id.to_stable_hash_key(hcx)
    }
}

impl<HirCtx: crate::HashStableContext> ToStableHashKey<HirCtx> for ForeignItemId {
    type KeyType = (DefPathHash, ItemLocalId);

    #[inline]
    fn to_stable_hash_key(&self, hcx: &HirCtx) -> (DefPathHash, ItemLocalId) {
        self.hir_id.to_stable_hash_key(hcx)
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
        hcx.hash_reference_to_item(self.id, hasher)
    }
}

impl<HirCtx: crate::HashStableContext> HashStable<HirCtx> for ForeignItemId {
    fn hash_stable(&self, hcx: &mut HirCtx, hasher: &mut StableHasher) {
        hcx.hash_reference_to_item(self.hir_id, hasher)
    }
}

impl<HirCtx: crate::HashStableContext> HashStable<HirCtx> for ImplItemId {
    fn hash_stable(&self, hcx: &mut HirCtx, hasher: &mut StableHasher) {
        hcx.hash_reference_to_item(self.hir_id, hasher)
    }
}

impl<HirCtx: crate::HashStableContext> HashStable<HirCtx> for TraitItemId {
    fn hash_stable(&self, hcx: &mut HirCtx, hasher: &mut StableHasher) {
        hcx.hash_reference_to_item(self.hir_id, hasher)
    }
}

impl<HirCtx: crate::HashStableContext> HashStable<HirCtx> for Mod<'_> {
    fn hash_stable(&self, hcx: &mut HirCtx, hasher: &mut StableHasher) {
        hcx.hash_hir_mod(self, hasher)
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
        let TraitItem { hir_id: _, ident, ref attrs, ref generics, ref kind, span } = *self;

        hcx.hash_hir_item_like(|hcx| {
            ident.name.hash_stable(hcx, hasher);
            attrs.hash_stable(hcx, hasher);
            generics.hash_stable(hcx, hasher);
            kind.hash_stable(hcx, hasher);
            span.hash_stable(hcx, hasher);
        });
    }
}

impl<HirCtx: crate::HashStableContext> HashStable<HirCtx> for ImplItem<'_> {
    fn hash_stable(&self, hcx: &mut HirCtx, hasher: &mut StableHasher) {
        let ImplItem {
            hir_id: _,
            ident,
            ref vis,
            defaultness,
            ref attrs,
            ref generics,
            ref kind,
            span,
        } = *self;

        hcx.hash_hir_item_like(|hcx| {
            ident.name.hash_stable(hcx, hasher);
            vis.hash_stable(hcx, hasher);
            defaultness.hash_stable(hcx, hasher);
            attrs.hash_stable(hcx, hasher);
            generics.hash_stable(hcx, hasher);
            kind.hash_stable(hcx, hasher);
            span.hash_stable(hcx, hasher);
        });
    }
}

impl<HirCtx: crate::HashStableContext> HashStable<HirCtx> for Item<'_> {
    fn hash_stable(&self, hcx: &mut HirCtx, hasher: &mut StableHasher) {
        let Item { ident, ref attrs, hir_id: _, ref kind, ref vis, span } = *self;

        hcx.hash_hir_item_like(|hcx| {
            ident.name.hash_stable(hcx, hasher);
            attrs.hash_stable(hcx, hasher);
            kind.hash_stable(hcx, hasher);
            vis.hash_stable(hcx, hasher);
            span.hash_stable(hcx, hasher);
        });
    }
}
