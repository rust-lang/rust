use rustc_data_structures::stable_hasher::{HashStable, StableHasher};

use crate::hir::{BodyId, Expr, ImplItemId, ItemId, Mod, TraitItemId, Ty, VisibilityKind};
use crate::hir_id::HirId;

/// Requirements for a `StableHashingContext` to be used in this crate.
/// This is a hack to allow using the `HashStable_Generic` derive macro
/// instead of implementing everything in librustc.
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
