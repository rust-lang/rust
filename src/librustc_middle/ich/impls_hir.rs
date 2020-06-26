//! This module contains `HashStable` implementations for various HIR data
//! types in no particular order.

use crate::ich::{NodeIdHashingMode, StableHashingContext};
use rustc_attr as attr;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher, ToStableHashKey};
use rustc_hir as hir;
use rustc_hir::def_id::{CrateNum, DefId, LocalDefId, CRATE_DEF_INDEX};
use rustc_hir::definitions::DefPathHash;
use smallvec::SmallVec;
use std::mem;

impl<'ctx> rustc_hir::HashStableContext for StableHashingContext<'ctx> {
    #[inline]
    fn hash_hir_id(&mut self, hir_id: hir::HirId, hasher: &mut StableHasher) {
        let hcx = self;
        match hcx.node_id_hashing_mode {
            NodeIdHashingMode::Ignore => {
                // Don't do anything.
            }
            NodeIdHashingMode::HashDefPath => {
                let hir::HirId { owner, local_id } = hir_id;

                hcx.local_def_path_hash(owner).hash_stable(hcx, hasher);
                local_id.hash_stable(hcx, hasher);
            }
        }
    }

    fn hash_body_id(&mut self, id: hir::BodyId, hasher: &mut StableHasher) {
        let hcx = self;
        if hcx.hash_bodies() {
            hcx.body_resolver.body(id).hash_stable(hcx, hasher);
        }
    }

    fn hash_reference_to_item(&mut self, id: hir::HirId, hasher: &mut StableHasher) {
        let hcx = self;

        hcx.with_node_id_hashing_mode(NodeIdHashingMode::HashDefPath, |hcx| {
            id.hash_stable(hcx, hasher);
        })
    }

    fn hash_hir_mod(&mut self, module: &hir::Mod<'_>, hasher: &mut StableHasher) {
        let hcx = self;
        let hir::Mod { inner: ref inner_span, ref item_ids } = *module;

        inner_span.hash_stable(hcx, hasher);

        // Combining the `DefPathHash`s directly is faster than feeding them
        // into the hasher. Because we use a commutative combine, we also don't
        // have to sort the array.
        let item_ids_hash = item_ids
            .iter()
            .map(|id| {
                let (def_path_hash, local_id) = id.id.to_stable_hash_key(hcx);
                debug_assert_eq!(local_id, hir::ItemLocalId::from_u32(0));
                def_path_hash.0
            })
            .fold(Fingerprint::ZERO, |a, b| a.combine_commutative(b));

        item_ids.len().hash_stable(hcx, hasher);
        item_ids_hash.hash_stable(hcx, hasher);
    }

    fn hash_hir_expr(&mut self, expr: &hir::Expr<'_>, hasher: &mut StableHasher) {
        self.while_hashing_hir_bodies(true, |hcx| {
            let hir::Expr { hir_id: _, ref span, ref kind, ref attrs } = *expr;

            span.hash_stable(hcx, hasher);
            kind.hash_stable(hcx, hasher);
            attrs.hash_stable(hcx, hasher);
        })
    }

    fn hash_hir_ty(&mut self, ty: &hir::Ty<'_>, hasher: &mut StableHasher) {
        self.while_hashing_hir_bodies(true, |hcx| {
            let hir::Ty { hir_id: _, ref kind, ref span } = *ty;

            kind.hash_stable(hcx, hasher);
            span.hash_stable(hcx, hasher);
        })
    }

    fn hash_hir_visibility_kind(
        &mut self,
        vis: &hir::VisibilityKind<'_>,
        hasher: &mut StableHasher,
    ) {
        let hcx = self;
        mem::discriminant(vis).hash_stable(hcx, hasher);
        match *vis {
            hir::VisibilityKind::Public | hir::VisibilityKind::Inherited => {
                // No fields to hash.
            }
            hir::VisibilityKind::Crate(sugar) => {
                sugar.hash_stable(hcx, hasher);
            }
            hir::VisibilityKind::Restricted { ref path, hir_id } => {
                hcx.with_node_id_hashing_mode(NodeIdHashingMode::HashDefPath, |hcx| {
                    hir_id.hash_stable(hcx, hasher);
                });
                path.hash_stable(hcx, hasher);
            }
        }
    }

    fn hash_hir_item_like<F: FnOnce(&mut Self)>(&mut self, f: F) {
        let prev_hash_node_ids = self.node_id_hashing_mode;
        self.node_id_hashing_mode = NodeIdHashingMode::Ignore;

        f(self);

        self.node_id_hashing_mode = prev_hash_node_ids;
    }

    #[inline]
    fn local_def_path_hash(&self, def_id: LocalDefId) -> DefPathHash {
        self.local_def_path_hash(def_id)
    }
}

impl<'a> ToStableHashKey<StableHashingContext<'a>> for DefId {
    type KeyType = DefPathHash;

    #[inline]
    fn to_stable_hash_key(&self, hcx: &StableHashingContext<'a>) -> DefPathHash {
        hcx.def_path_hash(*self)
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for LocalDefId {
    #[inline]
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        hcx.def_path_hash(self.to_def_id()).hash_stable(hcx, hasher);
    }
}

impl<'a> ToStableHashKey<StableHashingContext<'a>> for LocalDefId {
    type KeyType = DefPathHash;

    #[inline]
    fn to_stable_hash_key(&self, hcx: &StableHashingContext<'a>) -> DefPathHash {
        hcx.def_path_hash(self.to_def_id())
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for CrateNum {
    #[inline]
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        hcx.def_path_hash(DefId { krate: *self, index: CRATE_DEF_INDEX }).hash_stable(hcx, hasher);
    }
}

impl<'a> ToStableHashKey<StableHashingContext<'a>> for CrateNum {
    type KeyType = DefPathHash;

    #[inline]
    fn to_stable_hash_key(&self, hcx: &StableHashingContext<'a>) -> DefPathHash {
        let def_id = DefId { krate: *self, index: CRATE_DEF_INDEX };
        def_id.to_stable_hash_key(hcx)
    }
}

impl<'a> ToStableHashKey<StableHashingContext<'a>> for hir::ItemLocalId {
    type KeyType = hir::ItemLocalId;

    #[inline]
    fn to_stable_hash_key(&self, _: &StableHashingContext<'a>) -> hir::ItemLocalId {
        *self
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for hir::Body<'_> {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        let hir::Body { params, value, generator_kind } = self;

        hcx.with_node_id_hashing_mode(NodeIdHashingMode::Ignore, |hcx| {
            params.hash_stable(hcx, hasher);
            value.hash_stable(hcx, hasher);
            generator_kind.hash_stable(hcx, hasher);
        });
    }
}

impl<'a> ToStableHashKey<StableHashingContext<'a>> for hir::BodyId {
    type KeyType = (DefPathHash, hir::ItemLocalId);

    #[inline]
    fn to_stable_hash_key(
        &self,
        hcx: &StableHashingContext<'a>,
    ) -> (DefPathHash, hir::ItemLocalId) {
        let hir::BodyId { hir_id } = *self;
        hir_id.to_stable_hash_key(hcx)
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for hir::TraitCandidate {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        hcx.with_node_id_hashing_mode(NodeIdHashingMode::HashDefPath, |hcx| {
            let hir::TraitCandidate { def_id, import_ids } = self;

            def_id.hash_stable(hcx, hasher);
            import_ids.hash_stable(hcx, hasher);
        });
    }
}

impl<'a> ToStableHashKey<StableHashingContext<'a>> for hir::TraitCandidate {
    type KeyType = (DefPathHash, SmallVec<[DefPathHash; 1]>);

    fn to_stable_hash_key(&self, hcx: &StableHashingContext<'a>) -> Self::KeyType {
        let hir::TraitCandidate { def_id, import_ids } = self;

        (
            hcx.def_path_hash(*def_id),
            import_ids.iter().map(|def_id| hcx.local_def_path_hash(*def_id)).collect(),
        )
    }
}

impl<'hir> HashStable<StableHashingContext<'hir>> for attr::InlineAttr {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'hir>, hasher: &mut StableHasher) {
        mem::discriminant(self).hash_stable(hcx, hasher);
    }
}

impl<'hir> HashStable<StableHashingContext<'hir>> for attr::OptimizeAttr {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'hir>, hasher: &mut StableHasher) {
        mem::discriminant(self).hash_stable(hcx, hasher);
    }
}
