//! This module contains `HashStable` implementations for various HIR data
//! types in no particular order.

use crate::hir;
use crate::hir::map::DefPathHash;
use crate::hir::def_id::{DefId, LocalDefId, CrateNum, CRATE_DEF_INDEX};
use crate::ich::{StableHashingContext, NodeIdHashingMode, Fingerprint};
use rustc_data_structures::stable_hasher::{HashStable, ToStableHashKey,
                                           StableHasher, StableHasherResult};
use smallvec::SmallVec;
use std::mem;
use syntax::ast;
use syntax::attr;

impl<'a> HashStable<StableHashingContext<'a>> for DefId {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        hcx.def_path_hash(*self).hash_stable(hcx, hasher);
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
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
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
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        hcx.def_path_hash(DefId {
            krate: *self,
            index: CRATE_DEF_INDEX
        }).hash_stable(hcx, hasher);
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

impl<'a> ToStableHashKey<StableHashingContext<'a>>
for hir::ItemLocalId {
    type KeyType = hir::ItemLocalId;

    #[inline]
    fn to_stable_hash_key(&self,
                          _: &StableHashingContext<'a>)
                          -> hir::ItemLocalId {
        *self
    }
}

// The following implementations of HashStable for ItemId, TraitItemId, and
// ImplItemId deserve special attention. Normally we do not hash NodeIds within
// the HIR, since they just signify a HIR nodes own path. But ItemId et al
// are used when another item in the HIR is *referenced* and we certainly
// want to pick up on a reference changing its target, so we hash the NodeIds
// in "DefPath Mode".

impl<'a> HashStable<StableHashingContext<'a>> for hir::ItemId {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let hir::ItemId {
            id
        } = *self;

        hcx.with_node_id_hashing_mode(NodeIdHashingMode::HashDefPath, |hcx| {
            id.hash_stable(hcx, hasher);
        })
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for hir::TraitItemId {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let hir::TraitItemId {
            hir_id
        } = * self;

        hcx.with_node_id_hashing_mode(NodeIdHashingMode::HashDefPath, |hcx| {
            hir_id.hash_stable(hcx, hasher);
        })
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for hir::ImplItemId {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let hir::ImplItemId {
            hir_id
        } = * self;

        hcx.with_node_id_hashing_mode(NodeIdHashingMode::HashDefPath, |hcx| {
            hir_id.hash_stable(hcx, hasher);
        })
    }
}


impl_stable_hash_for!(struct ast::Label {
    ident
});

impl<'a> HashStable<StableHashingContext<'a>> for hir::Ty {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        hcx.while_hashing_hir_bodies(true, |hcx| {
            let hir::Ty {
                hir_id: _,
                ref node,
                ref span,
            } = *self;

            node.hash_stable(hcx, hasher);
            span.hash_stable(hcx, hasher);
        })
    }
}

impl_stable_hash_for_spanned!(hir::FieldPat);

impl_stable_hash_for_spanned!(hir::BinOpKind);

impl_stable_hash_for!(struct hir::Stmt {
    hir_id,
    node,
    span,
});


impl_stable_hash_for_spanned!(ast::Name);

impl<'a> HashStable<StableHashingContext<'a>> for hir::Expr {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        hcx.while_hashing_hir_bodies(true, |hcx| {
            let hir::Expr {
                hir_id: _,
                ref span,
                ref node,
                ref attrs
            } = *self;

            span.hash_stable(hcx, hasher);
            node.hash_stable(hcx, hasher);
            attrs.hash_stable(hcx, hasher);
        })
    }
}

impl_stable_hash_for_spanned!(usize);

impl_stable_hash_for_spanned!(ast::Ident);

impl_stable_hash_for!(struct ast::Ident {
    name,
    span,
});

impl<'a> HashStable<StableHashingContext<'a>> for hir::TraitItem {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let hir::TraitItem {
            hir_id: _,
            ident,
            ref attrs,
            ref generics,
            ref node,
            span
        } = *self;

        hcx.hash_hir_item_like(|hcx| {
            ident.name.hash_stable(hcx, hasher);
            attrs.hash_stable(hcx, hasher);
            generics.hash_stable(hcx, hasher);
            node.hash_stable(hcx, hasher);
            span.hash_stable(hcx, hasher);
        });
    }
}


impl<'a> HashStable<StableHashingContext<'a>> for hir::ImplItem {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let hir::ImplItem {
            hir_id: _,
            ident,
            ref vis,
            defaultness,
            ref attrs,
            ref generics,
            ref node,
            span
        } = *self;

        hcx.hash_hir_item_like(|hcx| {
            ident.name.hash_stable(hcx, hasher);
            vis.hash_stable(hcx, hasher);
            defaultness.hash_stable(hcx, hasher);
            attrs.hash_stable(hcx, hasher);
            generics.hash_stable(hcx, hasher);
            node.hash_stable(hcx, hasher);
            span.hash_stable(hcx, hasher);
        });
    }
}

impl_stable_hash_for!(enum ::syntax::ast::CrateSugar {
    JustCrate,
    PubCrate,
});

impl<'a> HashStable<StableHashingContext<'a>> for hir::VisibilityKind {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            hir::VisibilityKind::Public |
            hir::VisibilityKind::Inherited => {
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
}

impl_stable_hash_for_spanned!(hir::VisibilityKind);

impl<'a> HashStable<StableHashingContext<'a>> for hir::Mod {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let hir::Mod {
            inner: ref inner_span,
            ref item_ids,
        } = *self;

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
            }).fold(Fingerprint::ZERO, |a, b| {
                a.combine_commutative(b)
            });

        item_ids.len().hash_stable(hcx, hasher);
        item_ids_hash.hash_stable(hcx, hasher);
    }
}

impl_stable_hash_for_spanned!(hir::VariantKind);


impl<'a> HashStable<StableHashingContext<'a>> for hir::Item {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let hir::Item {
            ident,
            ref attrs,
            hir_id: _,
            ref node,
            ref vis,
            span
        } = *self;

        hcx.hash_hir_item_like(|hcx| {
            ident.name.hash_stable(hcx, hasher);
            attrs.hash_stable(hcx, hasher);
            node.hash_stable(hcx, hasher);
            vis.hash_stable(hcx, hasher);
            span.hash_stable(hcx, hasher);
        });
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for hir::Body {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let hir::Body {
            arguments,
            value,
            generator_kind,
        } = self;

        hcx.with_node_id_hashing_mode(NodeIdHashingMode::Ignore, |hcx| {
            arguments.hash_stable(hcx, hasher);
            value.hash_stable(hcx, hasher);
            generator_kind.hash_stable(hcx, hasher);
        });
    }
}

impl<'a> ToStableHashKey<StableHashingContext<'a>> for hir::BodyId {
    type KeyType = (DefPathHash, hir::ItemLocalId);

    #[inline]
    fn to_stable_hash_key(&self,
                          hcx: &StableHashingContext<'a>)
                          -> (DefPathHash, hir::ItemLocalId) {
        let hir::BodyId { hir_id } = *self;
        hir_id.to_stable_hash_key(hcx)
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for hir::def_id::DefIndex {

    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        hcx.local_def_path_hash(*self).hash_stable(hcx, hasher);
    }
}

impl<'a> ToStableHashKey<StableHashingContext<'a>>
for hir::def_id::DefIndex {
    type KeyType = DefPathHash;

    #[inline]
    fn to_stable_hash_key(&self, hcx: &StableHashingContext<'a>) -> DefPathHash {
         hcx.local_def_path_hash(*self)
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for crate::middle::lang_items::LangItem {
    fn hash_stable<W: StableHasherResult>(&self,
                                          _: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        ::std::hash::Hash::hash(self, hasher);
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for hir::TraitCandidate {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        hcx.with_node_id_hashing_mode(NodeIdHashingMode::HashDefPath, |hcx| {
            let hir::TraitCandidate {
                def_id,
                import_ids,
            } = self;

            def_id.hash_stable(hcx, hasher);
            import_ids.hash_stable(hcx, hasher);
        });
    }
}

impl<'a> ToStableHashKey<StableHashingContext<'a>> for hir::TraitCandidate {
    type KeyType = (DefPathHash, SmallVec<[(DefPathHash, hir::ItemLocalId); 1]>);

    fn to_stable_hash_key(&self,
                          hcx: &StableHashingContext<'a>)
                          -> Self::KeyType {
        let hir::TraitCandidate {
            def_id,
            import_ids,
        } = self;

        let import_keys = import_ids.iter().map(|node_id| hcx.node_to_hir_id(*node_id))
                                           .map(|hir_id| (hcx.local_def_path_hash(hir_id.owner),
                                                          hir_id.local_id)).collect();
        (hcx.def_path_hash(*def_id), import_keys)
    }
}

impl<'hir> HashStable<StableHashingContext<'hir>> for attr::InlineAttr {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'hir>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
    }
}

impl<'hir> HashStable<StableHashingContext<'hir>> for attr::OptimizeAttr {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'hir>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
    }
}
