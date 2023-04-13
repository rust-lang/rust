//! This module contains `HashStable` implementations for various HIR data
//! types in no particular order.

use crate::ich::hcx::BodyResolver;
use crate::ich::StableHashingContext;
use rustc_ast::ast;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_hir as hir;
use smallvec::SmallVec;

impl<'ctx> rustc_hir::HashStableContext for StableHashingContext<'ctx> {
    #[inline]
    fn hash_body_id(&mut self, id: hir::BodyId, hasher: &mut StableHasher) {
        let hcx = self;
        match hcx.body_resolver {
            BodyResolver::Forbidden => panic!("Hashing HIR bodies is forbidden."),
            BodyResolver::Ignore => {}
            BodyResolver::Traverse { owner, bodies } => {
                assert_eq!(id.hir_id.owner, owner);
                bodies[&id.hir_id.local_id].hash_stable(hcx, hasher);
            }
        }
    }
}

impl<'a, 'b> HashStable<StableHashingContext<'a>> for hir::ItemAttributes<'b> {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        if self.is_empty() {
            self.len().hash_stable(hcx, hasher);
            return;
        }

        // Some attributes are always ignored during hashing.
        let filtered: SmallVec<[&ast::Attribute; 8]> = self
            .values()
            .filter(|attr| {
                !attr.is_doc_comment()
                    && !attr.ident().map_or(false, |ident| hcx.is_ignored_attr(ident.name))
            })
            .collect();

        filtered.len().hash_stable(hcx, hasher);
        for attr in filtered {
            attr.hash_stable(hcx, hasher);
        }
    }
}
