//! This module contains `HashStable` implementations for various HIR data
//! types in no particular order.

use crate::ich::hcx::BodyResolver;
use crate::ich::StableHashingContext;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_hir as hir;

impl<'ctx> rustc_hir::HashStableContext for StableHashingContext<'ctx> {
    #[inline]
    fn hash_body_id(&mut self, id: hir::BodyId, hasher: &mut StableHasher) {
        let hcx = self;
        match hcx.body_resolver {
            BodyResolver::Forbidden => panic!("Hashing HIR bodies is forbidden."),
            BodyResolver::Traverse { hash_bodies: false, .. } => {}
            BodyResolver::Traverse { hash_bodies: true, owner, bodies } => {
                assert_eq!(id.hir_id.owner, owner);
                bodies[&id.hir_id.local_id].hash_stable(hcx, hasher);
            }
        }
    }

    fn hash_hir_expr(&mut self, expr: &hir::Expr<'_>, hasher: &mut StableHasher) {
        self.while_hashing_hir_bodies(true, |hcx| {
            let hir::Expr { hir_id, ref span, ref kind } = *expr;

            hir_id.hash_stable(hcx, hasher);
            span.hash_stable(hcx, hasher);
            kind.hash_stable(hcx, hasher);
        })
    }

    fn hash_hir_ty(&mut self, ty: &hir::Ty<'_>, hasher: &mut StableHasher) {
        self.while_hashing_hir_bodies(true, |hcx| {
            let hir::Ty { hir_id, ref kind, ref span } = *ty;

            hir_id.hash_stable(hcx, hasher);
            kind.hash_stable(hcx, hasher);
            span.hash_stable(hcx, hasher);
        })
    }
}
