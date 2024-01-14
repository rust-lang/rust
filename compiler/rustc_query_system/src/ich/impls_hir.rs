//! This module contains `HashStable` implementations for various HIR data
//! types in no particular order.

use crate::ich::hcx::BodyResolver;
use crate::ich::StableHashingContext;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_hir as hir;

impl<'ctx> rustc_hir::HashStableContext for StableHashingContext<'ctx> {
    #[inline]
    fn hash_body_id(&mut self, hir::BodyId { hir_id }: hir::BodyId, hasher: &mut StableHasher) {
        let hcx = self;
        match hcx.body_resolver {
            BodyResolver::Forbidden => panic!("Hashing HIR bodies is forbidden."),
            BodyResolver::Ignore => hir_id.hash_stable(hcx, hasher),
        }
    }
}
