//! This module contains `HashStable` implementations for various data types
//! from various crates in no particular order.

use rustc_ast as ast;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};

use super::StableHashingContext;

impl<'a> HashStable<StableHashingContext<'a>> for ast::NodeId {
    #[inline]
    fn hash_stable(&self, _: &mut StableHashingContext<'a>, _: &mut StableHasher) {
        panic!("Node IDs should not appear in incremental state");
    }
}

impl<'tcx> HashStable<StableHashingContext<'tcx>> for rustc_feature::Features {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'tcx>, hasher: &mut StableHasher) {
        // Unfortunately we cannot exhaustively list fields here, since the
        // struct has private fields (to ensure its invariant is maintained)
        self.enabled_lang_features().hash_stable(hcx, hasher);
        self.enabled_lib_features().hash_stable(hcx, hasher);
    }
}

impl<'tcx> HashStable<StableHashingContext<'tcx>> for rustc_feature::EnabledLangFeature {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'tcx>, hasher: &mut StableHasher) {
        let rustc_feature::EnabledLangFeature { gate_name, attr_sp, stable_since } = self;
        gate_name.hash_stable(hcx, hasher);
        attr_sp.hash_stable(hcx, hasher);
        stable_since.hash_stable(hcx, hasher);
    }
}

impl<'tcx> HashStable<StableHashingContext<'tcx>> for rustc_feature::EnabledLibFeature {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'tcx>, hasher: &mut StableHasher) {
        let rustc_feature::EnabledLibFeature { gate_name, attr_sp } = self;
        gate_name.hash_stable(hcx, hasher);
        attr_sp.hash_stable(hcx, hasher);
    }
}
