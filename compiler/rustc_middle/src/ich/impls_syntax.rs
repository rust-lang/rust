//! This module contains `HashStable` implementations for various data types
//! from various crates in no particular order.

use rustc_ast as ast;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_hir as hir;
use rustc_span::{Symbol, sym};
use smallvec::SmallVec;

use super::StableHashingContext;

impl<'a> HashStable<StableHashingContext<'a>> for ast::NodeId {
    #[inline]
    fn hash_stable(&self, _: &mut StableHashingContext<'a>, _: &mut StableHasher) {
        panic!("Node IDs should not appear in incremental state");
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for [hir::Attribute] {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        if self.is_empty() {
            self.len().hash_stable(hcx, hasher);
            return;
        }

        // Some attributes are always ignored during hashing.
        let filtered: SmallVec<[&hir::Attribute; 8]> = self
            .iter()
            .filter(|attr| {
                !attr.is_doc_comment()
                    // FIXME(jdonszelmann) have a better way to handle ignored attrs
                    && !attr.name().is_some_and(|ident| is_ignored_attr(ident))
            })
            .collect();

        filtered.len().hash_stable(hcx, hasher);
        for attr in filtered {
            attr.hash_stable(hcx, hasher);
        }
    }
}

#[inline]
fn is_ignored_attr(name: Symbol) -> bool {
    const IGNORED_ATTRIBUTES: &[Symbol] = &[
        sym::cfg_trace, // FIXME(#138844) should this really be ignored?
        sym::rustc_if_this_changed,
        sym::rustc_then_this_would_need,
        sym::rustc_clean,
        sym::rustc_partition_reused,
        sym::rustc_partition_codegened,
        sym::rustc_expected_cgu_reuse,
    ];
    IGNORED_ATTRIBUTES.contains(&name)
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
