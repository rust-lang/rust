//! This module contains `HashStable` implementations for various data types
//! from various crates in no particular order.

use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_hir as hir;
use rustc_span::SourceFile;
use smallvec::SmallVec;

use crate::ich::StableHashingContext;

impl<'ctx> rustc_target::HashStableContext for StableHashingContext<'ctx> {}
impl<'ctx> rustc_ast::HashStableContext for StableHashingContext<'ctx> {}

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
                    && !attr.ident().is_some_and(|ident| hcx.is_ignored_attr(ident.name))
            })
            .collect();

        filtered.len().hash_stable(hcx, hasher);
        for attr in filtered {
            attr.hash_stable(hcx, hasher);
        }
    }
}

impl<'ctx> rustc_hir::HashStableContext for StableHashingContext<'ctx> {
    fn hash_attr(&mut self, attr: &hir::Attribute, hasher: &mut StableHasher) {
        // Make sure that these have been filtered out.
        debug_assert!(!attr.ident().is_some_and(|ident| self.is_ignored_attr(ident.name)));
        debug_assert!(!attr.is_doc_comment());

        let hir::Attribute { kind, id: _, style, span } = attr;
        if let hir::AttrKind::Normal(item) = kind {
            item.hash_stable(self, hasher);
            style.hash_stable(self, hasher);
            span.hash_stable(self, hasher);
        } else {
            unreachable!();
        }
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for SourceFile {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        let SourceFile {
            name: _, // We hash the smaller stable_id instead of this
            stable_id,
            cnum,
            // Do not hash the source as it is not encoded
            src: _,
            ref src_hash,
            // Already includes src_hash, this is redundant
            checksum_hash: _,
            external_src: _,
            start_pos: _,
            source_len: _,
            lines: _,
            ref multibyte_chars,
            ref normalized_pos,
        } = *self;

        stable_id.hash_stable(hcx, hasher);

        src_hash.hash_stable(hcx, hasher);

        {
            // We are always in `Lines` form by the time we reach here.
            assert!(self.lines.read().is_lines());
            let lines = self.lines();
            // We only hash the relative position within this source_file
            lines.len().hash_stable(hcx, hasher);
            for &line in lines.iter() {
                line.hash_stable(hcx, hasher);
            }
        }

        // We only hash the relative position within this source_file
        multibyte_chars.len().hash_stable(hcx, hasher);
        for &char_pos in multibyte_chars.iter() {
            char_pos.hash_stable(hcx, hasher);
        }

        normalized_pos.len().hash_stable(hcx, hasher);
        for &char_pos in normalized_pos.iter() {
            char_pos.hash_stable(hcx, hasher);
        }

        cnum.hash_stable(hcx, hasher);
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
