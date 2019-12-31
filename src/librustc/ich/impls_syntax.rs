//! This module contains `HashStable` implementations for various data types
//! from libsyntax in no particular order.

use crate::ich::StableHashingContext;

use syntax::ast;
use syntax_pos::SourceFile;

use crate::hir::def_id::{CrateNum, DefId, CRATE_DEF_INDEX};

use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use smallvec::SmallVec;

impl<'ctx> rustc_target::HashStableContext for StableHashingContext<'ctx> {}

impl<'a> HashStable<StableHashingContext<'a>> for ast::Lifetime {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        self.id.hash_stable(hcx, hasher);
        self.ident.hash_stable(hcx, hasher);
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for [ast::Attribute] {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        if self.len() == 0 {
            self.len().hash_stable(hcx, hasher);
            return;
        }

        // Some attributes are always ignored during hashing.
        let filtered: SmallVec<[&ast::Attribute; 8]> = self
            .iter()
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

impl<'a> HashStable<StableHashingContext<'a>> for ast::Attribute {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        // Make sure that these have been filtered out.
        debug_assert!(!self.ident().map_or(false, |ident| hcx.is_ignored_attr(ident.name)));
        debug_assert!(!self.is_doc_comment());

        let ast::Attribute { kind, id: _, style, span } = self;
        if let ast::AttrKind::Normal(item) = kind {
            item.hash_stable(hcx, hasher);
            style.hash_stable(hcx, hasher);
            span.hash_stable(hcx, hasher);
        } else {
            unreachable!();
        }
    }
}

impl<'ctx> syntax::HashStableContext for StableHashingContext<'ctx> {}

impl<'a> HashStable<StableHashingContext<'a>> for SourceFile {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        let SourceFile {
            name: _, // We hash the smaller name_hash instead of this
            name_hash,
            name_was_remapped,
            unmapped_path: _,
            crate_of_origin,
            // Do not hash the source as it is not encoded
            src: _,
            src_hash,
            external_src: _,
            start_pos,
            end_pos: _,
            ref lines,
            ref multibyte_chars,
            ref non_narrow_chars,
            ref normalized_pos,
        } = *self;

        (name_hash as u64).hash_stable(hcx, hasher);
        name_was_remapped.hash_stable(hcx, hasher);

        DefId { krate: CrateNum::from_u32(crate_of_origin), index: CRATE_DEF_INDEX }
            .hash_stable(hcx, hasher);

        src_hash.hash_stable(hcx, hasher);

        // We only hash the relative position within this source_file
        lines.len().hash_stable(hcx, hasher);
        for &line in lines.iter() {
            stable_byte_pos(line, start_pos).hash_stable(hcx, hasher);
        }

        // We only hash the relative position within this source_file
        multibyte_chars.len().hash_stable(hcx, hasher);
        for &char_pos in multibyte_chars.iter() {
            stable_multibyte_char(char_pos, start_pos).hash_stable(hcx, hasher);
        }

        non_narrow_chars.len().hash_stable(hcx, hasher);
        for &char_pos in non_narrow_chars.iter() {
            stable_non_narrow_char(char_pos, start_pos).hash_stable(hcx, hasher);
        }

        normalized_pos.len().hash_stable(hcx, hasher);
        for &char_pos in normalized_pos.iter() {
            stable_normalized_pos(char_pos, start_pos).hash_stable(hcx, hasher);
        }
    }
}

fn stable_byte_pos(pos: ::syntax_pos::BytePos, source_file_start: ::syntax_pos::BytePos) -> u32 {
    pos.0 - source_file_start.0
}

fn stable_multibyte_char(
    mbc: ::syntax_pos::MultiByteChar,
    source_file_start: ::syntax_pos::BytePos,
) -> (u32, u32) {
    let ::syntax_pos::MultiByteChar { pos, bytes } = mbc;

    (pos.0 - source_file_start.0, bytes as u32)
}

fn stable_non_narrow_char(
    swc: ::syntax_pos::NonNarrowChar,
    source_file_start: ::syntax_pos::BytePos,
) -> (u32, u32) {
    let pos = swc.pos();
    let width = swc.width();

    (pos.0 - source_file_start.0, width as u32)
}

fn stable_normalized_pos(
    np: ::syntax_pos::NormalizedPos,
    source_file_start: ::syntax_pos::BytePos,
) -> (u32, u32) {
    let ::syntax_pos::NormalizedPos { pos, diff } = np;

    (pos.0 - source_file_start.0, diff)
}

impl<'tcx> HashStable<StableHashingContext<'tcx>> for rustc_feature::Features {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'tcx>, hasher: &mut StableHasher) {
        // Unfortunately we cannot exhaustively list fields here, since the
        // struct is macro generated.
        self.declared_lang_features.hash_stable(hcx, hasher);
        self.declared_lib_features.hash_stable(hcx, hasher);

        self.walk_feature_fields(|feature_name, value| {
            feature_name.hash_stable(hcx, hasher);
            value.hash_stable(hcx, hasher);
        });
    }
}
