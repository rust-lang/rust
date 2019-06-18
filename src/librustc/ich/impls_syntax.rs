//! This module contains `HashStable` implementations for various data types
//! from libsyntax in no particular order.

use crate::ich::StableHashingContext;

use std::hash as std_hash;
use std::mem;

use syntax::ast;
use syntax::feature_gate;
use syntax::parse::token;
use syntax::symbol::{InternedString, LocalInternedString};
use syntax::tokenstream;
use syntax_pos::SourceFile;

use crate::hir::def_id::{DefId, CrateNum, CRATE_DEF_INDEX};

use smallvec::SmallVec;
use rustc_data_structures::stable_hasher::{HashStable, ToStableHashKey,
                                           StableHasher, StableHasherResult};

impl<'a> HashStable<StableHashingContext<'a>> for InternedString {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        self.with(|s| s.hash_stable(hcx, hasher))
    }
}

impl<'a> ToStableHashKey<StableHashingContext<'a>> for InternedString {
    type KeyType = InternedString;

    #[inline]
    fn to_stable_hash_key(&self,
                          _: &StableHashingContext<'a>)
                          -> InternedString {
        self.clone()
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for LocalInternedString {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let s: &str = &**self;
        s.hash_stable(hcx, hasher);
    }
}

impl<'a> ToStableHashKey<StableHashingContext<'a>> for LocalInternedString {
    type KeyType = LocalInternedString;

    #[inline]
    fn to_stable_hash_key(&self,
                          _: &StableHashingContext<'a>)
                          -> LocalInternedString {
        self.clone()
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for ast::Name {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        self.as_str().hash_stable(hcx, hasher);
    }
}

impl<'a> ToStableHashKey<StableHashingContext<'a>> for ast::Name {
    type KeyType = InternedString;

    #[inline]
    fn to_stable_hash_key(&self,
                          _: &StableHashingContext<'a>)
                          -> InternedString {
        self.as_interned_str()
    }
}

impl_stable_hash_for!(enum ::syntax::ast::AsmDialect {
    Att,
    Intel
});

impl_stable_hash_for!(enum ::syntax::ext::base::MacroKind {
    Bang,
    Attr,
    Derive,
    ProcMacroStub,
});


impl_stable_hash_for!(enum ::rustc_target::spec::abi::Abi {
    Cdecl,
    Stdcall,
    Fastcall,
    Vectorcall,
    Thiscall,
    Aapcs,
    Win64,
    SysV64,
    PtxKernel,
    Msp430Interrupt,
    X86Interrupt,
    AmdGpuKernel,
    Rust,
    C,
    System,
    RustIntrinsic,
    RustCall,
    PlatformIntrinsic,
    Unadjusted
});

impl_stable_hash_for!(struct ::syntax::attr::Deprecation { since, note });
impl_stable_hash_for!(struct ::syntax::attr::Stability {
    level,
    feature,
    rustc_depr,
    promotable,
    allow_const_fn_ptr,
    const_stability
});

impl_stable_hash_for!(enum ::syntax::edition::Edition {
    Edition2015,
    Edition2018,
});

impl<'a> HashStable<StableHashingContext<'a>>
for ::syntax::attr::StabilityLevel {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            ::syntax::attr::StabilityLevel::Unstable { ref reason, ref issue } => {
                reason.hash_stable(hcx, hasher);
                issue.hash_stable(hcx, hasher);
            }
            ::syntax::attr::StabilityLevel::Stable { ref since } => {
                since.hash_stable(hcx, hasher);
            }
        }
    }
}

impl_stable_hash_for!(struct ::syntax::attr::RustcDeprecation { since, reason, suggestion });


impl_stable_hash_for!(enum ::syntax::attr::IntType {
    SignedInt(int_ty),
    UnsignedInt(uint_ty)
});

impl_stable_hash_for!(enum ::syntax::ast::LitIntType {
    Signed(int_ty),
    Unsigned(int_ty),
    Unsuffixed
});

impl_stable_hash_for!(struct ::syntax::ast::Lit {
    node,
    token,
    span
});

impl_stable_hash_for!(enum ::syntax::ast::LitKind {
    Str(value, style),
    ByteStr(value),
    Byte(value),
    Char(value),
    Int(value, lit_int_type),
    Float(value, float_ty),
    FloatUnsuffixed(value),
    Bool(value),
    Err(value)
});

impl_stable_hash_for_spanned!(::syntax::ast::LitKind);

impl_stable_hash_for!(enum ::syntax::ast::IntTy { Isize, I8, I16, I32, I64, I128 });
impl_stable_hash_for!(enum ::syntax::ast::UintTy { Usize, U8, U16, U32, U64, U128 });
impl_stable_hash_for!(enum ::syntax::ast::FloatTy { F32, F64 });
impl_stable_hash_for!(enum ::syntax::ast::Unsafety { Unsafe, Normal });
impl_stable_hash_for!(enum ::syntax::ast::Constness { Const, NotConst });
impl_stable_hash_for!(enum ::syntax::ast::Defaultness { Default, Final });
impl_stable_hash_for!(struct ::syntax::ast::Lifetime { id, ident });
impl_stable_hash_for!(enum ::syntax::ast::StrStyle { Cooked, Raw(pounds) });
impl_stable_hash_for!(enum ::syntax::ast::AttrStyle { Outer, Inner });

impl<'a> HashStable<StableHashingContext<'a>> for [ast::Attribute] {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        if self.len() == 0 {
            self.len().hash_stable(hcx, hasher);
            return
        }

        // Some attributes are always ignored during hashing.
        let filtered: SmallVec<[&ast::Attribute; 8]> = self
            .iter()
            .filter(|attr| {
                !attr.is_sugared_doc &&
                !attr.ident().map_or(false, |ident| hcx.is_ignored_attr(ident.name))
            })
            .collect();

        filtered.len().hash_stable(hcx, hasher);
        for attr in filtered {
            attr.hash_stable(hcx, hasher);
        }
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for ast::Path {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        self.segments.len().hash_stable(hcx, hasher);
        for segment in &self.segments {
            segment.ident.name.hash_stable(hcx, hasher);
        }
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for ast::Attribute {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        // Make sure that these have been filtered out.
        debug_assert!(!self.ident().map_or(false, |ident| hcx.is_ignored_attr(ident.name)));
        debug_assert!(!self.is_sugared_doc);

        let ast::Attribute {
            id: _,
            style,
            ref path,
            ref tokens,
            is_sugared_doc: _,
            span,
        } = *self;

        style.hash_stable(hcx, hasher);
        path.hash_stable(hcx, hasher);
        for tt in tokens.trees() {
            tt.hash_stable(hcx, hasher);
        }
        span.hash_stable(hcx, hasher);
    }
}

impl<'a> HashStable<StableHashingContext<'a>>
for tokenstream::TokenTree {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            tokenstream::TokenTree::Token(ref token) => {
                token.hash_stable(hcx, hasher);
            }
            tokenstream::TokenTree::Delimited(span, delim, ref tts) => {
                span.hash_stable(hcx, hasher);
                std_hash::Hash::hash(&delim, hasher);
                for sub_tt in tts.trees() {
                    sub_tt.hash_stable(hcx, hasher);
                }
            }
        }
    }
}

impl<'a> HashStable<StableHashingContext<'a>>
for tokenstream::TokenStream {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        for sub_tt in self.trees() {
            sub_tt.hash_stable(hcx, hasher);
        }
    }
}

impl_stable_hash_for!(enum token::LitKind {
    Bool,
    Byte,
    Char,
    Integer,
    Float,
    Str,
    ByteStr,
    StrRaw(n),
    ByteStrRaw(n),
    Err
});

impl_stable_hash_for!(struct token::Lit {
    kind,
    symbol,
    suffix
});

impl<'a> HashStable<StableHashingContext<'a>> for token::TokenKind {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            token::Eq |
            token::Lt |
            token::Le |
            token::EqEq |
            token::Ne |
            token::Ge |
            token::Gt |
            token::AndAnd |
            token::OrOr |
            token::Not |
            token::Tilde |
            token::At |
            token::Dot |
            token::DotDot |
            token::DotDotDot |
            token::DotDotEq |
            token::Comma |
            token::Semi |
            token::Colon |
            token::ModSep |
            token::RArrow |
            token::LArrow |
            token::FatArrow |
            token::Pound |
            token::Dollar |
            token::Question |
            token::SingleQuote |
            token::Whitespace |
            token::Comment |
            token::Eof => {}

            token::BinOp(bin_op_token) |
            token::BinOpEq(bin_op_token) => {
                std_hash::Hash::hash(&bin_op_token, hasher);
            }

            token::OpenDelim(delim_token) |
            token::CloseDelim(delim_token) => {
                std_hash::Hash::hash(&delim_token, hasher);
            }
            token::Literal(lit) => lit.hash_stable(hcx, hasher),

            token::Ident(name, is_raw) => {
                name.hash_stable(hcx, hasher);
                is_raw.hash_stable(hcx, hasher);
            }
            token::Lifetime(name) => name.hash_stable(hcx, hasher),

            token::Interpolated(_) => {
                bug!("interpolated tokens should not be present in the HIR")
            }

            token::DocComment(val) |
            token::Shebang(val) => val.hash_stable(hcx, hasher),
        }
    }
}

impl_stable_hash_for!(struct token::Token {
    kind,
    span
});

impl_stable_hash_for!(enum ::syntax::ast::NestedMetaItem {
    MetaItem(meta_item),
    Literal(lit)
});

impl_stable_hash_for!(struct ::syntax::ast::MetaItem {
    path,
    node,
    span
});

impl_stable_hash_for!(enum ::syntax::ast::MetaItemKind {
    Word,
    List(nested_items),
    NameValue(lit)
});

impl_stable_hash_for!(enum ::syntax_pos::hygiene::Transparency {
    Transparent,
    SemiTransparent,
    Opaque,
});

impl_stable_hash_for!(struct ::syntax_pos::hygiene::ExpnInfo {
    call_site,
    format,
    def_site,
    default_transparency,
    allow_internal_unstable,
    allow_internal_unsafe,
    local_inner_macros,
    edition
});

impl_stable_hash_for!(enum ::syntax_pos::hygiene::ExpnFormat {
    MacroAttribute(sym),
    MacroBang(sym),
    CompilerDesugaring(kind)
});

impl_stable_hash_for!(enum ::syntax_pos::hygiene::CompilerDesugaringKind {
    IfTemporary,
    Async,
    Await,
    QuestionMark,
    ExistentialType,
    ForLoop,
    TryBlock
});

impl_stable_hash_for!(enum ::syntax_pos::FileName {
    Real(pb),
    Macros(s),
    QuoteExpansion(s),
    Anon(s),
    MacroExpansion(s),
    ProcMacroSourceCode(s),
    CliCrateAttr(s),
    CfgSpec(s),
    Custom(s),
    DocTest(pb, line),
});

impl<'a> HashStable<StableHashingContext<'a>> for SourceFile {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
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
        } = *self;

        (name_hash as u64).hash_stable(hcx, hasher);
        name_was_remapped.hash_stable(hcx, hasher);

        DefId {
            krate: CrateNum::from_u32(crate_of_origin),
            index: CRATE_DEF_INDEX,
        }.hash_stable(hcx, hasher);

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
    }
}

fn stable_byte_pos(pos: ::syntax_pos::BytePos,
                   source_file_start: ::syntax_pos::BytePos)
                   -> u32 {
    pos.0 - source_file_start.0
}

fn stable_multibyte_char(mbc: ::syntax_pos::MultiByteChar,
                         source_file_start: ::syntax_pos::BytePos)
                         -> (u32, u32) {
    let ::syntax_pos::MultiByteChar {
        pos,
        bytes,
    } = mbc;

    (pos.0 - source_file_start.0, bytes as u32)
}

fn stable_non_narrow_char(swc: ::syntax_pos::NonNarrowChar,
                          source_file_start: ::syntax_pos::BytePos)
                          -> (u32, u32) {
    let pos = swc.pos();
    let width = swc.width();

    (pos.0 - source_file_start.0, width as u32)
}

impl<'tcx> HashStable<StableHashingContext<'tcx>> for feature_gate::Features {
    fn hash_stable<W: StableHasherResult>(
        &self,
        hcx: &mut StableHashingContext<'tcx>,
        hasher: &mut StableHasher<W>,
    ) {
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
