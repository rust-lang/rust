// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module contains `HashStable` implementations for various data types
//! from libsyntax in no particular order.

use ich::StableHashingContext;

use std::hash as std_hash;
use std::mem;

use syntax::ast;
use syntax::parse::token;
use syntax::symbol::InternedString;
use syntax::tokenstream;
use syntax_pos::{Span, FileMap};

use hir::def_id::{DefId, CrateNum, CRATE_DEF_INDEX};

use rustc_data_structures::stable_hasher::{HashStable, ToStableHashKey,
                                           StableHasher, StableHasherResult};
use rustc_data_structures::accumulate_vec::AccumulateVec;

impl<'gcx> HashStable<StableHashingContext<'gcx>> for InternedString {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        let s: &str = &**self;
        s.hash_stable(hcx, hasher);
    }
}

impl<'gcx> ToStableHashKey<StableHashingContext<'gcx>> for InternedString {
    type KeyType = InternedString;

    #[inline]
    fn to_stable_hash_key(&self,
                          _: &StableHashingContext<'gcx>)
                          -> InternedString {
        self.clone()
    }
}

impl<'gcx> HashStable<StableHashingContext<'gcx>> for ast::Name {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        self.as_str().hash_stable(hcx, hasher);
    }
}

impl<'gcx> ToStableHashKey<StableHashingContext<'gcx>> for ast::Name {
    type KeyType = InternedString;

    #[inline]
    fn to_stable_hash_key(&self,
                          _: &StableHashingContext<'gcx>)
                          -> InternedString {
        self.as_str()
    }
}

impl_stable_hash_for!(enum ::syntax::ast::AsmDialect {
    Att,
    Intel
});

impl_stable_hash_for!(enum ::syntax::ext::base::MacroKind {
    Bang,
    Attr,
    Derive
});


impl_stable_hash_for!(enum ::syntax::abi::Abi {
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
    rustc_const_unstable
});

impl<'gcx> HashStable<StableHashingContext<'gcx>>
for ::syntax::attr::StabilityLevel {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
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

impl_stable_hash_for!(struct ::syntax::attr::RustcDeprecation { since, reason });
impl_stable_hash_for!(struct ::syntax::attr::RustcConstUnstable { feature });


impl_stable_hash_for!(enum ::syntax::attr::IntType {
    SignedInt(int_ty),
    UnsignedInt(uint_ty)
});

impl_stable_hash_for!(enum ::syntax::ast::LitIntType {
    Signed(int_ty),
    Unsigned(int_ty),
    Unsuffixed
});

impl_stable_hash_for_spanned!(::syntax::ast::LitKind);
impl_stable_hash_for!(enum ::syntax::ast::LitKind {
    Str(value, style),
    ByteStr(value),
    Byte(value),
    Char(value),
    Int(value, lit_int_type),
    Float(value, float_ty),
    FloatUnsuffixed(value),
    Bool(value)
});

impl_stable_hash_for!(enum ::syntax::ast::IntTy { Is, I8, I16, I32, I64, I128 });
impl_stable_hash_for!(enum ::syntax::ast::UintTy { Us, U8, U16, U32, U64, U128 });
impl_stable_hash_for!(enum ::syntax::ast::FloatTy { F32, F64 });
impl_stable_hash_for!(enum ::syntax::ast::Unsafety { Unsafe, Normal });
impl_stable_hash_for!(enum ::syntax::ast::Constness { Const, NotConst });
impl_stable_hash_for!(enum ::syntax::ast::Defaultness { Default, Final });
impl_stable_hash_for!(struct ::syntax::ast::Lifetime { id, span, ident });
impl_stable_hash_for!(enum ::syntax::ast::StrStyle { Cooked, Raw(pounds) });
impl_stable_hash_for!(enum ::syntax::ast::AttrStyle { Outer, Inner });

impl<'gcx> HashStable<StableHashingContext<'gcx>> for [ast::Attribute] {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        if self.len() == 0 {
            self.len().hash_stable(hcx, hasher);
            return
        }

        // Some attributes are always ignored during hashing.
        let filtered: AccumulateVec<[&ast::Attribute; 8]> = self
            .iter()
            .filter(|attr| {
                !attr.is_sugared_doc &&
                attr.name().map(|name| !hcx.is_ignored_attr(name)).unwrap_or(true)
            })
            .collect();

        filtered.len().hash_stable(hcx, hasher);
        for attr in filtered {
            attr.hash_stable(hcx, hasher);
        }
    }
}

impl<'gcx> HashStable<StableHashingContext<'gcx>> for ast::Attribute {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        // Make sure that these have been filtered out.
        debug_assert!(self.name().map(|name| !hcx.is_ignored_attr(name)).unwrap_or(true));
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
        path.segments.len().hash_stable(hcx, hasher);
        for segment in &path.segments {
            segment.identifier.name.hash_stable(hcx, hasher);
        }
        for tt in tokens.trees() {
            tt.hash_stable(hcx, hasher);
        }
        span.hash_stable(hcx, hasher);
    }
}

impl<'gcx> HashStable<StableHashingContext<'gcx>>
for tokenstream::TokenTree {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            tokenstream::TokenTree::Token(span, ref token) => {
                span.hash_stable(hcx, hasher);
                hash_token(token, hcx, hasher, span);
            }
            tokenstream::TokenTree::Delimited(span, ref delimited) => {
                span.hash_stable(hcx, hasher);
                std_hash::Hash::hash(&delimited.delim, hasher);
                for sub_tt in delimited.stream().trees() {
                    sub_tt.hash_stable(hcx, hasher);
                }
            }
        }
    }
}

impl<'gcx> HashStable<StableHashingContext<'gcx>>
for tokenstream::TokenStream {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        for sub_tt in self.trees() {
            sub_tt.hash_stable(hcx, hasher);
        }
    }
}

fn hash_token<'gcx, W: StableHasherResult>(token: &token::Token,
                                           hcx: &mut StableHashingContext<'gcx>,
                                           hasher: &mut StableHasher<W>,
                                           error_reporting_span: Span) {
    mem::discriminant(token).hash_stable(hcx, hasher);
    match *token {
        token::Token::Eq |
        token::Token::Lt |
        token::Token::Le |
        token::Token::EqEq |
        token::Token::Ne |
        token::Token::Ge |
        token::Token::Gt |
        token::Token::AndAnd |
        token::Token::OrOr |
        token::Token::Not |
        token::Token::Tilde |
        token::Token::At |
        token::Token::Dot |
        token::Token::DotDot |
        token::Token::DotDotDot |
        token::Token::Comma |
        token::Token::Semi |
        token::Token::Colon |
        token::Token::ModSep |
        token::Token::RArrow |
        token::Token::LArrow |
        token::Token::FatArrow |
        token::Token::Pound |
        token::Token::Dollar |
        token::Token::Question |
        token::Token::Underscore |
        token::Token::Whitespace |
        token::Token::Comment |
        token::Token::Eof => {}

        token::Token::BinOp(bin_op_token) |
        token::Token::BinOpEq(bin_op_token) => {
            std_hash::Hash::hash(&bin_op_token, hasher);
        }

        token::Token::OpenDelim(delim_token) |
        token::Token::CloseDelim(delim_token) => {
            std_hash::Hash::hash(&delim_token, hasher);
        }
        token::Token::Literal(ref lit, ref opt_name) => {
            mem::discriminant(lit).hash_stable(hcx, hasher);
            match *lit {
                token::Lit::Byte(val) |
                token::Lit::Char(val) |
                token::Lit::Integer(val) |
                token::Lit::Float(val) |
                token::Lit::Str_(val) |
                token::Lit::ByteStr(val) => val.hash_stable(hcx, hasher),
                token::Lit::StrRaw(val, n) |
                token::Lit::ByteStrRaw(val, n) => {
                    val.hash_stable(hcx, hasher);
                    n.hash_stable(hcx, hasher);
                }
            };
            opt_name.hash_stable(hcx, hasher);
        }

        token::Token::Ident(ident) |
        token::Token::Lifetime(ident) => ident.name.hash_stable(hcx, hasher),

        token::Token::Interpolated(ref non_terminal) => {
            // FIXME(mw): This could be implemented properly. It's just a
            //            lot of work, since we would need to hash the AST
            //            in a stable way, in addition to the HIR.
            //            Since this is hardly used anywhere, just emit a
            //            warning for now.
            if hcx.sess().opts.debugging_opts.incremental.is_some() {
                let msg = format!("Quasi-quoting might make incremental \
                                   compilation very inefficient: {:?}",
                                  non_terminal);
                hcx.sess().span_warn(error_reporting_span, &msg[..]);
            }

            std_hash::Hash::hash(non_terminal, hasher);
        }

        token::Token::DocComment(val) |
        token::Token::Shebang(val) => val.hash_stable(hcx, hasher),
    }
}

impl_stable_hash_for_spanned!(::syntax::ast::NestedMetaItemKind);

impl_stable_hash_for!(enum ::syntax::ast::NestedMetaItemKind {
    MetaItem(meta_item),
    Literal(lit)
});

impl_stable_hash_for!(struct ::syntax::ast::MetaItem {
    name,
    node,
    span
});

impl_stable_hash_for!(enum ::syntax::ast::MetaItemKind {
    Word,
    List(nested_items),
    NameValue(lit)
});

impl<'gcx> HashStable<StableHashingContext<'gcx>> for FileMap {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        let FileMap {
            ref name,
            name_was_remapped,
            crate_of_origin,
            // Do not hash the source as it is not encoded
            src: _,
            src_hash,
            external_src: _,
            start_pos,
            end_pos: _,
            ref lines,
            ref multibyte_chars,
        } = *self;

        name.hash_stable(hcx, hasher);
        name_was_remapped.hash_stable(hcx, hasher);

        DefId {
            krate: CrateNum::from_u32(crate_of_origin),
            index: CRATE_DEF_INDEX,
        }.hash_stable(hcx, hasher);

        src_hash.hash_stable(hcx, hasher);

        // We only hash the relative position within this filemap
        let lines = lines.borrow();
        lines.len().hash_stable(hcx, hasher);
        for &line in lines.iter() {
            stable_byte_pos(line, start_pos).hash_stable(hcx, hasher);
        }

        // We only hash the relative position within this filemap
        let multibyte_chars = multibyte_chars.borrow();
        multibyte_chars.len().hash_stable(hcx, hasher);
        for &char_pos in multibyte_chars.iter() {
            stable_multibyte_char(char_pos, start_pos).hash_stable(hcx, hasher);
        }
    }
}

fn stable_byte_pos(pos: ::syntax_pos::BytePos,
                   filemap_start: ::syntax_pos::BytePos)
                   -> u32 {
    pos.0 - filemap_start.0
}

fn stable_multibyte_char(mbc: ::syntax_pos::MultiByteChar,
                         filemap_start: ::syntax_pos::BytePos)
                         -> (u32, u32) {
    let ::syntax_pos::MultiByteChar {
        pos,
        bytes,
    } = mbc;

    (pos.0 - filemap_start.0, bytes as u32)
}
