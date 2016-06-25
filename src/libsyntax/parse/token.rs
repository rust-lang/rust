// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use self::BinOpToken::*;
pub use self::Nonterminal::*;
pub use self::DelimToken::*;
pub use self::Lit::*;
pub use self::Token::*;

use ast::{self, BinOpKind};
use ext::mtwt;
use ptr::P;
use util::interner::{RcStr, StrInterner};
use util::interner;
use tokenstream;

use serialize::{Decodable, Decoder, Encodable, Encoder};
use std::fmt;
use std::ops::Deref;
use std::rc::Rc;

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Eq, Hash, Debug, Copy)]
pub enum BinOpToken {
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Caret,
    And,
    Or,
    Shl,
    Shr,
}

/// A delimiter token
#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Eq, Hash, Debug, Copy)]
pub enum DelimToken {
    /// A round parenthesis: `(` or `)`
    Paren,
    /// A square bracket: `[` or `]`
    Bracket,
    /// A curly brace: `{` or `}`
    Brace,
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Eq, Hash, Debug, Copy)]
pub enum SpecialMacroVar {
    /// `$crate` will be filled in with the name of the crate a macro was
    /// imported from, if any.
    CrateMacroVar,
}

impl SpecialMacroVar {
    pub fn as_str(self) -> &'static str {
        match self {
            SpecialMacroVar::CrateMacroVar => "crate",
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Eq, Hash, Debug, Copy)]
pub enum Lit {
    Byte(ast::Name),
    Char(ast::Name),
    Integer(ast::Name),
    Float(ast::Name),
    Str_(ast::Name),
    StrRaw(ast::Name, usize), /* raw str delimited by n hash symbols */
    ByteStr(ast::Name),
    ByteStrRaw(ast::Name, usize), /* raw byte str delimited by n hash symbols */
}

impl Lit {
    pub fn short_name(&self) -> &'static str {
        match *self {
            Byte(_) => "byte",
            Char(_) => "char",
            Integer(_) => "integer",
            Float(_) => "float",
            Str_(_) | StrRaw(..) => "string",
            ByteStr(_) | ByteStrRaw(..) => "byte string"
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Eq, Hash, Debug)]
pub enum Token {
    /* Expression-operator symbols. */
    Eq,
    Lt,
    Le,
    EqEq,
    Ne,
    Ge,
    Gt,
    AndAnd,
    OrOr,
    Not,
    Tilde,
    BinOp(BinOpToken),
    BinOpEq(BinOpToken),

    /* Structural symbols */
    At,
    Dot,
    DotDot,
    DotDotDot,
    Comma,
    Semi,
    Colon,
    ModSep,
    RArrow,
    LArrow,
    FatArrow,
    Pound,
    Dollar,
    Question,
    /// An opening delimiter, eg. `{`
    OpenDelim(DelimToken),
    /// A closing delimiter, eg. `}`
    CloseDelim(DelimToken),

    /* Literals */
    Literal(Lit, Option<ast::Name>),

    /* Name components */
    Ident(ast::Ident),
    Underscore,
    Lifetime(ast::Ident),

    /* For interpolation */
    Interpolated(Nonterminal),
    // Can be expanded into several tokens.
    /// Doc comment
    DocComment(ast::Name),
    // In left-hand-sides of MBE macros:
    /// Parse a nonterminal (name to bind, name of NT)
    MatchNt(ast::Ident, ast::Ident),
    // In right-hand-sides of MBE macros:
    /// A syntactic variable that will be filled in by macro expansion.
    SubstNt(ast::Ident),
    /// A macro variable with special meaning.
    SpecialVarNt(SpecialMacroVar),

    // Junk. These carry no data because we don't really care about the data
    // they *would* carry, and don't really want to allocate a new ident for
    // them. Instead, users could extract that from the associated span.

    /// Whitespace
    Whitespace,
    /// Comment
    Comment,
    Shebang(ast::Name),

    Eof,
}

impl Token {
    /// Returns `true` if the token starts with '>'.
    pub fn is_like_gt(&self) -> bool {
        match *self {
            BinOp(Shr) | BinOpEq(Shr) | Gt | Ge => true,
            _ => false,
        }
    }

    /// Returns `true` if the token can appear at the start of an expression.
    pub fn can_begin_expr(&self) -> bool {
        match *self {
            OpenDelim(_)                => true,
            Ident(..)                   => true,
            Underscore                  => true,
            Tilde                       => true,
            Literal(_, _)               => true,
            Not                         => true,
            BinOp(Minus)                => true,
            BinOp(Star)                 => true,
            BinOp(And)                  => true,
            BinOp(Or)                   => true, // in lambda syntax
            OrOr                        => true, // in lambda syntax
            AndAnd                      => true, // double borrow
            DotDot | DotDotDot          => true, // range notation
            ModSep                      => true,
            Interpolated(NtExpr(..))    => true,
            Interpolated(NtIdent(..))   => true,
            Interpolated(NtBlock(..))   => true,
            Interpolated(NtPath(..))    => true,
            Pound                       => true, // for expression attributes
            _                           => false,
        }
    }

    /// Returns `true` if the token is any literal
    pub fn is_lit(&self) -> bool {
        match *self {
            Literal(_, _) => true,
            _          => false,
        }
    }

    /// Returns `true` if the token is an identifier.
    pub fn is_ident(&self) -> bool {
        match *self {
            Ident(..)   => true,
            _           => false,
        }
    }

    /// Returns `true` if the token is interpolated.
    pub fn is_interpolated(&self) -> bool {
        match *self {
            Interpolated(..) => true,
            _                => false,
        }
    }

    /// Returns `true` if the token is an interpolated path.
    pub fn is_path(&self) -> bool {
        match *self {
            Interpolated(NtPath(..))    => true,
            _                           => false,
        }
    }

    /// Returns `true` if the token is a lifetime.
    pub fn is_lifetime(&self) -> bool {
        match *self {
            Lifetime(..) => true,
            _            => false,
        }
    }

    /// Returns `true` if the token is either the `mut` or `const` keyword.
    pub fn is_mutability(&self) -> bool {
        self.is_keyword(keywords::Mut) ||
        self.is_keyword(keywords::Const)
    }

    pub fn is_path_start(&self) -> bool {
        self == &ModSep || self == &Lt || self.is_path() ||
        self.is_path_segment_keyword() || self.is_ident() && !self.is_any_keyword()
    }

    /// Maps a token to its corresponding binary operator.
    pub fn to_binop(&self) -> Option<BinOpKind> {
        match *self {
            BinOp(Star)     => Some(BinOpKind::Mul),
            BinOp(Slash)    => Some(BinOpKind::Div),
            BinOp(Percent)  => Some(BinOpKind::Rem),
            BinOp(Plus)     => Some(BinOpKind::Add),
            BinOp(Minus)    => Some(BinOpKind::Sub),
            BinOp(Shl)      => Some(BinOpKind::Shl),
            BinOp(Shr)      => Some(BinOpKind::Shr),
            BinOp(And)      => Some(BinOpKind::BitAnd),
            BinOp(Caret)    => Some(BinOpKind::BitXor),
            BinOp(Or)       => Some(BinOpKind::BitOr),
            Lt              => Some(BinOpKind::Lt),
            Le              => Some(BinOpKind::Le),
            Ge              => Some(BinOpKind::Ge),
            Gt              => Some(BinOpKind::Gt),
            EqEq            => Some(BinOpKind::Eq),
            Ne              => Some(BinOpKind::Ne),
            AndAnd          => Some(BinOpKind::And),
            OrOr            => Some(BinOpKind::Or),
            _               => None,
        }
    }

    /// Returns `true` if the token is a given keyword, `kw`.
    pub fn is_keyword(&self, kw: keywords::Keyword) -> bool {
        match *self {
            Ident(id) => id.name == kw.name(),
            _ => false,
        }
    }

    pub fn is_path_segment_keyword(&self) -> bool {
        match *self {
            Ident(id) => id.name == keywords::Super.name() ||
                         id.name == keywords::SelfValue.name() ||
                         id.name == keywords::SelfType.name(),
            _ => false,
        }
    }

    /// Returns `true` if the token is either a strict or reserved keyword.
    pub fn is_any_keyword(&self) -> bool {
        self.is_strict_keyword() || self.is_reserved_keyword()
    }

    /// Returns `true` if the token is a strict keyword.
    pub fn is_strict_keyword(&self) -> bool {
        match *self {
            Ident(id) => id.name >= keywords::As.name() &&
                         id.name <= keywords::While.name(),
            _ => false,
        }
    }

    /// Returns `true` if the token is a keyword reserved for possible future use.
    pub fn is_reserved_keyword(&self) -> bool {
        match *self {
            Ident(id) => id.name >= keywords::Abstract.name() &&
                         id.name <= keywords::Yield.name(),
            _ => false,
        }
    }

    /// Hygienic identifier equality comparison.
    ///
    /// See `styntax::ext::mtwt`.
    pub fn mtwt_eq(&self, other : &Token) -> bool {
        match (self, other) {
            (&Ident(id1), &Ident(id2)) | (&Lifetime(id1), &Lifetime(id2)) =>
                mtwt::resolve(id1) == mtwt::resolve(id2),
            _ => *self == *other
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Eq, Hash)]
/// For interpolation during macro expansion.
pub enum Nonterminal {
    NtItem(P<ast::Item>),
    NtBlock(P<ast::Block>),
    NtStmt(P<ast::Stmt>),
    NtPat(P<ast::Pat>),
    NtExpr(P<ast::Expr>),
    NtTy(P<ast::Ty>),
    NtIdent(Box<ast::SpannedIdent>),
    /// Stuff inside brackets for attributes
    NtMeta(P<ast::MetaItem>),
    NtPath(Box<ast::Path>),
    NtTT(P<tokenstream::TokenTree>), // needs P'ed to break a circularity
    // These are not exposed to macros, but are used by quasiquote.
    NtArm(ast::Arm),
    NtImplItem(P<ast::ImplItem>),
    NtTraitItem(P<ast::TraitItem>),
    NtGenerics(ast::Generics),
    NtWhereClause(ast::WhereClause),
    NtArg(ast::Arg),
}

impl fmt::Debug for Nonterminal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            NtItem(..) => f.pad("NtItem(..)"),
            NtBlock(..) => f.pad("NtBlock(..)"),
            NtStmt(..) => f.pad("NtStmt(..)"),
            NtPat(..) => f.pad("NtPat(..)"),
            NtExpr(..) => f.pad("NtExpr(..)"),
            NtTy(..) => f.pad("NtTy(..)"),
            NtIdent(..) => f.pad("NtIdent(..)"),
            NtMeta(..) => f.pad("NtMeta(..)"),
            NtPath(..) => f.pad("NtPath(..)"),
            NtTT(..) => f.pad("NtTT(..)"),
            NtArm(..) => f.pad("NtArm(..)"),
            NtImplItem(..) => f.pad("NtImplItem(..)"),
            NtTraitItem(..) => f.pad("NtTraitItem(..)"),
            NtGenerics(..) => f.pad("NtGenerics(..)"),
            NtWhereClause(..) => f.pad("NtWhereClause(..)"),
            NtArg(..) => f.pad("NtArg(..)"),
        }
    }
}

// In this macro, there is the requirement that the name (the number) must be monotonically
// increasing by one in the special identifiers, starting at 0; the same holds for the keywords,
// except starting from the next number instead of zero.
macro_rules! declare_keywords {(
    $( ($index: expr, $konst: ident, $string: expr) )*
) => {
    pub mod keywords {
        use ast;
        #[derive(Clone, Copy, PartialEq, Eq)]
        pub struct Keyword {
            ident: ast::Ident,
        }
        impl Keyword {
            #[inline] pub fn ident(self) -> ast::Ident { self.ident }
            #[inline] pub fn name(self) -> ast::Name { self.ident.name }
        }
        $(
            #[allow(non_upper_case_globals)]
            pub const $konst: Keyword = Keyword {
                ident: ast::Ident::with_empty_ctxt(ast::Name($index))
            };
        )*
    }

    fn mk_fresh_ident_interner() -> IdentInterner {
        interner::StrInterner::prefill(&[$($string,)*])
    }
}}

// NB: leaving holes in the ident table is bad! a different ident will get
// interned with the id from the hole, but it will be between the min and max
// of the reserved words, and thus tagged as "reserved".
// After modifying this list adjust `is_strict_keyword`/`is_reserved_keyword`,
// this should be rarely necessary though if the keywords are kept in alphabetic order.
declare_keywords! {
    // Invalid identifier
    (0,  Invalid,        "")

    // Strict keywords used in the language.
    (1,  As,             "as")
    (2,  Box,            "box")
    (3,  Break,          "break")
    (4,  Const,          "const")
    (5,  Continue,       "continue")
    (6,  Crate,          "crate")
    (7,  Else,           "else")
    (8,  Enum,           "enum")
    (9,  Extern,         "extern")
    (10, False,          "false")
    (11, Fn,             "fn")
    (12, For,            "for")
    (13, If,             "if")
    (14, Impl,           "impl")
    (15, In,             "in")
    (16, Let,            "let")
    (17, Loop,           "loop")
    (18, Match,          "match")
    (19, Mod,            "mod")
    (20, Move,           "move")
    (21, Mut,            "mut")
    (22, Pub,            "pub")
    (23, Ref,            "ref")
    (24, Return,         "return")
    (25, SelfValue,      "self")
    (26, SelfType,       "Self")
    (27, Static,         "static")
    (28, Struct,         "struct")
    (29, Super,          "super")
    (30, Trait,          "trait")
    (31, True,           "true")
    (32, Type,           "type")
    (33, Unsafe,         "unsafe")
    (34, Use,            "use")
    (35, Where,          "where")
    (36, While,          "while")

    // Keywords reserved for future use.
    (37, Abstract,       "abstract")
    (38, Alignof,        "alignof")
    (39, Become,         "become")
    (40, Do,             "do")
    (41, Final,          "final")
    (42, Macro,          "macro")
    (43, Offsetof,       "offsetof")
    (44, Override,       "override")
    (45, Priv,           "priv")
    (46, Proc,           "proc")
    (47, Pure,           "pure")
    (48, Sizeof,         "sizeof")
    (49, Typeof,         "typeof")
    (50, Unsized,        "unsized")
    (51, Virtual,        "virtual")
    (52, Yield,          "yield")

    // Weak keywords, have special meaning only in specific contexts.
    (53, Default,        "default")
    (54, StaticLifetime, "'static")
    (55, Union,          "union")
}

// looks like we can get rid of this completely...
pub type IdentInterner = StrInterner;

// if an interner exists in TLS, return it. Otherwise, prepare a
// fresh one.
// FIXME(eddyb) #8726 This should probably use a thread-local reference.
pub fn get_ident_interner() -> Rc<IdentInterner> {
    thread_local!(static KEY: Rc<::parse::token::IdentInterner> = {
        Rc::new(mk_fresh_ident_interner())
    });
    KEY.with(|k| k.clone())
}

/// Reset the ident interner to its initial state.
pub fn reset_ident_interner() {
    let interner = get_ident_interner();
    interner.reset(mk_fresh_ident_interner());
}

/// Represents a string stored in the thread-local interner. Because the
/// interner lives for the life of the thread, this can be safely treated as an
/// immortal string, as long as it never crosses between threads.
///
/// FIXME(pcwalton): You must be careful about what you do in the destructors
/// of objects stored in TLS, because they may run after the interner is
/// destroyed. In particular, they must not access string contents. This can
/// be fixed in the future by just leaking all strings until thread death
/// somehow.
#[derive(Clone, PartialEq, Hash, PartialOrd, Eq, Ord)]
pub struct InternedString {
    string: RcStr,
}

impl InternedString {
    #[inline]
    pub fn new(string: &'static str) -> InternedString {
        InternedString {
            string: RcStr::new(string),
        }
    }

    #[inline]
    fn new_from_rc_str(string: RcStr) -> InternedString {
        InternedString {
            string: string,
        }
    }

    #[inline]
    pub fn new_from_name(name: ast::Name) -> InternedString {
        let interner = get_ident_interner();
        InternedString::new_from_rc_str(interner.get(name))
    }
}

impl Deref for InternedString {
    type Target = str;

    fn deref(&self) -> &str { &self.string }
}

impl fmt::Debug for InternedString {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.string, f)
    }
}

impl fmt::Display for InternedString {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.string, f)
    }
}

impl<'a> PartialEq<&'a str> for InternedString {
    #[inline(always)]
    fn eq(&self, other: & &'a str) -> bool {
        PartialEq::eq(&self.string[..], *other)
    }
    #[inline(always)]
    fn ne(&self, other: & &'a str) -> bool {
        PartialEq::ne(&self.string[..], *other)
    }
}

impl<'a> PartialEq<InternedString> for &'a str {
    #[inline(always)]
    fn eq(&self, other: &InternedString) -> bool {
        PartialEq::eq(*self, &other.string[..])
    }
    #[inline(always)]
    fn ne(&self, other: &InternedString) -> bool {
        PartialEq::ne(*self, &other.string[..])
    }
}

impl PartialEq<str> for InternedString {
    #[inline(always)]
    fn eq(&self, other: &str) -> bool {
        PartialEq::eq(&self.string[..], other)
    }
    #[inline(always)]
    fn ne(&self, other: &str) -> bool {
        PartialEq::ne(&self.string[..], other)
    }
}

impl PartialEq<InternedString> for str {
    #[inline(always)]
    fn eq(&self, other: &InternedString) -> bool {
        PartialEq::eq(self, &other.string[..])
    }
    #[inline(always)]
    fn ne(&self, other: &InternedString) -> bool {
        PartialEq::ne(self, &other.string[..])
    }
}

impl Decodable for InternedString {
    fn decode<D: Decoder>(d: &mut D) -> Result<InternedString, D::Error> {
        Ok(intern(d.read_str()?.as_ref()).as_str())
    }
}

impl Encodable for InternedString {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_str(&self.string)
    }
}

/// Interns and returns the string contents of an identifier, using the
/// thread-local interner.
#[inline]
pub fn intern_and_get_ident(s: &str) -> InternedString {
    intern(s).as_str()
}

/// Maps a string to its interned representation.
#[inline]
pub fn intern(s: &str) -> ast::Name {
    get_ident_interner().intern(s)
}

/// gensym's a new usize, using the current interner.
#[inline]
pub fn gensym(s: &str) -> ast::Name {
    get_ident_interner().gensym(s)
}

/// Maps a string to an identifier with an empty syntax context.
#[inline]
pub fn str_to_ident(s: &str) -> ast::Ident {
    ast::Ident::with_empty_ctxt(intern(s))
}

/// Maps a string to a gensym'ed identifier.
#[inline]
pub fn gensym_ident(s: &str) -> ast::Ident {
    ast::Ident::with_empty_ctxt(gensym(s))
}

// create a fresh name that maps to the same string as the old one.
// note that this guarantees that str_ptr_eq(ident_to_string(src),interner_get(fresh_name(src)));
// that is, that the new name and the old one are connected to ptr_eq strings.
pub fn fresh_name(src: ast::Ident) -> ast::Name {
    let interner = get_ident_interner();
    interner.gensym_copy(src.name)
    // following: debug version. Could work in final except that it's incompatible with
    // good error messages and uses of struct names in ambiguous could-be-binding
    // locations. Also definitely destroys the guarantee given above about ptr_eq.
    /*let num = rand::thread_rng().gen_uint_range(0,0xffff);
    gensym(format!("{}_{}",ident_to_string(src),num))*/
}

// create a fresh mark.
pub fn fresh_mark() -> ast::Mrk {
    gensym("mark").0
}

#[cfg(test)]
mod tests {
    use super::*;
    use ast;
    use ext::mtwt;

    fn mark_ident(id : ast::Ident, m : ast::Mrk) -> ast::Ident {
        ast::Ident::new(id.name, mtwt::apply_mark(m, id.ctxt))
    }

    #[test] fn mtwt_token_eq_test() {
        assert!(Gt.mtwt_eq(&Gt));
        let a = str_to_ident("bac");
        let a1 = mark_ident(a,92);
        assert!(Ident(a).mtwt_eq(&Ident(a1)));
    }
}
