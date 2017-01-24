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

use ast::{self};
use ptr::P;
use symbol::keywords;
use tokenstream;

use std::fmt;
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
    /// An empty delimiter
    NoDelim,
}

impl DelimToken {
    pub fn len(&self) -> u32 {
        if *self == NoDelim { 0 } else { 1 }
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
    Interpolated(Rc<Nonterminal>),
    // Can be expanded into several tokens.
    /// Doc comment
    DocComment(ast::Name),
    // In left-hand-sides of MBE macros:
    /// Parse a nonterminal (name to bind, name of NT)
    MatchNt(ast::Ident, ast::Ident),
    // In right-hand-sides of MBE macros:
    /// A syntactic variable that will be filled in by macro expansion.
    SubstNt(ast::Ident),

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
            OpenDelim(..)               => true,
            Ident(..)                   => true,
            Literal(..)                 => true,
            Not                         => true,
            BinOp(Minus)                => true,
            BinOp(Star)                 => true,
            BinOp(And)                  => true,
            BinOp(Or)                   => true, // in lambda syntax
            OrOr                        => true, // in lambda syntax
            AndAnd                      => true, // double borrow
            DotDot | DotDotDot          => true, // range notation
            Lt | BinOp(Shl)             => true, // associated path
            ModSep                      => true,
            Pound                       => true, // for expression attributes
            Interpolated(ref nt) => match **nt {
                NtExpr(..) => true,
                NtIdent(..) => true,
                NtBlock(..) => true,
                NtPath(..) => true,
                _ => false,
            },
            _ => false,
        }
    }

    /// Returns `true` if the token is any literal
    pub fn is_lit(&self) -> bool {
        match *self {
            Literal(..) => true,
            _           => false,
        }
    }

    /// Returns `true` if the token is an identifier.
    pub fn is_ident(&self) -> bool {
        match *self {
            Ident(..)   => true,
            _           => false,
        }
    }

    /// Returns `true` if the token is a documentation comment.
    pub fn is_doc_comment(&self) -> bool {
        match *self {
            DocComment(..)   => true,
            _                => false,
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
        if let Interpolated(ref nt) = *self {
            if let NtPath(..) = **nt {
                return true;
            }
        }
        false
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

    pub fn is_qpath_start(&self) -> bool {
        self == &Lt || self == &BinOp(Shl)
    }

    pub fn is_path_start(&self) -> bool {
        self == &ModSep || self.is_qpath_start() || self.is_path() ||
        self.is_path_segment_keyword() || self.is_ident() && !self.is_any_keyword()
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
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Eq, Hash)]
/// For interpolation during macro expansion.
pub enum Nonterminal {
    NtItem(P<ast::Item>),
    NtBlock(P<ast::Block>),
    NtStmt(ast::Stmt),
    NtPat(P<ast::Pat>),
    NtExpr(P<ast::Expr>),
    NtTy(P<ast::Ty>),
    NtIdent(ast::SpannedIdent),
    /// Stuff inside brackets for attributes
    NtMeta(ast::MetaItem),
    NtPath(ast::Path),
    NtTT(tokenstream::TokenTree),
    // These are not exposed to macros, but are used by quasiquote.
    NtArm(ast::Arm),
    NtImplItem(ast::ImplItem),
    NtTraitItem(ast::TraitItem),
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
