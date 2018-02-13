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
use parse::ParseSess;
use print::pprust;
use ptr::P;
use serialize::{Decodable, Decoder, Encodable, Encoder};
use symbol::keywords;
use syntax::parse::parse_stream_from_source_str;
use syntax_pos::{self, Span, FileName};
use tokenstream::{TokenStream, TokenTree};
use tokenstream;

use std::cell::Cell;
use std::{cmp, fmt};
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
    pub fn len(self) -> usize {
        if self == NoDelim { 0 } else { 1 }
    }

    pub fn is_empty(self) -> bool {
        self == NoDelim
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

fn ident_can_begin_expr(ident: ast::Ident) -> bool {
    let ident_token: Token = Ident(ident);

    !ident_token.is_reserved_ident() ||
    ident_token.is_path_segment_keyword() ||
    [
        keywords::Do.name(),
        keywords::Box.name(),
        keywords::Break.name(),
        keywords::Continue.name(),
        keywords::False.name(),
        keywords::For.name(),
        keywords::If.name(),
        keywords::Loop.name(),
        keywords::Match.name(),
        keywords::Move.name(),
        keywords::Return.name(),
        keywords::True.name(),
        keywords::Unsafe.name(),
        keywords::While.name(),
        keywords::Yield.name(),
    ].contains(&ident.name)
}

fn ident_can_begin_type(ident: ast::Ident) -> bool {
    let ident_token: Token = Ident(ident);

    !ident_token.is_reserved_ident() ||
    ident_token.is_path_segment_keyword() ||
    [
        keywords::For.name(),
        keywords::Impl.name(),
        keywords::Fn.name(),
        keywords::Unsafe.name(),
        keywords::Extern.name(),
        keywords::Typeof.name(),
    ].contains(&ident.name)
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
    DotDotEq,
    DotEq, // HACK(durka42) never produced by the parser, only used for libproc_macro
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

    // The `LazyTokenStream` is a pure function of the `Nonterminal`,
    // and so the `LazyTokenStream` can be ignored by Eq, Hash, etc.
    Interpolated(Rc<(Nonterminal, LazyTokenStream)>),
    // Can be expanded into several tokens.
    /// Doc comment
    DocComment(ast::Name),

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
    pub fn interpolated(nt: Nonterminal) -> Token {
        Token::Interpolated(Rc::new((nt, LazyTokenStream::new())))
    }

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
            Ident(ident)                => ident_can_begin_expr(ident), // value name or keyword
            OpenDelim(..)                     | // tuple, array or block
            Literal(..)                       | // literal
            Not                               | // operator not
            BinOp(Minus)                      | // unary minus
            BinOp(Star)                       | // dereference
            BinOp(Or) | OrOr                  | // closure
            BinOp(And)                        | // reference
            AndAnd                            | // double reference
            // DotDotDot is no longer supported, but we need some way to display the error
            DotDot | DotDotDot | DotDotEq     | // range notation
            Lt | BinOp(Shl)                   | // associated path
            ModSep                            | // global path
            Pound                             => true, // expression attributes
            Interpolated(ref nt) => match nt.0 {
                NtIdent(..) | NtExpr(..) | NtBlock(..) | NtPath(..) => true,
                _ => false,
            },
            _ => false,
        }
    }

    /// Returns `true` if the token can appear at the start of a type.
    pub fn can_begin_type(&self) -> bool {
        match *self {
            Ident(ident)                => ident_can_begin_type(ident), // type name or keyword
            OpenDelim(Paren)            | // tuple
            OpenDelim(Bracket)          | // array
            Underscore                  | // placeholder
            Not                         | // never
            BinOp(Star)                 | // raw pointer
            BinOp(And)                  | // reference
            AndAnd                      | // double reference
            Question                    | // maybe bound in trait object
            Lifetime(..)                | // lifetime bound in trait object
            Lt | BinOp(Shl)             | // associated path
            ModSep                      => true, // global path
            Interpolated(ref nt) => match nt.0 {
                NtIdent(..) | NtTy(..) | NtPath(..) | NtLifetime(..) => true,
                _ => false,
            },
            _ => false,
        }
    }

    /// Returns `true` if the token can appear at the start of a generic bound.
    pub fn can_begin_bound(&self) -> bool {
        self.is_path_start() || self.is_lifetime() || self.is_keyword(keywords::For) ||
        self == &Question || self == &OpenDelim(Paren)
    }

    /// Returns `true` if the token is any literal
    pub fn is_lit(&self) -> bool {
        match *self {
            Literal(..) => true,
            _           => false,
        }
    }

    pub fn ident(&self) -> Option<ast::Ident> {
        match *self {
            Ident(ident) => Some(ident),
            Interpolated(ref nt) => match nt.0 {
                NtIdent(ident) => Some(ident.node),
                _ => None,
            },
            _ => None,
        }
    }

    /// Returns `true` if the token is an identifier.
    pub fn is_ident(&self) -> bool {
        self.ident().is_some()
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
            if let NtPath(..) = nt.0 {
                return true;
            }
        }
        false
    }

    /// Returns a lifetime with the span and a dummy id if it is a lifetime,
    /// or the original lifetime if it is an interpolated lifetime, ignoring
    /// the span.
    pub fn lifetime(&self, span: Span) -> Option<ast::Lifetime> {
        match *self {
            Lifetime(ident) =>
                Some(ast::Lifetime { ident: ident, span: span, id: ast::DUMMY_NODE_ID }),
            Interpolated(ref nt) => match nt.0 {
                NtLifetime(lifetime) => Some(lifetime),
                _ => None,
            },
            _ => None,
        }
    }

    /// Returns `true` if the token is a lifetime.
    pub fn is_lifetime(&self) -> bool {
        self.lifetime(syntax_pos::DUMMY_SP).is_some()
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
        self.is_path_segment_keyword() || self.is_ident() && !self.is_reserved_ident()
    }

    /// Returns `true` if the token is a given keyword, `kw`.
    pub fn is_keyword(&self, kw: keywords::Keyword) -> bool {
        self.ident().map(|ident| ident.name == kw.name()).unwrap_or(false)
    }

    pub fn is_path_segment_keyword(&self) -> bool {
        match self.ident() {
            Some(id) => id.name == keywords::Super.name() ||
                        id.name == keywords::SelfValue.name() ||
                        id.name == keywords::SelfType.name() ||
                        id.name == keywords::Extern.name() ||
                        id.name == keywords::Crate.name() ||
                        id.name == keywords::DollarCrate.name(),
            None => false,
        }
    }

    // Returns true for reserved identifiers used internally for elided lifetimes,
    // unnamed method parameters, crate root module, error recovery etc.
    pub fn is_special_ident(&self) -> bool {
        match self.ident() {
            Some(id) => id.name <= keywords::DollarCrate.name(),
            _ => false,
        }
    }

    /// Returns `true` if the token is a keyword used in the language.
    pub fn is_used_keyword(&self) -> bool {
        match self.ident() {
            Some(id) => id.name >= keywords::As.name() && id.name <= keywords::While.name(),
            _ => false,
        }
    }

    /// Returns `true` if the token is a keyword reserved for possible future use.
    pub fn is_unused_keyword(&self) -> bool {
        match self.ident() {
            Some(id) => id.name >= keywords::Abstract.name() && id.name <= keywords::Yield.name(),
            _ => false,
        }
    }

    pub fn glue(self, joint: Token) -> Option<Token> {
        Some(match self {
            Eq => match joint {
                Eq => EqEq,
                Gt => FatArrow,
                _ => return None,
            },
            Lt => match joint {
                Eq => Le,
                Lt => BinOp(Shl),
                Le => BinOpEq(Shl),
                BinOp(Minus) => LArrow,
                _ => return None,
            },
            Gt => match joint {
                Eq => Ge,
                Gt => BinOp(Shr),
                Ge => BinOpEq(Shr),
                _ => return None,
            },
            Not => match joint {
                Eq => Ne,
                _ => return None,
            },
            BinOp(op) => match joint {
                Eq => BinOpEq(op),
                BinOp(And) if op == And => AndAnd,
                BinOp(Or) if op == Or => OrOr,
                Gt if op == Minus => RArrow,
                _ => return None,
            },
            Dot => match joint {
                Dot => DotDot,
                DotDot => DotDotDot,
                DotEq => DotDotEq,
                _ => return None,
            },
            DotDot => match joint {
                Dot => DotDotDot,
                Eq => DotDotEq,
                _ => return None,
            },
            Colon => match joint {
                Colon => ModSep,
                _ => return None,
            },

            Le | EqEq | Ne | Ge | AndAnd | OrOr | Tilde | BinOpEq(..) | At | DotDotDot | DotEq |
            DotDotEq | Comma | Semi | ModSep | RArrow | LArrow | FatArrow | Pound | Dollar |
            Question | OpenDelim(..) | CloseDelim(..) | Underscore => return None,

            Literal(..) | Ident(..) | Lifetime(..) | Interpolated(..) | DocComment(..) |
            Whitespace | Comment | Shebang(..) | Eof => return None,
        })
    }

    /// Returns tokens that are likely to be typed accidentally instead of the current token.
    /// Enables better error recovery when the wrong token is found.
    pub fn similar_tokens(&self) -> Option<Vec<Token>> {
        match *self {
            Comma => Some(vec![Dot, Lt]),
            Semi => Some(vec![Colon]),
            _ => None
        }
    }

    /// Returns `true` if the token is either a special identifier or a keyword.
    pub fn is_reserved_ident(&self) -> bool {
        self.is_special_ident() || self.is_used_keyword() || self.is_unused_keyword()
    }

    pub fn interpolated_to_tokenstream(&self, sess: &ParseSess, span: Span)
        -> TokenStream
    {
        let nt = match *self {
            Token::Interpolated(ref nt) => nt,
            _ => panic!("only works on interpolated tokens"),
        };

        // An `Interpolated` token means that we have a `Nonterminal`
        // which is often a parsed AST item. At this point we now need
        // to convert the parsed AST to an actual token stream, e.g.
        // un-parse it basically.
        //
        // Unfortunately there's not really a great way to do that in a
        // guaranteed lossless fashion right now. The fallback here is
        // to just stringify the AST node and reparse it, but this loses
        // all span information.
        //
        // As a result, some AST nodes are annotated with the token
        // stream they came from. Attempt to extract these lossless
        // token streams before we fall back to the stringification.
        let mut tokens = None;

        match nt.0 {
            Nonterminal::NtItem(ref item) => {
                tokens = prepend_attrs(sess, &item.attrs, item.tokens.as_ref(), span);
            }
            Nonterminal::NtTraitItem(ref item) => {
                tokens = prepend_attrs(sess, &item.attrs, item.tokens.as_ref(), span);
            }
            Nonterminal::NtImplItem(ref item) => {
                tokens = prepend_attrs(sess, &item.attrs, item.tokens.as_ref(), span);
            }
            Nonterminal::NtIdent(ident) => {
                let token = Token::Ident(ident.node);
                tokens = Some(TokenTree::Token(ident.span, token).into());
            }
            Nonterminal::NtLifetime(lifetime) => {
                let token = Token::Lifetime(lifetime.ident);
                tokens = Some(TokenTree::Token(lifetime.span, token).into());
            }
            Nonterminal::NtTT(ref tt) => {
                tokens = Some(tt.clone().into());
            }
            _ => {}
        }

        tokens.unwrap_or_else(|| {
            nt.1.force(|| {
                // FIXME(jseyfried): Avoid this pretty-print + reparse hack
                let source = pprust::token_to_string(self);
                parse_stream_from_source_str(FileName::MacroExpansion, source, sess, Some(span))
            })
        })
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
    NtVis(ast::Visibility),
    NtTT(TokenTree),
    // These are not exposed to macros, but are used by quasiquote.
    NtArm(ast::Arm),
    NtImplItem(ast::ImplItem),
    NtTraitItem(ast::TraitItem),
    NtGenerics(ast::Generics),
    NtWhereClause(ast::WhereClause),
    NtArg(ast::Arg),
    NtLifetime(ast::Lifetime),
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
            NtVis(..) => f.pad("NtVis(..)"),
            NtLifetime(..) => f.pad("NtLifetime(..)"),
        }
    }
}

pub fn is_op(tok: &Token) -> bool {
    match *tok {
        OpenDelim(..) | CloseDelim(..) | Literal(..) | DocComment(..) |
        Ident(..) | Underscore | Lifetime(..) | Interpolated(..) |
        Whitespace | Comment | Shebang(..) | Eof => false,
        _ => true,
    }
}

pub struct LazyTokenStream(Cell<Option<TokenStream>>);

impl Clone for LazyTokenStream {
    fn clone(&self) -> Self {
        let opt_stream = self.0.take();
        self.0.set(opt_stream.clone());
        LazyTokenStream(Cell::new(opt_stream))
    }
}

impl cmp::Eq for LazyTokenStream {}
impl PartialEq for LazyTokenStream {
    fn eq(&self, _other: &LazyTokenStream) -> bool {
        true
    }
}

impl fmt::Debug for LazyTokenStream {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.clone().0.into_inner(), f)
    }
}

impl LazyTokenStream {
    pub fn new() -> Self {
        LazyTokenStream(Cell::new(None))
    }

    pub fn force<F: FnOnce() -> TokenStream>(&self, f: F) -> TokenStream {
        let mut opt_stream = self.0.take();
        if opt_stream.is_none() {
            opt_stream = Some(f());
        }
        self.0.set(opt_stream.clone());
        opt_stream.clone().unwrap()
    }
}

impl Encodable for LazyTokenStream {
    fn encode<S: Encoder>(&self, _: &mut S) -> Result<(), S::Error> {
        Ok(())
    }
}

impl Decodable for LazyTokenStream {
    fn decode<D: Decoder>(_: &mut D) -> Result<LazyTokenStream, D::Error> {
        Ok(LazyTokenStream::new())
    }
}

impl ::std::hash::Hash for LazyTokenStream {
    fn hash<H: ::std::hash::Hasher>(&self, _hasher: &mut H) {}
}

fn prepend_attrs(sess: &ParseSess,
                 attrs: &[ast::Attribute],
                 tokens: Option<&tokenstream::TokenStream>,
                 span: syntax_pos::Span)
    -> Option<tokenstream::TokenStream>
{
    let tokens = tokens?;
    if attrs.len() == 0 {
        return Some(tokens.clone())
    }
    let mut builder = tokenstream::TokenStreamBuilder::new();
    for attr in attrs {
        assert_eq!(attr.style, ast::AttrStyle::Outer,
                   "inner attributes should prevent cached tokens from existing");
        // FIXME: Avoid this pretty-print + reparse hack as bove
        let name = FileName::MacroExpansion;
        let source = pprust::attr_to_string(attr);
        let stream = parse_stream_from_source_str(name, source, sess, Some(span));
        builder.push(stream);
    }
    builder.push(tokens.clone());
    Some(builder.build())
}
