pub use BinOpToken::*;
pub use DelimToken::*;
pub use LitKind::*;
pub use Nonterminal::*;
pub use TokenKind::*;

use crate::ast;
use crate::ptr::P;
use crate::tokenstream::TokenTree;

use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::sync::Lrc;
use rustc_macros::HashStable_Generic;
use rustc_span::hygiene::ExpnKind;
use rustc_span::source_map::SourceMap;
use rustc_span::symbol::{kw, sym};
use rustc_span::symbol::{Ident, Symbol};
use rustc_span::{self, FileName, RealFileName, Span, DUMMY_SP};
use std::borrow::Cow;
use std::{fmt, mem};

#[derive(Clone, Copy, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum CommentKind {
    Line,
    Block,
}

#[derive(Clone, PartialEq, Encodable, Decodable, Hash, Debug, Copy)]
#[derive(HashStable_Generic)]
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

/// A delimiter token.
#[derive(Clone, PartialEq, Eq, Encodable, Decodable, Hash, Debug, Copy)]
#[derive(HashStable_Generic)]
pub enum DelimToken {
    /// A round parenthesis (i.e., `(` or `)`).
    Paren,
    /// A square bracket (i.e., `[` or `]`).
    Bracket,
    /// A curly brace (i.e., `{` or `}`).
    Brace,
    /// An empty delimiter.
    NoDelim,
}

#[derive(Clone, Copy, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum LitKind {
    Bool, // AST only, must never appear in a `Token`
    Byte,
    Char,
    Integer,
    Float,
    Str,
    StrRaw(u16), // raw string delimited by `n` hash symbols
    ByteStr,
    ByteStrRaw(u16), // raw byte string delimited by `n` hash symbols
    Err,
}

/// A literal token.
#[derive(Clone, Copy, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub struct Lit {
    pub kind: LitKind,
    pub symbol: Symbol,
    pub suffix: Option<Symbol>,
}

impl fmt::Display for Lit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Lit { kind, symbol, suffix } = *self;
        match kind {
            Byte => write!(f, "b'{}'", symbol)?,
            Char => write!(f, "'{}'", symbol)?,
            Str => write!(f, "\"{}\"", symbol)?,
            StrRaw(n) => write!(
                f,
                "r{delim}\"{string}\"{delim}",
                delim = "#".repeat(n as usize),
                string = symbol
            )?,
            ByteStr => write!(f, "b\"{}\"", symbol)?,
            ByteStrRaw(n) => write!(
                f,
                "br{delim}\"{string}\"{delim}",
                delim = "#".repeat(n as usize),
                string = symbol
            )?,
            Integer | Float | Bool | Err => write!(f, "{}", symbol)?,
        }

        if let Some(suffix) = suffix {
            write!(f, "{}", suffix)?;
        }

        Ok(())
    }
}

impl LitKind {
    /// An English article for the literal token kind.
    pub fn article(self) -> &'static str {
        match self {
            Integer | Err => "an",
            _ => "a",
        }
    }

    pub fn descr(self) -> &'static str {
        match self {
            Bool => panic!("literal token contains `Lit::Bool`"),
            Byte => "byte",
            Char => "char",
            Integer => "integer",
            Float => "float",
            Str | StrRaw(..) => "string",
            ByteStr | ByteStrRaw(..) => "byte string",
            Err => "error",
        }
    }

    crate fn may_have_suffix(self) -> bool {
        matches!(self, Integer | Float | Err)
    }
}

impl Lit {
    pub fn new(kind: LitKind, symbol: Symbol, suffix: Option<Symbol>) -> Lit {
        Lit { kind, symbol, suffix }
    }
}

pub fn ident_can_begin_expr(name: Symbol, span: Span, is_raw: bool) -> bool {
    let ident_token = Token::new(Ident(name, is_raw), span);

    !ident_token.is_reserved_ident()
        || ident_token.is_path_segment_keyword()
        || [
            kw::Async,
            kw::Do,
            kw::Box,
            kw::Break,
            kw::Const,
            kw::Continue,
            kw::False,
            kw::For,
            kw::If,
            kw::Let,
            kw::Loop,
            kw::Match,
            kw::Move,
            kw::Return,
            kw::True,
            kw::Try,
            kw::Unsafe,
            kw::While,
            kw::Yield,
            kw::Static,
        ]
        .contains(&name)
}

fn ident_can_begin_type(name: Symbol, span: Span, is_raw: bool) -> bool {
    let ident_token = Token::new(Ident(name, is_raw), span);

    !ident_token.is_reserved_ident()
        || ident_token.is_path_segment_keyword()
        || [kw::Underscore, kw::For, kw::Impl, kw::Fn, kw::Unsafe, kw::Extern, kw::Typeof, kw::Dyn]
            .contains(&name)
}

#[derive(Clone, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum TokenKind {
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
    /// Used by proc macros for representing lifetimes, not generated by lexer right now.
    SingleQuote,
    /// An opening delimiter (e.g., `{`).
    OpenDelim(DelimToken),
    /// A closing delimiter (e.g., `}`).
    CloseDelim(DelimToken),

    /* Literals */
    Literal(Lit),

    /// Identifier token.
    /// Do not forget about `NtIdent` when you want to match on identifiers.
    /// It's recommended to use `Token::(ident,uninterpolate,uninterpolated_span)` to
    /// treat regular and interpolated identifiers in the same way.
    Ident(Symbol, /* is_raw */ bool),
    /// Lifetime identifier token.
    /// Do not forget about `NtLifetime` when you want to match on lifetime identifiers.
    /// It's recommended to use `Token::(lifetime,uninterpolate,uninterpolated_span)` to
    /// treat regular and interpolated lifetime identifiers in the same way.
    Lifetime(Symbol),

    Interpolated(Lrc<Nonterminal>),

    /// A doc comment token.
    /// `Symbol` is the doc comment's data excluding its "quotes" (`///`, `/**`, etc)
    /// similarly to symbols in string literal tokens.
    DocComment(CommentKind, ast::AttrStyle, Symbol),

    Eof,
}

// `TokenKind` is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(target_arch = "x86_64")]
rustc_data_structures::static_assert_size!(TokenKind, 16);

#[derive(Clone, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

impl TokenKind {
    pub fn lit(kind: LitKind, symbol: Symbol, suffix: Option<Symbol>) -> TokenKind {
        Literal(Lit::new(kind, symbol, suffix))
    }

    // An approximation to proc-macro-style single-character operators used by rustc parser.
    // If the operator token can be broken into two tokens, the first of which is single-character,
    // then this function performs that operation, otherwise it returns `None`.
    pub fn break_two_token_op(&self) -> Option<(TokenKind, TokenKind)> {
        Some(match *self {
            Le => (Lt, Eq),
            EqEq => (Eq, Eq),
            Ne => (Not, Eq),
            Ge => (Gt, Eq),
            AndAnd => (BinOp(And), BinOp(And)),
            OrOr => (BinOp(Or), BinOp(Or)),
            BinOp(Shl) => (Lt, Lt),
            BinOp(Shr) => (Gt, Gt),
            BinOpEq(Plus) => (BinOp(Plus), Eq),
            BinOpEq(Minus) => (BinOp(Minus), Eq),
            BinOpEq(Star) => (BinOp(Star), Eq),
            BinOpEq(Slash) => (BinOp(Slash), Eq),
            BinOpEq(Percent) => (BinOp(Percent), Eq),
            BinOpEq(Caret) => (BinOp(Caret), Eq),
            BinOpEq(And) => (BinOp(And), Eq),
            BinOpEq(Or) => (BinOp(Or), Eq),
            BinOpEq(Shl) => (Lt, Le),
            BinOpEq(Shr) => (Gt, Ge),
            DotDot => (Dot, Dot),
            DotDotDot => (Dot, DotDot),
            ModSep => (Colon, Colon),
            RArrow => (BinOp(Minus), Gt),
            LArrow => (Lt, BinOp(Minus)),
            FatArrow => (Eq, Gt),
            _ => return None,
        })
    }

    /// Returns tokens that are likely to be typed accidentally instead of the current token.
    /// Enables better error recovery when the wrong token is found.
    pub fn similar_tokens(&self) -> Option<Vec<TokenKind>> {
        match *self {
            Comma => Some(vec![Dot, Lt, Semi]),
            Semi => Some(vec![Colon, Comma]),
            _ => None,
        }
    }

    pub fn should_end_const_arg(&self) -> bool {
        matches!(self, Gt | Ge | BinOp(Shr) | BinOpEq(Shr))
    }
}

impl Token {
    pub fn new(kind: TokenKind, span: Span) -> Self {
        Token { kind, span }
    }

    /// Some token that will be thrown away later.
    pub fn dummy() -> Self {
        Token::new(TokenKind::Question, DUMMY_SP)
    }

    /// Recovers a `Token` from an `Ident`. This creates a raw identifier if necessary.
    pub fn from_ast_ident(ident: Ident) -> Self {
        Token::new(Ident(ident.name, ident.is_raw_guess()), ident.span)
    }

    /// Return this token by value and leave a dummy token in its place.
    pub fn take(&mut self) -> Self {
        mem::replace(self, Token::dummy())
    }

    /// For interpolated tokens, returns a span of the fragment to which the interpolated
    /// token refers. For all other tokens this is just a regular span.
    /// It is particularly important to use this for identifiers and lifetimes
    /// for which spans affect name resolution and edition checks.
    /// Note that keywords are also identifiers, so they should use this
    /// if they keep spans or perform edition checks.
    pub fn uninterpolated_span(&self) -> Span {
        match &self.kind {
            Interpolated(nt) => nt.span(),
            _ => self.span,
        }
    }

    pub fn is_op(&self) -> bool {
        !matches!(
            self.kind,
            OpenDelim(..)
                | CloseDelim(..)
                | Literal(..)
                | DocComment(..)
                | Ident(..)
                | Lifetime(..)
                | Interpolated(..)
                | Eof
        )
    }

    pub fn is_like_plus(&self) -> bool {
        matches!(self.kind, BinOp(Plus) | BinOpEq(Plus))
    }

    /// Returns `true` if the token can appear at the start of an expression.
    pub fn can_begin_expr(&self) -> bool {
        match self.uninterpolate().kind {
            Ident(name, is_raw)              =>
                ident_can_begin_expr(name, self.span, is_raw), // value name or keyword
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
            Lifetime(..)                      | // labeled loop
            Pound                             => true, // expression attributes
            Interpolated(ref nt) => matches!(**nt, NtLiteral(..) |
                NtExpr(..)    |
                NtBlock(..)   |
                NtPath(..)),
            _ => false,
        }
    }

    /// Returns `true` if the token can appear at the start of a type.
    pub fn can_begin_type(&self) -> bool {
        match self.uninterpolate().kind {
            Ident(name, is_raw)        =>
                ident_can_begin_type(name, self.span, is_raw), // type name or keyword
            OpenDelim(Paren)            | // tuple
            OpenDelim(Bracket)          | // array
            Not                         | // never
            BinOp(Star)                 | // raw pointer
            BinOp(And)                  | // reference
            AndAnd                      | // double reference
            Question                    | // maybe bound in trait object
            Lifetime(..)                | // lifetime bound in trait object
            Lt | BinOp(Shl)             | // associated path
            ModSep                      => true, // global path
            Interpolated(ref nt) => matches!(**nt, NtTy(..) | NtPath(..)),
            _ => false,
        }
    }

    /// Returns `true` if the token can appear at the start of a const param.
    pub fn can_begin_const_arg(&self) -> bool {
        match self.kind {
            OpenDelim(Brace) => true,
            Interpolated(ref nt) => matches!(**nt, NtExpr(..) | NtBlock(..) | NtLiteral(..)),
            _ => self.can_begin_literal_maybe_minus(),
        }
    }

    /// Returns `true` if the token can appear at the start of a generic bound.
    pub fn can_begin_bound(&self) -> bool {
        self.is_path_start()
            || self.is_lifetime()
            || self.is_keyword(kw::For)
            || self == &Question
            || self == &OpenDelim(Paren)
    }

    /// Returns `true` if the token is any literal.
    pub fn is_lit(&self) -> bool {
        matches!(self.kind, Literal(..))
    }

    /// Returns `true` if the token is any literal, a minus (which can prefix a literal,
    /// for example a '-42', or one of the boolean idents).
    ///
    /// In other words, would this token be a valid start of `parse_literal_maybe_minus`?
    ///
    /// Keep this in sync with and `Lit::from_token`, excluding unary negation.
    pub fn can_begin_literal_maybe_minus(&self) -> bool {
        match self.uninterpolate().kind {
            Literal(..) | BinOp(Minus) => true,
            Ident(name, false) if name.is_bool_lit() => true,
            Interpolated(ref nt) => match &**nt {
                NtLiteral(_) => true,
                NtExpr(e) => match &e.kind {
                    ast::ExprKind::Lit(_) => true,
                    ast::ExprKind::Unary(ast::UnOp::Neg, e) => {
                        matches!(&e.kind, ast::ExprKind::Lit(_))
                    }
                    _ => false,
                },
                _ => false,
            },
            _ => false,
        }
    }

    // A convenience function for matching on identifiers during parsing.
    // Turns interpolated identifier (`$i: ident`) or lifetime (`$l: lifetime`) token
    // into the regular identifier or lifetime token it refers to,
    // otherwise returns the original token.
    pub fn uninterpolate(&self) -> Cow<'_, Token> {
        match &self.kind {
            Interpolated(nt) => match **nt {
                NtIdent(ident, is_raw) => {
                    Cow::Owned(Token::new(Ident(ident.name, is_raw), ident.span))
                }
                NtLifetime(ident) => Cow::Owned(Token::new(Lifetime(ident.name), ident.span)),
                _ => Cow::Borrowed(self),
            },
            _ => Cow::Borrowed(self),
        }
    }

    /// Returns an identifier if this token is an identifier.
    pub fn ident(&self) -> Option<(Ident, /* is_raw */ bool)> {
        let token = self.uninterpolate();
        match token.kind {
            Ident(name, is_raw) => Some((Ident::new(name, token.span), is_raw)),
            _ => None,
        }
    }

    /// Returns a lifetime identifier if this token is a lifetime.
    pub fn lifetime(&self) -> Option<Ident> {
        let token = self.uninterpolate();
        match token.kind {
            Lifetime(name) => Some(Ident::new(name, token.span)),
            _ => None,
        }
    }

    /// Returns `true` if the token is an identifier.
    pub fn is_ident(&self) -> bool {
        self.ident().is_some()
    }

    /// Returns `true` if the token is a lifetime.
    pub fn is_lifetime(&self) -> bool {
        self.lifetime().is_some()
    }

    /// Returns `true` if the token is a identifier whose name is the given
    /// string slice.
    pub fn is_ident_named(&self, name: Symbol) -> bool {
        self.ident().map_or(false, |(ident, _)| ident.name == name)
    }

    /// Returns `true` if the token is an interpolated path.
    fn is_path(&self) -> bool {
        if let Interpolated(ref nt) = self.kind {
            if let NtPath(..) = **nt {
                return true;
            }
        }
        false
    }

    /// Would `maybe_whole_expr` in `parser.rs` return `Ok(..)`?
    /// That is, is this a pre-parsed expression dropped into the token stream
    /// (which happens while parsing the result of macro expansion)?
    pub fn is_whole_expr(&self) -> bool {
        if let Interpolated(ref nt) = self.kind {
            if let NtExpr(_) | NtLiteral(_) | NtPath(_) | NtIdent(..) | NtBlock(_) = **nt {
                return true;
            }
        }

        false
    }

    // Is the token an interpolated block (`$b:block`)?
    pub fn is_whole_block(&self) -> bool {
        if let Interpolated(ref nt) = self.kind {
            if let NtBlock(..) = **nt {
                return true;
            }
        }
        false
    }

    /// Returns `true` if the token is either the `mut` or `const` keyword.
    pub fn is_mutability(&self) -> bool {
        self.is_keyword(kw::Mut) || self.is_keyword(kw::Const)
    }

    pub fn is_qpath_start(&self) -> bool {
        self == &Lt || self == &BinOp(Shl)
    }

    pub fn is_path_start(&self) -> bool {
        self == &ModSep
            || self.is_qpath_start()
            || self.is_path()
            || self.is_path_segment_keyword()
            || self.is_ident() && !self.is_reserved_ident()
    }

    /// Returns `true` if the token is a given keyword, `kw`.
    pub fn is_keyword(&self, kw: Symbol) -> bool {
        self.is_non_raw_ident_where(|id| id.name == kw)
    }

    pub fn is_path_segment_keyword(&self) -> bool {
        self.is_non_raw_ident_where(Ident::is_path_segment_keyword)
    }

    // Returns true for reserved identifiers used internally for elided lifetimes,
    // unnamed method parameters, crate root module, error recovery etc.
    pub fn is_special_ident(&self) -> bool {
        self.is_non_raw_ident_where(Ident::is_special)
    }

    /// Returns `true` if the token is a keyword used in the language.
    pub fn is_used_keyword(&self) -> bool {
        self.is_non_raw_ident_where(Ident::is_used_keyword)
    }

    /// Returns `true` if the token is a keyword reserved for possible future use.
    pub fn is_unused_keyword(&self) -> bool {
        self.is_non_raw_ident_where(Ident::is_unused_keyword)
    }

    /// Returns `true` if the token is either a special identifier or a keyword.
    pub fn is_reserved_ident(&self) -> bool {
        self.is_non_raw_ident_where(Ident::is_reserved)
    }

    /// Returns `true` if the token is the identifier `true` or `false`.
    pub fn is_bool_lit(&self) -> bool {
        self.is_non_raw_ident_where(|id| id.name.is_bool_lit())
    }

    /// Returns `true` if the token is a non-raw identifier for which `pred` holds.
    pub fn is_non_raw_ident_where(&self, pred: impl FnOnce(Ident) -> bool) -> bool {
        match self.ident() {
            Some((id, false)) => pred(id),
            _ => false,
        }
    }

    pub fn glue(&self, joint: &Token) -> Option<Token> {
        let kind = match self.kind {
            Eq => match joint.kind {
                Eq => EqEq,
                Gt => FatArrow,
                _ => return None,
            },
            Lt => match joint.kind {
                Eq => Le,
                Lt => BinOp(Shl),
                Le => BinOpEq(Shl),
                BinOp(Minus) => LArrow,
                _ => return None,
            },
            Gt => match joint.kind {
                Eq => Ge,
                Gt => BinOp(Shr),
                Ge => BinOpEq(Shr),
                _ => return None,
            },
            Not => match joint.kind {
                Eq => Ne,
                _ => return None,
            },
            BinOp(op) => match joint.kind {
                Eq => BinOpEq(op),
                BinOp(And) if op == And => AndAnd,
                BinOp(Or) if op == Or => OrOr,
                Gt if op == Minus => RArrow,
                _ => return None,
            },
            Dot => match joint.kind {
                Dot => DotDot,
                DotDot => DotDotDot,
                _ => return None,
            },
            DotDot => match joint.kind {
                Dot => DotDotDot,
                Eq => DotDotEq,
                _ => return None,
            },
            Colon => match joint.kind {
                Colon => ModSep,
                _ => return None,
            },
            SingleQuote => match joint.kind {
                Ident(name, false) => Lifetime(Symbol::intern(&format!("'{}", name))),
                _ => return None,
            },

            Le | EqEq | Ne | Ge | AndAnd | OrOr | Tilde | BinOpEq(..) | At | DotDotDot
            | DotDotEq | Comma | Semi | ModSep | RArrow | LArrow | FatArrow | Pound | Dollar
            | Question | OpenDelim(..) | CloseDelim(..) | Literal(..) | Ident(..)
            | Lifetime(..) | Interpolated(..) | DocComment(..) | Eof => return None,
        };

        Some(Token::new(kind, self.span.to(joint.span)))
    }
}

impl PartialEq<TokenKind> for Token {
    fn eq(&self, rhs: &TokenKind) -> bool {
        self.kind == *rhs
    }
}

#[derive(Clone, Encodable, Decodable)]
/// For interpolation during macro expansion.
pub enum Nonterminal {
    NtItem(P<ast::Item>),
    NtBlock(P<ast::Block>),
    NtStmt(ast::Stmt),
    NtPat(P<ast::Pat>),
    NtExpr(P<ast::Expr>),
    NtTy(P<ast::Ty>),
    NtIdent(Ident, /* is_raw */ bool),
    NtLifetime(Ident),
    NtLiteral(P<ast::Expr>),
    /// Stuff inside brackets for attributes
    NtMeta(P<ast::AttrItem>),
    NtPath(ast::Path),
    NtVis(ast::Visibility),
    NtTT(TokenTree),
}

// `Nonterminal` is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(target_arch = "x86_64")]
rustc_data_structures::static_assert_size!(Nonterminal, 48);

#[derive(Debug, Copy, Clone, PartialEq, Encodable, Decodable)]
pub enum NonterminalKind {
    Item,
    Block,
    Stmt,
    Pat,
    Expr,
    Ty,
    Ident,
    Lifetime,
    Literal,
    Meta,
    Path,
    Vis,
    TT,
}

impl NonterminalKind {
    pub fn from_symbol(symbol: Symbol) -> Option<NonterminalKind> {
        Some(match symbol {
            sym::item => NonterminalKind::Item,
            sym::block => NonterminalKind::Block,
            sym::stmt => NonterminalKind::Stmt,
            sym::pat => NonterminalKind::Pat,
            sym::expr => NonterminalKind::Expr,
            sym::ty => NonterminalKind::Ty,
            sym::ident => NonterminalKind::Ident,
            sym::lifetime => NonterminalKind::Lifetime,
            sym::literal => NonterminalKind::Literal,
            sym::meta => NonterminalKind::Meta,
            sym::path => NonterminalKind::Path,
            sym::vis => NonterminalKind::Vis,
            sym::tt => NonterminalKind::TT,
            _ => return None,
        })
    }
    fn symbol(self) -> Symbol {
        match self {
            NonterminalKind::Item => sym::item,
            NonterminalKind::Block => sym::block,
            NonterminalKind::Stmt => sym::stmt,
            NonterminalKind::Pat => sym::pat,
            NonterminalKind::Expr => sym::expr,
            NonterminalKind::Ty => sym::ty,
            NonterminalKind::Ident => sym::ident,
            NonterminalKind::Lifetime => sym::lifetime,
            NonterminalKind::Literal => sym::literal,
            NonterminalKind::Meta => sym::meta,
            NonterminalKind::Path => sym::path,
            NonterminalKind::Vis => sym::vis,
            NonterminalKind::TT => sym::tt,
        }
    }
}

impl fmt::Display for NonterminalKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.symbol())
    }
}

impl Nonterminal {
    fn span(&self) -> Span {
        match self {
            NtItem(item) => item.span,
            NtBlock(block) => block.span,
            NtStmt(stmt) => stmt.span,
            NtPat(pat) => pat.span,
            NtExpr(expr) | NtLiteral(expr) => expr.span,
            NtTy(ty) => ty.span,
            NtIdent(ident, _) | NtLifetime(ident) => ident.span,
            NtMeta(attr_item) => attr_item.span(),
            NtPath(path) => path.span,
            NtVis(vis) => vis.span,
            NtTT(tt) => tt.span(),
        }
    }

    /// This nonterminal looks like some specific enums from
    /// `proc-macro-hack` and `procedural-masquerade` crates.
    /// We need to maintain some special pretty-printing behavior for them due to incorrect
    /// asserts in old versions of those crates and their wide use in the ecosystem.
    /// See issue #73345 for more details.
    /// FIXME(#73933): Remove this eventually.
    pub fn pretty_printing_compatibility_hack(&self) -> bool {
        let item = match self {
            NtItem(item) => item,
            NtStmt(stmt) => match &stmt.kind {
                ast::StmtKind::Item(item) => item,
                _ => return false,
            },
            _ => return false,
        };

        let name = item.ident.name;
        if name == sym::ProceduralMasqueradeDummyType || name == sym::ProcMacroHack {
            if let ast::ItemKind::Enum(enum_def, _) = &item.kind {
                if let [variant] = &*enum_def.variants {
                    return variant.ident.name == sym::Input;
                }
            }
        }
        false
    }

    // See issue #74616 for details
    pub fn ident_name_compatibility_hack(
        &self,
        orig_span: Span,
        source_map: &SourceMap,
    ) -> Option<(Ident, bool)> {
        if let NtIdent(ident, is_raw) = self {
            if let ExpnKind::Macro(_, macro_name) = orig_span.ctxt().outer_expn_data().kind {
                let filename = source_map.span_to_filename(orig_span);
                if let FileName::Real(RealFileName::Named(path)) = filename {
                    let matches_prefix = |prefix, filename| {
                        // Check for a path that ends with 'prefix*/src/<filename>'
                        let mut iter = path.components().rev();
                        iter.next().and_then(|p| p.as_os_str().to_str()) == Some(filename)
                            && iter.next().and_then(|p| p.as_os_str().to_str()) == Some("src")
                            && iter
                                .next()
                                .and_then(|p| p.as_os_str().to_str())
                                .map_or(false, |p| p.starts_with(prefix))
                    };

                    if (macro_name == sym::impl_macros
                        && matches_prefix("time-macros-impl", "lib.rs"))
                        || (macro_name == sym::arrays && matches_prefix("js-sys", "lib.rs"))
                    {
                        let snippet = source_map.span_to_snippet(orig_span);
                        if snippet.as_deref() == Ok("$name") {
                            return Some((*ident, *is_raw));
                        }
                    }

                    if macro_name == sym::tuple_from_req
                        && (matches_prefix("actix-web", "extract.rs")
                            || matches_prefix("actori-web", "extract.rs"))
                    {
                        let snippet = source_map.span_to_snippet(orig_span);
                        if snippet.as_deref() == Ok("$T") {
                            return Some((*ident, *is_raw));
                        }
                    }
                }
            }
        }
        None
    }
}

impl PartialEq for Nonterminal {
    fn eq(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (NtIdent(ident_lhs, is_raw_lhs), NtIdent(ident_rhs, is_raw_rhs)) => {
                ident_lhs == ident_rhs && is_raw_lhs == is_raw_rhs
            }
            (NtLifetime(ident_lhs), NtLifetime(ident_rhs)) => ident_lhs == ident_rhs,
            (NtTT(tt_lhs), NtTT(tt_rhs)) => tt_lhs == tt_rhs,
            // FIXME: Assume that all "complex" nonterminal are not equal, we can't compare them
            // correctly based on data from AST. This will prevent them from matching each other
            // in macros. The comparison will become possible only when each nonterminal has an
            // attached token stream from which it was parsed.
            _ => false,
        }
    }
}

impl fmt::Debug for Nonterminal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            NtItem(..) => f.pad("NtItem(..)"),
            NtBlock(..) => f.pad("NtBlock(..)"),
            NtStmt(..) => f.pad("NtStmt(..)"),
            NtPat(..) => f.pad("NtPat(..)"),
            NtExpr(..) => f.pad("NtExpr(..)"),
            NtTy(..) => f.pad("NtTy(..)"),
            NtIdent(..) => f.pad("NtIdent(..)"),
            NtLiteral(..) => f.pad("NtLiteral(..)"),
            NtMeta(..) => f.pad("NtMeta(..)"),
            NtPath(..) => f.pad("NtPath(..)"),
            NtTT(..) => f.pad("NtTT(..)"),
            NtVis(..) => f.pad("NtVis(..)"),
            NtLifetime(..) => f.pad("NtLifetime(..)"),
        }
    }
}

impl<CTX> HashStable<CTX> for Nonterminal
where
    CTX: crate::HashStableContext,
{
    fn hash_stable(&self, _hcx: &mut CTX, _hasher: &mut StableHasher) {
        panic!("interpolated tokens should not be present in the HIR")
    }
}
