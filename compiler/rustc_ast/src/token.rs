pub use BinOpToken::*;
pub use LitKind::*;
pub use TokenKind::*;

use crate::ast;
use crate::util::case::Case;

use rustc_macros::HashStable_Generic;
use rustc_span::symbol::{kw, sym};
#[allow(hidden_glob_reexports)]
use rustc_span::symbol::{Ident, Symbol};
use rustc_span::{self, edition::Edition, Span, DUMMY_SP};
use std::borrow::Cow;
use std::fmt;

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

/// Describes how a sequence of token trees is delimited.
/// Cannot use `proc_macro::Delimiter` directly because this
/// structure should implement some additional traits.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[derive(Encodable, Decodable, Hash, HashStable_Generic)]
pub enum Delimiter {
    /// `( ... )`
    Parenthesis,
    /// `{ ... }`
    Brace,
    /// `[ ... ]`
    Bracket,
    /// `Ø ... Ø`
    /// An invisible delimiter, that may, for example, appear around tokens coming from a
    /// "macro variable" `$var`. It is important to preserve operator priorities in cases like
    /// `$var * 3` where `$var` is `1 + 2`.
    /// Invisible delimiters might not survive roundtrip of a token stream through a string.
    Invisible(InvisibleSource),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Encodable, Decodable, Hash, HashStable_Generic)]
pub enum InvisibleSource {
    // From the expansion of a metavariable in a declarative macro.
    MetaVar(NonterminalKind),

    // Converted from `proc_macro::Delimiter` in
    // `proc_macro::Delimiter::to_internal`, i.e. returned by a proc macro.
    ProcMacro,
}

// Note that the suffix is *not* considered when deciding the `LitKind` in this
// type. This means that float literals like `1f32` are classified by this type
// as `Int`. Only upon conversion to `ast::LitKind` will such a literal be
// given the `Float` kind.
#[derive(Clone, Copy, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum LitKind {
    Bool, // AST only, must never appear in a `Token`
    Byte,
    Char,
    Integer, // e.g. `1`, `1u8`, `1f32`
    Float,   // e.g. `1.`, `1.0`, `1e3f32`
    Str,
    StrRaw(u8), // raw string delimited by `n` hash symbols
    ByteStr,
    ByteStrRaw(u8), // raw byte string delimited by `n` hash symbols
    CStr,
    CStrRaw(u8),
    Err,
}

/// A literal token.
#[derive(Clone, Copy, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub struct Lit {
    pub kind: LitKind,
    pub symbol: Symbol,
    pub suffix: Option<Symbol>,
}

impl Lit {
    pub fn new(kind: LitKind, symbol: Symbol, suffix: Option<Symbol>) -> Lit {
        Lit { kind, symbol, suffix }
    }

    /// Returns `true` if this is semantically a float literal. This includes
    /// ones like `1f32` that have an `Integer` kind but a float suffix.
    pub fn is_semantic_float(&self) -> bool {
        match self.kind {
            LitKind::Float => true,
            LitKind::Integer => match self.suffix {
                Some(sym) => sym == sym::f32 || sym == sym::f64,
                None => false,
            },
            _ => false,
        }
    }

    /// Keep this in sync with `Token::can_begin_literal_maybe_minus` and
    /// `Parser::maybe_parse_token_lit` (excluding unary negation).
    pub fn from_token(token: &Token) -> Option<Lit> {
        match token.uninterpolate().kind {
            Ident(name, false) if name.is_bool_lit() => Some(Lit::new(Bool, name, None)),
            Literal(token_lit) => Some(token_lit),
            OpenDelim(Delimiter::Invisible(source)) => {
                panic!("njn: from_token {source:?}");
            }
            _ => None,
        }
    }
}

impl fmt::Display for Lit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Lit { kind, symbol, suffix } = *self;
        match kind {
            Byte => write!(f, "b'{symbol}'")?,
            Char => write!(f, "'{symbol}'")?,
            Str => write!(f, "\"{symbol}\"")?,
            StrRaw(n) => write!(
                f,
                "r{delim}\"{string}\"{delim}",
                delim = "#".repeat(n as usize),
                string = symbol
            )?,
            ByteStr => write!(f, "b\"{symbol}\"")?,
            ByteStrRaw(n) => write!(
                f,
                "br{delim}\"{string}\"{delim}",
                delim = "#".repeat(n as usize),
                string = symbol
            )?,
            CStr => write!(f, "c\"{symbol}\"")?,
            CStrRaw(n) => {
                write!(f, "cr{delim}\"{symbol}\"{delim}", delim = "#".repeat(n as usize))?
            }
            Integer | Float | Bool | Err => write!(f, "{symbol}")?,
        }

        if let Some(suffix) = suffix {
            write!(f, "{suffix}")?;
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
            CStr | CStrRaw(..) => "C string",
            Err => "error",
        }
    }

    pub(crate) fn may_have_suffix(self) -> bool {
        matches!(self, Integer | Float | Err)
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

#[derive(Clone, Copy, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
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
    OpenDelim(Delimiter),
    /// A closing delimiter (e.g., `}`).
    CloseDelim(Delimiter),

    /* Literals */
    Literal(Lit),

    /// Identifier token.
    /// Do not forget about `InterpolatedIdent` when you want to match on identifiers.
    /// It's recommended to use `Token::(ident,uninterpolate,uninterpolated_span)` to
    /// treat regular and interpolated identifiers in the same way.
    Ident(Symbol, /* is_raw */ bool),
    /// This `Span` is the span of the original identifier passed to the
    /// declarative macro. The span in the `Token` is the span of the `ident`
    /// metavariable in the macro's RHS.
    InterpolatedIdent(Symbol, /* is_raw */ bool, Span),
    /// Lifetime identifier token.
    /// Do not forget about `InterpolatedLIfetime` when you want to match on lifetime identifiers.
    /// It's recommended to use `Token::(lifetime,uninterpolate,uninterpolated_span)` to
    /// treat regular and interpolated lifetime identifiers in the same way.
    Lifetime(Symbol),
    /// This `Span` is the span of the original lifetime passed to the
    /// declarative macro. The span in the `Token` is the span of the
    /// `lifetime` metavariable in the macro's RHS.
    InterpolatedLifetime(Symbol, Span),

    /// A doc comment token.
    /// `Symbol` is the doc comment's data excluding its "quotes" (`///`, `/**`, etc)
    /// similarly to symbols in string literal tokens.
    DocComment(CommentKind, ast::AttrStyle, Symbol),

    Eof,
}

#[derive(Clone, Copy, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

impl TokenKind {
    pub fn lit(kind: LitKind, symbol: Symbol, suffix: Option<Symbol>) -> TokenKind {
        Literal(Lit::new(kind, symbol, suffix))
    }

    /// An approximation to proc-macro-style single-character operators used by rustc parser.
    /// If the operator token can be broken into two tokens, the first of which is single-character,
    /// then this function performs that operation, otherwise it returns `None`.
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
            FatArrow => Some(vec![Eq, RArrow]),
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

    /// njn: phase this out in favour of Parser::uninterpolated_span
    /// For interpolated tokens, returns a span of the fragment to which the interpolated
    /// token refers. For all other tokens this is just a regular span.
    /// It is particularly important to use this for identifiers and lifetimes
    /// for which spans affect name resolution and edition checks.
    /// Note that keywords are also identifiers, so they should use this
    /// if they keep spans or perform edition checks.
    pub fn uninterpolated_span(&self) -> Span {
        match self.kind {
            InterpolatedIdent(_, _, uninterpolated_span)
            | InterpolatedLifetime(_, uninterpolated_span) => uninterpolated_span,
            OpenDelim(Delimiter::Invisible(source)) => {
                panic!("njn: uninterpolated_span {source:?}");
            }
            _ => self.span,
        }
    }

    pub fn is_range_separator(&self) -> bool {
        [DotDot, DotDotDot, DotDotEq].contains(&self.kind)
    }

    pub fn is_op(&self) -> bool {
        match self.kind {
            Eq | Lt | Le | EqEq | Ne | Ge | Gt | AndAnd | OrOr | Not | Tilde | BinOp(_)
            | BinOpEq(_) | At | Dot | DotDot | DotDotDot | DotDotEq | Comma | Semi | Colon
            | ModSep | RArrow | LArrow | FatArrow | Pound | Dollar | Question | SingleQuote => true,

            OpenDelim(..)
            | CloseDelim(..)
            | Literal(..)
            | DocComment(..)
            | Ident(..)
            | InterpolatedIdent(..)
            | Lifetime(..)
            | InterpolatedLifetime(..)
            | Eof => false,
        }
    }

    pub fn is_like_plus(&self) -> bool {
        matches!(self.kind, BinOp(Plus) | BinOpEq(Plus))
    }

    /// Returns `true` if the token can appear at the start of an expression.
    pub fn can_begin_expr(&self) -> bool {
        use Delimiter::*;
        match self.uninterpolate().kind {
            Ident(name, is_raw)              =>
                ident_can_begin_expr(name, self.span, is_raw), // value name or keyword
            OpenDelim(Parenthesis | Brace | Bracket) | // tuple, array or block
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
            OpenDelim(Delimiter::Invisible(InvisibleSource::MetaVar(
                NonterminalKind::Block |
                NonterminalKind::Expr |
                NonterminalKind::Literal |
                NonterminalKind::Path
            ))) |
            OpenDelim(Delimiter::Invisible(InvisibleSource::ProcMacro)) => true,
            _ => false,
        }
    }

    /// Returns `true` if the token can appear at the start of a pattern.
    ///
    /// Shamelessly borrowed from `can_begin_expr`, only used for diagnostics right now.
    pub fn can_begin_pattern(&self) -> bool {
        match self.uninterpolate().kind {
            Ident(name, is_raw) =>
                ident_can_begin_expr(name, self.span, is_raw), // value name or keyword
            | OpenDelim(Delimiter::Bracket | Delimiter::Parenthesis)  // tuple or array
            | Literal(..)                        // literal
            | BinOp(Minus)                       // unary minus
            | BinOp(And)                         // reference
            | AndAnd                             // double reference
            // DotDotDot is no longer supported
            | DotDot | DotDotDot | DotDotEq      // ranges
            | Lt | BinOp(Shl)                    // associated path
            | ModSep => true,                    // global path
            | OpenDelim(Delimiter::Invisible(InvisibleSource::MetaVar(
                NonterminalKind::Block |
                NonterminalKind::PatParam { .. } |
                NonterminalKind::PatWithOr |
                NonterminalKind::Path |
                NonterminalKind::Literal
            ))) |
            OpenDelim(Delimiter::Invisible(InvisibleSource::ProcMacro)) => true,
            _ => false,
        }
    }

    /// Returns `true` if the token can appear at the start of a type.
    pub fn can_begin_type(&self) -> bool {
        match self.uninterpolate().kind {
            Ident(name, is_raw)        =>
                ident_can_begin_type(name, self.span, is_raw), // type name or keyword
            OpenDelim(Delimiter::Parenthesis) | // tuple
            OpenDelim(Delimiter::Bracket)     | // array
            Not             | // never
            BinOp(Star)     | // raw pointer
            BinOp(And)      | // reference
            AndAnd          | // double reference
            Question        | // maybe bound in trait object
            Lifetime(..)    | // lifetime bound in trait object
            Lt | BinOp(Shl) | // associated path
            ModSep          => true, // global path
            OpenDelim(Delimiter::Invisible(InvisibleSource::MetaVar(
                NonterminalKind::Ty |
                NonterminalKind::Path
            ))) |
            OpenDelim(Delimiter::Invisible(InvisibleSource::ProcMacro)) => true,
            _ => false,
        }
    }

    /// Returns `true` if the token can appear at the start of a const param.
    pub fn can_begin_const_arg(&self) -> bool {
        match self.kind {
            OpenDelim(Delimiter::Brace) => true,
            OpenDelim(Delimiter::Invisible(InvisibleSource::MetaVar(
                NonterminalKind::Block | NonterminalKind::Expr | NonterminalKind::Literal,
            ))) => true,
            OpenDelim(Delimiter::Invisible(InvisibleSource::ProcMacro)) => true,
            _ => self.can_begin_literal_maybe_minus(),
        }
    }

    /// Returns `true` if the token can appear at the start of a generic bound.
    pub fn can_begin_bound(&self) -> bool {
        self.is_path_start()
            || self.is_lifetime()
            || self.is_keyword(kw::For)
            || self == &Question
            || self == &OpenDelim(Delimiter::Parenthesis)
    }

    /// Returns `true` if the token can appear at the start of an item.
    pub fn can_begin_item(&self) -> bool {
        match self.kind {
            Ident(name, _) => [
                kw::Fn,
                kw::Use,
                kw::Struct,
                kw::Enum,
                kw::Pub,
                kw::Trait,
                kw::Extern,
                kw::Impl,
                kw::Unsafe,
                kw::Const,
                kw::Static,
                kw::Union,
                kw::Macro,
                kw::Mod,
                kw::Type,
            ]
            .contains(&name),
            _ => false,
        }
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
    /// Keep this in sync with `Lit::from_token` and
    /// `Parser::maybe_parse_token_lit` (excluding unary negation).
    pub fn can_begin_literal_maybe_minus(&self) -> bool {
        match self.uninterpolate().kind {
            Literal(..) | BinOp(Minus) => true,
            Ident(name, false) if name.is_bool_lit() => true,
            OpenDelim(Delimiter::Invisible(InvisibleSource::MetaVar(
                NonterminalKind::Literal | NonterminalKind::Expr,
            )))
            | OpenDelim(Delimiter::Invisible(InvisibleSource::ProcMacro)) => true,
            _ => false,
        }
    }

    /// A convenience function for matching on identifiers during parsing.
    /// Turns interpolated identifier (`$i: ident`) or lifetime (`$l: lifetime`) token
    /// into the regular identifier or lifetime token it refers to,
    /// otherwise returns the original token.
    pub fn uninterpolate(&self) -> Cow<'_, Token> {
        match self.kind {
            InterpolatedIdent(name, is_raw, uninterpolated_span) => {
                Cow::Owned(Token::new(Ident(name, is_raw), uninterpolated_span))
            }
            InterpolatedLifetime(name, uninterpolated_span) => {
                Cow::Owned(Token::new(Lifetime(name), uninterpolated_span))
            }
            _ => Cow::Borrowed(self),
        }
    }

    /// Returns an identifier if this token is an identifier.
    #[inline]
    pub fn ident(&self) -> Option<(Ident, /* is_raw */ bool)> {
        // We avoid using `Token::uninterpolate` here because it's slow.
        match self.kind {
            Ident(name, is_raw) => Some((Ident::new(name, self.span), is_raw)),
            InterpolatedIdent(name, is_raw, uninterpolated_span) => {
                Some((Ident::new(name, uninterpolated_span), is_raw))
            }
            _ => None,
        }
    }

    /// Returns a lifetime identifier if this token is a lifetime.
    #[inline]
    pub fn lifetime(&self) -> Option<Ident> {
        // We avoid using `Token::uninterpolate` here because it's slow.
        match self.kind {
            Lifetime(name) => Some(Ident::new(name, self.span)),
            InterpolatedLifetime(name, uninterpolated_span) => {
                Some(Ident::new(name, uninterpolated_span))
            }
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

    /// Returns `true` if the token is an identifier whose name is the given
    /// string slice.
    pub fn is_ident_named(&self, name: Symbol) -> bool {
        self.ident().is_some_and(|(ident, _)| ident.name == name)
    }

    /// Would `maybe_reparse_metavar_expr` in `parser.rs` return `Ok(..)`?
    /// That is, is this a pre-parsed expression dropped into the token stream
    /// (which happens while parsing the result of macro expansion)?
    // njn: proc macro?
    pub fn is_metavar_expr(&self) -> bool {
        matches!(
            self.is_metavar_seq(),
            Some(
                NonterminalKind::Expr
                    | NonterminalKind::Literal
                    | NonterminalKind::Block
                    | NonterminalKind::Path
            )
        )
    }

    /// Are we at a block from a metavar (`$b:block`)?
    pub fn is_metavar_block(&self) -> bool {
        // njn: handle proc-macro here too?
        matches!(self.is_metavar_seq(), Some(NonterminalKind::Block))
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
            // njn: proc macro?
            || matches!(self.is_metavar_seq(), Some(NonterminalKind::Path))
            || self.is_path_segment_keyword()
            || self.is_ident() && !self.is_reserved_ident()
    }

    /// Returns `true` if the token is a given keyword, `kw`.
    pub fn is_keyword(&self, kw: Symbol) -> bool {
        self.is_non_raw_ident_where(|id| id.name == kw)
    }

    /// Returns `true` if the token is a given keyword, `kw` or if `case` is `Insensitive` and this token is an identifier equal to `kw` ignoring the case.
    pub fn is_keyword_case(&self, kw: Symbol, case: Case) -> bool {
        self.is_keyword(kw)
            || (case == Case::Insensitive
                && self.is_non_raw_ident_where(|id| {
                    id.name.as_str().to_lowercase() == kw.as_str().to_lowercase()
                }))
    }

    pub fn is_path_segment_keyword(&self) -> bool {
        self.is_non_raw_ident_where(Ident::is_path_segment_keyword)
    }

    /// Returns true for reserved identifiers used internally for elided lifetimes,
    /// unnamed method parameters, crate root module, error recovery etc.
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

    pub fn is_numeric_lit(&self) -> bool {
        matches!(
            self.kind,
            Literal(Lit { kind: LitKind::Integer, .. }) | Literal(Lit { kind: LitKind::Float, .. })
        )
    }

    /// Returns `true` if the token is a non-raw identifier for which `pred` holds.
    pub fn is_non_raw_ident_where(&self, pred: impl FnOnce(Ident) -> bool) -> bool {
        match self.ident() {
            Some((id, false)) => pred(id),
            _ => false,
        }
    }

    /// Is this an invisible open delimiter at the start of a token sequence
    /// from an expanded metavar?
    pub fn is_metavar_seq(&self) -> Option<NonterminalKind> {
        match self.kind {
            OpenDelim(Delimiter::Invisible(InvisibleSource::MetaVar(kind))) => Some(kind),
            _ => None,
        }
    }

    /// Is this an invisible open delimiter at the start of a token sequence
    /// from a proc macro?
    // njn: need to use this more
    pub fn is_proc_macro_seq(&self) -> bool {
        matches!(self.kind, OpenDelim(Delimiter::Invisible(InvisibleSource::ProcMacro)))
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
                Ident(name, false) => Lifetime(Symbol::intern(&format!("'{name}"))),
                _ => return None,
            },

            Le
            | EqEq
            | Ne
            | Ge
            | AndAnd
            | OrOr
            | Tilde
            | BinOpEq(..)
            | At
            | DotDotDot
            | DotDotEq
            | Comma
            | Semi
            | ModSep
            | RArrow
            | LArrow
            | FatArrow
            | Pound
            | Dollar
            | Question
            | OpenDelim(..)
            | CloseDelim(..)
            | Literal(..)
            | Ident(..)
            | InterpolatedIdent(..)
            | Lifetime(..)
            | InterpolatedLifetime(..)
            | DocComment(..)
            | Eof => return None,
        };

        Some(Token::new(kind, self.span.to(joint.span)))
    }
}

impl PartialEq<TokenKind> for Token {
    #[inline]
    fn eq(&self, rhs: &TokenKind) -> bool {
        self.kind == *rhs
    }
}

// njn: introduce cut-back version lacking Ident/Lifetime?
// - could that simplify the Pat cases too?
#[derive(Debug, Copy, Clone, PartialEq, Eq, Encodable, Decodable, Hash, HashStable_Generic)]
pub enum NonterminalKind {
    Item,
    Block,
    Stmt,
    PatParam {
        /// Keep track of whether the user used `:pat_param` or `:pat` and we inferred it from the
        /// edition of the span. This is used for diagnostics.
        inferred: bool,
    },
    PatWithOr,
    Expr,
    Ty,
    //njn: explain how these are never put in Invisible delims
    Ident,
    Lifetime,
    Literal,
    Meta,
    Path,
    Vis,
    TT,
}

impl NonterminalKind {
    /// The `edition` closure is used to get the edition for the given symbol. Doing
    /// `span.edition()` is expensive, so we do it lazily.
    pub fn from_symbol(
        symbol: Symbol,
        edition: impl FnOnce() -> Edition,
    ) -> Option<NonterminalKind> {
        Some(match symbol {
            sym::item => NonterminalKind::Item,
            sym::block => NonterminalKind::Block,
            sym::stmt => NonterminalKind::Stmt,
            sym::pat => match edition() {
                Edition::Edition2015 | Edition::Edition2018 => {
                    NonterminalKind::PatParam { inferred: true }
                }
                Edition::Edition2021 | Edition::Edition2024 => NonterminalKind::PatWithOr,
            },
            sym::pat_param => NonterminalKind::PatParam { inferred: false },
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
            NonterminalKind::PatParam { inferred: false } => sym::pat_param,
            NonterminalKind::PatParam { inferred: true } | NonterminalKind::PatWithOr => sym::pat,
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

// Some types are used a lot. Make sure they don't unintentionally get bigger.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
mod size_asserts {
    use super::*;
    use rustc_data_structures::static_assert_size;
    // tidy-alphabetical-start
    static_assert_size!(Lit, 12);
    static_assert_size!(LitKind, 2);
    static_assert_size!(Token, 24);
    static_assert_size!(TokenKind, 16);
    // tidy-alphabetical-end
}
