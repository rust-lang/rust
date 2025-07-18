use std::borrow::Cow;
use std::fmt;

pub use LitKind::*;
pub use NtExprKind::*;
pub use NtPatKind::*;
pub use TokenKind::*;
use rustc_macros::{Decodable, Encodable, HashStable_Generic};
use rustc_span::edition::Edition;
use rustc_span::{DUMMY_SP, ErrorGuaranteed, Span, kw, sym};
#[allow(clippy::useless_attribute)] // FIXME: following use of `hidden_glob_reexports` incorrectly triggers `useless_attribute` lint.
#[allow(hidden_glob_reexports)]
use rustc_span::{Ident, Symbol};

use crate::ast;
use crate::util::case::Case;

#[derive(Clone, Copy, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum CommentKind {
    Line,
    Block,
}

// This type must not implement `Hash` due to the unusual `PartialEq` impl below.
#[derive(Copy, Clone, Debug, Encodable, Decodable, HashStable_Generic)]
pub enum InvisibleOrigin {
    // From the expansion of a metavariable in a declarative macro.
    MetaVar(MetaVarKind),

    // Converted from `proc_macro::Delimiter` in
    // `proc_macro::Delimiter::to_internal`, i.e. returned by a proc macro.
    ProcMacro,
}

impl InvisibleOrigin {
    // Should the parser skip these invisible delimiters? Ideally this function
    // will eventually disappear and no invisible delimiters will be skipped.
    #[inline]
    pub fn skip(&self) -> bool {
        match self {
            InvisibleOrigin::MetaVar(_) => false,
            InvisibleOrigin::ProcMacro => true,
        }
    }
}

impl PartialEq for InvisibleOrigin {
    #[inline]
    fn eq(&self, _other: &InvisibleOrigin) -> bool {
        // When we had AST-based nonterminals we couldn't compare them, and the
        // old `Nonterminal` type had an `eq` that always returned false,
        // resulting in this restriction:
        // https://doc.rust-lang.org/nightly/reference/macros-by-example.html#forwarding-a-matched-fragment
        // This `eq` emulates that behaviour. We could consider lifting this
        // restriction now but there are still cases involving invisible
        // delimiters that make it harder than it first appears.
        false
    }
}

/// Annoyingly similar to `NonterminalKind`, but the slight differences are important.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Encodable, Decodable, Hash, HashStable_Generic)]
pub enum MetaVarKind {
    Item,
    Block,
    Stmt,
    Pat(NtPatKind),
    Expr {
        kind: NtExprKind,
        // This field is needed for `Token::can_begin_literal_maybe_minus`.
        can_begin_literal_maybe_minus: bool,
        // This field is needed for `Token::can_begin_string_literal`.
        can_begin_string_literal: bool,
    },
    Ty {
        is_path: bool,
    },
    Ident,
    Lifetime,
    Literal,
    Meta {
        /// Will `AttrItem::meta` succeed on this, if reparsed?
        has_meta_form: bool,
    },
    Path,
    Vis,
    TT,
}

impl fmt::Display for MetaVarKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sym = match self {
            MetaVarKind::Item => sym::item,
            MetaVarKind::Block => sym::block,
            MetaVarKind::Stmt => sym::stmt,
            MetaVarKind::Pat(PatParam { inferred: true } | PatWithOr) => sym::pat,
            MetaVarKind::Pat(PatParam { inferred: false }) => sym::pat_param,
            MetaVarKind::Expr { kind: Expr2021 { inferred: true } | Expr, .. } => sym::expr,
            MetaVarKind::Expr { kind: Expr2021 { inferred: false }, .. } => sym::expr_2021,
            MetaVarKind::Ty { .. } => sym::ty,
            MetaVarKind::Ident => sym::ident,
            MetaVarKind::Lifetime => sym::lifetime,
            MetaVarKind::Literal => sym::literal,
            MetaVarKind::Meta { .. } => sym::meta,
            MetaVarKind::Path => sym::path,
            MetaVarKind::Vis => sym::vis,
            MetaVarKind::TT => sym::tt,
        };
        write!(f, "{sym}")
    }
}

/// Describes how a sequence of token trees is delimited.
/// Cannot use `proc_macro::Delimiter` directly because this
/// structure should implement some additional traits.
#[derive(Copy, Clone, Debug, PartialEq, Encodable, Decodable, HashStable_Generic)]
pub enum Delimiter {
    /// `( ... )`
    Parenthesis,
    /// `{ ... }`
    Brace,
    /// `[ ... ]`
    Bracket,
    /// `∅ ... ∅`
    /// An invisible delimiter, that may, for example, appear around tokens coming from a
    /// "macro variable" `$var`. It is important to preserve operator priorities in cases like
    /// `$var * 3` where `$var` is `1 + 2`.
    /// Invisible delimiters might not survive roundtrip of a token stream through a string.
    Invisible(InvisibleOrigin),
}

impl Delimiter {
    // Should the parser skip these delimiters? Only happens for certain kinds
    // of invisible delimiters. Ideally this function will eventually disappear
    // and no invisible delimiters will be skipped.
    #[inline]
    pub fn skip(&self) -> bool {
        match self {
            Delimiter::Parenthesis | Delimiter::Bracket | Delimiter::Brace => false,
            Delimiter::Invisible(origin) => origin.skip(),
        }
    }

    // This exists because `InvisibleOrigin`s should be compared. It is only used for assertions.
    pub fn eq_ignoring_invisible_origin(&self, other: &Delimiter) -> bool {
        match (self, other) {
            (Delimiter::Parenthesis, Delimiter::Parenthesis) => true,
            (Delimiter::Brace, Delimiter::Brace) => true,
            (Delimiter::Bracket, Delimiter::Bracket) => true,
            (Delimiter::Invisible(_), Delimiter::Invisible(_)) => true,
            _ => false,
        }
    }

    pub fn as_open_token_kind(&self) -> TokenKind {
        match *self {
            Delimiter::Parenthesis => OpenParen,
            Delimiter::Brace => OpenBrace,
            Delimiter::Bracket => OpenBracket,
            Delimiter::Invisible(origin) => OpenInvisible(origin),
        }
    }

    pub fn as_close_token_kind(&self) -> TokenKind {
        match *self {
            Delimiter::Parenthesis => CloseParen,
            Delimiter::Brace => CloseBrace,
            Delimiter::Bracket => CloseBracket,
            Delimiter::Invisible(origin) => CloseInvisible(origin),
        }
    }
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
    Err(ErrorGuaranteed),
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
    /// `Parser::eat_token_lit` (excluding unary negation).
    pub fn from_token(token: &Token) -> Option<Lit> {
        match token.uninterpolate().kind {
            Ident(name, IdentIsRaw::No) if name.is_bool_lit() => Some(Lit::new(Bool, name, None)),
            Literal(token_lit) => Some(token_lit),
            OpenInvisible(InvisibleOrigin::MetaVar(
                MetaVarKind::Literal | MetaVarKind::Expr { .. },
            )) => {
                // Unreachable with the current test suite.
                panic!("from_token metavar");
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
            Integer | Float | Bool | Err(_) => write!(f, "{symbol}")?,
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
            Integer | Err(_) => "an",
            _ => "a",
        }
    }

    pub fn descr(self) -> &'static str {
        match self {
            Bool => "boolean",
            Byte => "byte",
            Char => "char",
            Integer => "integer",
            Float => "float",
            Str | StrRaw(..) => "string",
            ByteStr | ByteStrRaw(..) => "byte string",
            CStr | CStrRaw(..) => "C string",
            Err(_) => "error",
        }
    }

    pub(crate) fn may_have_suffix(self) -> bool {
        matches!(self, Integer | Float | Err(_))
    }
}

pub fn ident_can_begin_expr(name: Symbol, span: Span, is_raw: IdentIsRaw) -> bool {
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
            kw::Gen,
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
            kw::Safe,
            kw::Static,
        ]
        .contains(&name)
}

fn ident_can_begin_type(name: Symbol, span: Span, is_raw: IdentIsRaw) -> bool {
    let ident_token = Token::new(Ident(name, is_raw), span);

    !ident_token.is_reserved_ident()
        || ident_token.is_path_segment_keyword()
        || [kw::Underscore, kw::For, kw::Impl, kw::Fn, kw::Unsafe, kw::Extern, kw::Typeof, kw::Dyn]
            .contains(&name)
}

#[derive(PartialEq, Encodable, Decodable, Debug, Copy, Clone, HashStable_Generic)]
pub enum IdentIsRaw {
    No,
    Yes,
}

impl From<bool> for IdentIsRaw {
    fn from(b: bool) -> Self {
        if b { Self::Yes } else { Self::No }
    }
}

impl From<IdentIsRaw> for bool {
    fn from(is_raw: IdentIsRaw) -> bool {
        matches!(is_raw, IdentIsRaw::Yes)
    }
}

#[derive(Clone, Copy, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum TokenKind {
    /* Expression-operator symbols. */
    /// `=`
    Eq,
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `==`
    EqEq,
    /// `!=`
    Ne,
    /// `>=`
    Ge,
    /// `>`
    Gt,
    /// `&&`
    AndAnd,
    /// `||`
    OrOr,
    /// `!`
    Bang,
    /// `~`
    Tilde,
    // `+`
    Plus,
    // `-`
    Minus,
    // `*`
    Star,
    // `/`
    Slash,
    // `%`
    Percent,
    // `^`
    Caret,
    // `&`
    And,
    // `|`
    Or,
    // `<<`
    Shl,
    // `>>`
    Shr,
    // `+=`
    PlusEq,
    // `-=`
    MinusEq,
    // `*=`
    StarEq,
    // `/=`
    SlashEq,
    // `%=`
    PercentEq,
    // `^=`
    CaretEq,
    // `&=`
    AndEq,
    // `|=`
    OrEq,
    // `<<=`
    ShlEq,
    // `>>=`
    ShrEq,

    /* Structural symbols */
    /// `@`
    At,
    /// `.`
    Dot,
    /// `..`
    DotDot,
    /// `...`
    DotDotDot,
    /// `..=`
    DotDotEq,
    /// `,`
    Comma,
    /// `;`
    Semi,
    /// `:`
    Colon,
    /// `::`
    PathSep,
    /// `->`
    RArrow,
    /// `<-`
    LArrow,
    /// `=>`
    FatArrow,
    /// `#`
    Pound,
    /// `$`
    Dollar,
    /// `?`
    Question,
    /// Used by proc macros for representing lifetimes, not generated by lexer right now.
    SingleQuote,
    /// `(`
    OpenParen,
    /// `)`
    CloseParen,
    /// `{`
    OpenBrace,
    /// `}`
    CloseBrace,
    /// `[`
    OpenBracket,
    /// `]`
    CloseBracket,
    /// Invisible opening delimiter, produced by a macro.
    OpenInvisible(InvisibleOrigin),
    /// Invisible closing delimiter, produced by a macro.
    CloseInvisible(InvisibleOrigin),

    /* Literals */
    Literal(Lit),

    /// Identifier token.
    /// Do not forget about `NtIdent` when you want to match on identifiers.
    /// It's recommended to use `Token::{ident,uninterpolate}` and
    /// `Parser::token_uninterpolated_span` to treat regular and interpolated
    /// identifiers in the same way.
    Ident(Symbol, IdentIsRaw),
    /// This identifier (and its span) is the identifier passed to the
    /// declarative macro. The span in the surrounding `Token` is the span of
    /// the `ident` metavariable in the macro's RHS.
    NtIdent(Ident, IdentIsRaw),

    /// Lifetime identifier token.
    /// Do not forget about `NtLifetime` when you want to match on lifetime identifiers.
    /// It's recommended to use `Token::{ident,uninterpolate}` and
    /// `Parser::token_uninterpolated_span` to treat regular and interpolated
    /// identifiers in the same way.
    Lifetime(Symbol, IdentIsRaw),
    /// This identifier (and its span) is the lifetime passed to the
    /// declarative macro. The span in the surrounding `Token` is the span of
    /// the `lifetime` metavariable in the macro's RHS.
    NtLifetime(Ident, IdentIsRaw),

    /// A doc comment token.
    /// `Symbol` is the doc comment's data excluding its "quotes" (`///`, `/**`, etc)
    /// similarly to symbols in string literal tokens.
    DocComment(CommentKind, ast::AttrStyle, Symbol),

    /// End Of File
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

    /// An approximation to proc-macro-style single-character operators used by
    /// rustc parser. If the operator token can be broken into two tokens, the
    /// first of which has `n` (1 or 2) chars, then this function performs that
    /// operation, otherwise it returns `None`.
    pub fn break_two_token_op(&self, n: u32) -> Option<(TokenKind, TokenKind)> {
        assert!(n == 1 || n == 2);
        Some(match (self, n) {
            (Le, 1) => (Lt, Eq),
            (EqEq, 1) => (Eq, Eq),
            (Ne, 1) => (Bang, Eq),
            (Ge, 1) => (Gt, Eq),
            (AndAnd, 1) => (And, And),
            (OrOr, 1) => (Or, Or),
            (Shl, 1) => (Lt, Lt),
            (Shr, 1) => (Gt, Gt),
            (PlusEq, 1) => (Plus, Eq),
            (MinusEq, 1) => (Minus, Eq),
            (StarEq, 1) => (Star, Eq),
            (SlashEq, 1) => (Slash, Eq),
            (PercentEq, 1) => (Percent, Eq),
            (CaretEq, 1) => (Caret, Eq),
            (AndEq, 1) => (And, Eq),
            (OrEq, 1) => (Or, Eq),
            (ShlEq, 1) => (Lt, Le),  // `<` + `<=`
            (ShlEq, 2) => (Shl, Eq), // `<<` + `=`
            (ShrEq, 1) => (Gt, Ge),  // `>` + `>=`
            (ShrEq, 2) => (Shr, Eq), // `>>` + `=`
            (DotDot, 1) => (Dot, Dot),
            (DotDotDot, 1) => (Dot, DotDot), // `.` + `..`
            (DotDotDot, 2) => (DotDot, Dot), // `..` + `.`
            (DotDotEq, 2) => (DotDot, Eq),
            (PathSep, 1) => (Colon, Colon),
            (RArrow, 1) => (Minus, Gt),
            (LArrow, 1) => (Lt, Minus),
            (FatArrow, 1) => (Eq, Gt),
            _ => return None,
        })
    }

    /// Returns tokens that are likely to be typed accidentally instead of the current token.
    /// Enables better error recovery when the wrong token is found.
    pub fn similar_tokens(&self) -> &[TokenKind] {
        match self {
            Comma => &[Dot, Lt, Semi],
            Semi => &[Colon, Comma],
            Colon => &[Semi],
            FatArrow => &[Eq, RArrow, Ge, Gt],
            _ => &[],
        }
    }

    pub fn should_end_const_arg(&self) -> bool {
        matches!(self, Gt | Ge | Shr | ShrEq)
    }

    pub fn is_delim(&self) -> bool {
        self.open_delim().is_some() || self.close_delim().is_some()
    }

    pub fn open_delim(&self) -> Option<Delimiter> {
        match *self {
            OpenParen => Some(Delimiter::Parenthesis),
            OpenBrace => Some(Delimiter::Brace),
            OpenBracket => Some(Delimiter::Bracket),
            OpenInvisible(origin) => Some(Delimiter::Invisible(origin)),
            _ => None,
        }
    }

    pub fn close_delim(&self) -> Option<Delimiter> {
        match *self {
            CloseParen => Some(Delimiter::Parenthesis),
            CloseBrace => Some(Delimiter::Brace),
            CloseBracket => Some(Delimiter::Bracket),
            CloseInvisible(origin) => Some(Delimiter::Invisible(origin)),
            _ => None,
        }
    }

    pub fn is_close_delim_or_eof(&self) -> bool {
        match self {
            CloseParen | CloseBrace | CloseBracket | CloseInvisible(_) | Eof => true,
            _ => false,
        }
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
        Token::new(Ident(ident.name, ident.is_raw_guess().into()), ident.span)
    }

    pub fn is_range_separator(&self) -> bool {
        [DotDot, DotDotDot, DotDotEq].contains(&self.kind)
    }

    pub fn is_punct(&self) -> bool {
        match self.kind {
            Eq | Lt | Le | EqEq | Ne | Ge | Gt | AndAnd | OrOr | Bang | Tilde | Plus | Minus
            | Star | Slash | Percent | Caret | And | Or | Shl | Shr | PlusEq | MinusEq | StarEq
            | SlashEq | PercentEq | CaretEq | AndEq | OrEq | ShlEq | ShrEq | At | Dot | DotDot
            | DotDotDot | DotDotEq | Comma | Semi | Colon | PathSep | RArrow | LArrow
            | FatArrow | Pound | Dollar | Question | SingleQuote => true,

            OpenParen | CloseParen | OpenBrace | CloseBrace | OpenBracket | CloseBracket
            | OpenInvisible(_) | CloseInvisible(_) | Literal(..) | DocComment(..) | Ident(..)
            | NtIdent(..) | Lifetime(..) | NtLifetime(..) | Eof => false,
        }
    }

    pub fn is_like_plus(&self) -> bool {
        matches!(self.kind, Plus | PlusEq)
    }

    /// Returns `true` if the token can appear at the start of an expression.
    ///
    /// **NB**: Take care when modifying this function, since it will change
    /// the stable set of tokens that are allowed to match an expr nonterminal.
    pub fn can_begin_expr(&self) -> bool {
        match self.uninterpolate().kind {
            Ident(name, is_raw)              =>
                ident_can_begin_expr(name, self.span, is_raw), // value name or keyword
            OpenParen                         | // tuple
            OpenBrace                         | // block
            OpenBracket                       | // array
            Literal(..)                       | // literal
            Bang                              | // operator not
            Minus                             | // unary minus
            Star                              | // dereference
            Or | OrOr                         | // closure
            And                               | // reference
            AndAnd                            | // double reference
            // DotDotDot is no longer supported, but we need some way to display the error
            DotDot | DotDotDot | DotDotEq     | // range notation
            Lt | Shl                          | // associated path
            PathSep                           | // global path
            Lifetime(..)                      | // labeled loop
            Pound                             => true, // expression attributes
            OpenInvisible(InvisibleOrigin::MetaVar(
                MetaVarKind::Block |
                MetaVarKind::Expr { .. } |
                MetaVarKind::Literal |
                MetaVarKind::Path
            )) => true,
            _ => false,
        }
    }

    /// Returns `true` if the token can appear at the start of a pattern.
    ///
    /// Shamelessly borrowed from `can_begin_expr`, only used for diagnostics right now.
    pub fn can_begin_pattern(&self, pat_kind: NtPatKind) -> bool {
        match &self.uninterpolate().kind {
            // box, ref, mut, and other identifiers (can stricten)
            Ident(..) | NtIdent(..) |
            OpenParen |                          // tuple pattern
            OpenBracket |                        // slice pattern
            And |                                // reference
            Minus |                              // negative literal
            AndAnd |                             // double reference
            Literal(_) |                         // literal
            DotDot |                             // range pattern (future compat)
            DotDotDot |                          // range pattern (future compat)
            PathSep |                            // path
            Lt |                                 // path (UFCS constant)
            Shl => true,                         // path (double UFCS)
            Or => matches!(pat_kind, PatWithOr), // leading vert `|` or-pattern
            OpenInvisible(InvisibleOrigin::MetaVar(
                MetaVarKind::Expr { .. } |
                MetaVarKind::Literal |
                MetaVarKind::Meta { .. } |
                MetaVarKind::Pat(_) |
                MetaVarKind::Path |
                MetaVarKind::Ty { .. }
            )) => true,
            _ => false,
        }
    }

    /// Returns `true` if the token can appear at the start of a type.
    pub fn can_begin_type(&self) -> bool {
        match self.uninterpolate().kind {
            Ident(name, is_raw) =>
                ident_can_begin_type(name, self.span, is_raw), // type name or keyword
            OpenParen                         | // tuple
            OpenBracket                       | // array
            Bang                              | // never
            Star                              | // raw pointer
            And                               | // reference
            AndAnd                            | // double reference
            Question                          | // maybe bound in trait object
            Lifetime(..)                      | // lifetime bound in trait object
            Lt | Shl                          | // associated path
            PathSep => true,                    // global path
            OpenInvisible(InvisibleOrigin::MetaVar(
                MetaVarKind::Ty { .. } |
                MetaVarKind::Path
            )) => true,
            // For anonymous structs or unions, which only appear in specific positions
            // (type of struct fields or union fields), we don't consider them as regular types
            _ => false,
        }
    }

    /// Returns `true` if the token can appear at the start of a const param.
    pub fn can_begin_const_arg(&self) -> bool {
        match self.kind {
            OpenBrace | Literal(..) | Minus => true,
            Ident(name, IdentIsRaw::No) if name.is_bool_lit() => true,
            OpenInvisible(InvisibleOrigin::MetaVar(
                MetaVarKind::Expr { .. } | MetaVarKind::Block | MetaVarKind::Literal,
            )) => true,
            _ => false,
        }
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
                kw::Safe,
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
    /// Keep this in sync with `Lit::from_token` and `Parser::eat_token_lit`
    /// (excluding unary negation).
    pub fn can_begin_literal_maybe_minus(&self) -> bool {
        match self.uninterpolate().kind {
            Literal(..) | Minus => true,
            Ident(name, IdentIsRaw::No) if name.is_bool_lit() => true,
            OpenInvisible(InvisibleOrigin::MetaVar(mv_kind)) => match mv_kind {
                MetaVarKind::Literal => true,
                MetaVarKind::Expr { can_begin_literal_maybe_minus, .. } => {
                    can_begin_literal_maybe_minus
                }
                _ => false,
            },
            _ => false,
        }
    }

    pub fn can_begin_string_literal(&self) -> bool {
        match self.uninterpolate().kind {
            Literal(..) => true,
            OpenInvisible(InvisibleOrigin::MetaVar(mv_kind)) => match mv_kind {
                MetaVarKind::Literal => true,
                MetaVarKind::Expr { can_begin_string_literal, .. } => can_begin_string_literal,
                _ => false,
            },
            _ => false,
        }
    }

    /// A convenience function for matching on identifiers during parsing.
    /// Turns interpolated identifier (`$i: ident`) or lifetime (`$l: lifetime`) token
    /// into the regular identifier or lifetime token it refers to,
    /// otherwise returns the original token.
    pub fn uninterpolate(&self) -> Cow<'_, Token> {
        match self.kind {
            NtIdent(ident, is_raw) => Cow::Owned(Token::new(Ident(ident.name, is_raw), ident.span)),
            NtLifetime(ident, is_raw) => {
                Cow::Owned(Token::new(Lifetime(ident.name, is_raw), ident.span))
            }
            _ => Cow::Borrowed(self),
        }
    }

    /// Returns an identifier if this token is an identifier.
    #[inline]
    pub fn ident(&self) -> Option<(Ident, IdentIsRaw)> {
        // We avoid using `Token::uninterpolate` here because it's slow.
        match self.kind {
            Ident(name, is_raw) => Some((Ident::new(name, self.span), is_raw)),
            NtIdent(ident, is_raw) => Some((ident, is_raw)),
            _ => None,
        }
    }

    /// Returns a lifetime identifier if this token is a lifetime.
    #[inline]
    pub fn lifetime(&self) -> Option<(Ident, IdentIsRaw)> {
        // We avoid using `Token::uninterpolate` here because it's slow.
        match self.kind {
            Lifetime(name, is_raw) => Some((Ident::new(name, self.span), is_raw)),
            NtLifetime(ident, is_raw) => Some((ident, is_raw)),
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

    /// Is this a pre-parsed expression dropped into the token stream
    /// (which happens while parsing the result of macro expansion)?
    pub fn is_metavar_expr(&self) -> bool {
        matches!(
            self.is_metavar_seq(),
            Some(
                MetaVarKind::Expr { .. }
                    | MetaVarKind::Literal
                    | MetaVarKind::Path
                    | MetaVarKind::Block
            )
        )
    }

    /// Are we at a block from a metavar (`$b:block`)?
    pub fn is_metavar_block(&self) -> bool {
        matches!(self.is_metavar_seq(), Some(MetaVarKind::Block))
    }

    /// Returns `true` if the token is either the `mut` or `const` keyword.
    pub fn is_mutability(&self) -> bool {
        self.is_keyword(kw::Mut) || self.is_keyword(kw::Const)
    }

    pub fn is_qpath_start(&self) -> bool {
        self == &Lt || self == &Shl
    }

    pub fn is_path_start(&self) -> bool {
        self == &PathSep
            || self.is_qpath_start()
            || matches!(self.is_metavar_seq(), Some(MetaVarKind::Path))
            || self.is_path_segment_keyword()
            || self.is_non_reserved_ident()
    }

    /// Returns `true` if the token is a given keyword, `kw`.
    pub fn is_keyword(&self, kw: Symbol) -> bool {
        self.is_non_raw_ident_where(|id| id.name == kw)
    }

    /// Returns `true` if the token is a given keyword, `kw` or if `case` is `Insensitive` and this
    /// token is an identifier equal to `kw` ignoring the case.
    pub fn is_keyword_case(&self, kw: Symbol, case: Case) -> bool {
        self.is_keyword(kw)
            || (case == Case::Insensitive
                && self.is_non_raw_ident_where(|id| {
                    // Do an ASCII case-insensitive match, because all keywords are ASCII.
                    id.name.as_str().eq_ignore_ascii_case(kw.as_str())
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

    pub fn is_non_reserved_ident(&self) -> bool {
        self.ident().is_some_and(|(id, raw)| raw == IdentIsRaw::Yes || !Ident::is_reserved(id))
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

    /// Returns `true` if the token is the integer literal.
    pub fn is_integer_lit(&self) -> bool {
        matches!(self.kind, Literal(Lit { kind: LitKind::Integer, .. }))
    }

    /// Returns `true` if the token is a non-raw identifier for which `pred` holds.
    pub fn is_non_raw_ident_where(&self, pred: impl FnOnce(Ident) -> bool) -> bool {
        match self.ident() {
            Some((id, IdentIsRaw::No)) => pred(id),
            _ => false,
        }
    }

    /// Is this an invisible open delimiter at the start of a token sequence
    /// from an expanded metavar?
    pub fn is_metavar_seq(&self) -> Option<MetaVarKind> {
        match self.kind {
            OpenInvisible(InvisibleOrigin::MetaVar(kind)) => Some(kind),
            _ => None,
        }
    }

    pub fn glue(&self, joint: &Token) -> Option<Token> {
        let kind = match (&self.kind, &joint.kind) {
            (Eq, Eq) => EqEq,
            (Eq, Gt) => FatArrow,
            (Eq, _) => return None,

            (Lt, Eq) => Le,
            (Lt, Lt) => Shl,
            (Lt, Le) => ShlEq,
            (Lt, Minus) => LArrow,
            (Lt, _) => return None,

            (Gt, Eq) => Ge,
            (Gt, Gt) => Shr,
            (Gt, Ge) => ShrEq,
            (Gt, _) => return None,

            (Bang, Eq) => Ne,
            (Bang, _) => return None,

            (Plus, Eq) => PlusEq,
            (Plus, _) => return None,

            (Minus, Eq) => MinusEq,
            (Minus, Gt) => RArrow,
            (Minus, _) => return None,

            (Star, Eq) => StarEq,
            (Star, _) => return None,

            (Slash, Eq) => SlashEq,
            (Slash, _) => return None,

            (Percent, Eq) => PercentEq,
            (Percent, _) => return None,

            (Caret, Eq) => CaretEq,
            (Caret, _) => return None,

            (And, Eq) => AndEq,
            (And, And) => AndAnd,
            (And, _) => return None,

            (Or, Eq) => OrEq,
            (Or, Or) => OrOr,
            (Or, _) => return None,

            (Shl, Eq) => ShlEq,
            (Shl, _) => return None,

            (Shr, Eq) => ShrEq,
            (Shr, _) => return None,

            (Dot, Dot) => DotDot,
            (Dot, DotDot) => DotDotDot,
            (Dot, _) => return None,

            (DotDot, Dot) => DotDotDot,
            (DotDot, Eq) => DotDotEq,
            (DotDot, _) => return None,

            (Colon, Colon) => PathSep,
            (Colon, _) => return None,

            (SingleQuote, Ident(name, is_raw)) => {
                Lifetime(Symbol::intern(&format!("'{name}")), *is_raw)
            }
            (SingleQuote, _) => return None,

            (
                Le | EqEq | Ne | Ge | AndAnd | OrOr | Tilde | PlusEq | MinusEq | StarEq | SlashEq
                | PercentEq | CaretEq | AndEq | OrEq | ShlEq | ShrEq | At | DotDotDot | DotDotEq
                | Comma | Semi | PathSep | RArrow | LArrow | FatArrow | Pound | Dollar | Question
                | OpenParen | CloseParen | OpenBrace | CloseBrace | OpenBracket | CloseBracket
                | OpenInvisible(_) | CloseInvisible(_) | Literal(..) | Ident(..) | NtIdent(..)
                | Lifetime(..) | NtLifetime(..) | DocComment(..) | Eof,
                _,
            ) => {
                return None;
            }
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

#[derive(Debug, Copy, Clone, PartialEq, Eq, Encodable, Decodable, Hash, HashStable_Generic)]
pub enum NtPatKind {
    // Matches or-patterns. Was written using `pat` in edition 2021 or later.
    PatWithOr,
    // Doesn't match or-patterns.
    // - `inferred`: was written using `pat` in edition 2015 or 2018.
    // - `!inferred`: was written using `pat_param`.
    PatParam { inferred: bool },
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Encodable, Decodable, Hash, HashStable_Generic)]
pub enum NtExprKind {
    // Matches expressions using the post-edition 2024. Was written using
    // `expr` in edition 2024 or later.
    Expr,
    // Matches expressions using the pre-edition 2024 rules.
    // - `inferred`: was written using `expr` in edition 2021 or earlier.
    // - `!inferred`: was written using `expr_2021`.
    Expr2021 { inferred: bool },
}

/// A macro nonterminal, known in documentation as a fragment specifier.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Encodable, Decodable, Hash, HashStable_Generic)]
pub enum NonterminalKind {
    Item,
    Block,
    Stmt,
    Pat(NtPatKind),
    Expr(NtExprKind),
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
            sym::pat => {
                if edition().at_least_rust_2021() {
                    NonterminalKind::Pat(PatWithOr)
                } else {
                    NonterminalKind::Pat(PatParam { inferred: true })
                }
            }
            sym::pat_param => NonterminalKind::Pat(PatParam { inferred: false }),
            sym::expr => {
                if edition().at_least_rust_2024() {
                    NonterminalKind::Expr(Expr)
                } else {
                    NonterminalKind::Expr(Expr2021 { inferred: true })
                }
            }
            sym::expr_2021 => NonterminalKind::Expr(Expr2021 { inferred: false }),
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
            NonterminalKind::Pat(PatParam { inferred: true } | PatWithOr) => sym::pat,
            NonterminalKind::Pat(PatParam { inferred: false }) => sym::pat_param,
            NonterminalKind::Expr(Expr2021 { inferred: true } | Expr) => sym::expr,
            NonterminalKind::Expr(Expr2021 { inferred: false }) => sym::expr_2021,
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
#[cfg(target_pointer_width = "64")]
mod size_asserts {
    use rustc_data_structures::static_assert_size;

    use super::*;
    // tidy-alphabetical-start
    static_assert_size!(Lit, 12);
    static_assert_size!(LitKind, 2);
    static_assert_size!(Token, 24);
    static_assert_size!(TokenKind, 16);
    // tidy-alphabetical-end
}
