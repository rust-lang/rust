use std::borrow::Cow;
use std::fmt;
use std::sync::Arc;

pub use BinOpToken::*;
pub use LitKind::*;
pub use Nonterminal::*;
pub use NtExprKind::*;
pub use NtPatKind::*;
pub use TokenKind::*;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_macros::{Decodable, Encodable, HashStable_Generic};
use rustc_span::edition::Edition;
use rustc_span::{DUMMY_SP, ErrorGuaranteed, Span, kw, sym};
#[allow(clippy::useless_attribute)] // FIXME: following use of `hidden_glob_reexports` incorrectly triggers `useless_attribute` lint.
#[allow(hidden_glob_reexports)]
use rustc_span::{Ident, Symbol};

use crate::ast;
use crate::ptr::P;
use crate::util::case::Case;

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

// This type must not implement `Hash` due to the unusual `PartialEq` impl below.
#[derive(Copy, Clone, Debug, Encodable, Decodable, HashStable_Generic)]
pub enum InvisibleOrigin {
    // From the expansion of a metavariable in a declarative macro.
    MetaVar(MetaVarKind),

    // Converted from `proc_macro::Delimiter` in
    // `proc_macro::Delimiter::to_internal`, i.e. returned by a proc macro.
    ProcMacro,

    // Converted from `TokenKind::Interpolated` in
    // `TokenStream::flatten_token`. Treated similarly to `ProcMacro`.
    FlattenToken,
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
            Delimiter::Invisible(InvisibleOrigin::MetaVar(_)) => false,
            Delimiter::Invisible(InvisibleOrigin::FlattenToken | InvisibleOrigin::ProcMacro) => {
                true
            }
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

    /// Keep this in sync with `Token::can_begin_literal_maybe_minus` excluding unary negation.
    pub fn from_token(token: &Token) -> Option<Lit> {
        match token.uninterpolate().kind {
            Ident(name, IdentIsRaw::No) if name.is_bool_lit() => Some(Lit::new(Bool, name, None)),
            Literal(token_lit) => Some(token_lit),
            Interpolated(ref nt)
                if let NtExpr(expr) | NtLiteral(expr) = &**nt
                    && let ast::ExprKind::Lit(token_lit) = expr.kind =>
            {
                Some(token_lit)
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

// SAFETY: due to the `Clone` impl below, all fields of all variants other than
// `Interpolated` must impl `Copy`.
#[derive(PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
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
    Not,
    /// `~`
    Tilde,
    BinOp(BinOpToken),
    BinOpEq(BinOpToken),

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
    /// An opening delimiter (e.g., `{`).
    OpenDelim(Delimiter),
    /// A closing delimiter (e.g., `}`).
    CloseDelim(Delimiter),

    /* Literals */
    Literal(Lit),

    /// Identifier token.
    /// Do not forget about `NtIdent` when you want to match on identifiers.
    /// It's recommended to use `Token::(ident,uninterpolate,uninterpolated_span)` to
    /// treat regular and interpolated identifiers in the same way.
    Ident(Symbol, IdentIsRaw),
    /// This identifier (and its span) is the identifier passed to the
    /// declarative macro. The span in the surrounding `Token` is the span of
    /// the `ident` metavariable in the macro's RHS.
    NtIdent(Ident, IdentIsRaw),

    /// Lifetime identifier token.
    /// Do not forget about `NtLifetime` when you want to match on lifetime identifiers.
    /// It's recommended to use `Token::(lifetime,uninterpolate,uninterpolated_span)` to
    /// treat regular and interpolated lifetime identifiers in the same way.
    Lifetime(Symbol, IdentIsRaw),
    /// This identifier (and its span) is the lifetime passed to the
    /// declarative macro. The span in the surrounding `Token` is the span of
    /// the `lifetime` metavariable in the macro's RHS.
    NtLifetime(Ident, IdentIsRaw),

    /// An embedded AST node, as produced by a macro. This only exists for
    /// historical reasons. We'd like to get rid of it, for multiple reasons.
    /// - It's conceptually very strange. Saying a token can contain an AST
    ///   node is like saying, in natural language, that a word can contain a
    ///   sentence.
    /// - It requires special handling in a bunch of places in the parser.
    /// - It prevents `Token` from implementing `Copy`.
    /// It adds complexity and likely slows things down. Please don't add new
    /// occurrences of this token kind!
    ///
    /// The span in the surrounding `Token` is that of the metavariable in the
    /// macro's RHS. The span within the Nonterminal is that of the fragment
    /// passed to the macro at the call site.
    Interpolated(Arc<Nonterminal>),

    /// A doc comment token.
    /// `Symbol` is the doc comment's data excluding its "quotes" (`///`, `/**`, etc)
    /// similarly to symbols in string literal tokens.
    DocComment(CommentKind, ast::AttrStyle, Symbol),

    /// End Of File
    Eof,
}

impl Clone for TokenKind {
    fn clone(&self) -> Self {
        // `TokenKind` would impl `Copy` if it weren't for `Interpolated`. So
        // for all other variants, this implementation of `clone` is just like
        // a copy. This is faster than the `derive(Clone)` version which has a
        // separate path for every variant.
        match self {
            Interpolated(nt) => Interpolated(Arc::clone(nt)),
            _ => unsafe { std::ptr::read(self) },
        }
    }
}

#[derive(Clone, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
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
            (Ne, 1) => (Not, Eq),
            (Ge, 1) => (Gt, Eq),
            (AndAnd, 1) => (BinOp(And), BinOp(And)),
            (OrOr, 1) => (BinOp(Or), BinOp(Or)),
            (BinOp(Shl), 1) => (Lt, Lt),
            (BinOp(Shr), 1) => (Gt, Gt),
            (BinOpEq(Plus), 1) => (BinOp(Plus), Eq),
            (BinOpEq(Minus), 1) => (BinOp(Minus), Eq),
            (BinOpEq(Star), 1) => (BinOp(Star), Eq),
            (BinOpEq(Slash), 1) => (BinOp(Slash), Eq),
            (BinOpEq(Percent), 1) => (BinOp(Percent), Eq),
            (BinOpEq(Caret), 1) => (BinOp(Caret), Eq),
            (BinOpEq(And), 1) => (BinOp(And), Eq),
            (BinOpEq(Or), 1) => (BinOp(Or), Eq),
            (BinOpEq(Shl), 1) => (Lt, Le),         // `<` + `<=`
            (BinOpEq(Shl), 2) => (BinOp(Shl), Eq), // `<<` + `=`
            (BinOpEq(Shr), 1) => (Gt, Ge),         // `>` + `>=`
            (BinOpEq(Shr), 2) => (BinOp(Shr), Eq), // `>>` + `=`
            (DotDot, 1) => (Dot, Dot),
            (DotDotDot, 1) => (Dot, DotDot), // `.` + `..`
            (DotDotDot, 2) => (DotDot, Dot), // `..` + `.`
            (DotDotEq, 2) => (DotDot, Eq),
            (PathSep, 1) => (Colon, Colon),
            (RArrow, 1) => (BinOp(Minus), Gt),
            (LArrow, 1) => (Lt, BinOp(Minus)),
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
        Token::new(Ident(ident.name, ident.is_raw_guess().into()), ident.span)
    }

    /// For interpolated tokens, returns a span of the fragment to which the interpolated
    /// token refers. For all other tokens this is just a regular span.
    /// It is particularly important to use this for identifiers and lifetimes
    /// for which spans affect name resolution and edition checks.
    /// Note that keywords are also identifiers, so they should use this
    /// if they keep spans or perform edition checks.
    pub fn uninterpolated_span(&self) -> Span {
        match self.kind {
            NtIdent(ident, _) | NtLifetime(ident, _) => ident.span,
            Interpolated(ref nt) => nt.use_span(),
            _ => self.span,
        }
    }

    pub fn is_range_separator(&self) -> bool {
        [DotDot, DotDotDot, DotDotEq].contains(&self.kind)
    }

    pub fn is_punct(&self) -> bool {
        match self.kind {
            Eq | Lt | Le | EqEq | Ne | Ge | Gt | AndAnd | OrOr | Not | Tilde | BinOp(_)
            | BinOpEq(_) | At | Dot | DotDot | DotDotDot | DotDotEq | Comma | Semi | Colon
            | PathSep | RArrow | LArrow | FatArrow | Pound | Dollar | Question | SingleQuote => {
                true
            }

            OpenDelim(..) | CloseDelim(..) | Literal(..) | DocComment(..) | Ident(..)
            | NtIdent(..) | Lifetime(..) | NtLifetime(..) | Interpolated(..) | Eof => false,
        }
    }

    pub fn is_like_plus(&self) -> bool {
        matches!(self.kind, BinOp(Plus) | BinOpEq(Plus))
    }

    /// Returns `true` if the token can appear at the start of an expression.
    ///
    /// **NB**: Take care when modifying this function, since it will change
    /// the stable set of tokens that are allowed to match an expr nonterminal.
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
            PathSep                           | // global path
            Lifetime(..)                      | // labeled loop
            Pound                             => true, // expression attributes
            Interpolated(ref nt) =>
                matches!(&**nt,
                    NtBlock(..)   |
                    NtExpr(..)    |
                    NtLiteral(..)
                ),
            OpenDelim(Delimiter::Invisible(InvisibleOrigin::MetaVar(
                MetaVarKind::Block |
                MetaVarKind::Expr { .. } |
                MetaVarKind::Literal |
                MetaVarKind::Path
            ))) => true,
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
            OpenDelim(Delimiter::Parenthesis) |  // tuple pattern
            OpenDelim(Delimiter::Bracket) |      // slice pattern
            BinOp(And) |                  // reference
            BinOp(Minus) |                // negative literal
            AndAnd |                      // double reference
            Literal(_) |                  // literal
            DotDot |                      // range pattern (future compat)
            DotDotDot |                   // range pattern (future compat)
            PathSep |                     // path
            Lt |                          // path (UFCS constant)
            BinOp(Shl) => true,           // path (double UFCS)
            // leading vert `|` or-pattern
            BinOp(Or) => matches!(pat_kind, PatWithOr),
            Interpolated(nt) =>
                matches!(&**nt,
                    | NtExpr(..)
                    | NtLiteral(..)
                ),
            OpenDelim(Delimiter::Invisible(InvisibleOrigin::MetaVar(
                MetaVarKind::Expr { .. } |
                MetaVarKind::Literal |
                MetaVarKind::Meta { .. } |
                MetaVarKind::Pat(_) |
                MetaVarKind::Path |
                MetaVarKind::Ty { .. }
            ))) => true,
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
            Not                         | // never
            BinOp(Star)                 | // raw pointer
            BinOp(And)                  | // reference
            AndAnd                      | // double reference
            Question                    | // maybe bound in trait object
            Lifetime(..)                | // lifetime bound in trait object
            Lt | BinOp(Shl)             | // associated path
            PathSep                      => true, // global path
            OpenDelim(Delimiter::Invisible(InvisibleOrigin::MetaVar(
                MetaVarKind::Ty { .. } |
                MetaVarKind::Path
            ))) => true,
            // For anonymous structs or unions, which only appear in specific positions
            // (type of struct fields or union fields), we don't consider them as regular types
            _ => false,
        }
    }

    /// Returns `true` if the token can appear at the start of a const param.
    pub fn can_begin_const_arg(&self) -> bool {
        match self.kind {
            OpenDelim(Delimiter::Brace) | Literal(..) | BinOp(Minus) => true,
            Ident(name, IdentIsRaw::No) if name.is_bool_lit() => true,
            Interpolated(ref nt) => matches!(&**nt, NtExpr(..) | NtBlock(..) | NtLiteral(..)),
            OpenDelim(Delimiter::Invisible(InvisibleOrigin::MetaVar(
                MetaVarKind::Expr { .. } | MetaVarKind::Block | MetaVarKind::Literal,
            ))) => true,
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
    /// Keep this in sync with and `Lit::from_token`, excluding unary negation.
    pub fn can_begin_literal_maybe_minus(&self) -> bool {
        match self.uninterpolate().kind {
            Literal(..) | BinOp(Minus) => true,
            Ident(name, IdentIsRaw::No) if name.is_bool_lit() => true,
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
            OpenDelim(Delimiter::Invisible(InvisibleOrigin::MetaVar(mv_kind))) => match mv_kind {
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
            Interpolated(ref nt) => match &**nt {
                NtLiteral(_) => true,
                NtExpr(e) => match &e.kind {
                    ast::ExprKind::Lit(_) => true,
                    _ => false,
                },
                _ => false,
            },
            OpenDelim(Delimiter::Invisible(InvisibleOrigin::MetaVar(mv_kind))) => match mv_kind {
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
    pub fn is_whole_expr(&self) -> bool {
        if let Interpolated(nt) = &self.kind
            && let NtExpr(_) | NtLiteral(_) | NtBlock(_) = &**nt
        {
            true
        } else {
            matches!(self.is_metavar_seq(), Some(MetaVarKind::Path))
        }
    }

    /// Is the token an interpolated block (`$b:block`)?
    pub fn is_whole_block(&self) -> bool {
        if let Interpolated(nt) = &self.kind
            && let NtBlock(..) = &**nt
        {
            return true;
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
        self == &PathSep
            || self.is_qpath_start()
            || matches!(self.is_metavar_seq(), Some(MetaVarKind::Path))
            || self.is_path_segment_keyword()
            || self.is_ident() && !self.is_reserved_ident()
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

    /// Don't use this unless you're doing something very loose and heuristic-y.
    pub fn is_any_keyword(&self) -> bool {
        self.is_non_raw_ident_where(Ident::is_any_keyword)
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
            OpenDelim(Delimiter::Invisible(InvisibleOrigin::MetaVar(kind))) => Some(kind),
            _ => None,
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
                Colon => PathSep,
                _ => return None,
            },
            SingleQuote => match joint.kind {
                Ident(name, is_raw) => Lifetime(Symbol::intern(&format!("'{name}")), is_raw),
                _ => return None,
            },

            Le | EqEq | Ne | Ge | AndAnd | OrOr | Tilde | BinOpEq(..) | At | DotDotDot
            | DotDotEq | Comma | Semi | PathSep | RArrow | LArrow | FatArrow | Pound | Dollar
            | Question | OpenDelim(..) | CloseDelim(..) | Literal(..) | Ident(..) | NtIdent(..)
            | Lifetime(..) | NtLifetime(..) | Interpolated(..) | DocComment(..) | Eof => {
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

#[derive(Clone, Encodable, Decodable)]
/// For interpolation during macro expansion.
pub enum Nonterminal {
    NtItem(P<ast::Item>),
    NtBlock(P<ast::Block>),
    NtStmt(P<ast::Stmt>),
    NtExpr(P<ast::Expr>),
    NtLiteral(P<ast::Expr>),
}

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

impl Nonterminal {
    pub fn use_span(&self) -> Span {
        match self {
            NtItem(item) => item.span,
            NtBlock(block) => block.span,
            NtStmt(stmt) => stmt.span,
            NtExpr(expr) | NtLiteral(expr) => expr.span,
        }
    }

    pub fn descr(&self) -> &'static str {
        match self {
            NtItem(..) => "item",
            NtBlock(..) => "block",
            NtStmt(..) => "statement",
            NtExpr(..) => "expression",
            NtLiteral(..) => "literal",
        }
    }
}

impl PartialEq for Nonterminal {
    fn eq(&self, _rhs: &Self) -> bool {
        // FIXME: Assume that all nonterminals are not equal, we can't compare them
        // correctly based on data from AST. This will prevent them from matching each other
        // in macros. The comparison will become possible only when each nonterminal has an
        // attached token stream from which it was parsed.
        false
    }
}

impl fmt::Debug for Nonterminal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            NtItem(..) => f.pad("NtItem(..)"),
            NtBlock(..) => f.pad("NtBlock(..)"),
            NtStmt(..) => f.pad("NtStmt(..)"),
            NtExpr(..) => f.pad("NtExpr(..)"),
            NtLiteral(..) => f.pad("NtLiteral(..)"),
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

// Some types are used a lot. Make sure they don't unintentionally get bigger.
#[cfg(target_pointer_width = "64")]
mod size_asserts {
    use rustc_data_structures::static_assert_size;

    use super::*;
    // tidy-alphabetical-start
    static_assert_size!(Lit, 12);
    static_assert_size!(LitKind, 2);
    static_assert_size!(Nonterminal, 16);
    static_assert_size!(Token, 24);
    static_assert_size!(TokenKind, 16);
    // tidy-alphabetical-end
}
