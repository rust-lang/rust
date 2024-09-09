//! `tt` crate defines a `TokenTree` data structure: this is the interface (both
//! input and output) of macros. It closely mirrors `proc_macro` crate's
//! `TokenTree`.

#![cfg_attr(feature = "in-rust-tree", feature(rustc_private))]

#[cfg(not(feature = "in-rust-tree"))]
extern crate ra_ap_rustc_lexer as rustc_lexer;
#[cfg(feature = "in-rust-tree")]
extern crate rustc_lexer;

pub mod buffer;
pub mod iter;

use std::fmt;

use intern::Symbol;
use stdx::{impl_from, itertools::Itertools as _};

pub use text_size::{TextRange, TextSize};

#[derive(Clone, PartialEq, Debug)]
pub struct Lit {
    pub kind: LitKind,
    pub symbol: Symbol,
    pub suffix: Option<Symbol>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum IdentIsRaw {
    No,
    Yes,
}
impl IdentIsRaw {
    pub fn yes(self) -> bool {
        matches!(self, IdentIsRaw::Yes)
    }
    pub fn no(&self) -> bool {
        matches!(self, IdentIsRaw::No)
    }
    pub fn as_str(self) -> &'static str {
        match self {
            IdentIsRaw::No => "",
            IdentIsRaw::Yes => "r#",
        }
    }
    pub fn split_from_symbol(sym: &str) -> (Self, &str) {
        if let Some(sym) = sym.strip_prefix("r#") {
            (IdentIsRaw::Yes, sym)
        } else {
            (IdentIsRaw::No, sym)
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum LitKind {
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
    Err(()),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TokenTree<S = u32> {
    Leaf(Leaf<S>),
    Subtree(Subtree<S>),
}
impl_from!(Leaf<S>, Subtree<S> for TokenTree);
impl<S: Copy> TokenTree<S> {
    pub fn empty(span: S) -> Self {
        Self::Subtree(Subtree {
            delimiter: Delimiter::invisible_spanned(span),
            token_trees: Box::new([]),
        })
    }

    pub fn subtree_or_wrap(self, span: DelimSpan<S>) -> Subtree<S> {
        match self {
            TokenTree::Leaf(_) => Subtree {
                delimiter: Delimiter::invisible_delim_spanned(span),
                token_trees: Box::new([self]),
            },
            TokenTree::Subtree(s) => s,
        }
    }

    pub fn first_span(&self) -> S {
        match self {
            TokenTree::Leaf(l) => *l.span(),
            TokenTree::Subtree(s) => s.delimiter.open,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Leaf<S> {
    Literal(Literal<S>),
    Punct(Punct<S>),
    Ident(Ident<S>),
}

impl<S> Leaf<S> {
    pub fn span(&self) -> &S {
        match self {
            Leaf::Literal(it) => &it.span,
            Leaf::Punct(it) => &it.span,
            Leaf::Ident(it) => &it.span,
        }
    }
}
impl_from!(Literal<S>, Punct<S>, Ident<S> for Leaf);

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Subtree<S> {
    pub delimiter: Delimiter<S>,
    pub token_trees: Box<[TokenTree<S>]>,
}

impl<S: Copy> Subtree<S> {
    pub fn empty(span: DelimSpan<S>) -> Self {
        Subtree { delimiter: Delimiter::invisible_delim_spanned(span), token_trees: Box::new([]) }
    }

    /// This is slow, and should be avoided, as it will always reallocate!
    pub fn push(&mut self, subtree: TokenTree<S>) {
        let mut mutable_trees = std::mem::take(&mut self.token_trees).into_vec();

        // Reserve exactly space for one element, to avoid `into_boxed_slice` having to reallocate again.
        mutable_trees.reserve_exact(1);
        mutable_trees.push(subtree);

        self.token_trees = mutable_trees.into_boxed_slice();
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct SubtreeBuilder<S> {
    pub delimiter: Delimiter<S>,
    pub token_trees: Vec<TokenTree<S>>,
}

impl<S> SubtreeBuilder<S> {
    pub fn build(self) -> Subtree<S> {
        Subtree { delimiter: self.delimiter, token_trees: self.token_trees.into_boxed_slice() }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct DelimSpan<S> {
    pub open: S,
    pub close: S,
}

impl<Span: Copy> DelimSpan<Span> {
    pub fn from_single(sp: Span) -> Self {
        DelimSpan { open: sp, close: sp }
    }

    pub fn from_pair(open: Span, close: Span) -> Self {
        DelimSpan { open, close }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Delimiter<S> {
    pub open: S,
    pub close: S,
    pub kind: DelimiterKind,
}

impl<S: Copy> Delimiter<S> {
    pub const fn invisible_spanned(span: S) -> Self {
        Delimiter { open: span, close: span, kind: DelimiterKind::Invisible }
    }

    pub const fn invisible_delim_spanned(span: DelimSpan<S>) -> Self {
        Delimiter { open: span.open, close: span.close, kind: DelimiterKind::Invisible }
    }

    pub fn delim_span(&self) -> DelimSpan<S> {
        DelimSpan { open: self.open, close: self.close }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DelimiterKind {
    Parenthesis,
    Brace,
    Bracket,
    Invisible,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Literal<S> {
    // escaped
    pub symbol: Symbol,
    pub span: S,
    pub kind: LitKind,
    pub suffix: Option<Symbol>,
}

pub fn token_to_literal<S>(text: &str, span: S) -> Literal<S>
where
    S: Copy,
{
    use rustc_lexer::LiteralKind;

    let token = rustc_lexer::tokenize(text).next_tuple();
    let Some((rustc_lexer::Token {
        kind: rustc_lexer::TokenKind::Literal { kind, suffix_start },
        ..
    },)) = token
    else {
        return Literal {
            span,
            symbol: Symbol::intern(text),
            kind: LitKind::Err(()),
            suffix: None,
        };
    };

    let (kind, start_offset, end_offset) = match kind {
        LiteralKind::Int { .. } => (LitKind::Integer, 0, 0),
        LiteralKind::Float { .. } => (LitKind::Float, 0, 0),
        LiteralKind::Char { terminated } => (LitKind::Char, 1, terminated as usize),
        LiteralKind::Byte { terminated } => (LitKind::Byte, 2, terminated as usize),
        LiteralKind::Str { terminated } => (LitKind::Str, 1, terminated as usize),
        LiteralKind::ByteStr { terminated } => (LitKind::ByteStr, 2, terminated as usize),
        LiteralKind::CStr { terminated } => (LitKind::CStr, 2, terminated as usize),
        LiteralKind::RawStr { n_hashes } => (
            LitKind::StrRaw(n_hashes.unwrap_or_default()),
            2 + n_hashes.unwrap_or_default() as usize,
            1 + n_hashes.unwrap_or_default() as usize,
        ),
        LiteralKind::RawByteStr { n_hashes } => (
            LitKind::ByteStrRaw(n_hashes.unwrap_or_default()),
            3 + n_hashes.unwrap_or_default() as usize,
            1 + n_hashes.unwrap_or_default() as usize,
        ),
        LiteralKind::RawCStr { n_hashes } => (
            LitKind::CStrRaw(n_hashes.unwrap_or_default()),
            3 + n_hashes.unwrap_or_default() as usize,
            1 + n_hashes.unwrap_or_default() as usize,
        ),
    };

    let (lit, suffix) = text.split_at(suffix_start as usize);
    let lit = &lit[start_offset..lit.len() - end_offset];
    let suffix = match suffix {
        "" | "_" => None,
        suffix => Some(Symbol::intern(suffix)),
    };

    Literal { span, symbol: Symbol::intern(lit), kind, suffix }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Punct<S> {
    pub char: char,
    pub spacing: Spacing,
    pub span: S,
}

/// Indicates whether a token can join with the following token to form a
/// compound token. Used for conversions to `proc_macro::Spacing`. Also used to
/// guide pretty-printing, which is where the `JointHidden` value (which isn't
/// part of `proc_macro::Spacing`) comes in useful.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Spacing {
    /// The token cannot join with the following token to form a compound
    /// token.
    ///
    /// In token streams parsed from source code, the compiler will use `Alone`
    /// for any token immediately followed by whitespace, a non-doc comment, or
    /// EOF.
    ///
    /// When constructing token streams within the compiler, use this for each
    /// token that (a) should be pretty-printed with a space after it, or (b)
    /// is the last token in the stream. (In the latter case the choice of
    /// spacing doesn't matter because it is never used for the last token. We
    /// arbitrarily use `Alone`.)
    ///
    /// Converts to `proc_macro::Spacing::Alone`, and
    /// `proc_macro::Spacing::Alone` converts back to this.
    Alone,

    /// The token can join with the following token to form a compound token.
    ///
    /// In token streams parsed from source code, the compiler will use `Joint`
    /// for any token immediately followed by punctuation (as determined by
    /// `Token::is_punct`).
    ///
    /// When constructing token streams within the compiler, use this for each
    /// token that (a) should be pretty-printed without a space after it, and
    /// (b) is followed by a punctuation token.
    ///
    /// Converts to `proc_macro::Spacing::Joint`, and
    /// `proc_macro::Spacing::Joint` converts back to this.
    Joint,

    /// The token can join with the following token to form a compound token,
    /// but this will not be visible at the proc macro level. (This is what the
    /// `Hidden` means; see below.)
    ///
    /// In token streams parsed from source code, the compiler will use
    /// `JointHidden` for any token immediately followed by anything not
    /// covered by the `Alone` and `Joint` cases: an identifier, lifetime,
    /// literal, delimiter, doc comment.
    ///
    /// When constructing token streams, use this for each token that (a)
    /// should be pretty-printed without a space after it, and (b) is followed
    /// by a non-punctuation token.
    ///
    /// Converts to `proc_macro::Spacing::Alone`, but
    /// `proc_macro::Spacing::Alone` converts back to `token::Spacing::Alone`.
    /// Because of that, pretty-printing of `TokenStream`s produced by proc
    /// macros is unavoidably uglier (with more whitespace between tokens) than
    /// pretty-printing of `TokenStream`'s produced by other means (i.e. parsed
    /// source code, internally constructed token streams, and token streams
    /// produced by declarative macros).
    JointHidden,
}

/// Identifier or keyword.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Ident<S> {
    pub sym: Symbol,
    pub span: S,
    pub is_raw: IdentIsRaw,
}

impl<S> Ident<S> {
    pub fn new(text: &str, span: S) -> Self {
        // let raw_stripped = IdentIsRaw::split_from_symbol(text.as_ref());
        let (is_raw, text) = IdentIsRaw::split_from_symbol(text);
        Ident { sym: Symbol::intern(text), span, is_raw }
    }
}

fn print_debug_subtree<S: fmt::Debug>(
    f: &mut fmt::Formatter<'_>,
    subtree: &Subtree<S>,
    level: usize,
) -> fmt::Result {
    let align = "  ".repeat(level);

    let Delimiter { kind, open, close } = &subtree.delimiter;
    let delim = match kind {
        DelimiterKind::Invisible => "$$",
        DelimiterKind::Parenthesis => "()",
        DelimiterKind::Brace => "{}",
        DelimiterKind::Bracket => "[]",
    };

    write!(f, "{align}SUBTREE {delim} ",)?;
    fmt::Debug::fmt(&open, f)?;
    write!(f, " ")?;
    fmt::Debug::fmt(&close, f)?;
    if !subtree.token_trees.is_empty() {
        writeln!(f)?;
        for (idx, child) in subtree.token_trees.iter().enumerate() {
            print_debug_token(f, child, level + 1)?;
            if idx != subtree.token_trees.len() - 1 {
                writeln!(f)?;
            }
        }
    }

    Ok(())
}

fn print_debug_token<S: fmt::Debug>(
    f: &mut fmt::Formatter<'_>,
    tkn: &TokenTree<S>,
    level: usize,
) -> fmt::Result {
    let align = "  ".repeat(level);

    match tkn {
        TokenTree::Leaf(leaf) => match leaf {
            Leaf::Literal(lit) => {
                write!(
                    f,
                    "{}LITERAL {:?} {}{} {:#?}",
                    align,
                    lit.kind,
                    lit.symbol,
                    lit.suffix.as_ref().map(|it| it.as_str()).unwrap_or(""),
                    lit.span
                )?;
            }
            Leaf::Punct(punct) => {
                write!(
                    f,
                    "{}PUNCH   {} [{}] {:#?}",
                    align,
                    punct.char,
                    if punct.spacing == Spacing::Alone { "alone" } else { "joint" },
                    punct.span
                )?;
            }
            Leaf::Ident(ident) => {
                write!(
                    f,
                    "{}IDENT   {}{} {:#?}",
                    align,
                    ident.is_raw.as_str(),
                    ident.sym,
                    ident.span
                )?;
            }
        },
        TokenTree::Subtree(subtree) => {
            print_debug_subtree(f, subtree, level)?;
        }
    }

    Ok(())
}

impl<S: fmt::Debug> fmt::Debug for Subtree<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        print_debug_subtree(f, self, 0)
    }
}

impl<S> fmt::Display for TokenTree<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenTree::Leaf(it) => fmt::Display::fmt(it, f),
            TokenTree::Subtree(it) => fmt::Display::fmt(it, f),
        }
    }
}

impl<S> fmt::Display for Subtree<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (l, r) = match self.delimiter.kind {
            DelimiterKind::Parenthesis => ("(", ")"),
            DelimiterKind::Brace => ("{", "}"),
            DelimiterKind::Bracket => ("[", "]"),
            DelimiterKind::Invisible => ("", ""),
        };
        f.write_str(l)?;
        let mut needs_space = false;
        for tt in self.token_trees.iter() {
            if needs_space {
                f.write_str(" ")?;
            }
            needs_space = true;
            match tt {
                TokenTree::Leaf(Leaf::Punct(p)) => {
                    needs_space = p.spacing == Spacing::Alone;
                    fmt::Display::fmt(p, f)?;
                }
                tt => fmt::Display::fmt(tt, f)?,
            }
        }
        f.write_str(r)?;
        Ok(())
    }
}

impl<S> fmt::Display for Leaf<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Leaf::Ident(it) => fmt::Display::fmt(it, f),
            Leaf::Literal(it) => fmt::Display::fmt(it, f),
            Leaf::Punct(it) => fmt::Display::fmt(it, f),
        }
    }
}

impl<S> fmt::Display for Ident<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.is_raw.as_str(), f)?;
        fmt::Display::fmt(&self.sym, f)
    }
}

impl<S> fmt::Display for Literal<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            LitKind::Byte => write!(f, "b'{}'", self.symbol),
            LitKind::Char => write!(f, "'{}'", self.symbol),
            LitKind::Integer | LitKind::Float | LitKind::Err(_) => write!(f, "{}", self.symbol),
            LitKind::Str => write!(f, "\"{}\"", self.symbol),
            LitKind::ByteStr => write!(f, "b\"{}\"", self.symbol),
            LitKind::CStr => write!(f, "c\"{}\"", self.symbol),
            LitKind::StrRaw(num_of_hashes) => {
                let num_of_hashes = num_of_hashes as usize;
                write!(
                    f,
                    r#"r{0:#<num_of_hashes$}"{text}"{0:#<num_of_hashes$}"#,
                    "",
                    text = self.symbol
                )
            }
            LitKind::ByteStrRaw(num_of_hashes) => {
                let num_of_hashes = num_of_hashes as usize;
                write!(
                    f,
                    r#"br{0:#<num_of_hashes$}"{text}"{0:#<num_of_hashes$}"#,
                    "",
                    text = self.symbol
                )
            }
            LitKind::CStrRaw(num_of_hashes) => {
                let num_of_hashes = num_of_hashes as usize;
                write!(
                    f,
                    r#"cr{0:#<num_of_hashes$}"{text}"{0:#<num_of_hashes$}"#,
                    "",
                    text = self.symbol
                )
            }
        }?;
        if let Some(suffix) = &self.suffix {
            write!(f, "{}", suffix)?;
        }
        Ok(())
    }
}

impl<S> fmt::Display for Punct<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.char, f)
    }
}

impl<S> Subtree<S> {
    /// Count the number of tokens recursively
    pub fn count(&self) -> usize {
        let children_count = self
            .token_trees
            .iter()
            .map(|c| match c {
                TokenTree::Subtree(c) => c.count(),
                TokenTree::Leaf(_) => 0,
            })
            .sum::<usize>();

        self.token_trees.len() + children_count
    }
}

impl<S> Subtree<S> {
    /// A simple line string used for debugging
    pub fn as_debug_string(&self) -> String {
        let delim = match self.delimiter.kind {
            DelimiterKind::Brace => ("{", "}"),
            DelimiterKind::Bracket => ("[", "]"),
            DelimiterKind::Parenthesis => ("(", ")"),
            DelimiterKind::Invisible => ("$", "$"),
        };

        let mut res = String::new();
        res.push_str(delim.0);
        let mut last = None;
        for child in self.token_trees.iter() {
            let s = match child {
                TokenTree::Leaf(it) => {
                    let s = match it {
                        Leaf::Literal(it) => it.symbol.to_string(),
                        Leaf::Punct(it) => it.char.to_string(),
                        Leaf::Ident(it) => format!("{}{}", it.is_raw.as_str(), it.sym),
                    };
                    match (it, last) {
                        (Leaf::Ident(_), Some(&TokenTree::Leaf(Leaf::Ident(_)))) => {
                            " ".to_owned() + &s
                        }
                        (Leaf::Punct(_), Some(TokenTree::Leaf(Leaf::Punct(punct)))) => {
                            if punct.spacing == Spacing::Alone {
                                " ".to_owned() + &s
                            } else {
                                s
                            }
                        }
                        _ => s,
                    }
                }
                TokenTree::Subtree(it) => it.as_debug_string(),
            };
            res.push_str(&s);
            last = Some(child);
        }

        res.push_str(delim.1);
        res
    }
}

pub fn pretty<S>(tkns: &[TokenTree<S>]) -> String {
    fn tokentree_to_text<S>(tkn: &TokenTree<S>) -> String {
        match tkn {
            TokenTree::Leaf(Leaf::Ident(ident)) => {
                format!("{}{}", ident.is_raw.as_str(), ident.sym)
            }
            TokenTree::Leaf(Leaf::Literal(literal)) => format!("{literal}"),
            TokenTree::Leaf(Leaf::Punct(punct)) => format!("{}", punct.char),
            TokenTree::Subtree(subtree) => {
                let content = pretty(&subtree.token_trees);
                let (open, close) = match subtree.delimiter.kind {
                    DelimiterKind::Brace => ("{", "}"),
                    DelimiterKind::Bracket => ("[", "]"),
                    DelimiterKind::Parenthesis => ("(", ")"),
                    DelimiterKind::Invisible => ("", ""),
                };
                format!("{open}{content}{close}")
            }
        }
    }

    tkns.iter()
        .fold((String::new(), true), |(last, last_to_joint), tkn| {
            let s = [last, tokentree_to_text(tkn)].join(if last_to_joint { "" } else { " " });
            let mut is_joint = false;
            if let TokenTree::Leaf(Leaf::Punct(punct)) = tkn {
                if punct.spacing == Spacing::Joint {
                    is_joint = true;
                }
            }
            (s, is_joint)
        })
        .0
}
