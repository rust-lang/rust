//! `tt` crate defines a `TokenTree` data structure: this is the interface (both
//! input and output) of macros.
//!
//! The `TokenTree` is semantically a tree, but for performance reasons it is stored as a flat structure.

#![cfg_attr(feature = "in-rust-tree", feature(rustc_private))]

#[cfg(feature = "in-rust-tree")]
extern crate rustc_driver as _;

#[cfg(not(feature = "in-rust-tree"))]
extern crate ra_ap_rustc_lexer as rustc_lexer;
#[cfg(feature = "in-rust-tree")]
extern crate rustc_lexer;

pub mod buffer;
pub mod iter;
mod storage;

use std::{fmt, slice::SliceIndex};

use arrayvec::ArrayString;
use buffer::Cursor;
use intern::Symbol;
use stdx::{impl_from, itertools::Itertools as _};

pub use span::Span;
pub use text_size::{TextRange, TextSize};

use crate::storage::{CompressedSpanPart, SpanStorage};

pub use self::iter::{TtElement, TtIter};
pub use self::storage::{TopSubtree, TopSubtreeBuilder};

pub const MAX_GLUED_PUNCT_LEN: usize = 3;

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
pub enum TokenTree {
    Leaf(Leaf),
    Subtree(Subtree),
}
impl_from!(Leaf, Subtree for TokenTree);
impl TokenTree {
    pub fn first_span(&self) -> Span {
        match self {
            TokenTree::Leaf(l) => *l.span(),
            TokenTree::Subtree(s) => s.delimiter.open,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Leaf {
    Literal(Literal),
    Punct(Punct),
    Ident(Ident),
}

impl Leaf {
    pub fn span(&self) -> &Span {
        match self {
            Leaf::Literal(it) => &it.span,
            Leaf::Punct(it) => &it.span,
            Leaf::Ident(it) => &it.span,
        }
    }
}
impl_from!(Literal, Punct, Ident for Leaf);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Subtree {
    pub delimiter: Delimiter,
    /// Number of following token trees that belong to this subtree, excluding this subtree.
    pub len: u32,
}

impl Subtree {
    pub fn usize_len(&self) -> usize {
        self.len as usize
    }
}

#[rust_analyzer::macro_style(braces)]
macro_rules! dispatch_ref {
    (
        match $scrutinee:expr => $tt:ident => $body:expr
    ) => {
        match $scrutinee {
            $crate::TokenTreesReprRef::SpanStorage32($tt) => $body,
            $crate::TokenTreesReprRef::SpanStorage64($tt) => $body,
            $crate::TokenTreesReprRef::SpanStorage96($tt) => $body,
        }
    };
}
use dispatch_ref;

#[derive(Clone, Copy)]
enum TokenTreesReprRef<'a> {
    SpanStorage32(&'a [crate::storage::TokenTree<crate::storage::SpanStorage32>]),
    SpanStorage64(&'a [crate::storage::TokenTree<crate::storage::SpanStorage64>]),
    SpanStorage96(&'a [crate::storage::TokenTree<crate::storage::SpanStorage96>]),
}

impl<'a> TokenTreesReprRef<'a> {
    #[inline]
    fn get<I>(&self, index: I) -> Option<Self>
    where
        I: SliceIndex<
                [crate::storage::TokenTree<crate::storage::SpanStorage32>],
                Output = [crate::storage::TokenTree<crate::storage::SpanStorage32>],
            >,
        I: SliceIndex<
                [crate::storage::TokenTree<crate::storage::SpanStorage64>],
                Output = [crate::storage::TokenTree<crate::storage::SpanStorage64>],
            >,
        I: SliceIndex<
                [crate::storage::TokenTree<crate::storage::SpanStorage96>],
                Output = [crate::storage::TokenTree<crate::storage::SpanStorage96>],
            >,
    {
        Some(match self {
            TokenTreesReprRef::SpanStorage32(tt) => {
                TokenTreesReprRef::SpanStorage32(tt.get(index)?)
            }
            TokenTreesReprRef::SpanStorage64(tt) => {
                TokenTreesReprRef::SpanStorage64(tt.get(index)?)
            }
            TokenTreesReprRef::SpanStorage96(tt) => {
                TokenTreesReprRef::SpanStorage96(tt.get(index)?)
            }
        })
    }
}

#[derive(Clone, Copy)]
pub struct TokenTreesView<'a> {
    repr: TokenTreesReprRef<'a>,
    span_parts: &'a [CompressedSpanPart],
}

impl<'a> TokenTreesView<'a> {
    pub fn empty() -> Self {
        Self { repr: TokenTreesReprRef::SpanStorage32(&[]), span_parts: &[] }
    }

    pub fn iter(&self) -> TtIter<'a> {
        TtIter::new(*self)
    }

    pub fn cursor(&self) -> Cursor<'a> {
        Cursor::new(*self)
    }

    pub fn len(&self) -> usize {
        dispatch_ref! {
            match self.repr => tt => tt.len()
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn try_into_subtree(self) -> Option<SubtreeView<'a>> {
        let is_subtree = dispatch_ref! {
            match self.repr => tt => matches!(
                tt.first(),
                Some(crate::storage::TokenTree::Subtree { len, .. }) if (*len as usize) == (tt.len() - 1)
            )
        };
        if is_subtree { Some(SubtreeView(self)) } else { None }
    }

    pub fn strip_invisible(self) -> TokenTreesView<'a> {
        self.try_into_subtree().map(|subtree| subtree.strip_invisible()).unwrap_or(self)
    }

    pub fn split(
        self,
        mut split_fn: impl FnMut(TtElement<'a>) -> bool,
    ) -> impl Iterator<Item = TokenTreesView<'a>> {
        let mut subtree_iter = self.iter();
        let mut need_to_yield_even_if_empty = true;

        std::iter::from_fn(move || {
            if subtree_iter.is_empty() && !need_to_yield_even_if_empty {
                return None;
            };

            need_to_yield_even_if_empty = false;
            let savepoint = subtree_iter.savepoint();
            let mut result = subtree_iter.from_savepoint(savepoint);
            while let Some(tt) = subtree_iter.next() {
                if split_fn(tt) {
                    need_to_yield_even_if_empty = true;
                    break;
                }
                result = subtree_iter.from_savepoint(savepoint);
            }
            Some(result)
        })
    }

    pub fn first_span(&self) -> Option<Span> {
        Some(dispatch_ref! {
            match self.repr => tt => tt.first()?.first_span().span(self.span_parts)
        })
    }

    pub fn last_span(&self) -> Option<Span> {
        Some(dispatch_ref! {
            match self.repr => tt => tt.last()?.last_span().span(self.span_parts)
        })
    }

    pub fn iter_flat_tokens(self) -> impl ExactSizeIterator<Item = TokenTree> + use<'a> {
        (0..self.len()).map(move |idx| {
            dispatch_ref! {
                match self.repr => tt => tt[idx].to_api(self.span_parts)
            }
        })
    }
}

impl fmt::Debug for TokenTreesView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut iter = self.iter();
        while let Some(tt) = iter.next() {
            print_debug_token(f, 0, tt)?;
            if !iter.is_empty() {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

impl fmt::Display for TokenTreesView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        return token_trees_display(f, self.iter());

        fn subtree_display(
            subtree: &Subtree,
            f: &mut fmt::Formatter<'_>,
            iter: TtIter<'_>,
        ) -> fmt::Result {
            let (l, r) = match subtree.delimiter.kind {
                DelimiterKind::Parenthesis => ("(", ")"),
                DelimiterKind::Brace => ("{", "}"),
                DelimiterKind::Bracket => ("[", "]"),
                DelimiterKind::Invisible => ("", ""),
            };
            f.write_str(l)?;
            token_trees_display(f, iter)?;
            f.write_str(r)?;
            Ok(())
        }

        fn token_trees_display(f: &mut fmt::Formatter<'_>, iter: TtIter<'_>) -> fmt::Result {
            let mut needs_space = false;
            for child in iter {
                if needs_space {
                    f.write_str(" ")?;
                }
                needs_space = true;

                match child {
                    TtElement::Leaf(Leaf::Punct(p)) => {
                        needs_space = p.spacing == Spacing::Alone;
                        fmt::Display::fmt(&p, f)?;
                    }
                    TtElement::Leaf(leaf) => fmt::Display::fmt(&leaf, f)?,
                    TtElement::Subtree(subtree, subtree_iter) => {
                        subtree_display(&subtree, f, subtree_iter)?
                    }
                }
            }
            Ok(())
        }
    }
}

#[derive(Clone, Copy)]
// Invariant: always starts with `Subtree` that covers the entire thing.
pub struct SubtreeView<'a>(TokenTreesView<'a>);

impl<'a> SubtreeView<'a> {
    pub fn as_token_trees(self) -> TokenTreesView<'a> {
        self.0
    }

    pub fn iter(&self) -> TtIter<'a> {
        self.token_trees().iter()
    }

    pub fn top_subtree(&self) -> Subtree {
        dispatch_ref! {
            match self.0.repr => tt => {
                let crate::storage::TokenTree::Subtree { len, delim_kind, open_span, close_span } =
                    &tt[0]
                else {
                    unreachable!("the first token tree is always the top subtree");
                };
                Subtree {
                    delimiter: Delimiter {
                        open: open_span.span(self.0.span_parts),
                        close: close_span.span(self.0.span_parts),
                        kind: *delim_kind,
                    },
                    len: *len,
                }
            }
        }
    }

    pub fn strip_invisible(&self) -> TokenTreesView<'a> {
        if self.top_subtree().delimiter.kind == DelimiterKind::Invisible {
            self.token_trees()
        } else {
            self.0
        }
    }

    pub fn token_trees(&self) -> TokenTreesView<'a> {
        let repr = match self.0.repr {
            TokenTreesReprRef::SpanStorage32(token_trees) => {
                TokenTreesReprRef::SpanStorage32(&token_trees[1..])
            }
            TokenTreesReprRef::SpanStorage64(token_trees) => {
                TokenTreesReprRef::SpanStorage64(&token_trees[1..])
            }
            TokenTreesReprRef::SpanStorage96(token_trees) => {
                TokenTreesReprRef::SpanStorage96(&token_trees[1..])
            }
        };
        TokenTreesView { repr, ..self.0 }
    }
}

impl fmt::Debug for SubtreeView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

impl fmt::Display for SubtreeView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct DelimSpan {
    pub open: Span,
    pub close: Span,
}

impl DelimSpan {
    pub fn from_single(sp: Span) -> Self {
        DelimSpan { open: sp, close: sp }
    }

    pub fn from_pair(open: Span, close: Span) -> Self {
        DelimSpan { open, close }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Delimiter {
    pub open: Span,
    pub close: Span,
    pub kind: DelimiterKind,
}

impl Delimiter {
    pub const fn invisible_spanned(span: Span) -> Self {
        Delimiter { open: span, close: span, kind: DelimiterKind::Invisible }
    }

    pub const fn invisible_delim_spanned(span: DelimSpan) -> Self {
        Delimiter { open: span.open, close: span.close, kind: DelimiterKind::Invisible }
    }

    pub fn delim_span(&self) -> DelimSpan {
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
pub struct Literal {
    /// Escaped, text then suffix concatenated.
    pub text_and_suffix: Symbol,
    pub span: Span,
    pub kind: LitKind,
    pub suffix_len: u8,
}

impl Literal {
    #[inline]
    pub fn text_and_suffix(&self) -> (&str, &str) {
        let text_and_suffix = self.text_and_suffix.as_str();
        text_and_suffix.split_at(text_and_suffix.len() - usize::from(self.suffix_len))
    }

    #[inline]
    pub fn text(&self) -> &str {
        self.text_and_suffix().0
    }

    #[inline]
    pub fn suffix(&self) -> &str {
        self.text_and_suffix().1
    }

    pub fn new(text: &str, span: Span, kind: LitKind, suffix: &str) -> Self {
        const MAX_INLINE_CAPACITY: usize = 30;
        let text_and_suffix = if suffix.is_empty() {
            Symbol::intern(text)
        } else if (text.len() + suffix.len()) < MAX_INLINE_CAPACITY {
            let mut text_and_suffix = ArrayString::<MAX_INLINE_CAPACITY>::new();
            text_and_suffix.push_str(text);
            text_and_suffix.push_str(suffix);
            Symbol::intern(&text_and_suffix)
        } else {
            let mut text_and_suffix = String::with_capacity(text.len() + suffix.len());
            text_and_suffix.push_str(text);
            text_and_suffix.push_str(suffix);
            Symbol::intern(&text_and_suffix)
        };

        Self { text_and_suffix, span, kind, suffix_len: suffix.len().try_into().unwrap() }
    }

    #[inline]
    pub fn new_no_suffix(text: &str, span: Span, kind: LitKind) -> Self {
        Self { text_and_suffix: Symbol::intern(text), span, kind, suffix_len: 0 }
    }
}

pub fn token_to_literal(text: &str, span: Span) -> Literal {
    use rustc_lexer::LiteralKind;

    let token = rustc_lexer::tokenize(text, rustc_lexer::FrontmatterAllowed::No).next_tuple();
    let Some((rustc_lexer::Token {
        kind: rustc_lexer::TokenKind::Literal { kind, suffix_start },
        ..
    },)) = token
    else {
        return Literal::new_no_suffix(text, span, LitKind::Err(()));
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
        "" | "_" => "",
        // ill-suffixed literals
        _ if !matches!(kind, LitKind::Integer | LitKind::Float | LitKind::Err(_)) => {
            return Literal::new_no_suffix(text, span, LitKind::Err(()));
        }
        suffix => suffix,
    };

    Literal::new(lit, span, kind, suffix)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Punct {
    pub char: char,
    pub spacing: Spacing,
    pub span: Span,
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
pub struct Ident {
    pub sym: Symbol,
    pub span: Span,
    pub is_raw: IdentIsRaw,
}

impl Ident {
    pub fn new(text: &str, span: Span) -> Self {
        // let raw_stripped = IdentIsRaw::split_from_symbol(text.as_ref());
        let (is_raw, text) = IdentIsRaw::split_from_symbol(text);
        Ident { sym: Symbol::intern(text), span, is_raw }
    }
}

fn print_debug_subtree(
    f: &mut fmt::Formatter<'_>,
    subtree: &Subtree,
    level: usize,
    iter: TtIter<'_>,
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
    write!(f, "{open:#?}")?;
    write!(f, " ")?;
    write!(f, "{close:#?}")?;
    for child in iter {
        writeln!(f)?;
        print_debug_token(f, level + 1, child)?;
    }

    Ok(())
}

fn print_debug_token(f: &mut fmt::Formatter<'_>, level: usize, tt: TtElement<'_>) -> fmt::Result {
    let align = "  ".repeat(level);

    match tt {
        TtElement::Leaf(leaf) => match leaf {
            Leaf::Literal(lit) => {
                let (text, suffix) = lit.text_and_suffix();
                write!(f, "{}LITERAL {:?} {}{} {:#?}", align, lit.kind, text, suffix, lit.span)?;
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
        TtElement::Subtree(subtree, subtree_iter) => {
            print_debug_subtree(f, &subtree, level, subtree_iter)?;
        }
    }

    Ok(())
}

impl fmt::Debug for TopSubtree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.view(), f)
    }
}

impl fmt::Display for TopSubtree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.view(), f)
    }
}

impl fmt::Display for Leaf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Leaf::Ident(it) => fmt::Display::fmt(it, f),
            Leaf::Literal(it) => fmt::Display::fmt(it, f),
            Leaf::Punct(it) => fmt::Display::fmt(it, f),
        }
    }
}

impl fmt::Display for Ident {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.is_raw.as_str(), f)?;
        fmt::Display::fmt(&self.sym, f)
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (text, suffix) = self.text_and_suffix();
        match self.kind {
            LitKind::Byte => write!(f, "b'{}'", text),
            LitKind::Char => write!(f, "'{}'", text),
            LitKind::Integer | LitKind::Float | LitKind::Err(_) => write!(f, "{}", text),
            LitKind::Str => write!(f, "\"{}\"", text),
            LitKind::ByteStr => write!(f, "b\"{}\"", text),
            LitKind::CStr => write!(f, "c\"{}\"", text),
            LitKind::StrRaw(num_of_hashes) => {
                let num_of_hashes = num_of_hashes as usize;
                write!(f, r#"r{0:#<num_of_hashes$}"{text}"{0:#<num_of_hashes$}"#, "", text = text)
            }
            LitKind::ByteStrRaw(num_of_hashes) => {
                let num_of_hashes = num_of_hashes as usize;
                write!(f, r#"br{0:#<num_of_hashes$}"{text}"{0:#<num_of_hashes$}"#, "", text = text)
            }
            LitKind::CStrRaw(num_of_hashes) => {
                let num_of_hashes = num_of_hashes as usize;
                write!(f, r#"cr{0:#<num_of_hashes$}"{text}"{0:#<num_of_hashes$}"#, "", text = text)
            }
        }?;
        write!(f, "{suffix}")?;
        Ok(())
    }
}

impl fmt::Display for Punct {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.char, f)
    }
}

impl Subtree {
    /// Count the number of tokens recursively
    pub fn count(&self) -> usize {
        self.usize_len()
    }
}

pub fn pretty(tkns: TokenTreesView<'_>) -> String {
    return dispatch_ref! {
        match tkns.repr => tt => pretty_impl(tt)
    };

    use crate::storage::TokenTree;

    fn tokentree_to_text<S: SpanStorage>(tkn: &TokenTree<S>, tkns: &mut &[TokenTree<S>]) -> String {
        match tkn {
            TokenTree::Ident { sym, is_raw, .. } => format!("{}{}", is_raw.as_str(), sym),
            &TokenTree::Literal { ref text_and_suffix, kind, suffix_len, span: _ } => {
                format!(
                    "{}",
                    Literal {
                        text_and_suffix: text_and_suffix.clone(),
                        span: Span {
                            range: TextRange::empty(TextSize::new(0)),
                            anchor: span::SpanAnchor {
                                file_id: span::EditionedFileId::from_raw(0),
                                ast_id: span::FIXUP_ERASED_FILE_AST_ID_MARKER
                            },
                            ctx: span::SyntaxContext::root(span::Edition::Edition2015)
                        },
                        kind,
                        suffix_len
                    }
                )
            }
            TokenTree::Punct { char, .. } => format!("{}", char),
            TokenTree::Subtree { len, delim_kind, .. } => {
                let (subtree_content, rest) = tkns.split_at(*len as usize);
                let content = pretty_impl(subtree_content);
                *tkns = rest;
                let (open, close) = match *delim_kind {
                    DelimiterKind::Brace => ("{", "}"),
                    DelimiterKind::Bracket => ("[", "]"),
                    DelimiterKind::Parenthesis => ("(", ")"),
                    DelimiterKind::Invisible => ("", ""),
                };
                format!("{open}{content}{close}")
            }
        }
    }

    fn pretty_impl<S: SpanStorage>(mut tkns: &[TokenTree<S>]) -> String {
        let mut last = String::new();
        let mut last_to_joint = true;

        while let Some((tkn, rest)) = tkns.split_first() {
            tkns = rest;
            last = [last, tokentree_to_text(tkn, &mut tkns)].join(if last_to_joint {
                ""
            } else {
                " "
            });
            last_to_joint = false;
            if let TokenTree::Punct { spacing, .. } = tkn
                && *spacing == Spacing::Joint
            {
                last_to_joint = true;
            }
        }
        last
    }
}

#[derive(Debug)]
pub enum TransformTtAction<'a> {
    Keep,
    ReplaceWith(TokenTreesView<'a>),
}

impl TransformTtAction<'_> {
    #[inline]
    pub fn remove() -> Self {
        Self::ReplaceWith(TokenTreesView::empty())
    }
}

/// This function takes a token tree, and calls `callback` with each token tree in it.
/// Then it does what the callback says: keeps the tt or replaces it with a (possibly empty)
/// tts view.
pub fn transform_tt<'b>(
    tt: &mut TopSubtree,
    mut callback: impl FnMut(TokenTree) -> TransformTtAction<'b>,
) {
    let mut tt_vec = tt.as_token_trees().iter_flat_tokens().collect::<Vec<_>>();

    // We need to keep a stack of the currently open subtrees, because we need to update
    // them if we change the number of items in them.
    let mut subtrees_stack = Vec::new();
    let mut i = 0;
    while i < tt_vec.len() {
        'pop_finished_subtrees: while let Some(&subtree_idx) = subtrees_stack.last() {
            let TokenTree::Subtree(subtree) = &tt_vec[subtree_idx] else {
                unreachable!("non-subtree on subtrees stack");
            };
            if i >= subtree_idx + 1 + subtree.usize_len() {
                subtrees_stack.pop();
            } else {
                break 'pop_finished_subtrees;
            }
        }

        let current = match &tt_vec[i] {
            TokenTree::Leaf(leaf) => TokenTree::Leaf(match leaf {
                Leaf::Literal(leaf) => Leaf::Literal(leaf.clone()),
                Leaf::Punct(leaf) => Leaf::Punct(*leaf),
                Leaf::Ident(leaf) => Leaf::Ident(leaf.clone()),
            }),
            TokenTree::Subtree(subtree) => TokenTree::Subtree(*subtree),
        };
        let action = callback(current);
        match action {
            TransformTtAction::Keep => {
                // This cannot be shared with the replaced case, because then we may push the same subtree
                // twice, and will update it twice which will lead to errors.
                if let TokenTree::Subtree(_) = &tt_vec[i] {
                    subtrees_stack.push(i);
                }

                i += 1;
            }
            TransformTtAction::ReplaceWith(replacement) => {
                let old_len = 1 + match &tt_vec[i] {
                    TokenTree::Leaf(_) => 0,
                    TokenTree::Subtree(subtree) => subtree.usize_len(),
                };
                let len_diff = replacement.len() as i64 - old_len as i64;
                tt_vec.splice(i..i + old_len, replacement.iter_flat_tokens());
                // Skip the newly inserted replacement, we don't want to visit it.
                i += replacement.len();

                for &subtree_idx in &subtrees_stack {
                    let TokenTree::Subtree(subtree) = &mut tt_vec[subtree_idx] else {
                        unreachable!("non-subtree on subtrees stack");
                    };
                    subtree.len = (i64::from(subtree.len) + len_diff).try_into().unwrap();
                }
            }
        }
    }

    *tt = TopSubtree::from_serialized(tt_vec);
}
