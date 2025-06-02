//! `tt` crate defines a `TokenTree` data structure: this is the interface (both
//! input and output) of macros.
//!
//! The `TokenTree` is semantically a tree, but for performance reasons it is stored as a flat structure.

#![cfg_attr(feature = "in-rust-tree", feature(rustc_private))]

#[cfg(not(feature = "in-rust-tree"))]
extern crate ra_ap_rustc_lexer as rustc_lexer;
#[cfg(feature = "in-rust-tree")]
extern crate rustc_lexer;

pub mod buffer;
pub mod iter;

use std::fmt;

use buffer::Cursor;
use intern::Symbol;
use iter::{TtElement, TtIter};
use stdx::{impl_from, itertools::Itertools as _};

pub use text_size::{TextRange, TextSize};

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
pub enum TokenTree<S = u32> {
    Leaf(Leaf<S>),
    Subtree(Subtree<S>),
}
impl_from!(Leaf<S>, Subtree<S> for TokenTree);
impl<S: Copy> TokenTree<S> {
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Subtree<S> {
    pub delimiter: Delimiter<S>,
    /// Number of following token trees that belong to this subtree, excluding this subtree.
    pub len: u32,
}

impl<S> Subtree<S> {
    pub fn usize_len(&self) -> usize {
        self.len as usize
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct TopSubtree<S>(pub Box<[TokenTree<S>]>);

impl<S: Copy> TopSubtree<S> {
    pub fn empty(span: DelimSpan<S>) -> Self {
        Self(Box::new([TokenTree::Subtree(Subtree {
            delimiter: Delimiter::invisible_delim_spanned(span),
            len: 0,
        })]))
    }

    pub fn invisible_from_leaves<const N: usize>(delim_span: S, leaves: [Leaf<S>; N]) -> Self {
        let mut builder = TopSubtreeBuilder::new(Delimiter::invisible_spanned(delim_span));
        builder.extend(leaves);
        builder.build()
    }

    pub fn from_token_trees(delimiter: Delimiter<S>, token_trees: TokenTreesView<'_, S>) -> Self {
        let mut builder = TopSubtreeBuilder::new(delimiter);
        builder.extend_with_tt(token_trees);
        builder.build()
    }

    pub fn from_subtree(subtree: SubtreeView<'_, S>) -> Self {
        Self(subtree.0.into())
    }

    pub fn view(&self) -> SubtreeView<'_, S> {
        SubtreeView::new(&self.0)
    }

    pub fn iter(&self) -> TtIter<'_, S> {
        self.view().iter()
    }

    pub fn top_subtree(&self) -> &Subtree<S> {
        self.view().top_subtree()
    }

    pub fn top_subtree_delimiter_mut(&mut self) -> &mut Delimiter<S> {
        let TokenTree::Subtree(subtree) = &mut self.0[0] else {
            unreachable!("the first token tree is always the top subtree");
        };
        &mut subtree.delimiter
    }

    pub fn token_trees(&self) -> TokenTreesView<'_, S> {
        self.view().token_trees()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TopSubtreeBuilder<S> {
    unclosed_subtree_indices: Vec<usize>,
    token_trees: Vec<TokenTree<S>>,
    last_closed_subtree: Option<usize>,
}

impl<S: Copy> TopSubtreeBuilder<S> {
    pub fn new(top_delimiter: Delimiter<S>) -> Self {
        let mut result = Self {
            unclosed_subtree_indices: Vec::new(),
            token_trees: Vec::new(),
            last_closed_subtree: None,
        };
        let top_subtree = TokenTree::Subtree(Subtree { delimiter: top_delimiter, len: 0 });
        result.token_trees.push(top_subtree);
        result
    }

    pub fn open(&mut self, delimiter_kind: DelimiterKind, open_span: S) {
        self.unclosed_subtree_indices.push(self.token_trees.len());
        self.token_trees.push(TokenTree::Subtree(Subtree {
            delimiter: Delimiter {
                open: open_span,
                close: open_span, // Will be overwritten on close.
                kind: delimiter_kind,
            },
            len: 0,
        }));
    }

    pub fn close(&mut self, close_span: S) {
        let last_unclosed_index = self
            .unclosed_subtree_indices
            .pop()
            .expect("attempt to close a `tt::Subtree` when none is open");
        let subtree_len = (self.token_trees.len() - last_unclosed_index - 1) as u32;
        let TokenTree::Subtree(subtree) = &mut self.token_trees[last_unclosed_index] else {
            unreachable!("unclosed token tree is always a subtree");
        };
        subtree.len = subtree_len;
        subtree.delimiter.close = close_span;
        self.last_closed_subtree = Some(last_unclosed_index);
    }

    /// You cannot call this consecutively, it will only work once after close.
    pub fn remove_last_subtree_if_invisible(&mut self) {
        let Some(last_subtree_idx) = self.last_closed_subtree else { return };
        if let TokenTree::Subtree(Subtree {
            delimiter: Delimiter { kind: DelimiterKind::Invisible, .. },
            ..
        }) = self.token_trees[last_subtree_idx]
        {
            self.token_trees.remove(last_subtree_idx);
            self.last_closed_subtree = None;
        }
    }

    pub fn push(&mut self, leaf: Leaf<S>) {
        self.token_trees.push(TokenTree::Leaf(leaf));
    }

    pub fn extend(&mut self, leaves: impl IntoIterator<Item = Leaf<S>>) {
        self.token_trees.extend(leaves.into_iter().map(TokenTree::Leaf));
    }

    /// This does not check the token trees are valid, beware!
    pub fn extend_tt_dangerous(&mut self, tt: impl IntoIterator<Item = TokenTree<S>>) {
        self.token_trees.extend(tt);
    }

    pub fn extend_with_tt(&mut self, tt: TokenTreesView<'_, S>) {
        self.token_trees.extend(tt.0.iter().cloned());
    }

    /// Like [`Self::extend_with_tt()`], but makes sure the new tokens will never be
    /// joint with whatever comes after them.
    pub fn extend_with_tt_alone(&mut self, tt: TokenTreesView<'_, S>) {
        if let Some((last, before_last)) = tt.0.split_last() {
            self.token_trees.reserve(tt.0.len());
            self.token_trees.extend(before_last.iter().cloned());
            let last = if let TokenTree::Leaf(Leaf::Punct(last)) = last {
                let mut last = *last;
                last.spacing = Spacing::Alone;
                TokenTree::Leaf(Leaf::Punct(last))
            } else {
                last.clone()
            };
            self.token_trees.push(last);
        }
    }

    pub fn expected_delimiters(&self) -> impl Iterator<Item = &Delimiter<S>> {
        self.unclosed_subtree_indices.iter().rev().map(|&subtree_idx| {
            let TokenTree::Subtree(subtree) = &self.token_trees[subtree_idx] else {
                unreachable!("unclosed token tree is always a subtree")
            };
            &subtree.delimiter
        })
    }

    /// Builds, and remove the top subtree if it has only one subtree child.
    pub fn build_skip_top_subtree(mut self) -> TopSubtree<S> {
        let top_tts = TokenTreesView::new(&self.token_trees[1..]);
        match top_tts.try_into_subtree() {
            Some(_) => {
                assert!(
                    self.unclosed_subtree_indices.is_empty(),
                    "attempt to build an unbalanced `TopSubtreeBuilder`"
                );
                TopSubtree(self.token_trees.drain(1..).collect())
            }
            None => self.build(),
        }
    }

    pub fn build(mut self) -> TopSubtree<S> {
        assert!(
            self.unclosed_subtree_indices.is_empty(),
            "attempt to build an unbalanced `TopSubtreeBuilder`"
        );
        let total_len = self.token_trees.len() as u32;
        let TokenTree::Subtree(top_subtree) = &mut self.token_trees[0] else {
            unreachable!("first token tree is always a subtree");
        };
        top_subtree.len = total_len - 1;
        TopSubtree(self.token_trees.into_boxed_slice())
    }

    pub fn restore_point(&self) -> SubtreeBuilderRestorePoint {
        SubtreeBuilderRestorePoint {
            unclosed_subtree_indices_len: self.unclosed_subtree_indices.len(),
            token_trees_len: self.token_trees.len(),
            last_closed_subtree: self.last_closed_subtree,
        }
    }

    pub fn restore(&mut self, restore_point: SubtreeBuilderRestorePoint) {
        self.unclosed_subtree_indices.truncate(restore_point.unclosed_subtree_indices_len);
        self.token_trees.truncate(restore_point.token_trees_len);
        self.last_closed_subtree = restore_point.last_closed_subtree;
    }
}

#[derive(Clone, Copy)]
pub struct SubtreeBuilderRestorePoint {
    unclosed_subtree_indices_len: usize,
    token_trees_len: usize,
    last_closed_subtree: Option<usize>,
}

#[derive(Clone, Copy)]
pub struct TokenTreesView<'a, S>(&'a [TokenTree<S>]);

impl<'a, S: Copy> TokenTreesView<'a, S> {
    pub fn new(tts: &'a [TokenTree<S>]) -> Self {
        if cfg!(debug_assertions) {
            tts.iter().enumerate().for_each(|(idx, tt)| {
                if let TokenTree::Subtree(tt) = &tt {
                    // `<` and not `<=` because `Subtree.len` does not include the subtree node itself.
                    debug_assert!(
                        idx + tt.usize_len() < tts.len(),
                        "`TokenTreeView::new()` was given a cut-in-half list"
                    );
                }
            });
        }
        Self(tts)
    }

    pub fn iter(&self) -> TtIter<'a, S> {
        TtIter::new(self.0)
    }

    pub fn cursor(&self) -> Cursor<'a, S> {
        Cursor::new(self.0)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn try_into_subtree(self) -> Option<SubtreeView<'a, S>> {
        if let Some(TokenTree::Subtree(subtree)) = self.0.first() {
            if subtree.usize_len() == (self.0.len() - 1) {
                return Some(SubtreeView::new(self.0));
            }
        }
        None
    }

    pub fn strip_invisible(self) -> TokenTreesView<'a, S> {
        self.try_into_subtree().map(|subtree| subtree.strip_invisible()).unwrap_or(self)
    }

    /// This returns a **flat** structure of tokens (subtrees will be represented by a single node
    /// preceding their children), so it isn't suited for most use cases, only for matching leaves
    /// at the beginning/end with no subtrees before them. If you need a structured pass, use [`TtIter`].
    pub fn flat_tokens(&self) -> &'a [TokenTree<S>] {
        self.0
    }

    pub fn split(
        self,
        mut split_fn: impl FnMut(TtElement<'a, S>) -> bool,
    ) -> impl Iterator<Item = TokenTreesView<'a, S>> {
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
}

impl<S: fmt::Debug + Copy> fmt::Debug for TokenTreesView<'_, S> {
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

impl<S: Copy> fmt::Display for TokenTreesView<'_, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        return token_trees_display(f, self.iter());

        fn subtree_display<S>(
            subtree: &Subtree<S>,
            f: &mut fmt::Formatter<'_>,
            iter: TtIter<'_, S>,
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

        fn token_trees_display<S>(f: &mut fmt::Formatter<'_>, iter: TtIter<'_, S>) -> fmt::Result {
            let mut needs_space = false;
            for child in iter {
                if needs_space {
                    f.write_str(" ")?;
                }
                needs_space = true;

                match child {
                    TtElement::Leaf(Leaf::Punct(p)) => {
                        needs_space = p.spacing == Spacing::Alone;
                        fmt::Display::fmt(p, f)?;
                    }
                    TtElement::Leaf(leaf) => fmt::Display::fmt(leaf, f)?,
                    TtElement::Subtree(subtree, subtree_iter) => {
                        subtree_display(subtree, f, subtree_iter)?
                    }
                }
            }
            Ok(())
        }
    }
}

#[derive(Clone, Copy)]
// Invariant: always starts with `Subtree` that covers the entire thing.
pub struct SubtreeView<'a, S>(&'a [TokenTree<S>]);

impl<'a, S: Copy> SubtreeView<'a, S> {
    pub fn new(tts: &'a [TokenTree<S>]) -> Self {
        if cfg!(debug_assertions) {
            let TokenTree::Subtree(subtree) = &tts[0] else {
                panic!("first token tree must be a subtree in `SubtreeView`");
            };
            assert_eq!(
                subtree.usize_len(),
                tts.len() - 1,
                "subtree must cover the entire `SubtreeView`"
            );
        }
        Self(tts)
    }

    pub fn as_token_trees(self) -> TokenTreesView<'a, S> {
        TokenTreesView::new(self.0)
    }

    pub fn iter(&self) -> TtIter<'a, S> {
        TtIter::new(&self.0[1..])
    }

    pub fn top_subtree(&self) -> &'a Subtree<S> {
        let TokenTree::Subtree(subtree) = &self.0[0] else {
            unreachable!("the first token tree is always the top subtree");
        };
        subtree
    }

    pub fn strip_invisible(&self) -> TokenTreesView<'a, S> {
        if self.top_subtree().delimiter.kind == DelimiterKind::Invisible {
            TokenTreesView::new(&self.0[1..])
        } else {
            TokenTreesView::new(self.0)
        }
    }

    pub fn token_trees(&self) -> TokenTreesView<'a, S> {
        TokenTreesView::new(&self.0[1..])
    }
}

impl<S: fmt::Debug + Copy> fmt::Debug for SubtreeView<'_, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&TokenTreesView(self.0), f)
    }
}

impl<S: Copy> fmt::Display for SubtreeView<'_, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&TokenTreesView(self.0), f)
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
    iter: TtIter<'_, S>,
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

fn print_debug_token<S: fmt::Debug>(
    f: &mut fmt::Formatter<'_>,
    level: usize,
    tt: TtElement<'_, S>,
) -> fmt::Result {
    let align = "  ".repeat(level);

    match tt {
        TtElement::Leaf(leaf) => match leaf {
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
        TtElement::Subtree(subtree, subtree_iter) => {
            print_debug_subtree(f, subtree, level, subtree_iter)?;
        }
    }

    Ok(())
}

impl<S: fmt::Debug + Copy> fmt::Debug for TopSubtree<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.view(), f)
    }
}

impl<S: fmt::Display + Copy> fmt::Display for TopSubtree<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.view(), f)
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

impl<S> Literal<S> {
    pub fn display_no_minus(&self) -> impl fmt::Display {
        struct NoMinus<'a, S>(&'a Literal<S>);
        impl<S> fmt::Display for NoMinus<'_, S> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let symbol =
                    self.0.symbol.as_str().strip_prefix('-').unwrap_or(self.0.symbol.as_str());
                match self.0.kind {
                    LitKind::Byte => write!(f, "b'{symbol}'"),
                    LitKind::Char => write!(f, "'{symbol}'"),
                    LitKind::Integer | LitKind::Float | LitKind::Err(_) => write!(f, "{symbol}"),
                    LitKind::Str => write!(f, "\"{symbol}\""),
                    LitKind::ByteStr => write!(f, "b\"{symbol}\""),
                    LitKind::CStr => write!(f, "c\"{symbol}\""),
                    LitKind::StrRaw(num_of_hashes) => {
                        let num_of_hashes = num_of_hashes as usize;
                        write!(
                            f,
                            r#"r{0:#<num_of_hashes$}"{text}"{0:#<num_of_hashes$}"#,
                            "",
                            text = symbol
                        )
                    }
                    LitKind::ByteStrRaw(num_of_hashes) => {
                        let num_of_hashes = num_of_hashes as usize;
                        write!(
                            f,
                            r#"br{0:#<num_of_hashes$}"{text}"{0:#<num_of_hashes$}"#,
                            "",
                            text = symbol
                        )
                    }
                    LitKind::CStrRaw(num_of_hashes) => {
                        let num_of_hashes = num_of_hashes as usize;
                        write!(
                            f,
                            r#"cr{0:#<num_of_hashes$}"{text}"{0:#<num_of_hashes$}"#,
                            "",
                            text = symbol
                        )
                    }
                }?;
                if let Some(suffix) = &self.0.suffix {
                    write!(f, "{suffix}")?;
                }
                Ok(())
            }
        }
        NoMinus(self)
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
            write!(f, "{suffix}")?;
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
        self.usize_len()
    }
}

impl<S> TopSubtree<S> {
    /// A simple line string used for debugging
    pub fn subtree_as_debug_string(&self, subtree_idx: usize) -> String {
        fn debug_subtree<S>(
            output: &mut String,
            subtree: &Subtree<S>,
            iter: &mut std::slice::Iter<'_, TokenTree<S>>,
        ) {
            let delim = match subtree.delimiter.kind {
                DelimiterKind::Brace => ("{", "}"),
                DelimiterKind::Bracket => ("[", "]"),
                DelimiterKind::Parenthesis => ("(", ")"),
                DelimiterKind::Invisible => ("$", "$"),
            };

            output.push_str(delim.0);
            let mut last = None;
            let mut idx = 0;
            while idx < subtree.len {
                let child = iter.next().unwrap();
                debug_token_tree(output, child, last, iter);
                last = Some(child);
                idx += 1;
            }

            output.push_str(delim.1);
        }

        fn debug_token_tree<S>(
            output: &mut String,
            tt: &TokenTree<S>,
            last: Option<&TokenTree<S>>,
            iter: &mut std::slice::Iter<'_, TokenTree<S>>,
        ) {
            match tt {
                TokenTree::Leaf(it) => {
                    let s = match it {
                        Leaf::Literal(it) => it.symbol.to_string(),
                        Leaf::Punct(it) => it.char.to_string(),
                        Leaf::Ident(it) => format!("{}{}", it.is_raw.as_str(), it.sym),
                    };
                    match (it, last) {
                        (Leaf::Ident(_), Some(&TokenTree::Leaf(Leaf::Ident(_)))) => {
                            output.push(' ');
                            output.push_str(&s);
                        }
                        (Leaf::Punct(_), Some(TokenTree::Leaf(Leaf::Punct(punct)))) => {
                            if punct.spacing == Spacing::Alone {
                                output.push(' ');
                                output.push_str(&s);
                            } else {
                                output.push_str(&s);
                            }
                        }
                        _ => output.push_str(&s),
                    }
                }
                TokenTree::Subtree(it) => debug_subtree(output, it, iter),
            }
        }

        let mut res = String::new();
        debug_token_tree(
            &mut res,
            &self.0[subtree_idx],
            None,
            &mut self.0[subtree_idx + 1..].iter(),
        );
        res
    }
}

pub fn pretty<S>(mut tkns: &[TokenTree<S>]) -> String {
    fn tokentree_to_text<S>(tkn: &TokenTree<S>, tkns: &mut &[TokenTree<S>]) -> String {
        match tkn {
            TokenTree::Leaf(Leaf::Ident(ident)) => {
                format!("{}{}", ident.is_raw.as_str(), ident.sym)
            }
            TokenTree::Leaf(Leaf::Literal(literal)) => format!("{literal}"),
            TokenTree::Leaf(Leaf::Punct(punct)) => format!("{}", punct.char),
            TokenTree::Subtree(subtree) => {
                let (subtree_content, rest) = tkns.split_at(subtree.usize_len());
                let content = pretty(subtree_content);
                *tkns = rest;
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

    let mut last = String::new();
    let mut last_to_joint = true;

    while let Some((tkn, rest)) = tkns.split_first() {
        tkns = rest;
        last = [last, tokentree_to_text(tkn, &mut tkns)].join(if last_to_joint { "" } else { " " });
        last_to_joint = false;
        if let TokenTree::Leaf(Leaf::Punct(punct)) = tkn {
            if punct.spacing == Spacing::Joint {
                last_to_joint = true;
            }
        }
    }
    last
}
