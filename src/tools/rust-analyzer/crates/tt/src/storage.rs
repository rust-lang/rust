//! Spans are memory heavy, and we have a lot of token trees. Storing them straight
//! will waste a lot of memory. So instead we implement a clever compression mechanism:
//!
//! A `TopSubtree` has a list of [`CompressedSpanPart`], which are the parts of a span
//! that tend to be shared between tokens - namely, without the range. The main list
//! of token trees is kept in one of three versions, where we use the smallest version
//! we can for this tree:
//!
//!  1. In the most common version a span is just a `u32`. The bits are divided as follows:
//!     there are 4 bits that index into the [`CompressedSpanPart`] list. 20 bits
//!     store the range start, and 8 bits store the range length. In experiments,
//!     this accounts for 75%-85% of the spans.
//!  2. In the second version a span is 64 bits. 32 bits for the range start, 16 bits
//!     for the range length, and 16 bits for the span parts index. This is used in
//!     less than 2% of all `TopSubtree`s, but they account for 15%-25% of the spans:
//!     those are mostly token tree munchers, that generate a lot of `SyntaxContext`s
//!     (because they recurse a lot), which is why they can't fit in the first version,
//!     and tend to generate a lot of code.
//!  3. The third version is practically unused; 65,535 bytes for a token and 65,535
//!     unique span parts is more than enough for everybody. However, someone may still
//!     create a macro that requires more, therefore we have this version as a backup:
//!     it uses 96 bits, 32 for each of the range start, length and span parts index.

use std::fmt;

use intern::Symbol;
use rustc_hash::FxBuildHasher;
use span::{Span, SpanAnchor, SyntaxContext, TextRange, TextSize};

use crate::{
    DelimSpan, DelimiterKind, IdentIsRaw, LitKind, Spacing, SubtreeView, TokenTreesReprRef,
    TokenTreesView, TtIter, dispatch_ref,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct CompressedSpanPart {
    pub(crate) anchor: SpanAnchor,
    pub(crate) ctx: SyntaxContext,
}

impl CompressedSpanPart {
    #[inline]
    fn from_span(span: &Span) -> Self {
        Self { anchor: span.anchor, ctx: span.ctx }
    }

    #[inline]
    fn recombine(&self, range: TextRange) -> Span {
        Span { range, anchor: self.anchor, ctx: self.ctx }
    }
}

pub(crate) trait SpanStorage: Copy {
    fn can_hold(text_range: TextRange, span_parts_index: usize) -> bool;

    fn new(text_range: TextRange, span_parts_index: usize) -> Self;

    fn text_range(&self) -> TextRange;

    fn span_parts_index(&self) -> usize;

    #[inline]
    fn span(&self, span_parts: &[CompressedSpanPart]) -> Span {
        span_parts[self.span_parts_index()].recombine(self.text_range())
    }
}

#[inline]
const fn n_bits_mask(n: u32) -> u32 {
    (1 << n) - 1
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SpanStorage32(u32);

impl SpanStorage32 {
    const SPAN_PARTS_BIT: u32 = 4;
    const LEN_BITS: u32 = 8;
    const OFFSET_BITS: u32 = 20;
}

const _: () = assert!(
    (SpanStorage32::SPAN_PARTS_BIT + SpanStorage32::LEN_BITS + SpanStorage32::OFFSET_BITS)
        == u32::BITS
);

impl SpanStorage for SpanStorage32 {
    #[inline]
    fn can_hold(text_range: TextRange, span_parts_index: usize) -> bool {
        let offset = u32::from(text_range.start());
        let len = u32::from(text_range.len());
        let span_parts_index = span_parts_index as u32;

        offset <= n_bits_mask(Self::OFFSET_BITS)
            && len <= n_bits_mask(Self::LEN_BITS)
            && span_parts_index <= n_bits_mask(Self::SPAN_PARTS_BIT)
    }

    #[inline]
    fn new(text_range: TextRange, span_parts_index: usize) -> Self {
        let offset = u32::from(text_range.start());
        let len = u32::from(text_range.len());
        let span_parts_index = span_parts_index as u32;

        debug_assert!(offset <= n_bits_mask(Self::OFFSET_BITS));
        debug_assert!(len <= n_bits_mask(Self::LEN_BITS));
        debug_assert!(span_parts_index <= n_bits_mask(Self::SPAN_PARTS_BIT));

        Self(
            (offset << (Self::LEN_BITS + Self::SPAN_PARTS_BIT))
                | (len << Self::SPAN_PARTS_BIT)
                | span_parts_index,
        )
    }

    #[inline]
    fn text_range(&self) -> TextRange {
        let offset = TextSize::new(self.0 >> (Self::SPAN_PARTS_BIT + Self::LEN_BITS));
        let len = TextSize::new((self.0 >> Self::SPAN_PARTS_BIT) & n_bits_mask(Self::LEN_BITS));
        TextRange::at(offset, len)
    }

    #[inline]
    fn span_parts_index(&self) -> usize {
        (self.0 & n_bits_mask(Self::SPAN_PARTS_BIT)) as usize
    }
}

impl fmt::Debug for SpanStorage32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpanStorage32")
            .field("text_range", &self.text_range())
            .field("span_parts_index", &self.span_parts_index())
            .finish()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SpanStorage64 {
    offset: u32,
    len_and_parts: u32,
}

impl SpanStorage64 {
    const SPAN_PARTS_BIT: u32 = 16;
    const LEN_BITS: u32 = 16;
}

const _: () = assert!((SpanStorage64::SPAN_PARTS_BIT + SpanStorage64::LEN_BITS) == u32::BITS);

impl SpanStorage for SpanStorage64 {
    #[inline]
    fn can_hold(text_range: TextRange, span_parts_index: usize) -> bool {
        let len = u32::from(text_range.len());
        let span_parts_index = span_parts_index as u32;

        len <= n_bits_mask(Self::LEN_BITS) && span_parts_index <= n_bits_mask(Self::SPAN_PARTS_BIT)
    }

    #[inline]
    fn new(text_range: TextRange, span_parts_index: usize) -> Self {
        let offset = u32::from(text_range.start());
        let len = u32::from(text_range.len());
        let span_parts_index = span_parts_index as u32;

        debug_assert!(len <= n_bits_mask(Self::LEN_BITS));
        debug_assert!(span_parts_index <= n_bits_mask(Self::SPAN_PARTS_BIT));

        Self { offset, len_and_parts: (len << Self::SPAN_PARTS_BIT) | span_parts_index }
    }

    #[inline]
    fn text_range(&self) -> TextRange {
        let offset = TextSize::new(self.offset);
        let len = TextSize::new(self.len_and_parts >> Self::SPAN_PARTS_BIT);
        TextRange::at(offset, len)
    }

    #[inline]
    fn span_parts_index(&self) -> usize {
        (self.len_and_parts & n_bits_mask(Self::SPAN_PARTS_BIT)) as usize
    }
}

impl fmt::Debug for SpanStorage64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpanStorage64")
            .field("text_range", &self.text_range())
            .field("span_parts_index", &self.span_parts_index())
            .finish()
    }
}

impl From<SpanStorage32> for SpanStorage64 {
    #[inline]
    fn from(value: SpanStorage32) -> Self {
        SpanStorage64::new(value.text_range(), value.span_parts_index())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SpanStorage96 {
    offset: u32,
    len: u32,
    parts: u32,
}

impl SpanStorage for SpanStorage96 {
    #[inline]
    fn can_hold(_text_range: TextRange, _span_parts_index: usize) -> bool {
        true
    }

    #[inline]
    fn new(text_range: TextRange, span_parts_index: usize) -> Self {
        let offset = u32::from(text_range.start());
        let len = u32::from(text_range.len());
        let span_parts_index = span_parts_index as u32;

        Self { offset, len, parts: span_parts_index }
    }

    #[inline]
    fn text_range(&self) -> TextRange {
        let offset = TextSize::new(self.offset);
        let len = TextSize::new(self.len);
        TextRange::at(offset, len)
    }

    #[inline]
    fn span_parts_index(&self) -> usize {
        self.parts as usize
    }
}

impl fmt::Debug for SpanStorage96 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpanStorage96")
            .field("text_range", &self.text_range())
            .field("span_parts_index", &self.span_parts_index())
            .finish()
    }
}

impl From<SpanStorage32> for SpanStorage96 {
    #[inline]
    fn from(value: SpanStorage32) -> Self {
        SpanStorage96::new(value.text_range(), value.span_parts_index())
    }
}

impl From<SpanStorage64> for SpanStorage96 {
    #[inline]
    fn from(value: SpanStorage64) -> Self {
        SpanStorage96::new(value.text_range(), value.span_parts_index())
    }
}

// We don't use structs or enum nesting here to save padding.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum TokenTree<S> {
    Literal { text_and_suffix: Symbol, span: S, kind: LitKind, suffix_len: u8 },
    Punct { char: char, spacing: Spacing, span: S },
    Ident { sym: Symbol, span: S, is_raw: IdentIsRaw },
    Subtree { len: u32, delim_kind: DelimiterKind, open_span: S, close_span: S },
}

impl<S: SpanStorage> TokenTree<S> {
    #[inline]
    pub(crate) fn first_span(&self) -> &S {
        match self {
            TokenTree::Literal { span, .. } => span,
            TokenTree::Punct { span, .. } => span,
            TokenTree::Ident { span, .. } => span,
            TokenTree::Subtree { open_span, .. } => open_span,
        }
    }

    #[inline]
    pub(crate) fn last_span(&self) -> &S {
        match self {
            TokenTree::Literal { span, .. } => span,
            TokenTree::Punct { span, .. } => span,
            TokenTree::Ident { span, .. } => span,
            TokenTree::Subtree { close_span, .. } => close_span,
        }
    }

    #[inline]
    pub(crate) fn to_api(&self, span_parts: &[CompressedSpanPart]) -> crate::TokenTree {
        match self {
            TokenTree::Literal { text_and_suffix, span, kind, suffix_len } => {
                crate::TokenTree::Leaf(crate::Leaf::Literal(crate::Literal {
                    text_and_suffix: text_and_suffix.clone(),
                    span: span.span(span_parts),
                    kind: *kind,
                    suffix_len: *suffix_len,
                }))
            }
            TokenTree::Punct { char, spacing, span } => {
                crate::TokenTree::Leaf(crate::Leaf::Punct(crate::Punct {
                    char: *char,
                    spacing: *spacing,
                    span: span.span(span_parts),
                }))
            }
            TokenTree::Ident { sym, span, is_raw } => {
                crate::TokenTree::Leaf(crate::Leaf::Ident(crate::Ident {
                    sym: sym.clone(),
                    span: span.span(span_parts),
                    is_raw: *is_raw,
                }))
            }
            TokenTree::Subtree { len, delim_kind, open_span, close_span } => {
                crate::TokenTree::Subtree(crate::Subtree {
                    delimiter: crate::Delimiter {
                        open: open_span.span(span_parts),
                        close: close_span.span(span_parts),
                        kind: *delim_kind,
                    },
                    len: *len,
                })
            }
        }
    }

    #[inline]
    fn convert<U: From<S>>(self) -> TokenTree<U> {
        match self {
            TokenTree::Literal { text_and_suffix, span, kind, suffix_len } => {
                TokenTree::Literal { text_and_suffix, span: span.into(), kind, suffix_len }
            }
            TokenTree::Punct { char, spacing, span } => {
                TokenTree::Punct { char, spacing, span: span.into() }
            }
            TokenTree::Ident { sym, span, is_raw } => {
                TokenTree::Ident { sym, span: span.into(), is_raw }
            }
            TokenTree::Subtree { len, delim_kind, open_span, close_span } => TokenTree::Subtree {
                len,
                delim_kind,
                open_span: open_span.into(),
                close_span: close_span.into(),
            },
        }
    }
}

// This is used a lot, make sure it doesn't grow unintentionally.
const _: () = {
    assert!(size_of::<TokenTree<SpanStorage32>>() == 16);
    assert!(size_of::<TokenTree<SpanStorage64>>() == 24);
    assert!(size_of::<TokenTree<SpanStorage96>>() == 32);
};

#[rust_analyzer::macro_style(braces)]
macro_rules! dispatch {
    (
        match $scrutinee:expr => $tt:ident => $body:expr
    ) => {
        match $scrutinee {
            TopSubtreeRepr::SpanStorage32($tt) => $body,
            TopSubtreeRepr::SpanStorage64($tt) => $body,
            TopSubtreeRepr::SpanStorage96($tt) => $body,
        }
    };
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum TopSubtreeRepr {
    SpanStorage32(Box<[TokenTree<SpanStorage32>]>),
    SpanStorage64(Box<[TokenTree<SpanStorage64>]>),
    SpanStorage96(Box<[TokenTree<SpanStorage96>]>),
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct TopSubtree {
    repr: TopSubtreeRepr,
    span_parts: Box<[CompressedSpanPart]>,
}

impl TopSubtree {
    pub fn empty(span: DelimSpan) -> Self {
        Self {
            repr: TopSubtreeRepr::SpanStorage96(Box::new([TokenTree::Subtree {
                len: 0,
                delim_kind: DelimiterKind::Invisible,
                open_span: SpanStorage96::new(span.open.range, 0),
                close_span: SpanStorage96::new(span.close.range, 1),
            }])),
            span_parts: Box::new([
                CompressedSpanPart::from_span(&span.open),
                CompressedSpanPart::from_span(&span.close),
            ]),
        }
    }

    pub fn invisible_from_leaves<const N: usize>(
        delim_span: Span,
        leaves: [crate::Leaf; N],
    ) -> Self {
        let mut builder = TopSubtreeBuilder::new(crate::Delimiter::invisible_spanned(delim_span));
        builder.extend(leaves);
        builder.build()
    }

    pub fn from_token_trees(delimiter: crate::Delimiter, token_trees: TokenTreesView<'_>) -> Self {
        let mut builder = TopSubtreeBuilder::new(delimiter);
        builder.extend_with_tt(token_trees);
        builder.build()
    }

    pub fn from_serialized(tt: Vec<crate::TokenTree>) -> Self {
        let mut tt = tt.into_iter();
        let Some(crate::TokenTree::Subtree(top_subtree)) = tt.next() else {
            panic!("first must always come the top subtree")
        };
        let mut builder = TopSubtreeBuilder::new(top_subtree.delimiter);
        for tt in tt {
            builder.push_token_tree(tt);
        }
        builder.build()
    }

    pub fn from_subtree(subtree: SubtreeView<'_>) -> Self {
        let mut builder = TopSubtreeBuilder::new(subtree.top_subtree().delimiter);
        builder.extend_with_tt(subtree.token_trees());
        builder.build()
    }

    pub fn view(&self) -> SubtreeView<'_> {
        let repr = match &self.repr {
            TopSubtreeRepr::SpanStorage32(token_trees) => {
                TokenTreesReprRef::SpanStorage32(token_trees)
            }
            TopSubtreeRepr::SpanStorage64(token_trees) => {
                TokenTreesReprRef::SpanStorage64(token_trees)
            }
            TopSubtreeRepr::SpanStorage96(token_trees) => {
                TokenTreesReprRef::SpanStorage96(token_trees)
            }
        };
        SubtreeView(TokenTreesView { repr, span_parts: &self.span_parts })
    }

    pub fn iter(&self) -> TtIter<'_> {
        self.view().iter()
    }

    pub fn top_subtree(&self) -> crate::Subtree {
        self.view().top_subtree()
    }

    pub fn set_top_subtree_delimiter_kind(&mut self, kind: DelimiterKind) {
        dispatch! {
            match &mut self.repr => tt => {
                let TokenTree::Subtree { delim_kind, .. } = &mut tt[0] else {
                    unreachable!("the first token tree is always the top subtree");
                };
                *delim_kind = kind;
            }
        }
    }

    fn ensure_can_hold(&mut self, range: TextRange) {
        fn can_hold<S: SpanStorage>(_: &[TokenTree<S>], range: TextRange) -> bool {
            S::can_hold(range, 0)
        }
        let can_hold = dispatch! {
            match &self.repr => tt => can_hold(tt, range)
        };
        if can_hold {
            return;
        }

        // Otherwise, we do something very junky: recreate the entire tree. Hopefully this should be rare.
        let mut builder = TopSubtreeBuilder::new(self.top_subtree().delimiter);
        builder.extend_with_tt(self.token_trees());
        builder.ensure_can_hold(range, 0);
        *self = builder.build();
    }

    pub fn set_top_subtree_delimiter_span(&mut self, span: DelimSpan) {
        self.ensure_can_hold(span.open.range);
        self.ensure_can_hold(span.close.range);
        fn do_it<S: SpanStorage>(tt: &mut [TokenTree<S>], span: DelimSpan) {
            let TokenTree::Subtree { open_span, close_span, .. } = &mut tt[0] else {
                unreachable!()
            };
            *open_span = S::new(span.open.range, 0);
            *close_span = S::new(span.close.range, 0);
        }
        dispatch! {
            match &mut self.repr => tt => do_it(tt, span)
        }
        self.span_parts[0] = CompressedSpanPart::from_span(&span.open);
        self.span_parts[1] = CompressedSpanPart::from_span(&span.close);
    }

    /// Note: this cannot change spans.
    pub fn set_token(&mut self, idx: usize, leaf: crate::Leaf) {
        fn do_it<S: SpanStorage>(
            tt: &mut [TokenTree<S>],
            idx: usize,
            span_parts: &[CompressedSpanPart],
            leaf: crate::Leaf,
        ) {
            assert!(
                !matches!(tt[idx], TokenTree::Subtree { .. }),
                "`TopSubtree::set_token()` must be called on a leaf"
            );
            let existing_span_compressed = *tt[idx].first_span();
            let existing_span = existing_span_compressed.span(span_parts);
            assert_eq!(
                *leaf.span(),
                existing_span,
                "`TopSubtree::set_token()` cannot change spans"
            );
            match leaf {
                crate::Leaf::Literal(leaf) => {
                    tt[idx] = TokenTree::Literal {
                        text_and_suffix: leaf.text_and_suffix,
                        span: existing_span_compressed,
                        kind: leaf.kind,
                        suffix_len: leaf.suffix_len,
                    }
                }
                crate::Leaf::Punct(leaf) => {
                    tt[idx] = TokenTree::Punct {
                        char: leaf.char,
                        spacing: leaf.spacing,
                        span: existing_span_compressed,
                    }
                }
                crate::Leaf::Ident(leaf) => {
                    tt[idx] = TokenTree::Ident {
                        sym: leaf.sym,
                        span: existing_span_compressed,
                        is_raw: leaf.is_raw,
                    }
                }
            }
        }
        dispatch! {
            match &mut self.repr => tt => do_it(tt, idx, &self.span_parts, leaf)
        }
    }

    pub fn token_trees(&self) -> TokenTreesView<'_> {
        self.view().token_trees()
    }

    pub fn as_token_trees(&self) -> TokenTreesView<'_> {
        self.view().as_token_trees()
    }

    pub fn change_every_ast_id(&mut self, mut callback: impl FnMut(&mut span::ErasedFileAstId)) {
        for span_part in &mut self.span_parts {
            callback(&mut span_part.anchor.ast_id);
        }
    }
}

#[rust_analyzer::macro_style(braces)]
macro_rules! dispatch_builder {
    (
        match $scrutinee:expr => $tt:ident => $body:expr
    ) => {
        match $scrutinee {
            TopSubtreeBuilderRepr::SpanStorage32($tt) => $body,
            TopSubtreeBuilderRepr::SpanStorage64($tt) => $body,
            TopSubtreeBuilderRepr::SpanStorage96($tt) => $body,
        }
    };
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum TopSubtreeBuilderRepr {
    SpanStorage32(Vec<TokenTree<SpanStorage32>>),
    SpanStorage64(Vec<TokenTree<SpanStorage64>>),
    SpanStorage96(Vec<TokenTree<SpanStorage96>>),
}

type FxIndexSet<K> = indexmap::IndexSet<K, FxBuildHasher>;

/// In any tree, the first two subtree parts are reserved for the top subtree.
///
/// We do it because `TopSubtree` exposes an API to modify the top subtree, therefore it's more convenient
/// this way, and it's unlikely to affect memory usage.
const RESERVED_SPAN_PARTS_LEN: usize = 2;

#[derive(Debug, Clone)]
pub struct TopSubtreeBuilder {
    unclosed_subtree_indices: Vec<usize>,
    token_trees: TopSubtreeBuilderRepr,
    span_parts: FxIndexSet<CompressedSpanPart>,
    last_closed_subtree: Option<usize>,
    /// We need to keep those because they are not inside `span_parts`, see [`RESERVED_SPAN_PARTS_LEN`].
    top_subtree_spans: DelimSpan,
}

impl TopSubtreeBuilder {
    pub fn new(top_delimiter: crate::Delimiter) -> Self {
        let mut result = Self {
            unclosed_subtree_indices: Vec::new(),
            token_trees: TopSubtreeBuilderRepr::SpanStorage32(Vec::new()),
            span_parts: FxIndexSet::default(),
            last_closed_subtree: None,
            top_subtree_spans: top_delimiter.delim_span(),
        };
        result.ensure_can_hold(top_delimiter.open.range, 0);
        result.ensure_can_hold(top_delimiter.close.range, 1);
        fn push_first<S: SpanStorage>(tt: &mut Vec<TokenTree<S>>, top_delimiter: crate::Delimiter) {
            tt.push(TokenTree::Subtree {
                len: 0,
                delim_kind: top_delimiter.kind,
                open_span: S::new(top_delimiter.open.range, 0),
                close_span: S::new(top_delimiter.close.range, 1),
            });
        }
        dispatch_builder! {
            match &mut result.token_trees => tt => push_first(tt, top_delimiter)
        }
        result
    }

    fn span_part_index(&mut self, part: CompressedSpanPart) -> usize {
        self.span_parts.insert_full(part).0 + RESERVED_SPAN_PARTS_LEN
    }

    fn switch_repr<T: SpanStorage, U: From<T>>(repr: &mut Vec<TokenTree<T>>) -> Vec<TokenTree<U>> {
        let repr = std::mem::take(repr);
        repr.into_iter().map(|tt| tt.convert()).collect()
    }

    /// Ensures we have a representation that can hold these values.
    fn ensure_can_hold(&mut self, text_range: TextRange, span_parts_index: usize) {
        match &mut self.token_trees {
            TopSubtreeBuilderRepr::SpanStorage32(token_trees) => {
                if SpanStorage32::can_hold(text_range, span_parts_index) {
                    // Can hold.
                } else if SpanStorage64::can_hold(text_range, span_parts_index) {
                    self.token_trees =
                        TopSubtreeBuilderRepr::SpanStorage64(Self::switch_repr(token_trees));
                } else {
                    self.token_trees =
                        TopSubtreeBuilderRepr::SpanStorage96(Self::switch_repr(token_trees));
                }
            }
            TopSubtreeBuilderRepr::SpanStorage64(token_trees) => {
                if SpanStorage64::can_hold(text_range, span_parts_index) {
                    // Can hold.
                } else {
                    self.token_trees =
                        TopSubtreeBuilderRepr::SpanStorage96(Self::switch_repr(token_trees));
                }
            }
            TopSubtreeBuilderRepr::SpanStorage96(_) => {
                // Can hold anything.
            }
        }
    }

    /// Not to be exposed, this assumes the subtree's children will be filled in immediately.
    fn push_subtree(&mut self, subtree: crate::Subtree) {
        let open_span_parts_index =
            self.span_part_index(CompressedSpanPart::from_span(&subtree.delimiter.open));
        self.ensure_can_hold(subtree.delimiter.open.range, open_span_parts_index);
        let close_span_parts_index =
            self.span_part_index(CompressedSpanPart::from_span(&subtree.delimiter.close));
        self.ensure_can_hold(subtree.delimiter.close.range, close_span_parts_index);
        fn do_it<S: SpanStorage>(
            tt: &mut Vec<TokenTree<S>>,
            open_span_parts_index: usize,
            close_span_parts_index: usize,
            subtree: crate::Subtree,
        ) {
            let open_span = S::new(subtree.delimiter.open.range, open_span_parts_index);
            let close_span = S::new(subtree.delimiter.close.range, close_span_parts_index);
            tt.push(TokenTree::Subtree {
                len: subtree.len,
                delim_kind: subtree.delimiter.kind,
                open_span,
                close_span,
            });
        }
        dispatch_builder! {
            match &mut self.token_trees => tt => do_it(tt, open_span_parts_index, close_span_parts_index, subtree)
        }
    }

    pub fn open(&mut self, delimiter_kind: DelimiterKind, open_span: Span) {
        let span_parts_index = self.span_part_index(CompressedSpanPart::from_span(&open_span));
        self.ensure_can_hold(open_span.range, span_parts_index);
        fn do_it<S: SpanStorage>(
            token_trees: &mut Vec<TokenTree<S>>,
            delimiter_kind: DelimiterKind,
            range: TextRange,
            span_parts_index: usize,
        ) -> usize {
            let open_span = S::new(range, span_parts_index);
            token_trees.push(TokenTree::Subtree {
                len: 0,
                delim_kind: delimiter_kind,
                open_span,
                close_span: open_span, // Will be overwritten on close.
            });
            token_trees.len() - 1
        }
        let subtree_idx = dispatch_builder! {
            match &mut self.token_trees => tt => do_it(tt, delimiter_kind, open_span.range, span_parts_index)
        };
        self.unclosed_subtree_indices.push(subtree_idx);
    }

    pub fn close(&mut self, close_span: Span) {
        let span_parts_index = self.span_part_index(CompressedSpanPart::from_span(&close_span));
        let range = close_span.range;
        self.ensure_can_hold(range, span_parts_index);

        let last_unclosed_index = self
            .unclosed_subtree_indices
            .pop()
            .expect("attempt to close a `tt::Subtree` when none is open");
        fn do_it<S: SpanStorage>(
            token_trees: &mut [TokenTree<S>],
            last_unclosed_index: usize,
            range: TextRange,
            span_parts_index: usize,
        ) {
            let token_trees_len = token_trees.len();
            let TokenTree::Subtree { len, delim_kind: _, open_span: _, close_span } =
                &mut token_trees[last_unclosed_index]
            else {
                unreachable!("unclosed token tree is always a subtree");
            };
            *len = (token_trees_len - last_unclosed_index - 1) as u32;
            *close_span = S::new(range, span_parts_index);
        }
        dispatch_builder! {
            match &mut self.token_trees => tt => do_it(tt, last_unclosed_index, range, span_parts_index)
        }
        self.last_closed_subtree = Some(last_unclosed_index);
    }

    /// You cannot call this consecutively, it will only work once after close.
    pub fn remove_last_subtree_if_invisible(&mut self) {
        let Some(last_subtree_idx) = self.last_closed_subtree else { return };
        fn do_it<S: SpanStorage>(tt: &mut Vec<TokenTree<S>>, last_subtree_idx: usize) {
            if let TokenTree::Subtree { delim_kind: DelimiterKind::Invisible, .. } =
                tt[last_subtree_idx]
            {
                tt.remove(last_subtree_idx);
            }
        }
        dispatch_builder! {
            match &mut self.token_trees => tt => do_it(tt, last_subtree_idx)
        }
        self.last_closed_subtree = None;
    }

    fn push_literal(&mut self, leaf: crate::Literal) {
        let span_parts_index = self.span_part_index(CompressedSpanPart::from_span(&leaf.span));
        let range = leaf.span.range;
        self.ensure_can_hold(range, span_parts_index);
        fn do_it<S: SpanStorage>(
            tt: &mut Vec<TokenTree<S>>,
            range: TextRange,
            span_parts_index: usize,
            leaf: crate::Literal,
        ) {
            tt.push(TokenTree::Literal {
                text_and_suffix: leaf.text_and_suffix,
                span: S::new(range, span_parts_index),
                kind: leaf.kind,
                suffix_len: leaf.suffix_len,
            })
        }
        dispatch_builder! {
            match &mut self.token_trees => tt => do_it(tt, range, span_parts_index, leaf)
        }
    }

    fn push_punct(&mut self, leaf: crate::Punct) {
        let span_parts_index = self.span_part_index(CompressedSpanPart::from_span(&leaf.span));
        let range = leaf.span.range;
        self.ensure_can_hold(range, span_parts_index);
        fn do_it<S: SpanStorage>(
            tt: &mut Vec<TokenTree<S>>,
            range: TextRange,
            span_parts_index: usize,
            leaf: crate::Punct,
        ) {
            tt.push(TokenTree::Punct {
                char: leaf.char,
                spacing: leaf.spacing,
                span: S::new(range, span_parts_index),
            })
        }
        dispatch_builder! {
            match &mut self.token_trees => tt => do_it(tt, range, span_parts_index, leaf)
        }
    }

    fn push_ident(&mut self, leaf: crate::Ident) {
        let span_parts_index = self.span_part_index(CompressedSpanPart::from_span(&leaf.span));
        let range = leaf.span.range;
        self.ensure_can_hold(range, span_parts_index);
        fn do_it<S: SpanStorage>(
            tt: &mut Vec<TokenTree<S>>,
            range: TextRange,
            span_parts_index: usize,
            leaf: crate::Ident,
        ) {
            tt.push(TokenTree::Ident {
                sym: leaf.sym,
                span: S::new(range, span_parts_index),
                is_raw: leaf.is_raw,
            })
        }
        dispatch_builder! {
            match &mut self.token_trees => tt => do_it(tt, range, span_parts_index, leaf)
        }
    }

    pub fn push(&mut self, leaf: crate::Leaf) {
        match leaf {
            crate::Leaf::Literal(leaf) => self.push_literal(leaf),
            crate::Leaf::Punct(leaf) => self.push_punct(leaf),
            crate::Leaf::Ident(leaf) => self.push_ident(leaf),
        }
    }

    fn push_token_tree(&mut self, tt: crate::TokenTree) {
        match tt {
            crate::TokenTree::Leaf(leaf) => self.push(leaf),
            crate::TokenTree::Subtree(subtree) => self.push_subtree(subtree),
        }
    }

    pub fn extend(&mut self, leaves: impl IntoIterator<Item = crate::Leaf>) {
        leaves.into_iter().for_each(|leaf| self.push(leaf));
    }

    pub fn extend_with_tt(&mut self, tt: TokenTreesView<'_>) {
        fn do_it<S: SpanStorage>(
            this: &mut TopSubtreeBuilder,
            tt: &[TokenTree<S>],
            span_parts: &[CompressedSpanPart],
        ) {
            for tt in tt {
                this.push_token_tree(tt.to_api(span_parts));
            }
        }
        dispatch_ref! {
            match tt.repr => tt_repr => do_it(self, tt_repr, tt.span_parts)
        }
    }

    /// Like [`Self::extend_with_tt()`], but makes sure the new tokens will never be
    /// joint with whatever comes after them.
    pub fn extend_with_tt_alone(&mut self, tt: TokenTreesView<'_>) {
        self.extend_with_tt(tt);
        fn do_it<S: SpanStorage>(tt: &mut [TokenTree<S>]) {
            if let Some(TokenTree::Punct { spacing, .. }) = tt.last_mut() {
                *spacing = Spacing::Alone;
            }
        }
        if !tt.is_empty() {
            dispatch_builder! {
                match &mut self.token_trees => tt => do_it(tt)
            }
        }
    }

    pub fn expected_delimiters(&self) -> impl Iterator<Item = DelimiterKind> {
        self.unclosed_subtree_indices.iter().rev().map(|&subtree_idx| {
            dispatch_builder! {
                match &self.token_trees => tt => {
                    let TokenTree::Subtree { delim_kind, .. } = tt[subtree_idx] else {
                        unreachable!("unclosed token tree is always a subtree")
                    };
                    delim_kind
                }
            }
        })
    }

    /// Builds, and remove the top subtree if it has only one subtree child.
    pub fn build_skip_top_subtree(mut self) -> TopSubtree {
        fn remove_first_if_needed<S: SpanStorage>(
            tt: &mut Vec<TokenTree<S>>,
            top_delim_span: &mut DelimSpan,
            span_parts: &FxIndexSet<CompressedSpanPart>,
        ) {
            let tt_len = tt.len();
            let Some(TokenTree::Subtree { len, open_span, close_span, .. }) = tt.get_mut(1) else {
                return;
            };
            if (*len as usize) != (tt_len - 2) {
                // Subtree does not cover the whole tree (minus 2; itself, and the top span).
                return;
            }

            // Now we need to adjust the spans, because we assume that the first two spans are always reserved.
            let top_open_span = span_parts
                .get_index(open_span.span_parts_index() - RESERVED_SPAN_PARTS_LEN)
                .unwrap()
                .recombine(open_span.text_range());
            let top_close_span = span_parts
                .get_index(close_span.span_parts_index() - RESERVED_SPAN_PARTS_LEN)
                .unwrap()
                .recombine(close_span.text_range());
            *top_delim_span = DelimSpan { open: top_open_span, close: top_close_span };
            // Can't remove the top spans from the map, as maybe they're used by other things as well.
            // Now we need to reencode the spans, because their parts index changed:
            *open_span = S::new(open_span.text_range(), 0);
            *close_span = S::new(close_span.text_range(), 1);

            tt.remove(0);
        }
        dispatch_builder! {
            match &mut self.token_trees => tt => remove_first_if_needed(tt, &mut self.top_subtree_spans, &self.span_parts)
        }
        self.build()
    }

    pub fn build(mut self) -> TopSubtree {
        assert!(
            self.unclosed_subtree_indices.is_empty(),
            "attempt to build an unbalanced `TopSubtreeBuilder`"
        );
        fn finish_top_len<S: SpanStorage>(tt: &mut [TokenTree<S>]) {
            let total_len = tt.len() as u32;
            let TokenTree::Subtree { len, .. } = &mut tt[0] else {
                unreachable!("first token tree is always a subtree");
            };
            *len = total_len - 1;
        }
        dispatch_builder! {
            match &mut self.token_trees => tt => finish_top_len(tt)
        }

        let span_parts = [
            CompressedSpanPart::from_span(&self.top_subtree_spans.open),
            CompressedSpanPart::from_span(&self.top_subtree_spans.close),
        ]
        .into_iter()
        .chain(self.span_parts.iter().copied())
        .collect();

        let repr = match self.token_trees {
            TopSubtreeBuilderRepr::SpanStorage32(tt) => {
                TopSubtreeRepr::SpanStorage32(tt.into_boxed_slice())
            }
            TopSubtreeBuilderRepr::SpanStorage64(tt) => {
                TopSubtreeRepr::SpanStorage64(tt.into_boxed_slice())
            }
            TopSubtreeBuilderRepr::SpanStorage96(tt) => {
                TopSubtreeRepr::SpanStorage96(tt.into_boxed_slice())
            }
        };

        TopSubtree { repr, span_parts }
    }

    pub fn restore_point(&self) -> SubtreeBuilderRestorePoint {
        let token_trees_len = dispatch_builder! {
            match &self.token_trees => tt => tt.len()
        };
        SubtreeBuilderRestorePoint {
            unclosed_subtree_indices_len: self.unclosed_subtree_indices.len(),
            token_trees_len,
            last_closed_subtree: self.last_closed_subtree,
        }
    }

    pub fn restore(&mut self, restore_point: SubtreeBuilderRestorePoint) {
        self.unclosed_subtree_indices.truncate(restore_point.unclosed_subtree_indices_len);
        dispatch_builder! {
            match &mut self.token_trees => tt => tt.truncate(restore_point.token_trees_len)
        }
        self.last_closed_subtree = restore_point.last_closed_subtree;
    }
}

#[derive(Clone, Copy)]
pub struct SubtreeBuilderRestorePoint {
    unclosed_subtree_indices_len: usize,
    token_trees_len: usize,
    last_closed_subtree: Option<usize>,
}
