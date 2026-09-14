use std::borrow::Cow;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::num::NonZeroU32;
use std::sync::{Arc, LazyLock};

use rustc_data_structures::stable_hash::{StableHash, StableHashCtxt, StableHasher};
use rustc_index::static_assert_size;
use rustc_macros::{Decodable, Encodable, StableHash};
use rustc_serialize::{Decodable, Encodable};
use rustc_span::{Span, SpanDecoder, SpanEncoder};

use crate::token::{Delimiter, Token, TokenKind};
use crate::tokenstream::{
    DelimSpacing, DelimSpan, LazyAttrTokenStream, Spacing, TokenStream, TokenTree,
};
use crate::{Attribute, HasTokens};

/// Part of a `TokenArena`.
#[derive(Debug, Copy, Clone)]
pub enum ArenaTokenTree {
    /// A single token. Should never be `OpenDelim` or `CloseDelim`, because
    /// delimiters are implicitly represented by `DelimitedStart`/`DelimitedEnd`.
    Token(Token, Spacing),
    /// A delimited sequence of token trees.
    DelimitedStart(DelimitedBounds, DelimitedData),
}

impl ArenaTokenTree {
    /// Create a `TokenTree::Token` with alone spacing.
    #[inline]
    pub fn token_alone(kind: TokenKind, span: Span) -> ArenaTokenTree {
        ArenaTokenTree::Token(Token::new(kind, span), Spacing::Alone)
    }

    /// Create a `TokenTree::Token` with joint spacing.
    #[inline]
    pub fn token_joint(kind: TokenKind, span: Span) -> ArenaTokenTree {
        ArenaTokenTree::Token(Token::new(kind, span), Spacing::Joint)
    }

    pub fn uninterpolate(&self) -> Cow<'_, ArenaTokenTree> {
        match self {
            ArenaTokenTree::Token(token, spacing) => match token.uninterpolate() {
                Cow::Owned(token) => Cow::Owned(ArenaTokenTree::Token(token, *spacing)),
                Cow::Borrowed(_) => Cow::Borrowed(self),
            },
            _ => Cow::Borrowed(self),
        }
    }

    /// Convert an arena token tree to the tree-shaped token tree.
    pub fn to_token_tree(&self, arena: &ArenaTokenStream) -> TokenTree {
        match self {
            ArenaTokenTree::Token(token, spacing) => TokenTree::Token(*token, *spacing),
            ArenaTokenTree::DelimitedStart(bounds, data) => {
                let tts = arena
                    .iter_delimited_contents(bounds)
                    .map(|tt| tt.to_token_tree(arena))
                    .collect();
                TokenTree::Delimited(data.span, data.spacing, data.delimiter, TokenStream::new(tts))
            }
        }
    }

    /// Retrieves the `TokenTree`'s span.
    pub fn span(&self) -> Span {
        match self {
            Self::Token(token, _) => token.span,
            Self::DelimitedStart(_, data) => data.span.entire(),
        }
    }

    #[inline]
    pub fn to_delimited_data(&self) -> Option<&DelimitedData> {
        match self {
            ArenaTokenTree::Token(_, _) => None,
            ArenaTokenTree::DelimitedStart(_, data) => Some(data),
        }
    }

    #[inline]
    pub fn to_delimited_bounds(&self) -> Option<&DelimitedBounds> {
        match self {
            ArenaTokenTree::Token(_, _) => None,
            ArenaTokenTree::DelimitedStart(bounds, _) => Some(bounds),
        }
    }
}

static_assert_size!(ArenaTokenTree, 44);

#[derive(Debug, Default)]
pub struct ArenaTokenStreamBuilder {
    tokens: Vec<ArenaTokenTree>,
    /// Index of the current delimited sequence
    current_delimited_sequence: Option<AbsoluteTokenTreeIndex>,
    /// This is only useful to optimize glueing
    last_push_was_token: bool,
}

impl ArenaTokenStreamBuilder {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            tokens: Vec::with_capacity(capacity),
            current_delimited_sequence: None,
            last_push_was_token: false,
        }
    }

    pub fn tokens(&self) -> &[ArenaTokenTree] {
        &self.tokens
    }

    #[inline]
    pub fn push_token(&mut self, token: Token, spacing: Spacing) {
        self.tokens.push(ArenaTokenTree::Token(token, spacing));
        self.last_push_was_token = true;
    }

    #[inline]
    pub fn push_token_alone(&mut self, token: Token) {
        self.push_token(token, Spacing::Alone);
    }

    pub fn push_token_tree(&mut self, tt: &ArenaTokenTree, stream: &ArenaTokenStream) {
        match tt {
            ArenaTokenTree::Token(token, spacing) => {
                self.push_token(*token, *spacing);
            }
            ArenaTokenTree::DelimitedStart(bounds, data) => {
                self.push_delimited(
                    |builder| {
                        builder.fill_stream(stream.iter_delimited_contents(bounds));
                    },
                    *data,
                );
            }
        }
    }

    pub fn push_iter(&mut self, iter: ArenaTokenTreeIter<'_>) {
        self.fill_stream(iter);
    }

    pub fn push_stream(&mut self, stream: ArenaTokenStream) {
        self.tokens.reserve(stream.length());
        self.fill_stream(stream.iter_top_level_trees());
    }

    pub fn pop(&mut self) -> Option<ArenaTokenTree> {
        // Note: calling this function is fine even if we are within a delimited sequence.
        let tree = self.tokens.pop();
        if let Some(tree) = &tree {
            assert!(matches!(tree, ArenaTokenTree::Token(..)));
        }
        tree
    }

    // If `self` is not empty, try to glue `tt` onto its last top-level token. The return
    // value indicates if gluing took place.
    pub fn try_glue_to_last_top_level_token(&mut self, token: &Token, spacing: Spacing) -> bool {
        assert!(self.current_delimited_sequence.is_none());
        if let Some(ArenaTokenTree::Token(last_tok, Spacing::Joint | Spacing::JointHidden)) =
            self.tokens.last()
            // We can only do this if the last tree is a top-level token within the current
            // delimited sequence.
            // If there is no last top-level sequence, then the last token has to be top-level
            && self.last_push_was_token
            && let Some(glued_tok) = last_tok.glue(&token)
        {
            // ...then overwrite the last token tree in `vec` with the glued token.
            *self.tokens.last_mut().unwrap() = ArenaTokenTree::Token(glued_tok, spacing);
            true
        } else {
            false
        }
    }

    pub fn push_delimited<F, R>(&mut self, func: F, data: DelimitedData) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        let start = self.start_delimited();
        let ret = func(self);
        self.close_delimited(start, data);
        ret
    }

    pub fn start_delimited(&mut self) -> OpenDelimited {
        let index = AbsoluteTokenTreeIndex(self.length() as u32);
        let parent = self.current_delimited_sequence.replace(index);

        self.tokens.push(ArenaTokenTree::DelimitedStart(
            DelimitedBounds {
                start: index,
                length: NonZeroU32::MIN,
                parent,
                last_push_was_token: false,
            },
            DelimitedData {
                span: DelimSpan { open: Default::default(), close: Default::default() },
                spacing: DelimSpacing { open: Spacing::Alone, close: Spacing::Alone },
                delimiter: Delimiter::Parenthesis,
            },
        ));
        OpenDelimited { start: index }
    }

    pub fn close_delimited(&mut self, open: OpenDelimited, delimited_data: DelimitedData) {
        let last_push_was_token = self.last_push_was_token;
        self.last_push_was_token = false;

        let length = self.length();
        match &mut self.tokens[open.start.as_usize()] {
            ArenaTokenTree::Token(..) => {
                unreachable!("Called close_delimited on a token");
            }
            ArenaTokenTree::DelimitedStart(bounds, data) => {
                let len = length.saturating_sub(open.start.as_usize());
                bounds.length = NonZeroU32::new(len as u32).unwrap();
                *data = delimited_data;
                self.current_delimited_sequence = bounds.parent;
                bounds.last_push_was_token = last_push_was_token;
            }
        }
    }

    pub fn empty_delimited(&mut self, data: DelimitedData) {
        self.push_delimited(|_builder| {}, data);
    }

    /// Copy `stream` into this builder, while possibly adding additional tokens or skipping
    /// existing tokens.
    pub fn build_from_stream<F>(&mut self, stream: &ArenaTokenStream, mut func: F)
    where
        F: FnMut(&mut Self, &ArenaTokenTree) -> PerTreeOp,
    {
        fn fill(
            builder: &mut ArenaTokenStreamBuilder,
            func: &mut dyn FnMut(&mut ArenaTokenStreamBuilder, &ArenaTokenTree) -> PerTreeOp,
            tree: &ArenaTokenTree,
            stream: &ArenaTokenStream,
        ) {
            match func(builder, tree) {
                PerTreeOp::Continue => {}
                PerTreeOp::Skip => {
                    return;
                }
            }
            match tree {
                ArenaTokenTree::Token(token, spacing) => {
                    builder.push_token(*token, *spacing);
                }
                ArenaTokenTree::DelimitedStart(bounds, data) => {
                    builder.push_delimited(
                        |builder| {
                            fill_iter(builder, func, stream.iter_delimited_contents(bounds));
                        },
                        *data,
                    );
                }
            }
        }
        fn fill_iter(
            builder: &mut ArenaTokenStreamBuilder,
            func: &mut dyn FnMut(&mut ArenaTokenStreamBuilder, &ArenaTokenTree) -> PerTreeOp,
            iter: ArenaTokenTreeIter<'_>,
        ) {
            let stream = iter.stream().clone();
            for tree in iter {
                fill(builder, func, tree, &stream);
            }
        }
        fill_iter(self, &mut func, stream.iter_top_level_trees());
    }

    /// Insert trees from `builder` at the start of a delimited sequence specified by
    /// `bounds`.
    pub fn insert_at_start_of_delimited(
        &mut self,
        bounds: DelimitedBounds,
        builder: ArenaTokenStreamBuilder,
    ) {
        let start = bounds.start;
        let Some(ArenaTokenTree::DelimitedStart(..)) = self.get_innermost_elem_at(start) else {
            panic!("insert_at_start_of_delimited called with invalid bounds");
        };
        // Insert the trees
        let inserted_len = builder.tokens.len() as u32;
        let after_insertion = bounds.start.0 + 1 + inserted_len;
        self.tokens.splice(start.as_usize() + 1..start.as_usize() + 1, builder.tokens);

        // Fix-up the start indices and parents after what was inserted
        for tree in &mut self.tokens[after_insertion as usize..] {
            match tree {
                ArenaTokenTree::Token(_, _) => {}
                ArenaTokenTree::DelimitedStart(b, _) => {
                    b.start = AbsoluteTokenTreeIndex(b.start.0 + inserted_len);
                    b.parent = b.parent.map(|p| {
                        if p > start { AbsoluteTokenTreeIndex(p.0 + inserted_len) } else { p }
                    });
                }
            }
        }

        let start = start.as_usize();
        // Fix-up the length of trees before what was inserted, including the current delimited
        // sequence.
        for tree in self.tokens[..start + 1].iter_mut().rev() {
            match tree {
                ArenaTokenTree::Token(_, _) => {}
                ArenaTokenTree::DelimitedStart(b, _) => {
                    if b.index_of_next_token_tree().as_usize() > start {
                        b.length = b.length.checked_add(inserted_len).unwrap();
                    }
                }
            }
        }

        // Fix-up the start indices and parents in what was inserted
        let offset = bounds.start().next_index();
        for tree in &mut self.tokens[start + 1..after_insertion as usize] {
            match tree {
                ArenaTokenTree::Token(_, _) => {}
                ArenaTokenTree::DelimitedStart(b, _) => {
                    b.start = AbsoluteTokenTreeIndex(b.start.0 + offset.0);
                    b.parent = Some(match b.parent {
                        Some(p) => AbsoluteTokenTreeIndex(p.0 + offset.0),
                        None => {
                            // Reparent the inserted top-level delimited sequences to the current
                            // delimited sequence
                            bounds.start
                        }
                    });
                }
            }
        }
    }

    #[inline]
    pub fn get_innermost_elem_at(&self, index: AbsoluteTokenTreeIndex) -> Option<&ArenaTokenTree> {
        self.tokens.get(index.as_usize())
    }

    #[inline]
    pub fn current_index(&self) -> AbsoluteTokenTreeIndex {
        AbsoluteTokenTreeIndex(self.length() as u32)
    }

    #[inline]
    pub fn tree_count_since(&self, index: AbsoluteTokenTreeIndex) -> u32 {
        self.current_index().0.saturating_sub(index.0)
    }

    #[inline]
    pub fn finish(self) -> ArenaTokenStream {
        assert!(self.current_delimited_sequence.is_none());
        if self.tokens.is_empty() {
            ArenaTokenStream::default()
        } else {
            let range = TokenTreeRange::full(&self.tokens);
            ArenaTokenStream {
                tokens: Arc::new(self.tokens),
                range,
                last_push_was_token: self.last_push_was_token,
            }
        }
    }

    #[inline]
    pub fn length(&self) -> usize {
        self.tokens.len()
    }

    fn fill_stream(&mut self, iter: ArenaTokenTreeIter<'_>) {
        let stream = iter.stream().clone();
        self.tokens.reserve(iter.length());
        for tt in iter {
            self.push_token_tree(tt, &stream);
        }
    }
}

pub enum PerTreeOp {
    /// Continue processing the tree as normally.
    Continue,
    /// Skip the tree, do not insert it.
    Skip,
}

/// A shared empty token stream, used to avoid an unnecessary `Arc` allocation for every empty
/// token stream.
static EMPTY_TOKEN_STREAM: LazyLock<ArenaTokenStream> = LazyLock::new(|| ArenaTokenStream {
    tokens: Arc::new(Vec::new()),
    range: TokenTreeRange::empty(),
    last_push_was_token: false,
});

#[derive(Clone, Debug)]
pub struct ArenaTokenStream {
    tokens: Arc<Vec<ArenaTokenTree>>,
    range: TokenTreeRange,
    last_push_was_token: bool,
}

impl ArenaTokenStream {
    /// Note: using this function is potentially dangerous, because the caller has to ensure that
    /// if `tokens` contains any delimited sequences, their indices are lined up and do not refer
    /// to anything existing outside of the passed set of tokens.
    /// That is why the function is private.
    pub fn from_token_iter<I>(tokens: I) -> Self
    where
        I: IntoIterator<Item = (Token, Spacing)>,
    {
        let tokens = Arc::new(
            tokens
                .into_iter()
                .map(|(token, spacing)| ArenaTokenTree::Token(token, spacing))
                .collect::<Vec<_>>(),
        );
        let range = TokenTreeRange::full(&tokens);
        Self { tokens, range, last_push_was_token: true }
    }

    /// Create a new stream out of the token trees.
    /// We might need to copy out children trees out of `stream`, if `tokens` contains any
    /// delimited sequences.
    /// We also need to reparent those to fix-up the parent indices.
    pub fn new_reparented(trees: &[ArenaTokenTree], stream: &ArenaTokenStream) -> Self {
        let mut builder = ArenaTokenStreamBuilder::with_capacity(trees.len());
        // FIXME: implement this in a more performant way
        for tree in trees {
            builder.push_token_tree(tree, stream);
        }
        builder.finish()
    }

    pub fn from_ast(node: &(impl HasTokens + fmt::Debug)) -> Self {
        let tokens = node.tokens().unwrap_or_else(|| panic!("missing tokens for node: {:?}", node));
        let mut builder = ArenaTokenStreamBuilder::default();
        attrs_and_tokens_to_token_trees_arena(node.attrs(), tokens, &mut builder, 0);
        builder.finish()
    }

    pub fn to_token_stream(&self) -> TokenStream {
        let mut tokens = vec![];
        for tt in self.iter_top_level_trees() {
            tokens.push(tt.to_token_tree(self));
        }
        TokenStream::new(tokens)
    }

    /// Try to reuse the tokens of this stream into a builder, if we are the only copy.
    /// If it is not the only copy, clones the inner tokens.
    pub fn into_builder(self) -> ArenaTokenStreamBuilder {
        // Reuse the whole thing
        if self.range.start.0 == 0 && self.range.end.0 == self.tokens.len() as u32 {
            let last_push_was_token = self.last_push_was_token;
            ArenaTokenStreamBuilder {
                tokens: self.try_take_tokens(),
                current_delimited_sequence: None,
                last_push_was_token,
            }
        } else {
            // Copy out the given range
            let mut builder = ArenaTokenStreamBuilder::with_capacity(self.range.len());
            builder.push_stream(self);
            builder
        }
    }

    /// Try to reuse the tokens of this stream, if we are the only copy.
    /// If it is not the only copy, clones the inner tokens.
    fn try_take_tokens(mut self) -> Vec<ArenaTokenTree> {
        let tokens = Arc::make_mut(&mut self.tokens);
        std::mem::take(tokens)
    }

    /// Create a token stream containing a single token with alone spacing. The
    /// spacing used for the final token in a constructed stream doesn't matter
    /// because it's never used. In practice we arbitrarily use
    /// `Spacing::Alone`.
    pub fn token_alone(kind: TokenKind, span: Span) -> Self {
        Self {
            tokens: Arc::new(vec![ArenaTokenTree::token_alone(kind, span)]),
            range: TokenTreeRange::single(),
            last_push_was_token: true,
        }
    }

    /// Extract **the contents** of a delimited sequence out of this token stream.
    /// The delimited sequence start/end is **NOT** returend in the output.
    /// `stream` is the original token stream that contains the delimited sequence identified by
    /// `bounds`.
    pub fn separate_delimited_inner(
        bounds: DelimitedBounds,
        stream: &ArenaTokenStream,
    ) -> ArenaTokenStream {
        if bounds.is_empty() {
            return Self::default();
        }
        let range = TokenTreeRange::from_bounds_contents(&bounds);
        Self {
            tokens: stream.tokens.clone(),
            range,
            last_push_was_token: bounds.last_push_was_token,
        }
    }

    #[inline]
    pub fn range(&self) -> TokenTreeRange {
        self.range
    }

    #[inline]
    pub fn length(&self) -> usize {
        self.range.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.range.is_empty()
    }

    #[inline]
    pub fn get_parent_of(&self, bounds: DelimitedBounds) -> Option<DelimitedBounds> {
        let parent = bounds.parent?;
        match self.get_innermost_elem_at(parent) {
            Some(ArenaTokenTree::Token(..)) => {
                panic!("DelimitedBounds parent index points to a token. This is a bug.");
            }
            Some(ArenaTokenTree::DelimitedStart(bounds, _)) => Some(*bounds),
            None => None,
        }
    }

    #[inline]
    pub fn get_innermost_elem_at(&self, index: AbsoluteTokenTreeIndex) -> Option<&ArenaTokenTree> {
        if !self.range.contains(index) {
            return None;
        }
        self.tokens.get(index.as_usize())
    }

    /// Iterate top-level token trees of a delimited token sequence.
    /// Does not return the delimited sequence start itself.
    pub fn iter_delimited_contents(&self, bounds: &DelimitedBounds) -> ArenaTokenTreeIter<'_> {
        ArenaTokenTreeIter::new_delimited_contents(self, bounds)
    }

    /// Iterate over the delimited token sequence.
    /// Return the delimited sequence start itself.
    pub fn iter_delimited(&self, bounds: &DelimitedBounds) -> ArenaTokenTreeIter<'_> {
        ArenaTokenTreeIter::new_delimited(self, bounds)
    }

    /// Iterate over the top-level token trees of the whole stream.
    /// Does not recurse into delimited sequences.
    pub fn iter_top_level_trees(&self) -> ArenaTokenTreeIter<'_> {
        ArenaTokenTreeIter::new_top_level(self)
    }

    pub fn iter_all_trees(&self) -> impl Iterator<Item = &ArenaTokenTree> + DoubleEndedIterator {
        self.tokens.as_slice()[self.range.start.as_usize()..self.range.end.as_usize()].into_iter()
    }

    fn iter_flattened(&self) -> impl Iterator<Item = FlattenedTokenTree> {
        let mut index = self.range.start();
        std::iter::from_fn(move || {
            let Some(tree) = self.get_innermost_elem_at(index) else {
                return None;
            };
            index.bump_single();
            let item = match tree {
                ArenaTokenTree::Token(token, spacing) => {
                    FlattenedTokenTree::Token(*token, *spacing)
                }
                ArenaTokenTree::DelimitedStart(bounds, data) => {
                    // FIXME: is this correct? do we also have to take parents into account?
                    FlattenedTokenTree::Delimited { data: *data, length: bounds.length }
                }
            };
            Some(item)
        })
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Encodable, Decodable, StableHash)]
enum FlattenedTokenTree {
    Token(Token, Spacing),
    Delimited { data: DelimitedData, length: NonZeroU32 },
}

impl Default for ArenaTokenStream {
    fn default() -> Self {
        EMPTY_TOKEN_STREAM.clone()
    }
}

impl PartialEq for ArenaTokenStream {
    fn eq(&self, other: &Self) -> bool {
        self.iter_flattened().eq(other.iter_flattened())
    }
}

impl Eq for ArenaTokenStream {}

impl Hash for ArenaTokenStream {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for tree in self.iter_flattened() {
            tree.hash(state);
        }
    }
}

impl StableHash for ArenaTokenStream {
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        for tree in self.iter_flattened() {
            tree.stable_hash(hcx, hasher);
        }
    }
}

impl<S: SpanEncoder> Encodable<S> for ArenaTokenStream {
    fn encode(&self, encoder: &mut S) {
        let size = self.length();
        size.encode(encoder);
        for tree in self.iter_flattened() {
            tree.encode(encoder);
        }
    }
}

impl<D: SpanDecoder> Decodable<D> for ArenaTokenStream {
    fn decode(decoder: &mut D) -> Self {
        let mut remaining = usize::decode(decoder);
        if remaining == 0 {
            return Self::default();
        }

        let mut builder = ArenaTokenStreamBuilder::with_capacity(remaining);

        fn build<D: SpanDecoder>(
            decoder: &mut D,
            builder: &mut ArenaTokenStreamBuilder,
            remaining: &mut usize,
        ) {
            if *remaining == 0 {
                return;
            }
            let flattened = FlattenedTokenTree::decode(decoder);
            *remaining -= 1;
            match flattened {
                FlattenedTokenTree::Token(token, spacing) => {
                    builder.push_token(token, spacing);
                }
                FlattenedTokenTree::Delimited { data, length } => {
                    let mut remaining_children = length.get() as usize - 1;
                    builder.push_delimited(
                        |builder| {
                            while remaining_children > 0 {
                                build(decoder, builder, &mut remaining_children);
                            }
                        },
                        data,
                    );
                    *remaining -= (length.get() - 1) as usize;
                }
            }
        }

        while remaining > 0 {
            build(decoder, &mut builder, &mut remaining);
        }
        builder.finish()
    }
}

/// Absolute index into a flat list of arena token trees.
#[derive(
    Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Encodable, Decodable, StableHash
)]
pub struct AbsoluteTokenTreeIndex(u32);

impl AbsoluteTokenTreeIndex {
    #[inline]
    pub fn bump_single(&mut self) {
        self.0 += 1;
    }

    #[inline]
    pub fn bump_delimited(&mut self, bounds: &DelimitedBounds) {
        *self = bounds.index_of_next_token_tree();
    }

    #[inline]
    pub fn next_index(&self) -> Self {
        Self(self.0 + 1)
    }

    fn as_usize(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Encodable, Decodable)]
pub struct TokenTreeRange {
    /// Inclusive
    start: AbsoluteTokenTreeIndex,
    /// Exclusive
    end: AbsoluteTokenTreeIndex,
}

impl TokenTreeRange {
    fn empty() -> Self {
        Self { start: AbsoluteTokenTreeIndex(0), end: AbsoluteTokenTreeIndex(0) }
    }

    fn full(tokens: &[ArenaTokenTree]) -> Self {
        Self { start: AbsoluteTokenTreeIndex(0), end: AbsoluteTokenTreeIndex(tokens.len() as u32) }
    }

    fn single() -> Self {
        Self { start: AbsoluteTokenTreeIndex(0), end: AbsoluteTokenTreeIndex(1) }
    }

    /// Extract a range containing the *contents* of the delimited sequence, without its starting
    /// delimiter.
    fn from_bounds_contents(bounds: &DelimitedBounds) -> Self {
        Self { start: bounds.start.next_index(), end: bounds.index_of_next_token_tree() }
    }

    #[inline]
    pub fn start(&self) -> AbsoluteTokenTreeIndex {
        self.start
    }

    #[inline]
    pub fn end(&self) -> AbsoluteTokenTreeIndex {
        self.end
    }

    #[inline]
    pub fn one_past_end(&self) -> AbsoluteTokenTreeIndex {
        self.end.next_index()
    }

    #[inline]
    pub fn len(&self) -> usize {
        (self.end.0 - self.start.0) as usize
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    #[inline]
    fn contains(&self, index: AbsoluteTokenTreeIndex) -> bool {
        index >= self.start && index < self.end
    }
}

// Converts multiple attributes and the tokens for a target AST node into token trees, and appends
// them to `res`.
//
// Example: if the AST node is "fn f() { blah(); }", then:
// - Simple if no attributes are present, e.g. "fn f() { blah(); }"
// - Simple if only outer attribute are present, e.g. "#[outer1] #[outer2] fn f() { blah(); }"
// - Trickier if inner attributes are present, because they must be moved within the AST node's
//   tokens, e.g. "#[outer] fn f() { #![inner] blah() }"
pub fn attrs_and_tokens_to_token_trees_arena(
    attrs: &[Attribute],
    target_tokens: &LazyAttrTokenStream,
    builder: &mut ArenaTokenStreamBuilder,
    start: usize,
) {
    let idx = attrs.partition_point(|attr| matches!(attr.style, crate::AttrStyle::Outer));
    let (outer_attrs, inner_attrs) = attrs.split_at(idx);

    // Add outer attribute tokens.
    for attr in outer_attrs {
        attr.push_token_trees(builder);
    }

    // Add target AST node tokens.
    target_tokens.to_attr_token_stream().push_token_trees(builder);

    // Insert inner attribute tokens.
    if !inner_attrs.is_empty() {
        if let Some(bounds) = get_insertion_point(
            inner_attrs,
            AbsoluteTokenTreeIndex(start as u32),
            AbsoluteTokenTreeIndex(builder.tokens.len() as u32),
            builder,
        ) {
            // FIXME: implement this in a more efficient way
            let mut inner = ArenaTokenStreamBuilder::default();
            for attribute in inner_attrs {
                attribute.push_token_trees(&mut inner);
            }
            builder.insert_at_start_of_delimited(bounds, inner);
        } else {
            panic!("Failed to find trailing delimited group in: {builder:?}");
        }
    }

    // Inner attributes are only supported on blocks, functions, impls, and
    // modules. All of these have their inner attributes placed at the
    // beginning of the rightmost outermost braced group:
    // e.g. `fn foo() { #![my_attr] }`. (Note: the braces may be within
    // invisible delimiters.)
    //
    // Therefore, we can insert them back into the right location without
    // needing to do any extra position tracking.
    //
    // Note: Outline modules are an exception - they can have attributes like
    // `#![my_attr]` at the start of a file. Support for custom attributes in
    // this position is not properly implemented - we always synthesize fake
    // tokens, so we never reach this code.
    fn get_insertion_point(
        inner_attrs: &[Attribute],
        start: AbsoluteTokenTreeIndex,
        end: AbsoluteTokenTreeIndex,
        builder: &ArenaTokenStreamBuilder,
    ) -> Option<DelimitedBounds> {
        let is_top_level =
            |bounds: &DelimitedBounds| bounds.parent.map(|p| p < start).unwrap_or(true);

        // We need to iterate backwards, only in the range given to us
        for tree in builder.tokens[start.as_usize()..end.as_usize()].iter().rev() {
            // We need to find only the "top-level" trees in the given range
            // We recognize those by them either having no parent, or having a parent that is outside
            // the range.
            if let ArenaTokenTree::DelimitedStart(
                bounds,
                DelimitedData { delimiter: Delimiter::Brace, .. },
            ) = tree
            {
                if !is_top_level(bounds) {
                    continue;
                }
                // Found it: the rightmost, outermost braced group.
                return Some(*bounds);
            } else if let ArenaTokenTree::DelimitedStart(
                bounds,
                DelimitedData { delimiter: Delimiter::Invisible(_), .. },
            ) = tree
            {
                if !is_top_level(bounds) {
                    continue;
                }
                // Recurse inside invisible delimiters.
                // We iterate from the first tree inside of this delimited sequence
                let end = bounds.index_of_next_token_tree();
                if let Some(bounds) =
                    get_insertion_point(inner_attrs, bounds.start().next_index(), end, builder)
                {
                    return Some(bounds);
                }
            }
        }
        None
    }
}

#[derive(Clone)]
pub struct ArenaTokenTreeIter<'a> {
    stream: &'a ArenaTokenStream,
    index: AbsoluteTokenTreeIndex,
    end: AbsoluteTokenTreeIndex,
}

impl<'a> ArenaTokenTreeIter<'a> {
    fn new_top_level(stream: &'a ArenaTokenStream) -> Self {
        let index = stream.range.start();
        let end = stream.range.end();
        Self { stream, index, end }
    }

    fn new_delimited(stream: &'a ArenaTokenStream, bounds: &DelimitedBounds) -> Self {
        let index = bounds.start();
        let end = bounds.index_of_next_token_tree();
        Self { stream, index, end }
    }

    fn new_delimited_contents(stream: &'a ArenaTokenStream, bounds: &DelimitedBounds) -> Self {
        let index = bounds.start().next_index();
        let end = bounds.index_of_next_token_tree();
        Self { stream, index, end }
    }

    pub fn stream(&self) -> &'a ArenaTokenStream {
        self.stream
    }

    // Peeking could be done via `Peekable`, but most iterators need peeking,
    // and this is simple and avoids the need to use `peekable` and `Peekable`
    // at all the use sites.
    pub fn peek(&self) -> Option<&'a ArenaTokenTree> {
        if self.index >= self.end {
            return None;
        }
        self.stream.get_innermost_elem_at(self.index)
    }

    /// Returns true if the iterator has exactly single tree in it.
    pub fn has_single_tree(&self) -> bool {
        let mut iter = self.clone();
        if iter.next().is_none() {
            return false;
        }
        iter.next().is_none()
    }

    fn length(&self) -> usize {
        self.end.as_usize().saturating_sub(self.index.as_usize())
    }
}

impl<'a> Iterator for ArenaTokenTreeIter<'a> {
    type Item = &'a ArenaTokenTree;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.end {
            return None;
        }
        let item = self.stream.get_innermost_elem_at(self.index)?;
        match item {
            token @ ArenaTokenTree::Token(..) => {
                self.index.bump_single();
                Some(token)
            }
            tree @ ArenaTokenTree::DelimitedStart(bounds, _) => {
                self.index.bump_delimited(bounds);
                Some(tree)
            }
        }
    }
}

#[must_use]
pub struct OpenDelimited {
    start: AbsoluteTokenTreeIndex,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Encodable, Decodable, StableHash)]
pub struct DelimitedBounds {
    start: AbsoluteTokenTreeIndex,
    /// The length includes both the start token.
    /// So an empty delimited sequence has length 1.
    length: NonZeroU32,
    /// Index of the parent of the current delimited sequence.
    /// If this is the root delimited sequence, is `None`.
    parent: Option<AbsoluteTokenTreeIndex>,
    last_push_was_token: bool,
}

impl DelimitedBounds {
    #[inline]
    pub fn start(&self) -> AbsoluteTokenTreeIndex {
        self.start
    }

    /// Return the index of the next token tree that follows this delimited token sequence.
    #[inline]
    pub fn index_of_next_token_tree(&self) -> AbsoluteTokenTreeIndex {
        AbsoluteTokenTreeIndex(self.start.0 + self.length.get())
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.length.get() == 1
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Encodable, Decodable, StableHash)]
pub struct DelimitedData {
    pub span: DelimSpan,
    pub spacing: DelimSpacing,
    pub delimiter: Delimiter,
}
