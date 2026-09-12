use std::borrow::Cow;
use std::fmt;
use std::sync::Arc;

use rustc_data_structures::stable_hash::{StableHash, StableHashCtxt, StableHasher};
use rustc_index::static_assert_size;
use rustc_macros::{Decodable, Encodable, StableHash};
use rustc_span::Span;

use crate::token::{Delimiter, Token, TokenKind};
use crate::tokenstream::{
    DelimSpacing, DelimSpan, LazyAttrTokenStream, Spacing, TokenStream, TokenTree,
};
use crate::{Attribute, HasTokens};

/// Part of a `TokenArena`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Encodable, Decodable)]
#[derive(StableHash)] // FIXME: is this Ok?
pub enum ArenaTokenTree {
    /// A single token. Should never be `OpenDelim` or `CloseDelim`, because
    /// delimiters are implicitly represented by `DelimitedStart`/`DelimitedEnd`.
    Token(Token, Spacing),
    /// A delimited sequence of token trees.
    DelimitedStart(DelimitedBounds, DelimitedData),
}

impl ArenaTokenTree {
    /// Create a `TokenTree::Token` with alone spacing.
    pub fn token_alone(kind: TokenKind, span: Span) -> ArenaTokenTree {
        ArenaTokenTree::Token(Token::new(kind, span), Spacing::Alone)
    }

    /// Create a `TokenTree::Token` with joint spacing.
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
                let tts = arena.iter_delimited(bounds).map(|tt| tt.to_token_tree(arena)).collect();
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

    pub fn to_delimited_data(&self) -> Option<&DelimitedData> {
        match self {
            ArenaTokenTree::Token(_, _) => None,
            ArenaTokenTree::DelimitedStart(_, data) => Some(data),
        }
    }

    pub fn to_delimited_bounds(&self) -> Option<&DelimitedBounds> {
        match self {
            ArenaTokenTree::Token(_, _) => None,
            ArenaTokenTree::DelimitedStart(bounds, _) => Some(bounds),
        }
    }
}

static_assert_size!(ArenaTokenTree, 40);

#[derive(Debug, Default)]
pub struct ArenaTokenStreamBuilder {
    tokens: Vec<ArenaTokenTree>,
    /// Index of the current delimited sequence
    current_delimited_sequence: Option<usize>,
}

impl ArenaTokenStreamBuilder {
    pub fn with_capacity(capacity: usize) -> Self {
        Self { tokens: Vec::with_capacity(capacity), current_delimited_sequence: None }
    }

    pub fn push_token(&mut self, token: Token, spacing: Spacing) {
        self.tokens.push(ArenaTokenTree::Token(token, spacing));
    }

    pub fn push_token_alone(&mut self, token: Token) {
        self.tokens.push(ArenaTokenTree::Token(token, Spacing::Alone));
    }

    pub fn pop(&mut self) -> Option<ArenaTokenTree> {
        let tree = self.tokens.pop();
        if let Some(tree) = &tree {
            assert!(matches!(tree, ArenaTokenTree::Token(..)));
        }
        tree
    }

    pub fn push_token_tree(&mut self, tt: &TokenTree) {
        match tt {
            TokenTree::Token(token, spacing) => {
                self.tokens.push(ArenaTokenTree::Token(*token, *spacing));
            }
            TokenTree::Delimited(span, spacing, delimiter, stream) => {
                let start = self.start_delimited();
                self.fill(stream);
                self.close_delimited(
                    start,
                    DelimitedData { span: *span, spacing: *spacing, delimiter: *delimiter },
                );
            }
        }
    }

    pub fn start_delimited(&mut self) -> OpenDelimited {
        let index = self.length();
        let parent = self.current_delimited_sequence.replace(index);

        self.tokens.push(ArenaTokenTree::DelimitedStart(
            DelimitedBounds { start: index as u32, length: 0, parent: parent.map(|v| v as u32) },
            DelimitedData {
                span: DelimSpan { open: Default::default(), close: Default::default() },
                spacing: DelimSpacing { open: Spacing::Alone, close: Spacing::Alone },
                delimiter: Delimiter::Parenthesis,
            },
        ));
        OpenDelimited { start: index }
    }

    pub fn close_delimited(&mut self, open: OpenDelimited, delimited_data: DelimitedData) {
        let length = self.length();
        match &mut self.tokens[open.start] {
            ArenaTokenTree::Token(..) => {
                unreachable!("Called finish_delimited on a token");
            }
            ArenaTokenTree::DelimitedStart(bounds, data) => {
                let len = length.saturating_sub(open.start);
                bounds.length = len as u32;
                *data = delimited_data;
                self.current_delimited_sequence = bounds.parent.map(|v| v as usize);
            }
        }
    }

    pub fn empty_delimited(&mut self, delimited_data: DelimitedData) {
        let start = self.start_delimited();
        self.close_delimited(start, delimited_data);
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
                    let start = builder.start_delimited();
                    fill_iter(builder, func, stream.iter_delimited(bounds));
                    builder.close_delimited(start, *data);
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
        let start = bounds.start as usize;
        let Some(ArenaTokenTree::DelimitedStart(..)) = self.get_innermost_elem_at(start) else {
            panic!("insert_at_start_of_delimited called with invalid bounds");
        };
        // Insert the trees
        let inserted_len = builder.tokens.len() as u32;
        let after_insertion = bounds.start + inserted_len;
        self.tokens.splice(start..start, builder.tokens);

        // Fix-up the start indices and parents after what was inserted
        for tree in &mut self.tokens[after_insertion as usize..] {
            match tree {
                ArenaTokenTree::Token(_, _) => {}
                ArenaTokenTree::DelimitedStart(b, _) => {
                    b.start += inserted_len;
                    b.parent = b.parent.map(|p| {
                        assert!(p >= start as u32);
                        if p > start as u32 { p + inserted_len } else { p }
                    });
                }
            }
        }

        // Fix-up the length of trees before what was inserted, including the current delimited
        // sequence.
        for tree in self.tokens[..start + 1].iter_mut().rev() {
            match tree {
                ArenaTokenTree::Token(_, _) => {}
                ArenaTokenTree::DelimitedStart(b, _) => {
                    if b.index_of_next_token_tree() > start {
                        b.length += inserted_len;
                    }
                }
            }
        }

        // Fix-up the start indices and parents in what was inserted
        let offset = bounds.start + 1;
        for tree in &mut self.tokens[start + 1..after_insertion as usize] {
            match tree {
                ArenaTokenTree::Token(_, _) => {}
                ArenaTokenTree::DelimitedStart(b, _) => {
                    b.start += offset;
                    b.parent = b.parent.map(|p| p + offset);
                }
            }
        }
    }

    pub fn get_innermost_elem_at(&self, index: usize) -> Option<&ArenaTokenTree> {
        self.tokens.get(index)
    }

    pub fn finish(self) -> ArenaTokenStream {
        ArenaTokenStream { tokens: Arc::new(self.tokens) }
    }

    pub fn length(&self) -> usize {
        self.tokens.len()
    }

    fn fill(&mut self, stream: &TokenStream) {
        for tt in stream.iter() {
            self.push_token_tree(tt);
        }
    }
}

pub enum PerTreeOp {
    /// Continue processing the tree as normally.
    Continue,
    /// Skip the tree, do not insert it.
    Skip,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Encodable, Decodable)]
pub struct ArenaTokenStream {
    tokens: Arc<Vec<ArenaTokenTree>>,
}

impl ArenaTokenStream {
    /// Note: using this function is potentially dangerous, because the caller has to ensure that
    /// if `tokens` contains any delimited sequences, their indices are lined up and do not refer
    /// to anything existing outside of the passed set of tokens.
    /// That is why the function is private.
    pub fn from_token_vec(tokens: Vec<(Token, Spacing)>) -> Self {
        // FIXME: solve this in a better way
        Self {
            tokens: Arc::new(
                tokens
                    .into_iter()
                    .map(|(token, spacing)| ArenaTokenTree::Token(token, spacing))
                    .collect(),
            ),
        }
    }

    /// Create a new stream out of the token trees.
    /// We might need to copy out children trees out of `stream`, if `tokens` contains any
    /// delimited sequences.
    /// We also need to reparent those to fix-up the parent indices.
    pub fn new_reparented(trees: &[ArenaTokenTree], stream: &ArenaTokenStream) -> Self {
        let mut builder = ArenaTokenStreamBuilder::with_capacity(trees.len());
        // FIXME: implement this in a more performant way
        for tree in trees {
            let tree = tree.to_token_tree(stream);
            builder.push_token_tree(&tree);
        }
        builder.finish()
    }

    pub fn from_token(token: Token, spacing: Spacing) -> Self {
        Self { tokens: Arc::new(vec![ArenaTokenTree::Token(token, spacing)]) }
    }

    pub fn from_stream(stream: &TokenStream) -> Self {
        let mut arena = ArenaTokenStreamBuilder {
            tokens: Vec::with_capacity(stream.len()),
            current_delimited_sequence: None,
        };
        arena.fill(stream);
        arena.finish()
    }

    pub fn from_ast(node: &(impl HasTokens + fmt::Debug)) -> Self {
        let tokens = node.tokens().unwrap_or_else(|| panic!("missing tokens for node: {:?}", node));
        let mut builder = ArenaTokenStreamBuilder::default();
        attrs_and_tokens_to_token_trees_arena(node.attrs(), tokens, &mut builder);
        builder.finish()
    }

    pub fn to_token_stream(&self) -> TokenStream {
        let mut tokens = vec![];
        for tt in self.iter_top_level_trees() {
            tokens.push(tt.to_token_tree(self));
        }
        TokenStream::new(tokens)
    }

    /// Extract **the contents** of a delimited sequence out of this token stream.
    /// The delimited sequence start/end is **NOT** returend in the output.
    /// `stream` is the original token stream that contains the delimited sequence identified by
    /// `bounds`.
    pub fn separate_delimited_inner(
        bounds: DelimitedBounds,
        stream: &ArenaTokenStream,
    ) -> ArenaTokenStream {
        // eprintln!("separate delimited");
        // This could be implemented in a smarter way by reusing the original allocation
        // and storing an index with "view" into it.
        let start = bounds.start as usize + 1;
        let length = (bounds.length as usize).saturating_sub(1);

        let mut tokens = stream.tokens[start..start + length].to_vec();
        let start = start as u32;

        for tree in &mut tokens {
            match tree {
                ArenaTokenTree::Token(_, _) => {}
                ArenaTokenTree::DelimitedStart(b, _) => {
                    b.start -= start;
                    b.parent = b.parent.and_then(|p| {
                        if p < start {
                            // Top-level, now we will have no parent
                            None
                        } else {
                            Some(p - start)
                        }
                    });
                }
            }
        }
        Self { tokens: Arc::new(tokens) }
    }

    pub fn length(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn get_parent_of(&self, bounds: DelimitedBounds) -> Option<DelimitedBounds> {
        let parent = bounds.parent?;
        match self.tokens.get(parent as usize).expect("Parent index was not found") {
            ArenaTokenTree::Token(..) => {
                panic!("DelimitedBounds parent index points to a token. This is a bug.");
            }
            ArenaTokenTree::DelimitedStart(bounds, _) => Some(*bounds),
        }
    }

    pub fn get_innermost_elem_at(&self, index: usize) -> Option<&ArenaTokenTree> {
        self.tokens.get(index)
    }

    /// Iterate top-level token trees of a delimited token sequence.
    /// Does not return the delimited sequence start itself.
    pub fn iter_delimited(&self, bounds: &DelimitedBounds) -> ArenaTokenTreeIter<'_> {
        ArenaTokenTreeIter::new_delimited(self, bounds)
    }

    /// Iterate over the top-level token trees of the whole stream.
    /// Does not recurse into delimited sequences.
    pub fn iter_top_level_trees(&self) -> ArenaTokenTreeIter<'_> {
        ArenaTokenTreeIter::new_top_level(self)
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
        if let Some(bounds) = get_insertion_point(inner_attrs, 0, builder.tokens.len(), builder) {
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
        start: usize,
        end: usize,
        builder: &ArenaTokenStreamBuilder,
    ) -> Option<DelimitedBounds> {
        let is_top_level =
            |bounds: &DelimitedBounds| bounds.parent.map(|p| p < start as u32).unwrap_or(true);

        // We need to iterate backwards, only in the range given to us
        for tree in builder.tokens[start..end].iter().rev() {
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
                // Recurse inside invisible delimiters.
                // We iterate from the first tree inside of this delimited sequence
                let end = bounds.index_of_next_token_tree();
                if let Some(bounds) =
                    get_insertion_point(inner_attrs, bounds.start() + 1, end, builder)
                {
                    return Some(bounds);
                }
            }
        }
        None
    }
}

impl StableHash for ArenaTokenStream {
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        self.tokens.as_slice().stable_hash(hcx, hasher);
    }
}

pub struct ArenaTokenTreeIter<'a> {
    index: usize,
    end: usize,
    stream: &'a ArenaTokenStream,
}

impl<'a> ArenaTokenTreeIter<'a> {
    fn new_top_level(stream: &'a ArenaTokenStream) -> Self {
        Self { index: 0, end: stream.tokens.len(), stream }
    }

    fn new_delimited(stream: &'a ArenaTokenStream, bounds: &DelimitedBounds) -> Self {
        let index = (bounds.start + 1) as usize;
        let end = bounds.index_of_next_token_tree();
        Self { index, end, stream }
    }

    pub fn stream(&self) -> &'a ArenaTokenStream {
        self.stream
    }

    // Peeking could be done via `Peekable`, but most iterators need peeking,
    // and this is simple and avoids the need to use `peekable` and `Peekable`
    // at all the use sites.
    pub fn peek(&self) -> Option<&'a ArenaTokenTree> {
        self.stream.tokens.get(self.index)
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
                self.index += 1;
                Some(token)
            }
            tree @ ArenaTokenTree::DelimitedStart(bounds, _) => {
                self.index = bounds.index_of_next_token_tree();
                Some(tree)
            }
        }
    }
}

pub struct OpenDelimited {
    start: usize,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Encodable, Decodable, StableHash)]
pub struct DelimitedBounds {
    start: u32,
    /// The length includes both the start and the end token.
    /// So an empty delimited sequence has length 2.
    length: u32,
    /// Index of the parent of the current delimited sequence.
    /// If this is the root delimited sequence, is `None`.
    parent: Option<u32>,
}

impl DelimitedBounds {
    pub fn start(&self) -> usize {
        self.start as usize
    }

    /// Return the index of the next token tree that follows this delimited token sequence.
    pub fn index_of_next_token_tree(&self) -> usize {
        (self.start + self.length) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.length == 1
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Encodable, Decodable, StableHash)]
pub struct DelimitedData {
    pub span: DelimSpan,
    pub spacing: DelimSpacing,
    pub delimiter: Delimiter,
}
