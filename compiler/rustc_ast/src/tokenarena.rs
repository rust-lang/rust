use std::sync::Arc;

use rustc_data_structures::stable_hash::{StableHash, StableHashCtxt, StableHasher};
use rustc_index::static_assert_size;
use rustc_macros::{Decodable, Encodable, StableHash};
use rustc_span::Span;

use crate::token::{Delimiter, Token, TokenKind};
use crate::tokenstream::{DelimSpacing, DelimSpan, Spacing, TokenStream, TokenTree};

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

    /// Iter top-level token trees of a delimited token sequence.
    pub fn iter_delimited(&self, bounds: &DelimitedBounds) -> impl Iterator<Item = ArenaTokenTree> {
        let mut index = (bounds.start + 1) as usize;
        let end = bounds.index_of_next_token_tree();
        std::iter::from_fn(move || {
            if index >= end {
                return None;
            }
            let item = self.get_innermost_elem_at(index)?;
            match item {
                token @ ArenaTokenTree::Token(..) => {
                    index += 1;
                    Some(*token)
                }
                tree @ ArenaTokenTree::DelimitedStart(bounds, _) => {
                    index = bounds.index_of_next_token_tree();
                    Some(*tree)
                }
            }
        })
    }

    pub fn iter_top_level_trees(&self) -> impl Iterator<Item = ArenaTokenTree> {
        let mut index = 0;
        let end = self.tokens.len();
        std::iter::from_fn(move || {
            if index >= end {
                return None;
            }
            let item = self.get_innermost_elem_at(index)?;
            match item {
                token @ ArenaTokenTree::Token(..) => {
                    index += 1;
                    Some(*token)
                }
                tree @ ArenaTokenTree::DelimitedStart(bounds, _) => {
                    index = bounds.index_of_next_token_tree();
                    Some(*tree)
                }
            }
        })
    }
}

impl StableHash for ArenaTokenStream {
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        self.tokens.as_slice().stable_hash(hcx, hasher);
    }
}

pub struct OpenDelimited {
    start: usize,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Encodable, Decodable, StableHash)]
pub struct DelimitedBounds {
    pub start: u32,
    /// The length includes both the start and the end token.
    /// So an empty delimited sequence has length 2.
    pub length: u32,
    /// Index of the parent of the current delimited sequence.
    /// If this is the root delimited sequence, is `None`.
    pub parent: Option<u32>,
}

impl DelimitedBounds {
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
