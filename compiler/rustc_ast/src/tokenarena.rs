use rustc_index::static_assert_size;
use rustc_macros::{Decodable, Encodable, StableHash};
use rustc_span::Span;

use crate::token::{Delimiter, Token};
use crate::tokenstream::{DelimSpacing, DelimSpan, Spacing, TokenStream, TokenTree};

/// Part of a `TokenArena`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Encodable, Decodable, StableHash)]
pub enum ArenaTokenTree {
    /// A single token. Should never be `OpenDelim` or `CloseDelim`, because
    /// delimiters are implicitly represented by `DelimitedStart`/`DelimitedEnd`.
    Token(Token, Spacing),
    /// A delimited sequence of token trees.
    DelimitedStart(DelimitedBounds, DelimitedData),
    // TODO: get rid of this and represent it implicitly
    DelimitedEnd,
}

impl ArenaTokenTree {
    /// Convert an arena token tree to the tree-shaped token tree.
    pub fn to_token_tree(&self, arena: &TokenArena) -> TokenTree {
        match self {
            ArenaTokenTree::Token(token, spacing) => TokenTree::Token(*token, *spacing),
            ArenaTokenTree::DelimitedStart(bounds, data) => {
                let tts = arena.iter_delimited(bounds).map(|tt| tt.to_token_tree(arena)).collect();
                TokenTree::Delimited(data.span, data.spacing, data.delimiter, TokenStream::new(tts))
            }
            ArenaTokenTree::DelimitedEnd => unreachable!(),
        }
    }

    /// Retrieves the `TokenTree`'s span.
    pub fn span(&self) -> Span {
        match self {
            Self::Token(token, _) => token.span,
            Self::DelimitedStart(_, data) => data.span.entire(),
            _ => unreachable!(),
        }
    }

    pub fn to_delimited_data(&self) -> Option<&DelimitedData> {
        match self {
            ArenaTokenTree::Token(_, _) => None,
            ArenaTokenTree::DelimitedStart(_, data) => Some(data),
            ArenaTokenTree::DelimitedEnd => unreachable!(),
        }
    }

    pub fn to_delimited_bounds(&self) -> Option<&DelimitedBounds> {
        match self {
            ArenaTokenTree::Token(_, _) => None,
            ArenaTokenTree::DelimitedStart(bounds, _) => Some(bounds),
            ArenaTokenTree::DelimitedEnd => unreachable!(),
        }
    }
}

static_assert_size!(ArenaTokenTree, 36);

#[derive(Debug, Default, PartialEq, Eq, Hash, Encodable, Decodable)]
pub struct TokenArena {
    tokens: Vec<ArenaTokenTree>,
}

impl TokenArena {
    pub fn new(tokens: Vec<ArenaTokenTree>) -> Self {
        Self { tokens }
    }

    pub fn push(&mut self, token: ArenaTokenTree) {
        self.tokens.push(token);
    }

    /// Iter top-level token trees of a delimited token sequence.
    pub fn iter_delimited(&self, bounds: &DelimitedBounds) -> impl Iterator<Item = ArenaTokenTree> {
        let mut index = (bounds.start + 1) as usize;
        let end = bounds.index_of_next_token_tree().saturating_sub(1);
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
                ArenaTokenTree::DelimitedEnd => unreachable!(),
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
                ArenaTokenTree::DelimitedEnd => unreachable!(),
            }
        })
    }

    pub fn start_delimited(&mut self) -> OpenDelimited {
        let index = self.length();
        self.tokens.push(ArenaTokenTree::DelimitedStart(
            DelimitedBounds { start: index as u32, length: 0 },
            DelimitedData {
                span: DelimSpan { open: Default::default(), close: Default::default() },
                spacing: DelimSpacing { open: Spacing::Alone, close: Spacing::Alone },
                delimiter: Delimiter::Parenthesis,
            },
        ));
        OpenDelimited { start: index }
    }

    pub fn finish_delimited(&mut self, open: OpenDelimited, delimited_data: DelimitedData) {
        self.tokens.push(ArenaTokenTree::DelimitedEnd);
        let length = self.length();
        match &mut self.tokens[open.start] {
            tree @ (ArenaTokenTree::Token(..) | ArenaTokenTree::DelimitedEnd) => {
                unreachable!("Called finish_delimited on an invalid tree type {tree:?}")
            }
            ArenaTokenTree::DelimitedStart(bounds, data) => {
                let len = length.saturating_sub(open.start);
                bounds.length = len as u32;
                *data = delimited_data;
            }
        }
    }

    pub fn get_innermost_elem_at(&self, index: usize) -> Option<&ArenaTokenTree> {
        self.tokens.get(index)
    }

    pub fn length(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn from_stream(stream: &TokenStream) -> Self {
        let mut arena = TokenArena { tokens: Vec::with_capacity(stream.len()) };
        arena.fill(stream);
        arena
    }

    fn fill(&mut self, stream: &TokenStream) {
        for item in stream.iter() {
            match item {
                TokenTree::Token(token, spacing) => {
                    self.tokens.push(ArenaTokenTree::Token(*token, *spacing));
                }
                TokenTree::Delimited(span, spacing, delimiter, stream) => {
                    let start = self.start_delimited();
                    self.fill(stream);
                    self.finish_delimited(
                        start,
                        DelimitedData { span: *span, spacing: *spacing, delimiter: *delimiter },
                    );
                }
            }
        }
    }

    pub fn to_token_stream(&self) -> TokenStream {
        let mut tokens = vec![];
        for tt in self.iter_top_level_trees() {
            tokens.push(tt.to_token_tree(self));
        }
        TokenStream::new(tokens)
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
}

impl DelimitedBounds {
    /// Return the index of the next token tree that follows this delimited token sequence.
    pub fn index_of_next_token_tree(&self) -> usize {
        (self.start + self.length) as usize
    }

    /// Return the index of the closing delimiter of this token sequence.
    pub fn index_of_closing_delimiter(&self) -> usize {
        self.index_of_next_token_tree().saturating_sub(1)
    }

    pub fn is_empty(&self) -> bool {
        self.length == 2
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Encodable, Decodable, StableHash)]
pub struct DelimitedData {
    pub span: DelimSpan,
    pub spacing: DelimSpacing,
    pub delimiter: Delimiter,
}
