use rustc_macros::{Decodable, Encodable, StableHash};

use crate::token::{Delimiter, Token};
use crate::tokenstream::{DelimSpacing, DelimSpan, Spacing, TokenStream, TokenTree};

/// Part of a `TokenArena`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Encodable, Decodable, StableHash)]
pub enum ArenaTokenTree {
    /// A single token. Should never be `OpenDelim` or `CloseDelim`, because
    /// delimiters are implicitly represented by `Delimited`.
    Token(Token, Spacing),
    /// A delimited sequence of token trees.
    Delimited(DelimitedBounds, DelimitedData),
}

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

    pub fn start_delimited(&mut self) -> OpenDelimited {
        let index = self.length();
        self.tokens.push(ArenaTokenTree::Delimited(
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
        let length = self.length();
        match &mut self.tokens[open.start] {
            ArenaTokenTree::Token(_, _) => unreachable!("Called finish_delimited on a token"),
            ArenaTokenTree::Delimited(bounds, data) => {
                let len = length.saturating_sub(open.start);
                bounds.length = len as u32;
                *data = delimited_data;
            }
        }
    }

    pub fn get_item_at(&self, index: usize) -> Option<&ArenaTokenTree> {
        self.tokens.get(index)
    }

    pub fn length(&self) -> usize {
        self.tokens.len()
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
        fn to_token_stream(arena: &TokenArena, start: usize, length: usize) -> TokenStream {
            let mut tokens = Vec::new();
            let mut index = start;
            let end = start + length;
            while index < end {
                match &arena.tokens[index] {
                    ArenaTokenTree::Token(a, b) => {
                        tokens.push(TokenTree::Token(*a, *b));
                        index += 1;
                    }
                    ArenaTokenTree::Delimited(bounds, data) => {
                        let tokenstream = to_token_stream(
                            arena,
                            (bounds.start + 1) as usize,
                            (bounds.length as usize).saturating_sub(1),
                        );
                        tokens.push(TokenTree::Delimited(
                            data.span,
                            data.spacing,
                            data.delimiter,
                            tokenstream,
                        ));
                        index += bounds.length as usize;
                    }
                }
            }

            TokenStream::new(tokens)
        }
        to_token_stream(&self, 0, self.tokens.len())
    }
}

pub struct OpenDelimited {
    start: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Encodable, Decodable, StableHash)]
pub struct DelimitedBounds {
    pub start: u32,
    pub length: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Encodable, Decodable, StableHash)]
pub struct DelimitedData {
    pub span: DelimSpan,
    pub spacing: DelimSpacing,
    pub delimiter: Delimiter,
}
