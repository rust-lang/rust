use crate::{
    SyntaxKind, SyntaxKind::EOF, TextRange, TextUnit,
    parsing::{
        parser_impl::TokenSource,
        lexer::Token,
    },
};

use std::ops::{Add, AddAssign};

impl<'t> TokenSource for ParserInput<'t> {
    fn token_kind(&self, pos: InputPosition) -> SyntaxKind {
        let idx = pos.0 as usize;
        if !(idx < self.tokens.len()) {
            return EOF;
        }
        self.tokens[idx].kind
    }
    fn is_token_joint_to_next(&self, pos: InputPosition) -> bool {
        let idx_curr = pos.0 as usize;
        let idx_next = pos.0 as usize;
        if !(idx_next < self.tokens.len()) {
            return true;
        }
        self.start_offsets[idx_curr] + self.tokens[idx_curr].len == self.start_offsets[idx_next]
    }
    fn is_keyword(&self, pos: InputPosition, kw: &str) -> bool {
        let idx = pos.0 as usize;
        if !(idx < self.tokens.len()) {
            return false;
        }
        let range = TextRange::offset_len(self.start_offsets[idx], self.tokens[idx].len);

        self.text[range] == *kw
    }
}

pub(crate) struct ParserInput<'t> {
    text: &'t str,
    /// start position of each token(expect whitespace and comment)
    /// ```non-rust
    ///  struct Foo;
    /// ^------^---
    /// |      |  ^-
    /// 0      7  10
    /// ```
    /// (token, start_offset): `[(struct, 0), (Foo, 7), (;, 10)]`
    start_offsets: Vec<TextUnit>,
    /// non-whitespace/comment tokens
    /// ```non-rust
    /// struct Foo {}
    /// ^^^^^^ ^^^ ^^
    /// ```
    /// tokens: `[struct, Foo, {, }]`
    tokens: Vec<Token>,
}

impl<'t> ParserInput<'t> {
    /// Generate input from tokens(expect comment and whitespace).
    pub fn new(text: &'t str, raw_tokens: &'t [Token]) -> ParserInput<'t> {
        let mut tokens = Vec::new();
        let mut start_offsets = Vec::new();
        let mut len = 0.into();
        for &token in raw_tokens.iter() {
            if !token.kind.is_trivia() {
                tokens.push(token);
                start_offsets.push(len);
            }
            len += token.len;
        }

        ParserInput { text, start_offsets, tokens }
    }
}

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub(crate) struct InputPosition(u32);

impl InputPosition {
    pub fn new() -> Self {
        InputPosition(0)
    }
}

impl Add<u32> for InputPosition {
    type Output = InputPosition;

    fn add(self, rhs: u32) -> InputPosition {
        InputPosition(self.0 + rhs)
    }
}

impl AddAssign<u32> for InputPosition {
    fn add_assign(&mut self, rhs: u32) {
        self.0 += rhs
    }
}
