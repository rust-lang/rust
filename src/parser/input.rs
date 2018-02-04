use {SyntaxKind, TextRange, TextUnit, Token};
use syntax_kinds::EOF;
use super::is_insignificant;

use std::ops::{Add, AddAssign};

pub(crate) struct ParserInput<'t> {
    #[allow(unused)]
    text: &'t str,
    #[allow(unused)]
    start_offsets: Vec<TextUnit>,
    tokens: Vec<Token>, // non-whitespace tokens
}

impl<'t> ParserInput<'t> {
    pub fn new(text: &'t str, raw_tokens: &'t [Token]) -> ParserInput<'t> {
        let mut tokens = Vec::new();
        let mut start_offsets = Vec::new();
        let mut len = TextUnit::new(0);
        for &token in raw_tokens.iter() {
            if !is_insignificant(token.kind) {
                tokens.push(token);
                start_offsets.push(len);
            }
            len += token.len;
        }

        ParserInput {
            text,
            start_offsets,
            tokens,
        }
    }

    pub fn kind(&self, pos: InputPosition) -> SyntaxKind {
        let idx = pos.0 as usize;
        if !(idx < self.tokens.len()) {
            return EOF;
        }
        self.tokens[idx].kind
    }

    #[allow(unused)]
    pub fn text(&self, pos: InputPosition) -> &'t str {
        let idx = pos.0 as usize;
        if !(idx < self.tokens.len()) {
            return "";
        }
        let range = TextRange::from_len(self.start_offsets[idx], self.tokens[idx].len);
        &self.text[range]
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
