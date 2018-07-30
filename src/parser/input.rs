use {lexer::Token, SyntaxKind, SyntaxKind::EOF, TextRange, TextUnit};

use std::ops::{Add, AddAssign};

pub(crate) struct ParserInput<'t> {
    text: &'t str,
    start_offsets: Vec<TextUnit>,
    tokens: Vec<Token>, // non-whitespace tokens
}

impl<'t> ParserInput<'t> {
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
        let range = TextRange::offset_len(self.start_offsets[idx], self.tokens[idx].len);
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
