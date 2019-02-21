use ra_parser::TokenSource;

use crate::{
    SyntaxKind, SyntaxKind::EOF, TextRange, TextUnit,
    parsing::lexer::Token,
};

impl<'t> TokenSource for ParserInput<'t> {
    fn token_kind(&self, pos: usize) -> SyntaxKind {
        if !(pos < self.tokens.len()) {
            return EOF;
        }
        self.tokens[pos].kind
    }
    fn is_token_joint_to_next(&self, pos: usize) -> bool {
        if !(pos + 1 < self.tokens.len()) {
            return true;
        }
        self.start_offsets[pos] + self.tokens[pos].len == self.start_offsets[pos + 1]
    }
    fn is_keyword(&self, pos: usize, kw: &str) -> bool {
        if !(pos < self.tokens.len()) {
            return false;
        }
        let range = TextRange::offset_len(self.start_offsets[pos], self.tokens[pos].len);

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
