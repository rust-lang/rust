//! FIXME: write short doc here

use ra_parser::Token as PToken;
use ra_parser::TokenSource;

use crate::{parsing::lexer::Token, SyntaxKind::EOF, TextRange, TextUnit};

pub(crate) struct TextTokenSource<'t> {
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

    /// Current token and position
    curr: (PToken, usize),
}

impl<'t> TokenSource for TextTokenSource<'t> {
    fn current(&self) -> PToken {
        self.curr.0
    }

    fn lookahead_nth(&self, n: usize) -> PToken {
        mk_token(self.curr.1 + n, &self.start_offsets, &self.tokens)
    }

    fn bump(&mut self) {
        if self.curr.0.kind == EOF {
            return;
        }

        let pos = self.curr.1 + 1;
        self.curr = (mk_token(pos, &self.start_offsets, &self.tokens), pos);
    }

    fn is_keyword(&self, kw: &str) -> bool {
        let pos = self.curr.1;
        if !(pos < self.tokens.len()) {
            return false;
        }
        let range = TextRange::offset_len(self.start_offsets[pos], self.tokens[pos].len);
        self.text[range] == *kw
    }
}

fn mk_token(pos: usize, start_offsets: &[TextUnit], tokens: &[Token]) -> PToken {
    let kind = tokens.get(pos).map(|t| t.kind).unwrap_or(EOF);
    let is_jointed_to_next = if pos + 1 < start_offsets.len() {
        start_offsets[pos] + tokens[pos].len == start_offsets[pos + 1]
    } else {
        false
    };

    PToken { kind, is_jointed_to_next }
}

impl<'t> TextTokenSource<'t> {
    /// Generate input from tokens(expect comment and whitespace).
    pub fn new(text: &'t str, raw_tokens: &'t [Token]) -> TextTokenSource<'t> {
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

        let first = mk_token(0, &start_offsets, &tokens);
        TextTokenSource { text, start_offsets, tokens, curr: (first, 0) }
    }
}
