//! See `TextTokenSource` docs.

use parser::TokenSource;

use crate::{parsing::lexer::Token, SyntaxKind::EOF, TextRange, TextSize};

/// Implementation of `parser::TokenSource` that takes tokens from source code text.
pub(crate) struct TextTokenSource<'t> {
    text: &'t str,
    /// token and its start position (non-whitespace/comment tokens)
    /// ```non-rust
    ///  struct Foo;
    ///  ^------^--^-
    ///  |      |    \________
    ///  |      \____         \
    ///  |           \         |
    ///  (struct, 0) (Foo, 7) (;, 10)
    /// ```
    /// `[(struct, 0), (Foo, 7), (;, 10)]`
    token_offset_pairs: Vec<(Token, TextSize)>,

    /// Current token and position
    curr: (parser::Token, usize),
}

impl<'t> TokenSource for TextTokenSource<'t> {
    fn current(&self) -> parser::Token {
        self.curr.0
    }

    fn lookahead_nth(&self, n: usize) -> parser::Token {
        mk_token(self.curr.1 + n, &self.token_offset_pairs)
    }

    fn bump(&mut self) {
        if self.curr.0.kind == EOF {
            return;
        }

        let pos = self.curr.1 + 1;
        self.curr = (mk_token(pos, &self.token_offset_pairs), pos);
    }

    fn is_keyword(&self, kw: &str) -> bool {
        self.token_offset_pairs
            .get(self.curr.1)
            .map(|(token, offset)| &self.text[TextRange::at(*offset, token.len)] == kw)
            .unwrap_or(false)
    }
}

fn mk_token(pos: usize, token_offset_pairs: &[(Token, TextSize)]) -> parser::Token {
    let (kind, is_jointed_to_next) = match token_offset_pairs.get(pos) {
        Some((token, offset)) => (
            token.kind,
            token_offset_pairs
                .get(pos + 1)
                .map(|(_, next_offset)| offset + token.len == *next_offset)
                .unwrap_or(false),
        ),
        None => (EOF, false),
    };
    parser::Token { kind, is_jointed_to_next }
}

impl<'t> TextTokenSource<'t> {
    /// Generate input from tokens(expect comment and whitespace).
    pub fn new(text: &'t str, raw_tokens: &'t [Token]) -> TextTokenSource<'t> {
        let token_offset_pairs: Vec<_> = raw_tokens
            .iter()
            .filter_map({
                let mut len = 0.into();
                move |token| {
                    let pair = if token.kind.is_trivia() { None } else { Some((*token, len)) };
                    len += token.len;
                    pair
                }
            })
            .collect();

        let first = mk_token(0, &token_offset_pairs);
        TextTokenSource { text, token_offset_pairs, curr: (first, 0) }
    }
}
