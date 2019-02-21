use ra_parser::{TreeSink, ParseError};

use crate::{
    SmolStr, SyntaxError, SyntaxErrorKind, TextUnit, TextRange,
    SyntaxKind::{self, *},
    parsing::Token,
    syntax_node::{GreenNode, RaTypes},
};

use rowan::GreenNodeBuilder;

pub(crate) struct TreeBuilder<'a> {
    text: &'a str,
    tokens: &'a [Token],
    text_pos: TextUnit,
    token_pos: usize,
    errors: Vec<SyntaxError>,
    inner: GreenNodeBuilder<RaTypes>,
}

impl<'a> TreeSink for TreeBuilder<'a> {
    fn leaf(&mut self, kind: SyntaxKind, n_tokens: u8) {
        self.eat_trivias();
        let n_tokens = n_tokens as usize;
        let len = self.tokens[self.token_pos..self.token_pos + n_tokens]
            .iter()
            .map(|it| it.len)
            .sum::<TextUnit>();
        self.do_leaf(kind, len, n_tokens);
    }

    fn start_branch(&mut self, kind: SyntaxKind, root: bool) {
        if root {
            self.inner.start_internal(kind);
            return;
        }
        let n_trivias =
            self.tokens[self.token_pos..].iter().take_while(|it| it.kind.is_trivia()).count();
        let leading_trivias = &self.tokens[self.token_pos..self.token_pos + n_trivias];
        let mut trivia_end =
            self.text_pos + leading_trivias.iter().map(|it| it.len).sum::<TextUnit>();

        let n_attached_trivias = {
            let leading_trivias = leading_trivias.iter().rev().map(|it| {
                let next_end = trivia_end - it.len;
                let range = TextRange::from_to(next_end, trivia_end);
                trivia_end = next_end;
                (it.kind, &self.text[range])
            });
            n_attached_trivias(kind, leading_trivias)
        };
        self.eat_n_trivias(n_trivias - n_attached_trivias);
        self.inner.start_internal(kind);
        self.eat_n_trivias(n_attached_trivias);
    }

    fn finish_branch(&mut self, root: bool) {
        if root {
            self.eat_trivias()
        }
        self.inner.finish_internal();
    }

    fn error(&mut self, error: ParseError) {
        let error = SyntaxError::new(SyntaxErrorKind::ParseError(error), self.text_pos);
        self.errors.push(error)
    }
}

impl<'a> TreeBuilder<'a> {
    pub(super) fn new(text: &'a str, tokens: &'a [Token]) -> TreeBuilder<'a> {
        TreeBuilder {
            text,
            tokens,
            text_pos: 0.into(),
            token_pos: 0,
            errors: Vec::new(),
            inner: GreenNodeBuilder::new(),
        }
    }

    pub(super) fn finish(self) -> (GreenNode, Vec<SyntaxError>) {
        (self.inner.finish(), self.errors)
    }

    fn eat_trivias(&mut self) {
        while let Some(&token) = self.tokens.get(self.token_pos) {
            if !token.kind.is_trivia() {
                break;
            }
            self.do_leaf(token.kind, token.len, 1);
        }
    }

    fn eat_n_trivias(&mut self, n: usize) {
        for _ in 0..n {
            let token = self.tokens[self.token_pos];
            assert!(token.kind.is_trivia());
            self.do_leaf(token.kind, token.len, 1);
        }
    }

    fn do_leaf(&mut self, kind: SyntaxKind, len: TextUnit, n_tokens: usize) {
        let range = TextRange::offset_len(self.text_pos, len);
        let text: SmolStr = self.text[range].into();
        self.text_pos += len;
        self.token_pos += n_tokens;
        self.inner.leaf(kind, text);
    }
}

fn n_attached_trivias<'a>(
    kind: SyntaxKind,
    trivias: impl Iterator<Item = (SyntaxKind, &'a str)>,
) -> usize {
    match kind {
        CONST_DEF | TYPE_DEF | STRUCT_DEF | ENUM_DEF | ENUM_VARIANT | FN_DEF | TRAIT_DEF
        | MODULE | NAMED_FIELD_DEF => {
            let mut res = 0;
            for (i, (kind, text)) in trivias.enumerate() {
                match kind {
                    WHITESPACE => {
                        if text.contains("\n\n") {
                            break;
                        }
                    }
                    COMMENT => {
                        res = i + 1;
                    }
                    _ => (),
                }
            }
            res
        }
        _ => 0,
    }
}
