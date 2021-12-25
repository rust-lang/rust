//! See [`TextTreeSink`].

use std::mem;

use parser::LexedStr;

use crate::{
    ast,
    syntax_node::GreenNode,
    SyntaxError,
    SyntaxKind::{self, *},
    SyntaxTreeBuilder, TextRange,
};

pub(crate) fn build_tree(
    lexed: LexedStr<'_>,
    parser_output: parser::Output,
    synthetic_root: bool,
) -> (GreenNode, Vec<SyntaxError>, bool) {
    let mut builder = Builder::new(lexed);

    if synthetic_root {
        builder.enter(SyntaxKind::SOURCE_FILE);
    }

    for event in parser_output.iter() {
        match event {
            parser::Step::Token { kind, n_input_tokens: n_raw_tokens } => {
                builder.token(kind, n_raw_tokens)
            }
            parser::Step::Enter { kind } => builder.enter(kind),
            parser::Step::Exit => builder.exit(),
            parser::Step::Error { msg } => {
                let text_pos = builder.lexed.text_start(builder.pos).try_into().unwrap();
                builder.inner.error(msg.to_string(), text_pos);
            }
        }
    }
    if synthetic_root {
        builder.exit()
    }
    builder.build()
}

struct Builder<'a> {
    lexed: LexedStr<'a>,
    pos: usize,
    state: State,
    inner: SyntaxTreeBuilder,
}

enum State {
    PendingStart,
    Normal,
    PendingFinish,
}

impl<'a> Builder<'a> {
    fn new(lexed: parser::LexedStr<'a>) -> Self {
        Self { lexed, pos: 0, state: State::PendingStart, inner: SyntaxTreeBuilder::default() }
    }

    fn build(mut self) -> (GreenNode, Vec<SyntaxError>, bool) {
        match mem::replace(&mut self.state, State::Normal) {
            State::PendingFinish => {
                self.eat_trivias();
                self.inner.finish_node();
            }
            State::PendingStart | State::Normal => unreachable!(),
        }

        let (node, mut errors) = self.inner.finish_raw();
        for (i, err) in self.lexed.errors() {
            let text_range = self.lexed.text_range(i);
            let text_range = TextRange::new(
                text_range.start.try_into().unwrap(),
                text_range.end.try_into().unwrap(),
            );
            errors.push(SyntaxError::new(err, text_range))
        }

        let is_eof = self.pos == self.lexed.len();

        (node, errors, is_eof)
    }

    fn token(&mut self, kind: SyntaxKind, n_tokens: u8) {
        match mem::replace(&mut self.state, State::Normal) {
            State::PendingStart => unreachable!(),
            State::PendingFinish => self.inner.finish_node(),
            State::Normal => (),
        }
        self.eat_trivias();
        self.do_token(kind, n_tokens as usize);
    }

    fn enter(&mut self, kind: SyntaxKind) {
        match mem::replace(&mut self.state, State::Normal) {
            State::PendingStart => {
                self.inner.start_node(kind);
                // No need to attach trivias to previous node: there is no
                // previous node.
                return;
            }
            State::PendingFinish => self.inner.finish_node(),
            State::Normal => (),
        }

        let n_trivias =
            (self.pos..self.lexed.len()).take_while(|&it| self.lexed.kind(it).is_trivia()).count();
        let leading_trivias = self.pos..self.pos + n_trivias;
        let n_attached_trivias = n_attached_trivias(
            kind,
            leading_trivias.rev().map(|it| (self.lexed.kind(it), self.lexed.text(it))),
        );
        self.eat_n_trivias(n_trivias - n_attached_trivias);
        self.inner.start_node(kind);
        self.eat_n_trivias(n_attached_trivias);
    }

    fn exit(&mut self) {
        match mem::replace(&mut self.state, State::PendingFinish) {
            State::PendingStart => unreachable!(),
            State::PendingFinish => self.inner.finish_node(),
            State::Normal => (),
        }
    }

    fn eat_trivias(&mut self) {
        while self.pos < self.lexed.len() {
            let kind = self.lexed.kind(self.pos);
            if !kind.is_trivia() {
                break;
            }
            self.do_token(kind, 1);
        }
    }

    fn eat_n_trivias(&mut self, n: usize) {
        for _ in 0..n {
            let kind = self.lexed.kind(self.pos);
            assert!(kind.is_trivia());
            self.do_token(kind, 1);
        }
    }

    fn do_token(&mut self, kind: SyntaxKind, n_tokens: usize) {
        let text = &self.lexed.range_text(self.pos..self.pos + n_tokens);
        self.pos += n_tokens;
        self.inner.token(kind, text);
    }
}

fn n_attached_trivias<'a>(
    kind: SyntaxKind,
    trivias: impl Iterator<Item = (SyntaxKind, &'a str)>,
) -> usize {
    match kind {
        CONST | ENUM | FN | IMPL | MACRO_CALL | MACRO_DEF | MACRO_RULES | MODULE | RECORD_FIELD
        | STATIC | STRUCT | TRAIT | TUPLE_FIELD | TYPE_ALIAS | UNION | USE | VARIANT => {
            let mut res = 0;
            let mut trivias = trivias.enumerate().peekable();

            while let Some((i, (kind, text))) = trivias.next() {
                match kind {
                    WHITESPACE if text.contains("\n\n") => {
                        // we check whether the next token is a doc-comment
                        // and skip the whitespace in this case
                        if let Some((COMMENT, peek_text)) = trivias.peek().map(|(_, pair)| pair) {
                            let comment_kind = ast::CommentKind::from_text(peek_text);
                            if comment_kind.doc == Some(ast::CommentPlacement::Outer) {
                                continue;
                            }
                        }
                        break;
                    }
                    COMMENT => {
                        let comment_kind = ast::CommentKind::from_text(text);
                        if comment_kind.doc == Some(ast::CommentPlacement::Inner) {
                            break;
                        }
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
