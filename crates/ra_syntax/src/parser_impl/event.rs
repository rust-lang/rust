//! This module provides a way to construct a `File`.
//! It is intended to be completely decoupled from the
//! parser, so as to allow to evolve the tree representation
//! and the parser algorithm independently.
//!
//! The `Sink` trait is the bridge between the parser and the
//! tree builder: the parser produces a stream of events like
//! `start node`, `finish node`, and `FileBuilder` converts
//! this stream to a real tree.
use std::mem;
use {
    TextUnit, TextRange, SmolStr,
    lexer::Token,
    parser_impl::Sink,
    SyntaxKind::{self, TOMBSTONE},
};


/// `Parser` produces a flat list of `Event`s.
/// They are converted to a tree-structure in
/// a separate pass, via `TreeBuilder`.
#[derive(Debug)]
pub(crate) enum Event {
    /// This event signifies the start of the node.
    /// It should be either abandoned (in which case the
    /// `kind` is `TOMBSTONE`, and the event is ignored),
    /// or completed via a `Finish` event.
    ///
    /// All tokens between a `Start` and a `Finish` would
    /// become the children of the respective node.
    ///
    /// For left-recursive syntactic constructs, the parser produces
    /// a child node before it sees a parent. `forward_parent`
    /// exists to allow to tweak parent-child relationships.
    ///
    /// Consider this path
    ///
    /// foo::bar
    ///
    /// The events for it would look like this:
    ///
    ///
    /// START(PATH) IDENT('foo') FINISH START(PATH) COLONCOLON IDENT('bar') FINISH
    ///       |                          /\
    ///       |                          |
    ///       +------forward-parent------+
    ///
    /// And the tree would look like this
    ///
    ///    +--PATH---------+
    ///    |   |           |
    ///    |   |           |
    ///    |  '::'       'bar'
    ///    |
    ///   PATH
    ///    |
    ///   'foo'
    ///
    /// See also `CompletedMarker::precede`.
    Start {
        kind: SyntaxKind,
        forward_parent: Option<u32>,
    },

    /// Complete the previous `Start` event
    Finish,

    /// Produce a single leaf-element.
    /// `n_raw_tokens` is used to glue complex contextual tokens.
    /// For example, lexer tokenizes `>>` as `>`, `>`, and
    /// `n_raw_tokens = 2` is used to produced a single `>>`.
    Token {
        kind: SyntaxKind,
        n_raw_tokens: u8,
    },

    Error {
        msg: String,
    },
}

pub(super) struct EventProcessor<'a, S: Sink> {
    sink: S,
    text_pos: TextUnit,
    text: &'a str,
    token_pos: usize,
    tokens: &'a [Token],
    events: &'a mut [Event],
}

impl<'a, S: Sink> EventProcessor<'a, S> {
    pub(super) fn new(sink: S, text: &'a str, tokens: &'a[Token], events: &'a mut [Event]) -> EventProcessor<'a, S> {
        EventProcessor {
            sink,
            text_pos: 0.into(),
            text,
            token_pos: 0,
            tokens,
            events
        }
    }

    pub(super) fn process(mut self) -> S {
        fn tombstone() -> Event {
            Event::Start { kind: TOMBSTONE, forward_parent: None }
        }
        let mut depth = 0;
        let mut forward_parents = Vec::new();

        for i in 0..self.events.len() {
            match mem::replace(&mut self.events[i], tombstone()) {
                Event::Start {
                    kind: TOMBSTONE, ..
                } => (),

                Event::Start { kind, forward_parent } => {
                    forward_parents.push(kind);
                    let mut idx = i;
                    let mut fp = forward_parent;
                    while let Some(fwd) = fp {
                        idx += fwd as usize;
                        fp = match mem::replace(&mut self.events[idx], tombstone()) {
                            Event::Start {
                                kind,
                                forward_parent,
                            } => {
                                forward_parents.push(kind);
                                forward_parent
                            },
                            _ => unreachable!(),
                        };
                    }
                    for kind in forward_parents.drain(..).rev() {
                        if depth > 0 {
                            self.eat_ws();
                        }
                        depth += 1;
                        self.sink.start_internal(kind);
                    }
                }
                Event::Finish => {
                    depth -= 1;
                    if depth == 0 {
                        self.eat_ws();
                    }

                    self.sink.finish_internal();
                }
                Event::Token {
                    kind,
                    n_raw_tokens,
                } => {
                    self.eat_ws();
                    let n_raw_tokens = n_raw_tokens as usize;
                    let len = self.tokens[self.token_pos..self.token_pos + n_raw_tokens]
                        .iter()
                        .map(|it| it.len)
                        .sum::<TextUnit>();
                    self.leaf(kind, len, n_raw_tokens);
                }
                Event::Error { msg } => self.sink.error(msg, self.text_pos),
            }
        }
        self.sink
    }

    fn eat_ws(&mut self) {
        while let Some(&token) = self.tokens.get(self.token_pos) {
            if !token.kind.is_trivia() {
                break;
            }
            self.leaf(token.kind, token.len, 1);
        }
    }

    fn leaf(&mut self, kind: SyntaxKind, len: TextUnit, n_tokens: usize) {
        let range = TextRange::offset_len(self.text_pos, len);
        let text: SmolStr = self.text[range].into();
        self.text_pos += len;
        self.token_pos += n_tokens;
        self.sink.leaf(kind, text);
    }
}
