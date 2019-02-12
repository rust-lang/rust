//! This module provides a way to construct a `File`.
//! It is intended to be completely decoupled from the
//! parser, so as to allow to evolve the tree representation
//! and the parser algorithm independently.
//!
//! The `Sink` trait is the bridge between the parser and the
//! tree builder: the parser produces a stream of events like
//! `start node`, `finish node`, and `FileBuilder` converts
//! this stream to a real tree.
use crate::{
    lexer::Token,
    parser_impl::Sink,
    SmolStr,
    SyntaxKind::{self, *},
    TextRange, TextUnit,
    syntax_node::syntax_error::{
        ParseError,
        SyntaxError,
        SyntaxErrorKind,
    },
};
use std::mem;

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
    /// saves the position of current event's parent.
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
        msg: ParseError,
    },
}

impl Event {
    pub(crate) fn tombstone() -> Self {
        Event::Start { kind: TOMBSTONE, forward_parent: None }
    }
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
    pub(super) fn new(
        sink: S,
        text: &'a str,
        tokens: &'a [Token],
        events: &'a mut [Event],
    ) -> EventProcessor<'a, S> {
        EventProcessor { sink, text_pos: 0.into(), text, token_pos: 0, tokens, events }
    }

    /// Generate the syntax tree with the control of events.
    pub(super) fn process(mut self) -> S {
        let mut forward_parents = Vec::new();

        for i in 0..self.events.len() {
            match mem::replace(&mut self.events[i], Event::tombstone()) {
                Event::Start { kind: TOMBSTONE, .. } => (),

                Event::Start { kind, forward_parent } => {
                    // For events[A, B, C], B is A's forward_parent, C is B's forward_parent,
                    // in the normal control flow, the parent-child relation: `A -> B -> C`,
                    // while with the magic forward_parent, it writes: `C <- B <- A`.

                    // append `A` into parents.
                    forward_parents.push(kind);
                    let mut idx = i;
                    let mut fp = forward_parent;
                    while let Some(fwd) = fp {
                        idx += fwd as usize;
                        // append `A`'s forward_parent `B`
                        fp = match mem::replace(&mut self.events[idx], Event::tombstone()) {
                            Event::Start { kind, forward_parent } => {
                                forward_parents.push(kind);
                                forward_parent
                            }
                            _ => unreachable!(),
                        };
                        // append `B`'s forward_parent `C` in the next stage.
                    }

                    for kind in forward_parents.drain(..).rev() {
                        self.start(kind);
                    }
                }
                Event::Finish => {
                    let is_last = i == self.events.len() - 1;
                    self.finish(is_last);
                }
                Event::Token { kind, n_raw_tokens } => {
                    self.eat_trivias();
                    let n_raw_tokens = n_raw_tokens as usize;
                    let len = self.tokens[self.token_pos..self.token_pos + n_raw_tokens]
                        .iter()
                        .map(|it| it.len)
                        .sum::<TextUnit>();
                    self.leaf(kind, len, n_raw_tokens);
                }
                Event::Error { msg } => self
                    .sink
                    .error(SyntaxError::new(SyntaxErrorKind::ParseError(msg), self.text_pos)),
            }
        }
        self.sink
    }

    /// Add the node into syntax tree but discard the comments/whitespaces.
    fn start(&mut self, kind: SyntaxKind) {
        if kind == SOURCE_FILE {
            self.sink.start_branch(kind);
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
        self.sink.start_branch(kind);
        self.eat_n_trivias(n_attached_trivias);
    }

    fn finish(&mut self, is_last: bool) {
        if is_last {
            self.eat_trivias()
        }
        self.sink.finish_branch();
    }

    fn eat_trivias(&mut self) {
        while let Some(&token) = self.tokens.get(self.token_pos) {
            if !token.kind.is_trivia() {
                break;
            }
            self.leaf(token.kind, token.len, 1);
        }
    }

    fn eat_n_trivias(&mut self, n: usize) {
        for _ in 0..n {
            let token = self.tokens[self.token_pos];
            assert!(token.kind.is_trivia());
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
