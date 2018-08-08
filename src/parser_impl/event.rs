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


pub(super) fn process<'a, S: Sink<'a>>(builder: &mut S, tokens: &[Token], mut events: Vec<Event>) {
    fn tombstone() -> Event {
        Event::Start { kind: TOMBSTONE, forward_parent: None }
    }
    let eat_ws = |idx: &mut usize, builder: &mut S| {
        while let Some(token) = tokens.get(*idx) {
            if !token.kind.is_trivia() {
                break;
            }
            builder.leaf(token.kind, token.len);
            *idx += 1
        }
    };

    let events: &mut [Event] = &mut events;
    let mut depth = 0;
    let mut forward_parents = Vec::new();
    let mut next_tok_idx = 0;
    for i in 0..events.len() {
        match mem::replace(&mut events[i], tombstone()) {
            Event::Start {
                kind: TOMBSTONE, ..
            } => (),

            Event::Start { kind, forward_parent } => {
                forward_parents.push(kind);
                let mut idx = i;
                let mut fp = forward_parent;
                while let Some(fwd) = fp {
                    idx += fwd as usize;
                    fp = match mem::replace(&mut events[idx], tombstone()) {
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
                        eat_ws(&mut next_tok_idx, builder);
                    }
                    depth += 1;
                    builder.start_internal(kind);
                }
            }
            Event::Finish => {
                depth -= 1;
                if depth == 0 {
                    eat_ws(&mut next_tok_idx, builder);
                }

                builder.finish_internal();
            }
            Event::Token {
                kind,
                mut n_raw_tokens,
            } => {
                eat_ws(&mut next_tok_idx, builder);
                let mut len = 0.into();
                for _ in 0..n_raw_tokens {
                    len += tokens[next_tok_idx].len;
                    next_tok_idx += 1;
                }
                builder.leaf(kind, len);
            }
            Event::Error { msg } => builder.error(msg),
        }
    }
}
