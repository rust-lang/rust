//! This module provides a way to construct a `File`.
//! It is intended to be completely decoupled from the
//! parser, so as to allow to evolve the tree representation
//! and the parser algorithm independently.
//!
//! The `Sink` trait is the bridge between the parser and the
//! tree builder: the parser produces a stream of events like
//! `start node`, `finish node`, and `FileBuilder` converts
//! this stream to a real tree.
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

pub(super) fn process<'a>(builder: &mut impl Sink<'a>, tokens: &[Token], events: Vec<Event>) {
    let mut idx = 0;

    let mut holes = Vec::new();
    let mut forward_parents = Vec::new();

    for (i, event) in events.iter().enumerate() {
        if holes.last() == Some(&i) {
            holes.pop();
            continue;
        }

        match event {
            &Event::Start {
                kind: TOMBSTONE, ..
            } => (),

            &Event::Start { .. } => {
                forward_parents.clear();
                let mut idx = i;
                loop {
                    let (kind, fwd) = match events[idx] {
                        Event::Start {
                            kind,
                            forward_parent,
                        } => (kind, forward_parent),
                        _ => unreachable!(),
                    };
                    forward_parents.push((idx, kind));
                    if let Some(fwd) = fwd {
                        idx += fwd as usize;
                    } else {
                        break;
                    }
                }
                for &(idx, kind) in forward_parents.iter().into_iter().rev() {
                    builder.start_internal(kind);
                    holes.push(idx);
                }
                holes.pop();
            }
            &Event::Finish => {
                while idx < tokens.len() {
                    let token = tokens[idx];
                    if token.kind.is_trivia() {
                        idx += 1;
                        builder.leaf(token.kind, token.len);
                    } else {
                        break;
                    }
                }
                builder.finish_internal()
            }
            &Event::Token {
                kind,
                mut n_raw_tokens,
            } => {
                // FIXME: currently, we attach whitespace to some random node
                // this should be done in a sensible manner instead
                loop {
                    let token = tokens[idx];
                    if !token.kind.is_trivia() {
                        break;
                    }
                    builder.leaf(token.kind, token.len);
                    idx += 1
                }
                let mut len = 0.into();
                for _ in 0..n_raw_tokens {
                    len += tokens[idx].len;
                    idx += 1;
                }
                builder.leaf(kind, len);
            }
            &Event::Error { ref msg } => builder.error(msg.clone()),
        }
    }
}
