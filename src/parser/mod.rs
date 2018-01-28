use {File, FileBuilder, Sink, SyntaxKind, Token};

use syntax_kinds::*;

mod event_parser;
use self::event_parser::Event;

/// Parse a sequence of tokens into the representative node tree
pub fn parse(text: String, tokens: &[Token]) -> File {
    let events = event_parser::parse(&text, tokens);
    from_events_to_file(text, tokens, events)
}

fn from_events_to_file(text: String, tokens: &[Token], events: Vec<Event>) -> File {
    let mut builder = FileBuilder::new(text);
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
                    if is_insignificant(token.kind) {
                        idx += 1;
                        builder.leaf(token.kind, token.len);
                    } else {
                        break;
                    }
                }
                builder.finish_internal()
            }
            &Event::Token {
                kind: _,
                mut n_raw_tokens,
            } => loop {
                let token = tokens[idx];
                if !is_insignificant(token.kind) {
                    n_raw_tokens -= 1;
                }
                idx += 1;
                builder.leaf(token.kind, token.len);
                if n_raw_tokens == 0 {
                    break;
                }
            },
            &Event::Error { ref message } => builder.error().message(message.clone()).emit(),
        }
    }
    builder.finish()
}

fn is_insignificant(kind: SyntaxKind) -> bool {
    match kind {
        WHITESPACE | COMMENT => true,
        _ => false,
    }
}
