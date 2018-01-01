use {Token, File, FileBuilder, Sink, SyntaxKind};

use syntax_kinds::*;

mod event_parser;
use self::event_parser::Event;


pub fn parse(text: String, tokens: &[Token]) -> File {
    let events = event_parser::parse(&text, tokens);
    from_events_to_file(text, tokens, events)
}

fn from_events_to_file(
    text: String,
    tokens: &[Token],
    events: Vec<Event>,
) -> File {
    let mut builder = FileBuilder::new(text);
    let mut idx = 0;
    for event in events {
        match event {
            Event::Start { kind } => builder.start_internal(kind),
            Event::Finish => {
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
            },
            Event::Token { kind, mut n_raw_tokens } => loop {
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