use {File, SyntaxKind, Token};

use syntax_kinds::*;

#[macro_use]
mod parser;
mod event;
mod grammar;
use self::event::Event;

/// Parse a sequence of tokens into the representative node tree
pub fn parse(text: String, tokens: &[Token]) -> File {
    let events = parse_into_events(&text, tokens);
    event::to_file(text, tokens, events)
}

pub(crate) fn parse_into_events<'t>(text: &'t str, raw_tokens: &'t [Token]) -> Vec<Event> {
    let mut parser = parser::Parser::new(text, raw_tokens);
    grammar::file(&mut parser);
    parser.into_events()
}

fn is_insignificant(kind: SyntaxKind) -> bool {
    match kind {
        WHITESPACE | COMMENT => true,
        _ => false,
    }
}
