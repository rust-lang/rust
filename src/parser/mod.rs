use {File, SyntaxKind, Token};

use syntax_kinds::*;

#[macro_use]
mod parser;
mod input;
mod event;
mod grammar;
use self::event::Event;

/// Parse a sequence of tokens into the representative node tree
pub fn parse(text: String, tokens: &[Token]) -> File {
    let events = {
        let input = input::ParserInput::new(&text, tokens);
        let mut parser = parser::Parser::new(&input);
        grammar::file(&mut parser);
        parser.into_events()
    };
    event::to_file(text, tokens, events)
}

fn is_insignificant(kind: SyntaxKind) -> bool {
    match kind {
        WHITESPACE | COMMENT => true,
        _ => false,
    }
}
