use {Token, SyntaxKind};

#[macro_use]
mod parser;
mod grammar;

#[derive(Debug)]
pub(crate) enum Event {
    Start {
        kind: SyntaxKind,
        forward_parent: Option<u32>,
    },
    Finish,
    Token {
        kind: SyntaxKind,
        n_raw_tokens: u8,
    },
    Error {
        message: String,
    },
}

pub(crate) fn parse<'t>(text: &'t str, raw_tokens: &'t [Token]) -> Vec<Event> {
    let mut parser = parser::Parser::new(text, raw_tokens);
    grammar::file(&mut parser);
    parser.into_events()
}
