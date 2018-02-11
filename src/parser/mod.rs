use {File, SyntaxKind, Token};

use syntax_kinds::*;

#[macro_use]
mod token_set;
mod parser;
mod input;
mod event;
mod grammar;

/// Parse a sequence of tokens into the representative node tree
pub fn parse(text: String, tokens: &[Token]) -> File {
    let events = {
        let input = input::ParserInput::new(&text, tokens);
        let parser_impl = parser::imp::ParserImpl::new(&input);
        let mut parser = parser::Parser(parser_impl);
        grammar::file(&mut parser);
        parser.0.into_events()
    };
    event::to_file(text, tokens, events)
}

fn is_insignificant(kind: SyntaxKind) -> bool {
    match kind {
        WHITESPACE | COMMENT => true,
        _ => false,
    }
}
