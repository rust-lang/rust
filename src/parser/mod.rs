#[macro_use]
mod token_set;
mod event;
mod grammar;
mod input;
mod parser;

use {lexer::Token, parser::event::process};

pub(crate) use self::event::Sink;

/// Parse a sequence of tokens into the representative node tree
pub(crate) fn parse<S: Sink>(text: String, tokens: &[Token]) -> S::Tree {
    let events = {
        let input = input::ParserInput::new(&text, tokens);
        let parser_impl = parser::imp::ParserImpl::new(&input);
        let mut parser = parser::Parser(parser_impl);
        grammar::file(&mut parser);
        parser.0.into_events()
    };
    let mut sink = S::new(text);
    process(&mut sink, tokens, events);
    sink.finish()
}
