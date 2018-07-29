#[macro_use]
mod token_set;
mod parser;
mod input;
mod event;
mod grammar;

use std::sync::Arc;
use {
    Token,
    yellow::SyntaxNode,
    syntax_kinds::*
};
use GreenBuilder;
use parser::event::process;


/// Parse a sequence of tokens into the representative node tree
pub fn parse_green(text: String, tokens: &[Token]) -> SyntaxNode {
    let events = {
        let input = input::ParserInput::new(&text, tokens);
        let parser_impl = parser::imp::ParserImpl::new(&input);
        let mut parser = parser::Parser(parser_impl);
        grammar::file(&mut parser);
        parser.0.into_events()
    };
    let mut builder = GreenBuilder::new(text);
    process(&mut builder, tokens, events);
    let (green, errors) = builder.finish();
    SyntaxNode::new(Arc::new(green), errors)
}

fn is_insignificant(kind: SyntaxKind) -> bool {
    match kind {
        WHITESPACE | COMMENT => true,
        _ => false,
    }
}
