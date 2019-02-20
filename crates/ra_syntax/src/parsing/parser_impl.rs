pub(super) mod event;
pub(super) mod input;

use crate::parsing::{
    TreeSink, TokenSource,
    lexer::Token,
    parser_api::Parser,
    parser_impl::event::EventProcessor,
};

/// Parse a sequence of tokens into the representative node tree
pub(super) fn parse_with<S: TreeSink>(
    sink: S,
    text: &str,
    tokens: &[Token],
    parser: fn(&mut Parser),
) -> S::Tree {
    let mut events = {
        let input = input::ParserInput::new(text, tokens);
        let mut parser_api = Parser::new(&input);
        parser(&mut parser_api);
        parser_api.finish()
    };
    EventProcessor::new(sink, text, tokens, &mut events).process().finish()
}
