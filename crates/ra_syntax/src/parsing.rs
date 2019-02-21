//! Lexing, bridging to ra_parser (which does the actual parsing) and
//! incremental reparsing.

mod lexer;
mod input;
mod builder;
mod reparsing;

use crate::{
    SyntaxError,
    syntax_node::GreenNode,
    parsing::{
        builder::TreeBuilder,
        input::ParserInput,
    },
};

pub use self::lexer::{tokenize, Token};

pub(crate) use self::reparsing::incremental_reparse;

pub(crate) fn parse_text(text: &str) -> (GreenNode, Vec<SyntaxError>) {
    let tokens = tokenize(&text);
    let token_source = ParserInput::new(text, &tokens);
    let mut tree_sink = TreeBuilder::new(text, &tokens);
    ra_parser::parse(&token_source, &mut tree_sink);
    tree_sink.finish()
}
