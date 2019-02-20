#[macro_use]
mod token_set;
mod builder;
mod lexer;
mod parser_impl;
mod parser_api;
mod grammar;
mod reparsing;

use crate::{
    SyntaxError,
    parsing::builder::GreenBuilder,
    syntax_node::GreenNode,
};

pub use self::lexer::{tokenize, Token};

pub(crate) use self::reparsing::incremental_reparse;

pub(crate) fn parse_text(text: &str) -> (GreenNode, Vec<SyntaxError>) {
    let tokens = tokenize(&text);
    let (green, errors) =
        parser_impl::parse_with(GreenBuilder::new(), text, &tokens, grammar::root);
    (green, errors)
}
