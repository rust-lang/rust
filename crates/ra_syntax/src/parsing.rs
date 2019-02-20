#[macro_use]
mod token_set;
mod builder;
mod lexer;
mod event;
mod input;
mod parser;
mod grammar;
mod reparsing;

use crate::{
    SyntaxKind, SmolStr, SyntaxError,
    parsing::{
        builder::GreenBuilder,
        input::ParserInput,
        event::EventProcessor,
        parser::Parser,
    },
    syntax_node::GreenNode,
};

pub use self::lexer::{tokenize, Token};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParseError(pub String);

pub(crate) use self::reparsing::incremental_reparse;

pub(crate) fn parse_text(text: &str) -> (GreenNode, Vec<SyntaxError>) {
    let tokens = tokenize(&text);
    parse_with(GreenBuilder::default(), text, &tokens, grammar::root)
}

fn parse_with<S: TreeSink>(
    tree_sink: S,
    text: &str,
    tokens: &[Token],
    f: fn(&mut Parser),
) -> S::Tree {
    let mut events = {
        let input = ParserInput::new(text, &tokens);
        let mut p = Parser::new(&input);
        f(&mut p);
        p.finish()
    };
    EventProcessor::new(tree_sink, text, tokens, &mut events).process().finish()
}

/// `TreeSink` abstracts details of a particular syntax tree implementation.
trait TreeSink {
    type Tree;

    /// Adds new leaf to the current branch.
    fn leaf(&mut self, kind: SyntaxKind, text: SmolStr);

    /// Start new branch and make it current.
    fn start_branch(&mut self, kind: SyntaxKind);

    /// Finish current branch and restore previous
    /// branch as current.
    fn finish_branch(&mut self);

    fn error(&mut self, error: ParseError);

    /// Complete tree building. Make sure that
    /// `start_branch` and `finish_branch` calls
    /// are paired!
    fn finish(self) -> Self::Tree;
}

/// `TokenSource` abstracts the source of the tokens parser operates one.
///
/// Hopefully this will allow us to treat text and token trees in the same way!
trait TokenSource {
    fn token_kind(&self, pos: usize) -> SyntaxKind;
    fn is_token_joint_to_next(&self, pos: usize) -> bool;
    fn is_keyword(&self, pos: usize, kw: &str) -> bool;
}
