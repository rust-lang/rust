mod builder;
mod lexer;
mod input;
mod reparsing;

use ra_parser::{parse, ParseError};

use crate::{
    SyntaxKind, SyntaxError,
    parsing::{
        builder::TreeBuilder,
        input::ParserInput,
    },
    syntax_node::GreenNode,
};

pub use self::lexer::{tokenize, Token};

pub(crate) use self::reparsing::incremental_reparse;

pub(crate) fn parse_text(text: &str) -> (GreenNode, Vec<SyntaxError>) {
    let tokens = tokenize(&text);
    let token_source = ParserInput::new(text, &tokens);
    let mut tree_sink = TreeBuilder::new(text, &tokens);
    parse(&token_source, &mut tree_sink);
    tree_sink.finish()
}

/// `TreeSink` abstracts details of a particular syntax tree implementation.
trait TreeSink {
    type Tree;

    /// Adds new leaf to the current branch.
    fn leaf(&mut self, kind: SyntaxKind, n_tokens: u8);

    /// Start new branch and make it current.
    fn start_branch(&mut self, kind: SyntaxKind, root: bool);

    /// Finish current branch and restore previous
    /// branch as current.
    fn finish_branch(&mut self, root: bool);

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
