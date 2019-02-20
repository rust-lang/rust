#[macro_use]
mod token_set;
mod builder;
mod lexer;
mod event;
mod input;
mod parser_api;
mod grammar;
mod reparsing;

use crate::{
    SyntaxError, SyntaxKind, SmolStr,
    parsing::{
        builder::GreenBuilder,
        input::ParserInput,
        event::EventProcessor,
        parser_api::Parser,
    },
    syntax_node::GreenNode,
};

pub use self::lexer::{tokenize, Token};

pub(crate) use self::reparsing::incremental_reparse;

pub(crate) fn parse_text(text: &str) -> (GreenNode, Vec<SyntaxError>) {
    let tokens = tokenize(&text);
    parse_with(GreenBuilder::new(), text, &tokens, grammar::root)
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

    fn error(&mut self, error: SyntaxError);

    /// Complete tree building. Make sure that
    /// `start_branch` and `finish_branch` calls
    /// are paired!
    fn finish(self) -> Self::Tree;
}

/// `TokenSource` abstracts the source of the tokens parser operates one.
///
/// Hopefully this will allow us to treat text and token trees in the same way!
trait TokenSource {
    fn token_kind(&self, pos: TokenPos) -> SyntaxKind;
    fn is_token_joint_to_next(&self, pos: TokenPos) -> bool;
    fn is_keyword(&self, pos: TokenPos, kw: &str) -> bool;
}

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Default)]
pub(crate) struct TokenPos(pub u32);

impl std::ops::Add<u32> for TokenPos {
    type Output = TokenPos;

    fn add(self, rhs: u32) -> TokenPos {
        TokenPos(self.0 + rhs)
    }
}

impl std::ops::AddAssign<u32> for TokenPos {
    fn add_assign(&mut self, rhs: u32) {
        self.0 += rhs
    }
}
