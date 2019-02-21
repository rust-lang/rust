#[macro_use]
mod token_set;
mod syntax_kind;
mod event;
mod parser;
mod grammar;

pub(crate) use token_set::TokenSet;

pub use syntax_kind::SyntaxKind;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParseError(pub String);

/// `TreeSink` abstracts details of a particular syntax tree implementation.
pub trait TreeSink {
    /// Adds new leaf to the current branch.
    fn leaf(&mut self, kind: SyntaxKind, n_tokens: u8);

    /// Start new branch and make it current.
    fn start_branch(&mut self, kind: SyntaxKind, root: bool);

    /// Finish current branch and restore previous
    /// branch as current.
    fn finish_branch(&mut self, root: bool);

    fn error(&mut self, error: ParseError);
}

/// `TokenSource` abstracts the source of the tokens parser operates one.
///
/// Hopefully this will allow us to treat text and token trees in the same way!
pub trait TokenSource {
    fn token_kind(&self, pos: usize) -> SyntaxKind;
    fn is_token_joint_to_next(&self, pos: usize) -> bool;
    fn is_keyword(&self, pos: usize, kw: &str) -> bool;
}

pub fn parse(token_source: &dyn TokenSource, tree_sink: &mut dyn TreeSink) {
    let mut p = parser::Parser::new(token_source);
    grammar::root(&mut p);
    let events = p.finish();
    event::process(tree_sink, events);
}

pub struct Reparser(fn(&mut parser::Parser));

impl Reparser {
    pub fn for_node(
        node: SyntaxKind,
        first_child: Option<SyntaxKind>,
        parent: Option<SyntaxKind>,
    ) -> Option<Reparser> {
        grammar::reparser(node, first_child, parent).map(Reparser)
    }
}

pub fn reparse(token_source: &dyn TokenSource, tree_sink: &mut dyn TreeSink, reparser: Reparser) {
    let Reparser(r) = reparser;
    let mut p = parser::Parser::new(token_source);
    r(&mut p);
    let events = p.finish();
    event::process(tree_sink, events);
}
