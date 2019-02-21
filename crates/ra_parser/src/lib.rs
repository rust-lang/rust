//! The Rust parser.
//!
//! The parser doesn't know about concrete representation of tokens and syntax
//! trees. Abstract `TokenSource` and `TreeSink` traits are used instead. As a
//! consequence, this crates does not contain a lexer.
//!
//! The `Parser` struct from the `parser` module is a cursor into the sequence
//! of tokens. Parsing routines use `Parser` to inspect current state and
//! advance the parsing.
//!
//! The actual parsing happens in the `grammar` module.
//!
//! Tests for this crate live in `ra_syntax` crate.

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

/// `TokenSource` abstracts the source of the tokens parser operates one.
///
/// Hopefully this will allow us to treat text and token trees in the same way!
pub trait TokenSource {
    /// What is the current token?
    fn token_kind(&self, pos: usize) -> SyntaxKind;
    /// Is the current token joined to the next one (`> >` vs `>>`).
    fn is_token_joint_to_next(&self, pos: usize) -> bool;
    /// Is the current token a specified keyword?
    fn is_keyword(&self, pos: usize, kw: &str) -> bool;
}

/// `TreeSink` abstracts details of a particular syntax tree implementation.
pub trait TreeSink {
    /// Adds new leaf to the current branch.
    fn leaf(&mut self, kind: SyntaxKind, n_tokens: u8);

    /// Start new branch and make it current.
    fn start_branch(&mut self, kind: SyntaxKind);

    /// Finish current branch and restore previous
    /// branch as current.
    fn finish_branch(&mut self);

    fn error(&mut self, error: ParseError);
}

/// Parse given tokens into the given sink as a rust file.
pub fn parse(token_source: &dyn TokenSource, tree_sink: &mut dyn TreeSink) {
    let mut p = parser::Parser::new(token_source);
    grammar::root(&mut p);
    let events = p.finish();
    event::process(tree_sink, events);
}

/// A parsing function for a specific braced-block.
pub struct Reparser(fn(&mut parser::Parser));

impl Reparser {
    /// If the node is a braced block, return the corresponding `Reparser`.
    pub fn for_node(
        node: SyntaxKind,
        first_child: Option<SyntaxKind>,
        parent: Option<SyntaxKind>,
    ) -> Option<Reparser> {
        grammar::reparser(node, first_child, parent).map(Reparser)
    }

    /// Re-parse given tokens using this `Reparser`.
    ///
    /// Tokens must start with `{`, end with `}` and form a valid brace
    /// sequence.
    pub fn parse(self, token_source: &dyn TokenSource, tree_sink: &mut dyn TreeSink) {
        let Reparser(r) = self;
        let mut p = parser::Parser::new(token_source);
        r(&mut p);
        let events = p.finish();
        event::process(tree_sink, events);
    }
}
