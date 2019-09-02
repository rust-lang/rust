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
#[macro_use]
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
    fn current(&self) -> Token;

    /// Lookahead n token
    fn lookahead_nth(&self, n: usize) -> Token;

    /// bump cursor to next token
    fn bump(&mut self);

    /// Is the current token a specified keyword?
    fn is_keyword(&self, kw: &str) -> bool;
}

/// `TokenCursor` abstracts the cursor of `TokenSource` operates one.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Token {
    /// What is the current token?
    pub kind: SyntaxKind,

    /// Is the current token joined to the next one (`> >` vs `>>`).
    pub is_jointed_to_next: bool,
}

/// `TreeSink` abstracts details of a particular syntax tree implementation.
pub trait TreeSink {
    /// Adds new token to the current branch.
    fn token(&mut self, kind: SyntaxKind, n_tokens: u8);

    /// Start new branch and make it current.
    fn start_node(&mut self, kind: SyntaxKind);

    /// Finish current branch and restore previous
    /// branch as current.
    fn finish_node(&mut self);

    fn error(&mut self, error: ParseError);
}

fn parse_from_tokens<F>(token_source: &mut dyn TokenSource, tree_sink: &mut dyn TreeSink, f: F)
where
    F: FnOnce(&mut parser::Parser),
{
    let mut p = parser::Parser::new(token_source);
    f(&mut p);
    let events = p.finish();
    event::process(tree_sink, events);
}

/// Parse given tokens into the given sink as a rust file.
pub fn parse(token_source: &mut dyn TokenSource, tree_sink: &mut dyn TreeSink) {
    parse_from_tokens(token_source, tree_sink, grammar::root);
}

/// Parse given tokens into the given sink as a path
pub fn parse_path(token_source: &mut dyn TokenSource, tree_sink: &mut dyn TreeSink) {
    parse_from_tokens(token_source, tree_sink, grammar::fragments::path);
}

/// Parse given tokens into the given sink as a expression
pub fn parse_expr(token_source: &mut dyn TokenSource, tree_sink: &mut dyn TreeSink) {
    parse_from_tokens(token_source, tree_sink, grammar::fragments::expr);
}

/// Parse given tokens into the given sink as a ty
pub fn parse_ty(token_source: &mut dyn TokenSource, tree_sink: &mut dyn TreeSink) {
    parse_from_tokens(token_source, tree_sink, grammar::fragments::type_);
}

/// Parse given tokens into the given sink as a pattern
pub fn parse_pat(token_source: &mut dyn TokenSource, tree_sink: &mut dyn TreeSink) {
    parse_from_tokens(token_source, tree_sink, grammar::fragments::pattern);
}

/// Parse given tokens into the given sink as a statement
pub fn parse_stmt(
    token_source: &mut dyn TokenSource,
    tree_sink: &mut dyn TreeSink,
    with_semi: bool,
) {
    parse_from_tokens(token_source, tree_sink, |p| grammar::fragments::stmt(p, with_semi));
}

/// Parse given tokens into the given sink as a block
pub fn parse_block(token_source: &mut dyn TokenSource, tree_sink: &mut dyn TreeSink) {
    parse_from_tokens(token_source, tree_sink, grammar::fragments::block);
}

pub fn parse_meta(token_source: &mut dyn TokenSource, tree_sink: &mut dyn TreeSink) {
    parse_from_tokens(token_source, tree_sink, grammar::fragments::meta_item);
}

/// Parse given tokens into the given sink as an item
pub fn parse_item(token_source: &mut dyn TokenSource, tree_sink: &mut dyn TreeSink) {
    parse_from_tokens(token_source, tree_sink, grammar::fragments::item);
}

/// Parse given tokens into the given sink as an visibility qualifier
pub fn parse_vis(token_source: &mut dyn TokenSource, tree_sink: &mut dyn TreeSink) {
    parse_from_tokens(token_source, tree_sink, grammar::fragments::opt_visibility);
}

pub fn parse_macro_items(token_source: &mut dyn TokenSource, tree_sink: &mut dyn TreeSink) {
    parse_from_tokens(token_source, tree_sink, grammar::fragments::macro_items);
}

pub fn parse_macro_stmts(token_source: &mut dyn TokenSource, tree_sink: &mut dyn TreeSink) {
    parse_from_tokens(token_source, tree_sink, grammar::fragments::macro_stmts);
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
    pub fn parse(self, token_source: &mut dyn TokenSource, tree_sink: &mut dyn TreeSink) {
        let Reparser(r) = self;
        let mut p = parser::Parser::new(token_source);
        r(&mut p);
        let events = p.finish();
        event::process(tree_sink, events);
    }
}
