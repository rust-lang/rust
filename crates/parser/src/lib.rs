//! The Rust parser.
//!
//! The parser doesn't know about concrete representation of tokens and syntax
//! trees. Abstract [`TokenSource`] and [`TreeSink`] traits are used instead.
//! As a consequence, this crate does not contain a lexer.
//!
//! The [`Parser`] struct from the [`parser`] module is a cursor into the
//! sequence of tokens.  Parsing routines use [`Parser`] to inspect current
//! state and advance the parsing.
//!
//! The actual parsing happens in the [`grammar`] module.
//!
//! Tests for this crate live in the `syntax` crate.
//!
//! [`Parser`]: crate::parser::Parser
#![allow(rustdoc::private_intra_doc_links)]

mod token_set;
mod syntax_kind;
mod event;
mod parser;
mod grammar;

pub(crate) use token_set::TokenSet;

pub use syntax_kind::SyntaxKind;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParseError(pub Box<String>);

/// `TokenSource` abstracts the source of the tokens parser operates on.
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

/// `Token` abstracts the cursor of `TokenSource` operates on.
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

/// rust-analyzer parser allows you to choose one of the possible entry points.
///
/// The primary consumer of this API are declarative macros, `$x:expr` matchers
/// are implemented by calling into the parser with non-standard entry point.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum ParserEntryPoint {
    SourceFile,
    Path,
    Expr,
    Statement,
    StatementOptionalSemi,
    Type,
    Pattern,
    Item,
    Block,
    Visibility,
    MetaItem,
    Items,
    Statements,
    Attr,
}

/// Parse given tokens into the given sink as a rust file.
pub fn parse_source_file(token_source: &mut dyn TokenSource, tree_sink: &mut dyn TreeSink) {
    parse(token_source, tree_sink, ParserEntryPoint::SourceFile);
}

pub fn parse(
    token_source: &mut dyn TokenSource,
    tree_sink: &mut dyn TreeSink,
    entry_point: ParserEntryPoint,
) {
    let entry_point: fn(&'_ mut parser::Parser) = match entry_point {
        ParserEntryPoint::SourceFile => grammar::entry_points::source_file,
        ParserEntryPoint::Path => grammar::entry_points::path,
        ParserEntryPoint::Expr => grammar::entry_points::expr,
        ParserEntryPoint::Type => grammar::entry_points::type_,
        ParserEntryPoint::Pattern => grammar::entry_points::pattern,
        ParserEntryPoint::Item => grammar::entry_points::item,
        ParserEntryPoint::Block => grammar::entry_points::block_expr,
        ParserEntryPoint::Visibility => grammar::entry_points::visibility,
        ParserEntryPoint::MetaItem => grammar::entry_points::meta_item,
        ParserEntryPoint::Statement => grammar::entry_points::stmt,
        ParserEntryPoint::StatementOptionalSemi => grammar::entry_points::stmt_optional_semi,
        ParserEntryPoint::Items => grammar::entry_points::macro_items,
        ParserEntryPoint::Statements => grammar::entry_points::macro_stmts,
        ParserEntryPoint::Attr => grammar::entry_points::attr,
    };

    let mut p = parser::Parser::new(token_source);
    entry_point(&mut p);
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
    pub fn parse(self, token_source: &mut dyn TokenSource, tree_sink: &mut dyn TreeSink) {
        let Reparser(r) = self;
        let mut p = parser::Parser::new(token_source);
        r(&mut p);
        let events = p.finish();
        event::process(tree_sink, events);
    }
}
