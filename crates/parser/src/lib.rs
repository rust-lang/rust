//! The Rust parser.
//!
//! NOTE: The crate is undergoing refactors, don't believe everything the docs
//! say :-)
//!
//! The parser doesn't know about concrete representation of tokens and syntax
//! trees. Abstract [`TokenSource`] and [`TreeSink`] traits are used instead. As
//! a consequence, this crate does not contain a lexer.
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

mod lexed_str;
mod token_set;
mod syntax_kind;
mod event;
mod parser;
mod grammar;
mod input;
mod output;
mod shortcuts;

#[cfg(test)]
mod tests;

pub(crate) use token_set::TokenSet;

pub use crate::{
    input::Input,
    lexed_str::LexedStr,
    output::{Output, Step},
    shortcuts::StrStep,
    syntax_kind::SyntaxKind,
};

/// Parse a syntactic construct at the *start* of the input.
///
/// This is used by macro-by-example parser to implement things like `$i:item`.
///
/// Note that this is generally non-optional -- the result is intentionally not
/// `Option<Output>`. The way MBE work, by the time we *try* to parse `$e:expr`
/// we already commit to expression. In other words, this API by design can't be
/// used to implement "rollback and try another alternative" logic.
pub enum PrefixEntryPoint {
    Vis,
}

impl PrefixEntryPoint {
    pub fn parse(self, input: &Input) -> Output {
        let entry_point: fn(&'_ mut parser::Parser) = match self {
            PrefixEntryPoint::Vis => grammar::entry::prefix::vis,
        };
        let mut p = parser::Parser::new(input);
        entry_point(&mut p);
        let events = p.finish();
        event::process(events)
    }
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
pub fn parse_source_file(inp: &Input) -> Output {
    parse(inp, ParserEntryPoint::SourceFile)
}

/// Parses the given [`Input`] into [`Output`] assuming that the top-level
/// syntactic construct is the given [`ParserEntryPoint`].
///
/// Both input and output here are fairly abstract. The overall flow is that the
/// caller has some "real" tokens, converts them to [`Input`], parses them to
/// [`Output`], and then converts that into a "real" tree. The "real" tree is
/// made of "real" tokens, so this all hinges on rather tight coordination of
/// indices between the four stages.
pub fn parse(inp: &Input, entry_point: ParserEntryPoint) -> Output {
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

    let mut p = parser::Parser::new(inp);
    entry_point(&mut p);
    let events = p.finish();
    event::process(events)
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
    pub fn parse(self, tokens: &Input) -> Output {
        let Reparser(r) = self;
        let mut p = parser::Parser::new(tokens);
        r(&mut p);
        let events = p.finish();
        event::process(events)
    }
}
