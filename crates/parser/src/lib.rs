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

/// Parse a prefix of the input as a given syntactic construct.
///
/// This is used by macro-by-example parser to implement things like `$i:item`
/// and the naming of variants follows the naming of macro fragments.
///
/// Note that this is generally non-optional -- the result is intentionally not
/// `Option<Output>`. The way MBE work, by the time we *try* to parse `$e:expr`
/// we already commit to expression. In other words, this API by design can't be
/// used to implement "rollback and try another alternative" logic.
#[derive(Debug)]
pub enum PrefixEntryPoint {
    Vis,
    Block,
    Stmt,
    Pat,
    Ty,
    Expr,
    Path,
    Item,
    MetaItem,
}

impl PrefixEntryPoint {
    pub fn parse(&self, input: &Input) -> Output {
        let entry_point: fn(&'_ mut parser::Parser) = match self {
            PrefixEntryPoint::Vis => grammar::entry::prefix::vis,
            PrefixEntryPoint::Block => grammar::entry::prefix::block,
            PrefixEntryPoint::Stmt => grammar::entry::prefix::stmt,
            PrefixEntryPoint::Pat => grammar::entry::prefix::pat,
            PrefixEntryPoint::Ty => grammar::entry::prefix::ty,
            PrefixEntryPoint::Expr => grammar::entry::prefix::expr,
            PrefixEntryPoint::Path => grammar::entry::prefix::path,
            PrefixEntryPoint::Item => grammar::entry::prefix::item,
            PrefixEntryPoint::MetaItem => grammar::entry::prefix::meta_item,
        };
        let mut p = parser::Parser::new(input);
        entry_point(&mut p);
        let events = p.finish();
        event::process(events)
    }
}

/// Parse the whole of the input as a given syntactic construct.
///
/// This covers two main use-cases:
///
///   * Parsing a Rust file.
///   * Parsing a result of macro expansion.
///
/// That is, for something like
///
/// ```
/// quick_check! {
///    fn prop() {}
/// }
/// ```
///
/// the input to the macro will be parsed with [`PrefixEntryPoint::Item`], and
/// the result will be [`TopEntryPoint::Items`].
///
/// This *should* (but currently doesn't) guarantee that all input is consumed.
#[derive(Debug)]
pub enum TopEntryPoint {
    SourceFile,
    MacroStmts,
    MacroItems,
    Pattern,
    Type,
    Expr,
    MetaItem,
}

impl TopEntryPoint {
    pub fn parse(&self, input: &Input) -> Output {
        let entry_point: fn(&'_ mut parser::Parser) = match self {
            TopEntryPoint::SourceFile => grammar::entry::top::source_file,
            TopEntryPoint::MacroStmts => grammar::entry::top::macro_stmts,
            TopEntryPoint::MacroItems => grammar::entry::top::macro_items,
            // FIXME
            TopEntryPoint::Pattern => grammar::entry::prefix::pat,
            TopEntryPoint::Type => grammar::entry::prefix::ty,
            TopEntryPoint::Expr => grammar::entry::prefix::expr,
            TopEntryPoint::MetaItem => grammar::entry::prefix::meta_item,
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
    Path,
    Expr,
    StatementOptionalSemi,
    Pattern,
    Item,
    Attr,
}

/// Parse given tokens into the given sink as a rust file.
pub fn parse_source_file(input: &Input) -> Output {
    TopEntryPoint::SourceFile.parse(input)
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
        ParserEntryPoint::Path => grammar::entry::prefix::path,
        ParserEntryPoint::Expr => grammar::entry::prefix::expr,
        ParserEntryPoint::Pattern => grammar::entry::prefix::pat,
        ParserEntryPoint::Item => grammar::entry::prefix::item,
        ParserEntryPoint::StatementOptionalSemi => grammar::entry_points::stmt_optional_semi,
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
