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

#![warn(rust_2018_idioms, unused_lifetimes, semicolon_in_expressions_from_macros)]
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
/// the result will be [`TopEntryPoint::MacroItems`].
///
/// [`TopEntryPoint::parse`] makes a guarantee that
///   * all input is consumed
///   * the result is a valid tree (there's one root node)
#[derive(Debug)]
pub enum TopEntryPoint {
    SourceFile,
    MacroStmts,
    MacroItems,
    Pattern,
    Type,
    Expr,
    /// Edge case -- macros generally don't expand to attributes, with the
    /// exception of `cfg_attr` which does!
    MetaItem,
}

impl TopEntryPoint {
    pub fn parse(&self, input: &Input) -> Output {
        let entry_point: fn(&'_ mut parser::Parser<'_>) = match self {
            TopEntryPoint::SourceFile => grammar::entry::top::source_file,
            TopEntryPoint::MacroStmts => grammar::entry::top::macro_stmts,
            TopEntryPoint::MacroItems => grammar::entry::top::macro_items,
            TopEntryPoint::Pattern => grammar::entry::top::pattern,
            TopEntryPoint::Type => grammar::entry::top::type_,
            TopEntryPoint::Expr => grammar::entry::top::expr,
            TopEntryPoint::MetaItem => grammar::entry::top::meta_item,
        };
        let mut p = parser::Parser::new(input);
        entry_point(&mut p);
        let events = p.finish();
        let res = event::process(events);

        if cfg!(debug_assertions) {
            let mut depth = 0;
            let mut first = true;
            for step in res.iter() {
                assert!(depth > 0 || first);
                first = false;
                match step {
                    Step::Enter { .. } => depth += 1,
                    Step::Exit => depth -= 1,
                    Step::FloatSplit { ends_in_dot: has_pseudo_dot } => {
                        depth -= 1 + !has_pseudo_dot as usize
                    }
                    Step::Token { .. } | Step::Error { .. } => (),
                }
            }
            assert!(!first, "no tree at all");
            assert_eq!(depth, 0, "unbalanced tree");
        }

        res
    }
}

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
    PatTop,
    Ty,
    Expr,
    Path,
    Item,
    MetaItem,
}

impl PrefixEntryPoint {
    pub fn parse(&self, input: &Input) -> Output {
        let entry_point: fn(&'_ mut parser::Parser<'_>) = match self {
            PrefixEntryPoint::Vis => grammar::entry::prefix::vis,
            PrefixEntryPoint::Block => grammar::entry::prefix::block,
            PrefixEntryPoint::Stmt => grammar::entry::prefix::stmt,
            PrefixEntryPoint::Pat => grammar::entry::prefix::pat,
            PrefixEntryPoint::PatTop => grammar::entry::prefix::pat_top,
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

/// A parsing function for a specific braced-block.
pub struct Reparser(fn(&mut parser::Parser<'_>));

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
