//! An experimental implementation of [Rust RFC#2256 libsyntax2.0][rfc#2256].
//!
//! The intent is to be an IDE-ready parser, i.e. one that offers
//!
//! - easy and fast incremental re-parsing,
//! - graceful handling of errors, and
//! - maintains all information in the source file.
//!
//! For more information, see [the RFC][rfc#2265], or [the working draft][RFC.md].
//!
//!   [rfc#2256]: <https://github.com/rust-lang/rfcs/pull/2256>
//!   [RFC.md]: <https://github.com/matklad/libsyntax2/blob/master/docs/RFC.md>

#![forbid(
    missing_debug_implementations,
    unconditional_recursion,
    future_incompatible
)]
#![deny(bad_style, missing_docs)]
#![allow(missing_docs)]
//#![warn(unreachable_pub)] // rust-lang/rust#47816

extern crate itertools;
extern crate unicode_xid;
extern crate drop_bomb;
extern crate parking_lot;
extern crate smol_str;
extern crate text_unit;

#[cfg(test)]
#[macro_use]
extern crate test_utils;

pub mod algo;
pub mod ast;
mod lexer;
#[macro_use]
mod token_set;
mod parser_api;
mod grammar;
mod parser_impl;
mod reparsing;

mod syntax_kinds;
mod yellow;
/// Utilities for simple uses of the parser.
pub mod utils;
pub mod text_utils;

pub use {
    text_unit::{TextRange, TextUnit},
    smol_str::SmolStr,
    ast::AstNode,
    lexer::{tokenize, Token},
    syntax_kinds::SyntaxKind,
    yellow::{SyntaxNode, SyntaxNodeRef, OwnedRoot, RefRoot, TreeRoot, SyntaxError},
    reparsing::AtomEdit,
};

use {
    yellow::{GreenNode, SyntaxRoot},
};

#[derive(Clone, Debug, Hash)]
pub struct File {
    root: SyntaxNode
}

impl File {
    fn new(green: GreenNode, errors: Vec<SyntaxError>) -> File {
        let root = SyntaxRoot::new(green, errors);
        let root = SyntaxNode::new_owned(root);
        if cfg!(debug_assertions) {
            utils::validate_block_structure(root.borrowed());
        }
        File { root }
    }
    pub fn parse(text: &str) -> File {
        let tokens = tokenize(&text);
        let (green, errors) = parser_impl::parse_with::<yellow::GreenBuilder>(
            text, &tokens, grammar::root,
        );
        File::new(green, errors)
    }
    pub fn reparse(&self, edit: &AtomEdit) -> File {
        self.incremental_reparse(edit).unwrap_or_else(|| self.full_reparse(edit))
    }
    pub fn incremental_reparse(&self, edit: &AtomEdit) -> Option<File> {
        reparsing::incremental_reparse(self.syntax(), edit, self.errors())
            .map(|(green_node, errors)| File::new(green_node, errors))
    }
    fn full_reparse(&self, edit: &AtomEdit) -> File {
        let text = text_utils::replace_range(self.syntax().text().to_string(), edit.delete, &edit.insert);
        File::parse(&text)
    }
    pub fn ast(&self) -> ast::Root {
        ast::Root::cast(self.syntax()).unwrap()
    }
    pub fn syntax(&self) -> SyntaxNodeRef {
        self.root.borrowed()
    }
    pub fn errors(&self) -> Vec<SyntaxError> {
        self.syntax().root.syntax_root().errors.clone()
    }
}
