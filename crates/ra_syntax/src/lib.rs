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

#![forbid(missing_debug_implementations, unconditional_recursion, future_incompatible)]
#![deny(bad_style, missing_docs)]
#![allow(missing_docs)]
//#![warn(unreachable_pub)] // rust-lang/rust#47816

mod syntax_node;
mod syntax_text;
mod syntax_error;
mod parsing;
mod string_lexing;
mod validation;
mod ptr;

pub mod algo;
pub mod ast;
/// Utilities for simple uses of the parser.
pub mod utils;

pub use rowan::{SmolStr, TextRange, TextUnit};
pub use ra_parser::SyntaxKind;
pub use crate::{
    ast::AstNode,
    syntax_error::{SyntaxError, SyntaxErrorKind, Location},
    syntax_text::SyntaxText,
    syntax_node::{Direction,  SyntaxNode, WalkEvent, TreeArc},
    ptr::{SyntaxNodePtr, AstPtr},
    parsing::{tokenize, Token},
};

use ra_text_edit::AtomTextEdit;
use crate::syntax_node::GreenNode;

/// `SourceFile` represents a parse tree for a single Rust file.
pub use crate::ast::SourceFile;

impl SourceFile {
    fn new(green: GreenNode, errors: Vec<SyntaxError>) -> TreeArc<SourceFile> {
        let root = SyntaxNode::new(green, errors);
        if cfg!(debug_assertions) {
            utils::validate_block_structure(&root);
        }
        assert_eq!(root.kind(), SyntaxKind::SOURCE_FILE);
        TreeArc::cast(root)
    }

    pub fn parse(text: &str) -> TreeArc<SourceFile> {
        let (green, errors) = parsing::parse_text(text);
        SourceFile::new(green, errors)
    }

    pub fn reparse(&self, edit: &AtomTextEdit) -> TreeArc<SourceFile> {
        self.incremental_reparse(edit).unwrap_or_else(|| self.full_reparse(edit))
    }

    pub fn incremental_reparse(&self, edit: &AtomTextEdit) -> Option<TreeArc<SourceFile>> {
        parsing::incremental_reparse(self.syntax(), edit, self.errors())
            .map(|(green_node, errors)| SourceFile::new(green_node, errors))
    }

    fn full_reparse(&self, edit: &AtomTextEdit) -> TreeArc<SourceFile> {
        let text = edit.apply(self.syntax().text().to_string());
        SourceFile::parse(&text)
    }

    pub fn errors(&self) -> Vec<SyntaxError> {
        let mut errors = self.syntax.root_data().clone();
        errors.extend(validation::validate(self));
        errors
    }
}
