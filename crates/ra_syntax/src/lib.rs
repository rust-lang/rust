//! Syntax Tree library used throughout the rust analyzer.
//!
//! Properties:
//!   - easy and fast incremental re-parsing
//!   - graceful handling of errors
//!   - full-fidelity representation (*any* text can be precisely represented as
//!     a syntax tree)
//!
//! For more information, see the [RFC]. Current implementation is inspired by
//! the [Swift] one.
//!
//! The most interesting modules here are `syntax_node` (which defines concrete
//! syntax tree) and `ast` (which defines abstract syntax tree on top of the
//! CST). The actual parser live in a separate `ra_parser` crate, thought the
//! lexer lives in this crate.
//!
//! [RFC]: <https://github.com/rust-lang/rfcs/pull/2256>
//! [Swift]: <https://github.com/apple/swift/blob/13d593df6f359d0cb2fc81cfaac273297c539455/lib/Syntax/README.md>

mod syntax_node;
mod syntax_text;
mod syntax_error;
mod parsing;
mod string_lexing;
mod validation;
mod ptr;

pub mod algo;
pub mod ast;

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
            validation::validate_block_structure(&root);
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

pub fn check_fuzz_invariants(text: &str) {
    let file = SourceFile::parse(text);
    let root = file.syntax();
    validation::validate_block_structure(root);
    let _ = file.errors();
}
