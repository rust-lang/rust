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

extern crate unicode_xid;
extern crate text_unit;

mod tree;
mod lexer;
mod parser;
mod yellow;

pub mod syntax_kinds;
pub use text_unit::{TextRange, TextUnit};
pub use tree::{File, Node, SyntaxKind, Token};
pub(crate) use tree::{ErrorMsg, FileBuilder, Sink, GreenBuilder};
pub use lexer::{next_token, tokenize};
pub use yellow::SyntaxNode;
pub(crate) use yellow::SError;
pub use parser::{parse, parse_green};

/// Utilities for simple uses of the parser.
pub mod utils {
    use std::fmt::Write;

    use {File, Node, SyntaxNode};
    use std::collections::BTreeSet;
    use SError;

    /// Parse a file and create a string representation of the resulting parse tree.
    pub fn dump_tree(file: &File) -> String {
        let mut result = String::new();
        go(file.root(), &mut result, 0);
        return result;

        fn go(node: Node, buff: &mut String, level: usize) {
            buff.push_str(&String::from("  ").repeat(level));
            write!(buff, "{:?}\n", node).unwrap();
            let my_errors = node.errors().filter(|e| e.after_child().is_none());
            let parent_errors = node.parent()
                .into_iter()
                .flat_map(|n| n.errors())
                .filter(|e| e.after_child() == Some(node));

            for err in my_errors {
                buff.push_str(&String::from("  ").repeat(level));
                write!(buff, "err: `{}`\n", err.message()).unwrap();
            }

            for child in node.children() {
                go(child, buff, level + 1)
            }

            for err in parent_errors {
                buff.push_str(&String::from("  ").repeat(level));
                write!(buff, "err: `{}`\n", err.message()).unwrap();
            }
        }
    }

    /// Parse a file and create a string representation of the resulting parse tree.
    pub fn dump_tree_green(syntax: &SyntaxNode) -> String {
        let mut errors: BTreeSet<_> = syntax.root.errors.iter().cloned().collect();
        let mut result = String::new();
        go(syntax, &mut result, 0, &mut errors);
        return result;

        fn go(node: &SyntaxNode, buff: &mut String, level: usize, errors: &mut BTreeSet<SError>) {
            buff.push_str(&String::from("  ").repeat(level));
            write!(buff, "{:?}\n", node).unwrap();
//            let my_errors = node.errors().filter(|e| e.after_child().is_none());
//            let parent_errors = node.parent()
//                .into_iter()
//                .flat_map(|n| n.errors())
//                .filter(|e| e.after_child() == Some(node));
//
            let my_errors: Vec<_> = errors.iter().filter(|e| e.offset == node.range().start())
                .cloned().collect();
            for err in my_errors {
                errors.remove(&err);
                buff.push_str(&String::from("  ").repeat(level));
                write!(buff, "err: `{}`\n", err.message).unwrap();
            }

            for child in node.children().iter() {
                go(child, buff, level + 1, errors)
            }

            let my_errors: Vec<_> = errors.iter().filter(|e| e.offset == node.range().end())
                .cloned().collect();
            for err in my_errors {
                errors.remove(&err);
                buff.push_str(&String::from("  ").repeat(level));
                write!(buff, "err: `{}`\n", err.message).unwrap();
            }
        }
    }
}
