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
#![deny(bad_style, unsafe_code, missing_docs)]
//#![warn(unreachable_pub)] // rust-lang/rust#47816

extern crate unicode_xid;
extern crate text_unit;

mod tree;
mod lexer;
mod parser;

pub mod syntax_kinds;
pub use text_unit::{TextRange, TextUnit};
pub use tree::{File, Node, SyntaxKind, Token};
pub(crate) use tree::{ErrorMsg, FileBuilder, Sink};
pub use lexer::{next_token, tokenize};
pub use parser::parse;

/// Utilities for simple uses of the parser.
pub mod utils {
    use std::fmt::Write;

    use {File, Node};

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
}
