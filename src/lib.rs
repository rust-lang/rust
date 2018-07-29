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

mod lexer;
mod parser;
mod yellow;
mod syntax_kinds;

pub use {
    text_unit::{TextRange, TextUnit},
    syntax_kinds::SyntaxKind,
    yellow::{SyntaxNode},
    lexer::{tokenize, Token},
};

pub(crate) use {
    yellow::SyntaxError
};

pub fn parse(text: String) -> SyntaxNode {
    let tokens = tokenize(&text);
    parser::parse::<yellow::GreenBuilder>(text, &tokens)
}


/// Utilities for simple uses of the parser.
pub mod utils {
    use std::{
        fmt::Write,
        collections::BTreeSet
    };

    use {SyntaxNode, SyntaxError};

    /// Parse a file and create a string representation of the resulting parse tree.
    pub fn dump_tree_green(syntax: &SyntaxNode) -> String {
        let mut errors: BTreeSet<_> = syntax.root.errors.iter().cloned().collect();
        let mut result = String::new();
        go(syntax, &mut result, 0, &mut errors);
        return result;

        fn go(node: &SyntaxNode, buff: &mut String, level: usize, errors: &mut BTreeSet<SyntaxError>) {
            buff.push_str(&String::from("  ").repeat(level));
            write!(buff, "{:?}\n", node).unwrap();
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
