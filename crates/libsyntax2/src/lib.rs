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

pub mod algo;
pub mod ast;
mod lexer;
#[macro_use]
mod parser_api;
mod grammar;
mod parser_impl;

mod syntax_kinds;
mod yellow;
/// Utilities for simple uses of the parser.
pub mod utils;
pub mod text_utils;

pub use {
    text_unit::{TextRange, TextUnit},
    smol_str::SmolStr,
    ast::{AstNode, ParsedFile},
    lexer::{tokenize, Token},
    syntax_kinds::SyntaxKind,
    yellow::{SyntaxNode, SyntaxNodeRef, OwnedRoot, RefRoot, TreeRoot, SyntaxError},
};


pub fn parse(text: &str) -> SyntaxNode {
    let tokens = tokenize(&text);
    let res = parser_impl::parse::<yellow::GreenBuilder>(text, &tokens);
    validate_block_structure(res.borrowed());
    res
}

#[cfg(not(debug_assertions))]
fn validate_block_structure(_: SyntaxNodeRef) {}

#[cfg(debug_assertions)]
fn validate_block_structure(root: SyntaxNodeRef) {
    let mut stack = Vec::new();
    for node in algo::walk::preorder(root) {
        match node.kind() {
            SyntaxKind::L_CURLY => {
                stack.push(node)
            }
            SyntaxKind::R_CURLY => {
                if let Some(pair) = stack.pop() {
                    assert_eq!(
                        node.parent(),
                        pair.parent(),
                        "unpaired curleys:\n{}",
                        utils::dump_tree(root),
                    );
                    assert!(
                        node.next_sibling().is_none() && pair.prev_sibling().is_none(),
                        "floating curlys at {:?}\nfile:\n{}\nerror:\n{}\n",
                        node,
                        root.text(),
                        node.text(),
                    );
                }
            }
            _ => (),
        }
    }
}
