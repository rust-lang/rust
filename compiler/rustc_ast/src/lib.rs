//! The Rust Abstract Syntax Tree (AST).
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

// tidy-alphabetical-start
#![allow(internal_features)]
#![doc(
    html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/",
    test(attr(deny(warnings)))
)]
#![doc(rust_logo)]
#![feature(array_windows)]
#![feature(associated_type_defaults)]
#![feature(box_patterns)]
#![feature(if_let_guard)]
#![feature(macro_metavar_expr)]
#![feature(negative_impls)]
#![feature(never_type)]
#![feature(rustdoc_internals)]
#![feature(stmt_expr_attributes)]
#![recursion_limit = "256"]
// tidy-alphabetical-end

pub mod util {
    pub mod case;
    pub mod classify;
    pub mod comments;
    pub mod literal;
    pub mod parser;
    pub mod unicode;
}

pub mod ast;
pub mod ast_traits;
pub mod attr;
pub mod entry;
pub mod expand;
pub mod format;
pub mod mut_visit;
pub mod node_id;
pub mod ptr;
pub mod token;
pub mod tokenstream;
pub mod visit;

pub use self::ast::*;
pub use self::ast_traits::{AstNodeWrapper, HasAttrs, HasNodeId, HasTokens};

/// Requirements for a `StableHashingContext` to be used in this crate.
/// This is a hack to allow using the `HashStable_Generic` derive macro
/// instead of implementing everything in `rustc_middle`.
pub trait HashStableContext: rustc_span::HashStableContext {}
