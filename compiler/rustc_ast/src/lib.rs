//! The Rust Abstract Syntax Tree (AST).
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![doc(
    html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/",
    test(attr(deny(warnings)))
)]
#![feature(box_patterns)]
#![feature(crate_visibility_modifier)]
#![feature(if_let_guard)]
#![feature(label_break_value)]
#![feature(nll)]
#![feature(min_specialization)]
#![recursion_limit = "256"]
#![feature(slice_internals)]

#[macro_use]
extern crate rustc_macros;

pub mod util {
    pub mod classify;
    pub mod comments;
    pub mod literal;
    pub mod parser;
    pub mod unicode;
}

pub mod ast;
pub mod ast_like;
pub mod attr;
pub mod entry;
pub mod expand;
pub mod mut_visit;
pub mod node_id;
pub mod ptr;
pub mod token;
pub mod tokenstream;
pub mod visit;

pub use self::ast::*;
pub use self::ast_like::{AstLike, AstLikeWrapper};

use rustc_data_structures::stable_hasher::{HashStable, StableHasher};

/// Requirements for a `StableHashingContext` to be used in this crate.
/// This is a hack to allow using the `HashStable_Generic` derive macro
/// instead of implementing everything in `rustc_middle`.
pub trait HashStableContext: rustc_span::HashStableContext {
    fn hash_attr(&mut self, _: &ast::Attribute, hasher: &mut StableHasher);
}

impl<AstCtx: crate::HashStableContext> HashStable<AstCtx> for ast::Attribute {
    fn hash_stable(&self, hcx: &mut AstCtx, hasher: &mut StableHasher) {
        hcx.hash_attr(self, hasher)
    }
}
