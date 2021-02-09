//! The Rust parser and macro expander.
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![doc(
    html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/",
    test(attr(deny(warnings)))
)]
#![feature(box_syntax)]
#![feature(const_fn)] // For the `transmute` in `P::new`
#![feature(const_fn_transmute)]
#![feature(const_panic)]
#![feature(crate_visibility_modifier)]
#![feature(iterator_fold_self)]
#![feature(label_break_value)]
#![feature(nll)]
#![feature(or_patterns)]
#![recursion_limit = "256"]

#[macro_use]
extern crate rustc_macros;

#[macro_export]
macro_rules! unwrap_or {
    ($opt:expr, $default:expr) => {
        match $opt {
            Some(x) => x,
            None => $default,
        }
    };
}

pub mod util {
    pub mod classify;
    pub mod comments;
    pub mod literal;
    pub mod parser;
}

pub mod ast;
pub mod attr;
pub mod crate_disambiguator;
pub mod entry;
pub mod expand;
pub mod mut_visit;
pub mod node_id;
pub mod ptr;
pub mod token;
pub mod tokenstream;
pub mod visit;

pub use self::ast::*;

use rustc_data_structures::stable_hasher::{HashStable, StableHasher};

/// Requirements for a `StableHashingContext` to be used in this crate.
/// This is a hack to allow using the `HashStable_Generic` derive macro
/// instead of implementing everything in librustc_middle.
pub trait HashStableContext: rustc_span::HashStableContext {
    fn hash_attr(&mut self, _: &ast::Attribute, hasher: &mut StableHasher);
}

impl<AstCtx: crate::HashStableContext> HashStable<AstCtx> for ast::Attribute {
    fn hash_stable(&self, hcx: &mut AstCtx, hasher: &mut StableHasher) {
        hcx.hash_attr(self, hasher)
    }
}
