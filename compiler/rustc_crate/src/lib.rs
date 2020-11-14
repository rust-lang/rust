#![feature(int_error_matching)]
#![feature(once_cell)]
#![feature(or_patterns)]

#[macro_use]
extern crate rustc_macros;

pub mod cstore;

/// Requirements for a `StableHashingContext` to be used in this crate.
/// This is a hack to allow using the `HashStable_Generic` derive macro
/// instead of implementing everything in librustc_middle.
pub trait HashStableContext: rustc_ast::HashStableContext + rustc_hir::HashStableContext {}
