//@ aux-crate:interner=interner.rs
// https://github.com/rust-lang/rust/pull/122247
extern crate interner;
#[doc(inline)]
pub use interner::*;
