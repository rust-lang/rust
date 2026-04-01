#![deny(keyword_idents)] // Should affect the submodule, but doesn't.
//@ edition: 2015
//@ known-bug: #132218
//@ check-pass (known bug; should be check-fail)

// Because `keyword_idents_2018` and `keyword_idents_2024` are pre-expansion
// lints, configuring them via lint attributes doesn't propagate to submodules
// in other files.
// <https://github.com/rust-lang/rust/issues/132218>

#[path = "./auxiliary/multi_file_submod.rs"]
mod multi_file_submod;

fn main() {}
