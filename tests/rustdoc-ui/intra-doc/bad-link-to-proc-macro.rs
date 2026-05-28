//@ compile-flags: --crate-type=proc-macro --document-private-items
#![deny(rustdoc::broken_intra_doc_links)]

//! Link to [`m`].
//~^ ERROR `m` is both a module and a macro

// test a further edge case related to https://github.com/rust-lang/rust/issues/91274

// we need to make sure that when there is actually an ambiguity
// in a proc-macro crate, we print out a sensible error.
// because proc macro crates can't normally export modules,
// this can only happen in --document-private-items mode.

extern crate proc_macro;

mod m {}

#[proc_macro]
pub fn m(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    input
}
