// Ensure that we reject attributes on the string literal of `#![crate_type]` as the result of the
// expansion wouldn't get used anyway as we need the crate type before macro expansion.
//
// We also test the behavior under print request `file-names` as that's non-normal execution which
// doesn't reach the majority of compiler passes.
//
// issue: <https://github.com/rust-lang/rust/issues/163330>

//@ revisions: normal print
//@[print] compile-flags: --print=file-names

//@ proc-macro: replace.rs
//@ edition: 2018..

#![feature(stmt_expr_attributes, proc_macro_hygiene)]

#![crate_type = #[replace::with("bin")] "lib"]
//~^ ERROR attributes are not allowed in this context

fn main() {}
