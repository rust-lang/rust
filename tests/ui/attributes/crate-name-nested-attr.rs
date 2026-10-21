// Ensure that we reject attributes on the string literal of `#![crate_name]` as the result of the
// expansion wouldn't get used anyway as we need the crate name before macro expansion.
//
// We also test the behavior under print request `crate-name` as that's non-normal execution which
// doesn't reach the majority of compiler passes.
//
// issue: <https://github.com/rust-lang/rust/issues/163330>

//@ revisions: normal print
//@[print] compile-flags: --print=crate-name

//@ proc-macro: replace.rs
//@ edition: 2018..

#![feature(stmt_expr_attributes, proc_macro_hygiene)]

#![crate_name = #[replace::with("alias")] "name"]
//~^ ERROR attributes are not allowed in this context

fn main() {}
