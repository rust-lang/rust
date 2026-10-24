// Ensure that we reject attributes on the string literal of `#![recursion_limit]` as the result of
// the expansion wouldn't get used anyway as we obviously need the recursion limit before macro
// expansion.
// issue: <https://github.com/rust-lang/rust/issues/163330>

//@ proc-macro: replace.rs
//@ edition: 2018..

#![feature(stmt_expr_attributes, proc_macro_hygiene)]

#![recursion_limit = #[replace::with("0")] "64"]
//~^ ERROR attributes are not allowed in this context

fn main() {}
