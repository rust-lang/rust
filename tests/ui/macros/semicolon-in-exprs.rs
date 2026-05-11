//! Regression test for https://github.com/rust-lang/rust/issues/156084.
//! This test can probably be removed again once
//! `semicolon_in_expressions_from_macros` is a hard error.
//@ check-pass
//@ aux-build:semicolon-in-exprs.rs
//@ edition: 2021

extern crate semicolon_in_exprs;

macro_rules! inner {
    [$($x:expr),*] => { [$($x),*] };
}
fn main() {
    let _v: Vec<i32> = semicolon_in_exprs::outer!(inner).into_iter().collect();
}
