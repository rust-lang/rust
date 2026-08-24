//! This test used to get stuck in an infinite
//! recursion during normalization.
//!
//! issue: https://github.com/rust-lang/rust/issues/133901
//@ ignore-parallel-frontend query cycle
#![feature(trait_alias)]
fn foo<T: Baz<i32>>() {}
trait Baz<A> = Baz<Option<A>>;
//~^ ERROR: cycle detected when computing the implied clauses of `Baz`

fn main() {}
