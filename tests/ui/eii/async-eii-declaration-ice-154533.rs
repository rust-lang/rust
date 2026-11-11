//! Regression test for <https://github.com/rust-lang/rust/issues/154533>.
//!
//! An `async fn` declared with `#[eii]` used to ICE with
//! "trait bounds on EII not yet supported" under the new solver.

//@ edition: 2021
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

#![feature(extern_item_impls)]

#[eii]
async fn hello_impl() {}
//~^ ERROR functions in `extern` blocks cannot have `async` qualifier
//~| ERROR function `hello_impl` has a type that is incompatible with the declaration of `#[hello_impl]`

fn main() {}
