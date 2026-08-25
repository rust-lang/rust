//! Regression test to check that outer attributes applied to the first module item is applied to
//! its attached module item only, and not also to other subsequent module items
//!
//! Commit: <https://github.com/rust-lang/rust/commit/7aee9f7b56f8d96f9444ebb1d06e32e024b81974>

//@ check-pass
//@ compile-flags: --cfg=first
//@ no-auto-check-cfg

#[cfg(first)]
mod hello {}

#[cfg(not_set)]
mod hello {}

fn main() {}
