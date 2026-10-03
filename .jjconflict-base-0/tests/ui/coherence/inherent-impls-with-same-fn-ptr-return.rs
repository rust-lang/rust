//! Regression test for <https://github.com/rust-lang/rust/issues/41298>.
//! Two impl blocks with same generics caused ICE during coherence overlap
//! check.
//@ check-pass

#![allow(dead_code)]
struct Function<T, F> { t: T, f: F }

impl<T, R> Function<T, fn() -> R> { fn foo() { } }
impl<T, R> Function<T, fn() -> R> { fn bar() { } }

fn main() { }
