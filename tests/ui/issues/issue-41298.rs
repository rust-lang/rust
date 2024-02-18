//@ check-pass
#![allow(dead_code)]
struct Function<T, F> { t: T, f: F }

impl<T, R> Function<T, fn() -> R> { fn foo() { } }
impl<T, R> Function<T, fn() -> R> { fn bar() { } }

fn main() { }
