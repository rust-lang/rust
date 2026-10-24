//@ check-pass
// Issue #143131: `#[derive(Clone)]` should accept a function pointer field whose
// input type contains a placeholder lifetime.

#![allow(dead_code)]

trait SomeTrait {
    type SomeType<'a>;
}

#[derive(Clone)]
struct Foo<T: SomeTrait> {
    x: fn(T::SomeType<'_>),
}

fn main() {}
