//@ run-pass
// Issue #122622: `#[derive(Clone)]` should work for HRTB function type taking an associated type
#![allow(dead_code)]
trait SomeTrait {
    type SomeType<'a>;
}

#[derive(Clone)]
struct Foo<T: SomeTrait> {
    x: for<'a> fn(T::SomeType<'a>)
}

fn main() {}
