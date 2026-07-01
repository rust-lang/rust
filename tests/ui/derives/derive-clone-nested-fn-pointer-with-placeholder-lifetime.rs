//@ check-pass
// Type parameters inside nested function pointer signatures should not
// contribute bounds for derived Clone impls.

#![allow(dead_code)]

trait SomeTrait {
    type SomeType<'a>;
}

#[derive(Clone)]
struct Foo<T: SomeTrait> {
    x: Option<Result<u32, fn(T::SomeType<'_>)>>,
}

fn main() {}
