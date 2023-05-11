// check-pass
// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// revisions: current next

#![feature(return_position_impl_trait_in_trait)]
#![allow(incomplete_features)]

trait Foo<T> {
    fn foo<F2: Foo<T>>(self) -> impl Foo<T>;
}

struct Bar;

impl Foo<u8> for Bar {
    fn foo<F2: Foo<u8>>(self) -> impl Foo<u8> {
        self
    }
}

fn main() {}
