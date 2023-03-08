// check-pass
// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// revisions: current next

#![feature(return_position_impl_trait_in_trait)]
#![allow(incomplete_features)]

struct Wrapper<T>(T);

trait Foo {
    fn bar() -> Wrapper<impl Sized>;
}

impl Foo for () {
    fn bar() -> Wrapper<i32> {
        Wrapper(0)
    }
}

fn main() {}
