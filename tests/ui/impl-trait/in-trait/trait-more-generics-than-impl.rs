#![feature(return_position_impl_trait_in_trait)]
#![allow(incomplete_features)]

struct S;

trait Foo {
    fn bar<T>() -> impl Sized;
}

impl Foo for S {
    fn bar() -> impl Sized {}
    //~^ ERROR method `bar` has 0 type parameters but its trait declaration has 1 type parameter
}

fn main() {
    S::bar();
}
