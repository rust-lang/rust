#![feature(return_position_impl_trait_in_trait)]
#![allow(incomplete_features)]

struct U;

trait Foo {
    fn bar(&self) -> impl Sized;
}

impl Foo for U {
    fn bar<T>(&self) {}
    //~^ ERROR method `bar` has 1 type parameter but its trait declaration has 0 type parameters
}

fn main() {
    U.bar();
}
