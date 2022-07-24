#![feature(const_trait_impl)]

trait Foo {
    fn a(&self);
}
trait Bar: ~const Foo {}

const fn foo<T: Bar>(x: &T) {
    x.a();
    //~^ ERROR the trait bound
    //~| ERROR cannot call
}

fn main() {}
