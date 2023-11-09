#![allow(warnings)]

trait Trait<T> {
    fn foo(_: T) {}
}

pub struct Foo<T = Box<Trait<DefaultFoo>>>;  //~ ERROR cycle detected
//~^ ERROR `T` is never used
//~| ERROR `Trait` cannot be made into an object
type DefaultFoo = Foo;

fn main() {
}
