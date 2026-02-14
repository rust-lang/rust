#![allow(warnings)]

trait Trait<T> {
    fn foo(_: T) {}
}

pub struct Foo<T = Box<dyn Trait<DefaultFoo>>>;
//~^ ERROR cycle detected when computing type of `Foo::T`
//~| ERROR type parameter `T` is never used
type DefaultFoo = Foo;

fn main() {
}
