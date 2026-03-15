#![allow(warnings)]
//@ ignore-parallel-frontend query cycle
trait Trait<T> {
    fn foo(_: T) {}
}

pub struct Foo<T = Box<dyn Trait<DefaultFoo>>>;
//~^ ERROR cycle detected when computing type of `Foo::T`
type DefaultFoo = Foo;

fn main() {
}
