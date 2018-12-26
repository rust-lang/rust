#![allow(warnings)]

trait Trait<T> {
    fn foo(_: T) {}
}

pub struct Foo<T = Box<Trait<DefaultFoo>>>;  //~ ERROR cycle detected
type DefaultFoo = Foo;

fn main() {
}
