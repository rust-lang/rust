//! Regression test for <https://github.com/rust-lang/rust/issues/34373>.
//! Test cyclic default type param through alias doesn't ICE.

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
