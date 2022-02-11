#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

use std::fmt::Debug;

type FooX = impl Debug;
//~^ ERROR: could not find defining uses

trait Foo<A> {}

impl Foo<()> for () {}
impl Foo<u32> for () {}

fn foo() -> impl Foo<FooX> {
    //~^ ERROR: the trait bound `(): Foo<impl Debug>` is not satisfied [E0277]
    ()
}

fn main() {}
