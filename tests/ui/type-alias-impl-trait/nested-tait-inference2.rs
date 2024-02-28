#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

use std::fmt::Debug;

type FooX = impl Debug;

trait Foo<A> {}

impl Foo<()> for () {}
impl Foo<u32> for () {}

fn foo() -> impl Foo<FooX> {
    //~^ ERROR trait `Foo<FooX>` is not implemented for `()`
    ()
}

fn main() {}
