#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

use std::fmt::Debug;

type FooX = impl Debug;

trait Foo<A> {}

impl Foo<FooX> for () {}

#[define_opaque(FooX)]
fn foo() -> impl Foo<FooX> {
    //~^ ERROR: item does not constrain
    ()
}

fn main() {}
