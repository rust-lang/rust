#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

use std::fmt::Debug;

type FooX = impl Debug;
//~^ ERROR unconstrained opaque type

trait Foo<A> { }

impl Foo<FooX> for () { }

fn foo() -> impl Foo<FooX> {
    ()
}

fn main() { }
