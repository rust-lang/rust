#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

use std::fmt::Debug;

type FooX = impl Debug;

trait Foo<A> { }

impl Foo<FooX> for () { }
//~^ cannot implement trait on type alias impl trait

fn foo() -> impl Foo<FooX> {
    ()
}

fn main() { }
