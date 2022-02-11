// check-pass

#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

use std::fmt::Debug;

type FooX = impl Debug;

trait Foo<A> { }

impl Foo<()> for () { }

fn foo() -> impl Foo<FooX> {
    ()
}

fn main() { }
