#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

use std::fmt::Debug;

type FooX = impl Debug;

trait Foo<A> { }

impl Foo<()> for () { }

fn foo() -> impl Foo<FooX> {
    //~^ ERROR: the trait bound `(): Foo<FooX>` is not satisfied
    // FIXME(type-alias-impl-trait): We could probably make this work.
    ()
}

fn main() { }
