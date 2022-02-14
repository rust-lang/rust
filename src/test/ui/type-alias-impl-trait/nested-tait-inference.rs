#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

use std::fmt::Debug;

type FooX = impl Debug;

trait Foo<A> { }

impl Foo<()> for () { }

fn foo() -> impl Foo<FooX> {
    // FIXME(type-alias-impl-trait): We could probably make this work.
    ()
    //~^ ERROR: the trait bound `(): Foo<FooX>` is not satisfied
}

fn main() { }
