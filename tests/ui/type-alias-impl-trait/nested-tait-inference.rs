#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] check-pass

use std::fmt::Debug;

type FooX = impl Debug;

trait Foo<A> {}

impl Foo<()> for () {}

#[define_opaque(FooX)]
fn foo() -> impl Foo<FooX> {
    //[current]~^ ERROR: the trait bound `(): Foo<FooX>` is not satisfied
    // FIXME(type-alias-impl-trait): We could probably make this work.
    ()
}

fn main() {}
