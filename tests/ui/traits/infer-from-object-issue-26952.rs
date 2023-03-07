// run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
// Test that when we match a trait reference like `Foo<A>: Foo<_#0t>`,
// we unify with `_#0t` with `A`. In this code, if we failed to do
// that, then you get an unconstrained type-variable in `call`.
//
// Also serves as a regression test for issue #26952, though the test
// was derived from another reported regression with the same cause.

use std::marker::PhantomData;

trait Trait<A> { fn foo(&self); }

struct Type<A> { a: PhantomData<A> }

fn as_trait<A>(t: &Type<A>) -> &dyn Trait<A> { loop {  } }

fn want<A,T:Trait<A>+?Sized>(t: &T) { }

fn call<A>(p: Type<A>) {
    let q = as_trait(&p);
    want(q); // parameter A to `want` *would* be unconstrained
}

fn main() { }
